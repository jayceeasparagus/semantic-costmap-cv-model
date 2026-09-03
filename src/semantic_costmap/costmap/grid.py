"""Rasterize painted 3D points into a vehicle-relative navigation grid."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from semantic_costmap.config import (
    BACKGROUND_CLASS_ID,
    NUM_CLASSES,
    class_colors,
    class_costs,
)
from semantic_costmap.fusion import PaintedPointCloud


UNKNOWN_COST = 255
LETHAL_COST = 254


@dataclass(frozen=True)
class CostmapConfig:
    resolution: float = 0.20
    x_min: float = 0.0
    x_max: float = 50.0
    y_min: float = -20.0
    y_max: float = 20.0
    semantic_z_min: float = -1.5
    semantic_z_max: float = 2.0
    obstacle_z_min: float = -0.45
    obstacle_z_max: float = 1.5
    minimum_confidence: float = 0.50
    minimum_points_per_cell: int = 1
    raytrace_free_space: bool = True
    raytrace_max_range: float = 50.0
    ground_interpolation_iterations: int = 2
    ground_interpolation_min_neighbors: int = 3
    obstacle_marking_radius_m: float = 0.30

    @property
    def width(self) -> int:
        return int(round((self.x_max - self.x_min) / self.resolution))

    @property
    def height(self) -> int:
        return int(round((self.y_max - self.y_min) / self.resolution))

    def validate(self) -> None:
        if self.resolution <= 0.0:
            raise ValueError("resolution must be positive")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("grid maximums must exceed minimums")
        if self.minimum_points_per_cell < 1:
            raise ValueError("minimum_points_per_cell must be at least one")
        if self.raytrace_max_range <= 0.0:
            raise ValueError("raytrace_max_range must be positive")
        if self.ground_interpolation_iterations < 0:
            raise ValueError("ground_interpolation_iterations cannot be negative")
        if not 1 <= self.ground_interpolation_min_neighbors <= 8:
            raise ValueError("ground_interpolation_min_neighbors must be 1-8")
        if self.obstacle_marking_radius_m < 0.0:
            raise ValueError("obstacle_marking_radius_m cannot be negative")


@dataclass(frozen=True)
class SemanticCostmap:
    costs: np.ndarray
    class_ids: np.ndarray
    evidence_count: np.ndarray
    obstacle_mask: np.ndarray
    config: CostmapConfig
    semantic_mask: np.ndarray | None = None
    interpolated_free_mask: np.ndarray | None = None
    observed_free_mask: np.ndarray | None = None
    obstacle_seed_mask: np.ndarray | None = None


def _cell_indices(
    points: np.ndarray,
    config: CostmapConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = np.floor(
        (points[:, 0] - config.x_min) / config.resolution
    ).astype(np.int32)
    rows = np.floor(
        (points[:, 1] - config.y_min) / config.resolution
    ).astype(np.int32)
    valid = (
        (columns >= 0)
        & (columns < config.width)
        & (rows >= 0)
        & (rows < config.height)
    )
    return rows, columns, valid


def _neighbor_count(mask: np.ndarray) -> np.ndarray:
    """Count occupied cells in each cell's eight-connected neighborhood."""

    padded = np.pad(mask.astype(np.uint8), 1)
    counts = np.zeros(mask.shape, dtype=np.uint8)
    for row_offset in range(3):
        for column_offset in range(3):
            if row_offset == 1 and column_offset == 1:
                continue
            counts += padded[
                row_offset : row_offset + mask.shape[0],
                column_offset : column_offset + mask.shape[1],
            ]
    return counts


def _interpolate_drivable_cells(
    costs: np.ndarray,
    class_ids: np.ndarray,
    iterations: int,
    minimum_neighbors: int,
) -> None:
    """Fill only small unknown gaps surrounded by observed drivable cells."""

    drivable = (class_ids == 0) & (costs != UNKNOWN_COST)
    for _ in range(iterations):
        fill = (
            (costs == UNKNOWN_COST)
            & (_neighbor_count(drivable) >= minimum_neighbors)
        )
        if not fill.any():
            break
        costs[fill] = 0
        class_ids[fill] = 0
        drivable |= fill


def _raytraced_free_mask(
    raw_points_vehicle: np.ndarray,
    config: CostmapConfig,
    origin_vehicle: np.ndarray,
) -> np.ndarray:
    """Mark cells traversed before each LiDAR return as observed free space."""

    mask = np.zeros((config.height, config.width), dtype=bool)
    origin_vehicle = np.asarray(origin_vehicle, dtype=np.float64)
    if origin_vehicle.shape != (3,) or not np.isfinite(origin_vehicle).all():
        raise ValueError("origin_vehicle must contain three finite values")
    origin = origin_vehicle.reshape(1, 3)
    start_rows, start_columns, origin_valid = _cell_indices(origin, config)
    if not origin_valid[0]:
        return mask

    endpoints = np.asarray(raw_points_vehicle[:, :2], dtype=np.float64)
    finite = np.isfinite(endpoints).all(axis=1)
    endpoints = endpoints[finite]
    origin_xy = origin_vehicle[:2]
    directions = endpoints - origin_xy
    distances = np.linalg.norm(directions, axis=1)
    usable = distances > 1e-9
    directions = directions[usable]
    distances = distances[usable]
    endpoints = origin_xy + directions * np.minimum(
        1.0, config.raytrace_max_range / distances
    )[:, None]

    # Clip all rays to the rectangular grid from the calibrated origin.
    clipping_scale = np.ones(len(endpoints), dtype=np.float64)
    epsilon = config.resolution * 1e-6
    limits = (
        (0, config.x_min + epsilon, config.x_max - epsilon),
        (1, config.y_min + epsilon, config.y_max - epsilon),
    )
    for axis, lower, upper in limits:
        component = directions[:, axis]
        positive = component > 0.0
        negative = component < 0.0
        clipping_scale[positive] = np.minimum(
            clipping_scale[positive],
            (upper - origin_xy[axis]) / component[positive],
        )
        clipping_scale[negative] = np.minimum(
            clipping_scale[negative],
            (lower - origin_xy[axis]) / component[negative],
        )
    usable = clipping_scale > 0.0
    endpoints = origin_xy + directions[usable] * clipping_scale[usable, None]

    endpoint_xyz = np.column_stack((endpoints, np.zeros(len(endpoints))))
    end_rows, end_columns, endpoint_valid = _cell_indices(endpoint_xyz, config)
    endpoint_cells = np.unique(
        np.column_stack((end_rows[endpoint_valid], end_columns[endpoint_valid])),
        axis=0,
    )
    if not len(endpoint_cells):
        return mask

    start_row = int(start_rows[0])
    start_column = int(start_columns[0])
    row_delta = endpoint_cells[:, 0] - start_row
    column_delta = endpoint_cells[:, 1] - start_column
    step_count = np.maximum(np.abs(row_delta), np.abs(column_delta))

    # Vectorized digital differential analyzer: each loop advances every ray.
    for step in range(int(step_count.max())):
        active = step < step_count
        fraction = step / step_count[active]
        rows = np.rint(start_row + row_delta[active] * fraction).astype(np.int32)
        columns = np.rint(
            start_column + column_delta[active] * fraction
        ).astype(np.int32)
        mask[rows, columns] = True
    return mask


def build_semantic_costmap(
    painted: PaintedPointCloud,
    config: CostmapConfig | None = None,
    raw_points_vehicle: np.ndarray | None = None,
    raytrace_origin_vehicle: np.ndarray | None = None,
) -> SemanticCostmap:
    """Aggregate semantic probabilities and conservative LiDAR obstacles."""

    config = config or CostmapConfig()
    config.validate()
    shape = (config.height, config.width)
    evidence = np.zeros(shape, dtype=np.uint16)
    class_sums = np.zeros((BACKGROUND_CLASS_ID, *shape), dtype=np.float64)

    points = painted.points_vehicle
    rows, columns, in_grid = _cell_indices(points, config)
    semantic_valid = (
        in_grid
        & (points[:, 2] >= config.semantic_z_min)
        & (points[:, 2] <= config.semantic_z_max)
        & (painted.confidence >= config.minimum_confidence)
        & (painted.class_ids != BACKGROUND_CLASS_ID)
    )

    semantic_rows = rows[semantic_valid]
    semantic_columns = columns[semantic_valid]
    navigation_probabilities = painted.probabilities[
        semantic_valid, :BACKGROUND_CLASS_ID
    ].astype(np.float64)
    probability_totals = navigation_probabilities.sum(axis=1, keepdims=True)
    navigation_probabilities = np.divide(
        navigation_probabilities,
        probability_totals,
        out=np.zeros_like(navigation_probabilities),
        where=probability_totals > 1e-12,
    )

    np.add.at(evidence, (semantic_rows, semantic_columns), 1)
    for class_id in range(BACKGROUND_CLASS_ID):
        np.add.at(
            class_sums[class_id],
            (semantic_rows, semantic_columns),
            navigation_probabilities[:, class_id],
        )

    populated = evidence >= config.minimum_points_per_cell
    semantic_mask = populated.copy()
    costs = np.full(shape, UNKNOWN_COST, dtype=np.uint8)
    class_ids = np.full(shape, BACKGROUND_CLASS_ID, dtype=np.uint8)
    if populated.any():
        class_ids[populated] = class_sums[:, populated].argmax(axis=0)
        semantic_cost_values = np.asarray(
            class_costs()[:BACKGROUND_CLASS_ID],
            dtype=np.float64,
        )
        total_weight = class_sums[:, populated].sum(axis=0)
        weighted_cost = (
            semantic_cost_values[:, None] * class_sums[:, populated]
        ).sum(axis=0) / total_weight
        costs[populated] = np.rint(weighted_cost).astype(np.uint8)

    if config.ground_interpolation_iterations > 0:
        before_interpolation = costs == UNKNOWN_COST
        _interpolate_drivable_cells(
            costs,
            class_ids,
            config.ground_interpolation_iterations,
            config.ground_interpolation_min_neighbors,
        )
        interpolated_free_mask = before_interpolation & (costs == 0)
    else:
        interpolated_free_mask = np.zeros(shape, dtype=bool)

    if raw_points_vehicle is None:
        raw_points_vehicle = points
    raw_points_vehicle = np.asarray(raw_points_vehicle, dtype=np.float64)
    if raw_points_vehicle.ndim != 2 or raw_points_vehicle.shape[1] != 3:
        raise ValueError("raw_points_vehicle must have shape (N, 3)")

    if raytrace_origin_vehicle is None:
        raytrace_origin_vehicle = np.zeros(3, dtype=np.float64)
    if config.raytrace_free_space:
        free_mask = _raytraced_free_mask(
            raw_points_vehicle,
            config,
            raytrace_origin_vehicle,
        )
        newly_observed_free = free_mask & (costs == UNKNOWN_COST)
        costs[newly_observed_free] = 0
        observed_free_mask = newly_observed_free
    else:
        observed_free_mask = np.zeros(shape, dtype=bool)

    obstacle_rows, obstacle_columns, obstacle_in_grid = _cell_indices(
        raw_points_vehicle,
        config,
    )
    obstacle_points = (
        obstacle_in_grid
        & (raw_points_vehicle[:, 2] >= config.obstacle_z_min)
        & (raw_points_vehicle[:, 2] <= config.obstacle_z_max)
    )
    obstacle_seed_mask = np.zeros(shape, dtype=bool)
    obstacle_seed_mask[
        obstacle_rows[obstacle_points],
        obstacle_columns[obstacle_points],
    ] = True
    obstacle_mask = _dilate_mask(
        obstacle_seed_mask,
        config.obstacle_marking_radius_m,
        config.resolution,
    )
    costs[obstacle_mask] = LETHAL_COST

    return SemanticCostmap(
        costs=costs,
        class_ids=class_ids,
        evidence_count=evidence,
        obstacle_mask=obstacle_mask,
        config=config,
        semantic_mask=semantic_mask,
        interpolated_free_mask=interpolated_free_mask,
        observed_free_mask=observed_free_mask,
        obstacle_seed_mask=obstacle_seed_mask,
    )


def _dilate_mask(
    mask: np.ndarray,
    radius_m: float,
    resolution: float,
) -> np.ndarray:
    """Expand occupied cells by a circular metric radius."""

    mask = np.asarray(mask, dtype=bool)
    if radius_m <= 0.0 or not mask.any():
        return mask.copy()
    radius_cells = int(np.ceil(radius_m / resolution))
    offsets = [
        (row, column)
        for row in range(-radius_cells, radius_cells + 1)
        for column in range(-radius_cells, radius_cells + 1)
        if (row * row + column * column) ** 0.5 <= radius_m / resolution
    ]
    expanded = np.zeros_like(mask)
    source_rows, source_columns = np.nonzero(mask)
    for row_offset, column_offset in offsets:
        rows = source_rows + row_offset
        columns = source_columns + column_offset
        valid = (
            (rows >= 0)
            & (rows < mask.shape[0])
            & (columns >= 0)
            & (columns < mask.shape[1])
        )
        expanded[rows[valid], columns[valid]] = True
    return expanded


def costmap_to_rgb(
    costs: np.ndarray,
    observed_free_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a raw cost grid into a human-readable RGB image."""
    image = np.empty((*costs.shape, 3), dtype=np.uint8)
    unknown = costs == UNKNOWN_COST
    lethal = costs == LETHAL_COST
    known = ~(unknown | lethal)

    image[unknown] = (70, 70, 70)
    image[lethal] = (220, 0, 0)
    if known.any():
        fraction = costs[known].astype(np.float32) / LETHAL_COST
        colors = np.zeros((known.sum(), 3), dtype=np.uint8)
        colors[:, 0] = (255 * fraction).astype(np.uint8)
        colors[:, 1] = (200 * (1.0 - fraction)).astype(np.uint8)
        image[known] = colors
    if observed_free_mask is not None:
        observed_free_mask = np.asarray(observed_free_mask, dtype=bool)
        if observed_free_mask.shape != costs.shape:
            raise ValueError("observed_free_mask must match costs shape")
        image[observed_free_mask & (costs == 0)] = (0, 150, 220)
    return np.flipud(image)


def semantic_map_to_rgb(
    class_ids: np.ndarray,
    known_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Render semantic class IDs with the project class palette."""

    class_ids = np.asarray(class_ids)
    if class_ids.ndim != 2:
        raise ValueError("class_ids must be a two-dimensional array")
    if class_ids.size and (
        class_ids.min() < 0 or class_ids.max() >= NUM_CLASSES
    ):
        raise ValueError("class_ids contains an unsupported class")
    image = np.asarray(class_colors(), dtype=np.uint8)[class_ids]
    if known_mask is not None:
        known_mask = np.asarray(known_mask, dtype=bool)
        if known_mask.shape != class_ids.shape:
            raise ValueError("known_mask must match class_ids")
        image[~known_mask] = (70, 70, 70)
    return np.flipud(image)


def save_costmap(
    output_directory: str | Path,
    costmap: SemanticCostmap,
) -> dict[str, Path]:
    """Save arrays, a Nav2-map-server file, metadata, and a debug image."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    npz_path = output_directory / "semantic_costmap.npz"
    pgm_path = output_directory / "semantic_costmap.pgm"
    yaml_path = output_directory / "semantic_costmap.yaml"
    preview_path = output_directory / "semantic_costmap_preview.png"

    np.savez_compressed(
        npz_path,
        costs=costmap.costs,
        class_ids=costmap.class_ids,
        evidence_count=costmap.evidence_count,
        obstacle_mask=costmap.obstacle_mask,
        semantic_mask=costmap.semantic_mask,
        interpolated_free_mask=costmap.interpolated_free_mask,
        observed_free_mask=costmap.observed_free_mask,
        obstacle_seed_mask=costmap.obstacle_seed_mask,
        **asdict(costmap.config),
    )
    Image.fromarray(np.flipud(costmap.costs), mode="L").save(pgm_path)
    Image.fromarray(
        costmap_to_rgb(costmap.costs, costmap.observed_free_mask),
        mode="RGB",
    ).save(preview_path)
    metadata = {
        "image": pgm_path.name,
        "mode": "raw",
        "resolution": costmap.config.resolution,
        "origin": [costmap.config.x_min, costmap.config.y_min, 0.0],
        "negate": 0,
        "occupied_thresh": 0.99,
        "free_thresh": 0.01,
    }
    yaml_path.write_text(yaml.safe_dump(metadata, sort_keys=False))
    return {
        "arrays": npz_path,
        "map": pgm_path,
        "metadata": yaml_path,
        "preview": preview_path,
    }
