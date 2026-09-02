"""Rasterize painted 3D points into a vehicle-relative navigation grid."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from semantic_costmap.config import BACKGROUND_CLASS_ID, class_costs
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


@dataclass(frozen=True)
class SemanticCostmap:
    costs: np.ndarray
    class_ids: np.ndarray
    evidence_count: np.ndarray
    obstacle_mask: np.ndarray
    config: CostmapConfig


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


def build_semantic_costmap(
    painted: PaintedPointCloud,
    config: CostmapConfig | None = None,
    raw_points_vehicle: np.ndarray | None = None,
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

    if raw_points_vehicle is None:
        raw_points_vehicle = points
    raw_points_vehicle = np.asarray(raw_points_vehicle, dtype=np.float64)
    if raw_points_vehicle.ndim != 2 or raw_points_vehicle.shape[1] != 3:
        raise ValueError("raw_points_vehicle must have shape (N, 3)")
    obstacle_rows, obstacle_columns, obstacle_in_grid = _cell_indices(
        raw_points_vehicle,
        config,
    )
    obstacle_points = (
        obstacle_in_grid
        & (raw_points_vehicle[:, 2] >= config.obstacle_z_min)
        & (raw_points_vehicle[:, 2] <= config.obstacle_z_max)
    )
    obstacle_mask = np.zeros(shape, dtype=bool)
    obstacle_mask[
        obstacle_rows[obstacle_points],
        obstacle_columns[obstacle_points],
    ] = True
    costs[obstacle_mask] = LETHAL_COST

    return SemanticCostmap(
        costs=costs,
        class_ids=class_ids,
        evidence_count=evidence,
        obstacle_mask=obstacle_mask,
        config=config,
    )


def costmap_to_rgb(costs: np.ndarray) -> np.ndarray:
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
        **asdict(costmap.config),
    )
    Image.fromarray(np.flipud(costmap.costs), mode="L").save(pgm_path)
    Image.fromarray(costmap_to_rgb(costmap.costs), mode="RGB").save(preview_path)
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
