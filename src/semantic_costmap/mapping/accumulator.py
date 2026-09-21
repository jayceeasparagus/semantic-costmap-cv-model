"""Accumulate local semantic evidence in a persistent map frame."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from semantic_costmap.config import BACKGROUND_CLASS_ID, NUM_CLASSES, class_costs

if TYPE_CHECKING:
    from semantic_costmap.costmap import SemanticCostmap


UNKNOWN_COST = 255


DYNAMIC_CLASS_ID = 3


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class GlobalMapConfig:
    resolution: float = 0.20
    x_min: float = -100.0
    x_max: float = 100.0
    y_min: float = -100.0
    y_max: float = 100.0
    dynamic_decay_seconds: float = 2.0
    map_gap_fill_iterations: int = 1
    map_gap_fill_min_neighbors: int = 5

    @property
    def width(self) -> int:
        return int(round((self.x_max - self.x_min) / self.resolution))

    @property
    def height(self) -> int:
        return int(round((self.y_max - self.y_min) / self.resolution))


class PoseAwareAccumulator:
    """Maintain static costs and a separately decaying dynamic layer."""

    def __init__(self, config: GlobalMapConfig | None = None) -> None:
        self.config = config or GlobalMapConfig()
        if self.config.resolution <= 0.0:
            raise ValueError("resolution must be positive")
        shape = (self.config.height, self.config.width)
        self.static_costs = np.full(shape, -1, dtype=np.int16)
        self.dynamic_costs = np.full(shape, -1, dtype=np.int16)
        self.static_class_scores = np.zeros(
            (NUM_CLASSES, *shape),
            dtype=np.float32,
        )
        self.static_confidence = np.zeros(shape, dtype=np.float32)
        self.dynamic_confidence = np.zeros(shape, dtype=np.float32)
        self.static_class_ids = np.full(
            shape,
            BACKGROUND_CLASS_ID,
            dtype=np.uint8,
        )
        self.dynamic_class_ids = np.full(
            shape,
            BACKGROUND_CLASS_ID,
            dtype=np.uint8,
        )
        self.dynamic_last_seen = np.full(shape, -np.inf, dtype=np.float64)
        if self.config.dynamic_decay_seconds <= 0.0:
            raise ValueError("dynamic_decay_seconds must be positive")
        if self.config.map_gap_fill_iterations < 0:
            raise ValueError("map_gap_fill_iterations cannot be negative")
        if not 1 <= self.config.map_gap_fill_min_neighbors <= 8:
            raise ValueError("map_gap_fill_min_neighbors must be 1-8")

    def _indices(
        self,
        points_map: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        columns = np.floor(
            (points_map[:, 0] - self.config.x_min) / self.config.resolution
        ).astype(np.int32)
        rows = np.floor(
            (points_map[:, 1] - self.config.y_min) / self.config.resolution
        ).astype(np.int32)
        valid = (
            (columns >= 0)
            & (columns < self.config.width)
            & (rows >= 0)
            & (rows < self.config.height)
        )
        return rows, columns, valid

    def _refresh_static_cells(
        self,
        rows: np.ndarray,
        columns: np.ndarray,
    ) -> None:
        """Recompute labels from confidence-weighted class evidence."""

        if not len(rows):
            return
        unique_cells = np.unique(
            np.column_stack((rows, columns)),
            axis=0,
        )
        rows = unique_cells[:, 0]
        columns = unique_cells[:, 1]
        scores = self.static_class_scores[:, rows, columns]
        total_weight = scores.sum(axis=0)
        valid = total_weight > 1e-6
        if not valid.any():
            return
        scores = scores[:, valid]
        rows = rows[valid]
        columns = columns[valid]
        total_weight = total_weight[valid]
        cost_values = np.asarray(
            class_costs(background_value=0),
            dtype=np.float32,
        )
        winners = (scores + cost_values[:, None] * 1e-6).argmax(axis=0)
        self.static_class_ids[rows, columns] = winners.astype(np.uint8)
        self.static_confidence[rows, columns] = np.clip(
            scores.max(axis=0) / total_weight,
            0.0,
            1.0,
        )

    def update_map_points(
        self,
        points_map: np.ndarray,
        class_ids: np.ndarray,
        costs: np.ndarray,
        timestamp: float,
        confidence: np.ndarray | None = None,
    ) -> None:
        """Insert already transformed semantic points into the global grid."""

        points_map = np.asarray(points_map, dtype=np.float64)
        class_ids = np.asarray(class_ids)
        costs = np.asarray(costs)
        if points_map.ndim != 2 or points_map.shape[1] < 2:
            raise ValueError("points_map must have shape (N, 2+) ")
        if len(points_map) != len(class_ids) or len(points_map) != len(costs):
            raise ValueError("points, class IDs, and costs must have equal length")
        if confidence is None:
            confidence = np.ones(len(points_map), dtype=np.float32)
        confidence = np.asarray(confidence, dtype=np.float32)
        if confidence.ndim != 1 or len(confidence) != len(points_map):
            raise ValueError("confidence must have one value per point")
        if not np.isfinite(confidence).all():
            raise ValueError("confidence must contain finite values")
        confidence = np.clip(confidence, 0.0, 1.0)

        rows, columns, in_map = self._indices(points_map)
        usable = in_map & (costs >= 0) & (class_ids != BACKGROUND_CLASS_ID)
        static = usable & (class_ids != DYNAMIC_CLASS_ID)
        dynamic = usable & (class_ids == DYNAMIC_CLASS_ID)
        static_rows = rows[static]
        static_columns = columns[static]
        static_costs = costs[static].astype(np.int16)
        static_confidence = np.maximum(confidence[static], 0.05)
        np.maximum.at(
            self.static_costs,
            (static_rows, static_columns),
            static_costs,
        )
        np.add.at(
            self.static_class_scores,
            (
                class_ids[static].astype(np.intp),
                static_rows,
                static_columns,
            ),
            static_confidence,
        )
        self._refresh_static_cells(static_rows, static_columns)
        dynamic_rows = rows[dynamic]
        dynamic_columns = columns[dynamic]
        dynamic_costs = costs[dynamic].astype(np.int16)
        dynamic_confidence = np.maximum(confidence[dynamic], 0.05)
        np.maximum.at(
            self.dynamic_costs,
            (dynamic_rows, dynamic_columns),
            dynamic_costs,
        )
        np.maximum.at(
            self.dynamic_confidence,
            (dynamic_rows, dynamic_columns),
            dynamic_confidence,
        )
        dynamic_winners = dynamic_costs >= self.dynamic_costs[
            dynamic_rows,
            dynamic_columns,
        ]
        self.dynamic_class_ids[
            dynamic_rows[dynamic_winners],
            dynamic_columns[dynamic_winners],
        ] = class_ids[dynamic][dynamic_winners].astype(np.uint8)
        np.maximum.at(
            self.dynamic_last_seen,
            (dynamic_rows, dynamic_columns),
            float(timestamp),
        )

    def update_local_costmap(
        self,
        local: "SemanticCostmap",
        pose: Pose2D,
        timestamp: float,
    ) -> None:
        """Transform local grid-cell centers through a 2D map-to-base pose."""

        known_rows, known_columns = np.nonzero(local.costs != UNKNOWN_COST)
        local_x = (
            local.config.x_min
            + (known_columns.astype(np.float64) + 0.5) * local.config.resolution
        )
        local_y = (
            local.config.y_min
            + (known_rows.astype(np.float64) + 0.5) * local.config.resolution
        )
        cosine = np.cos(pose.yaw)
        sine = np.sin(pose.yaw)
        map_x = pose.x + cosine * local_x - sine * local_y
        map_y = pose.y + sine * local_x + cosine * local_y
        map_points = np.column_stack((map_x, map_y))
        classes = local.class_ids[known_rows, known_columns].copy()
        raw_only = (
            local.obstacle_mask[known_rows, known_columns]
            & (classes == BACKGROUND_CLASS_ID)
        )
        classes[raw_only] = 2
        self.update_map_points(
            map_points,
            classes,
            local.costs[known_rows, known_columns],
            timestamp,
            confidence=(
                np.ones(len(known_rows), dtype=np.float32)
                if local.confidence is None
                else local.confidence[known_rows, known_columns]
            ),
        )

    def grid(self, timestamp: float) -> np.ndarray:
        """Return static costs overlaid with non-expired dynamic observations."""

        self._fill_drivable_gaps()
        result = self.static_costs.copy()
        dynamic_active = (
            (float(timestamp) - self.dynamic_last_seen)
            <= self.config.dynamic_decay_seconds
        )
        result[dynamic_active] = np.maximum(
            result[dynamic_active],
            self.dynamic_costs[dynamic_active],
        )
        output = np.full(result.shape, UNKNOWN_COST, dtype=np.uint8)
        known = result >= 0
        output[known] = result[known].astype(np.uint8)
        return output

    def semantic_grid(self, timestamp: float) -> np.ndarray:
        """Return the accumulated semantic class ID at each global cell."""

        self._fill_drivable_gaps()
        result = self.static_class_ids.copy()
        static_known = self.static_costs >= 0
        dynamic_active = (
            (float(timestamp) - self.dynamic_last_seen)
            <= self.config.dynamic_decay_seconds
        ) & (self.dynamic_costs >= 0)
        dynamic_wins = dynamic_active & (
            self.dynamic_costs >= self.static_costs
        )
        result[dynamic_wins] = self.dynamic_class_ids[dynamic_wins]
        result[~(static_known | dynamic_active)] = BACKGROUND_CLASS_ID
        return result

    def semantic_known_mask(self, timestamp: float) -> np.ndarray:
        """Return which global cells contain a non-background observation."""

        dynamic_active = (
            (float(timestamp) - self.dynamic_last_seen)
            <= self.config.dynamic_decay_seconds
        ) & (self.dynamic_costs >= 0)
        return (self.static_costs >= 0) | dynamic_active

    def confidence_grid(self, timestamp: float) -> np.ndarray:
        """Return confidence for the currently visible semantic map."""

        self._fill_drivable_gaps()
        result = self.static_confidence.copy()
        dynamic_active = (
            (float(timestamp) - self.dynamic_last_seen)
            <= self.config.dynamic_decay_seconds
        ) & (self.dynamic_costs >= 0)
        result[dynamic_active] = np.maximum(
            result[dynamic_active],
            self.dynamic_confidence[dynamic_active],
        )
        known = (self.static_costs >= 0) | dynamic_active
        result[~known] = 0.0
        return result

    def _fill_drivable_gaps(self) -> None:
        """Fill small unknown gaps surrounded by static drivable cells."""

        for _ in range(self.config.map_gap_fill_iterations):
            drivable = (self.static_costs == 0) & (
                self.static_class_ids == 0
            )
            padded = np.pad(drivable.astype(np.uint8), 1)
            neighbors = np.zeros(drivable.shape, dtype=np.uint8)
            for row_offset in range(3):
                for column_offset in range(3):
                    if row_offset == 1 and column_offset == 1:
                        continue
                    neighbors += padded[
                        row_offset : row_offset + drivable.shape[0],
                        column_offset : column_offset + drivable.shape[1],
                    ]
            fill = (
                (self.static_costs < 0)
                & (neighbors >= self.config.map_gap_fill_min_neighbors)
            )
            if not fill.any():
                break
            self.static_costs[fill] = 0
            self.static_class_ids[fill] = 0
            self.static_confidence[fill] = 0.5
