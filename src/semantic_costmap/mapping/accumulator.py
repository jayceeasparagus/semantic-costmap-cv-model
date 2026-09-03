"""Accumulate local semantic evidence in a persistent map frame."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from semantic_costmap.config import BACKGROUND_CLASS_ID

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

    def update_map_points(
        self,
        points_map: np.ndarray,
        class_ids: np.ndarray,
        costs: np.ndarray,
        timestamp: float,
    ) -> None:
        """Insert already transformed semantic points into the global grid."""

        points_map = np.asarray(points_map, dtype=np.float64)
        class_ids = np.asarray(class_ids)
        costs = np.asarray(costs)
        if points_map.ndim != 2 or points_map.shape[1] < 2:
            raise ValueError("points_map must have shape (N, 2+) ")
        if len(points_map) != len(class_ids) or len(points_map) != len(costs):
            raise ValueError("points, class IDs, and costs must have equal length")

        rows, columns, in_map = self._indices(points_map)
        usable = in_map & (costs >= 0) & (class_ids != BACKGROUND_CLASS_ID)
        static = usable & (class_ids != DYNAMIC_CLASS_ID)
        dynamic = usable & (class_ids == DYNAMIC_CLASS_ID)
        static_rows = rows[static]
        static_columns = columns[static]
        static_costs = costs[static].astype(np.int16)
        np.maximum.at(
            self.static_costs,
            (static_rows, static_columns),
            static_costs,
        )
        static_winners = static_costs >= self.static_costs[
            static_rows,
            static_columns,
        ]
        self.static_class_ids[
            static_rows[static_winners],
            static_columns[static_winners],
        ] = class_ids[static][static_winners].astype(np.uint8)
        dynamic_rows = rows[dynamic]
        dynamic_columns = columns[dynamic]
        dynamic_costs = costs[dynamic].astype(np.int16)
        np.maximum.at(
            self.dynamic_costs,
            (dynamic_rows, dynamic_columns),
            dynamic_costs,
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
        )

    def grid(self, timestamp: float) -> np.ndarray:
        """Return static costs overlaid with non-expired dynamic observations."""

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
