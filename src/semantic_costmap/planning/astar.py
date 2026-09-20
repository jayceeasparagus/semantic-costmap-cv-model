"""A small local A* planner for vehicle-relative semantic costmaps."""

from dataclasses import dataclass
from heapq import heappop, heappush
import math

import numpy as np

from semantic_costmap.costmap.grid import LETHAL_COST, UNKNOWN_COST, SemanticCostmap


@dataclass(frozen=True)
class RoutePlannerConfig:
    """Parameters for choosing a safe forward goal in one frame."""

    goal_forward_m: float = 12.0
    minimum_goal_forward_m: float = 5.0
    goal_lateral_tolerance_m: float = 5.0
    start_forward_search_m: float = 1.0
    start_lateral_search_m: float = 1.5
    unknown_penalty: float = 8.0
    semantic_cost_scale: float = 0.01
    goal_search_step_m: float = 1.0

    def validate(self) -> None:
        if self.goal_forward_m <= 0.0:
            raise ValueError("goal_forward_m must be positive")
        if not 0.0 < self.minimum_goal_forward_m <= self.goal_forward_m:
            raise ValueError(
                "minimum_goal_forward_m must be in (0, goal_forward_m]"
            )
        if self.goal_lateral_tolerance_m <= 0.0:
            raise ValueError("goal_lateral_tolerance_m must be positive")
        if self.start_forward_search_m < 0.0:
            raise ValueError("start_forward_search_m cannot be negative")
        if self.start_lateral_search_m < 0.0:
            raise ValueError("start_lateral_search_m cannot be negative")
        if self.unknown_penalty <= 1.0:
            raise ValueError("unknown_penalty must be greater than one")
        if self.semantic_cost_scale < 0.0:
            raise ValueError("semantic_cost_scale cannot be negative")
        if self.goal_search_step_m <= 0.0:
            raise ValueError("goal_search_step_m must be positive")


@dataclass(frozen=True)
class RouteResult:
    """A planned route expressed in grid cells and vehicle coordinates."""

    found: bool
    cells: np.ndarray
    points_vehicle: np.ndarray
    start_cell: tuple[int, int] | None
    goal_cell: tuple[int, int] | None
    path_cost: float
    expanded_nodes: int
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly route diagnostics for playback logs."""

        return {
            "found": self.found,
            "start_cell": self.start_cell,
            "goal_cell": self.goal_cell,
            "path_cost": self.path_cost,
            "expanded_nodes": self.expanded_nodes,
            "point_count": int(len(self.points_vehicle)),
            "reason": self.reason,
        }


class LocalRoutePlanner:
    """Plan a short route from the vehicle to a safe forward goal."""

    _NEIGHBORS = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )

    def __init__(self, config: RoutePlannerConfig | None = None) -> None:
        self.config = config or RoutePlannerConfig()
        self.config.validate()

    def plan(self, costmap: SemanticCostmap) -> RouteResult:
        """Plan one route using only the supplied frame's costmap."""

        costs = np.asarray(costmap.costs)
        if costs.ndim != 2:
            raise ValueError("costmap costs must be two-dimensional")
        if costs.shape != (costmap.config.height, costmap.config.width):
            raise ValueError("costmap shape does not match its configuration")

        start = self._find_start(costs, costmap)
        if start is None:
            return self._failure("no traversable start cell")

        for goal in self._goal_candidates(costs, costmap):
            cells, path_cost, expanded = self._astar(costs, start, goal)
            if cells is not None:
                points = self._cells_to_vehicle(cells, costmap)
                return RouteResult(
                    found=True,
                    cells=cells,
                    points_vehicle=points,
                    start_cell=start,
                    goal_cell=goal,
                    path_cost=path_cost,
                    expanded_nodes=expanded,
                )

        return RouteResult(
            found=False,
            cells=np.empty((0, 2), dtype=np.int32),
            points_vehicle=np.empty((0, 2), dtype=np.float64),
            start_cell=start,
            goal_cell=None,
            path_cost=math.inf,
            expanded_nodes=0,
            reason="no traversable forward goal was reachable",
        )

    def _failure(self, reason: str) -> RouteResult:
        return RouteResult(
            found=False,
            cells=np.empty((0, 2), dtype=np.int32),
            points_vehicle=np.empty((0, 2), dtype=np.float64),
            start_cell=None,
            goal_cell=None,
            path_cost=math.inf,
            expanded_nodes=0,
            reason=reason,
        )

    def _find_start(
        self,
        costs: np.ndarray,
        costmap: SemanticCostmap,
    ) -> tuple[int, int] | None:
        center_row = self._vehicle_to_cell(0.0, 0.0, costmap)
        max_column = self._vehicle_to_cell(
            self.config.start_forward_search_m,
            0.0,
            costmap,
        )[1]
        max_row_delta = int(
            round(self.config.start_lateral_search_m / costmap.config.resolution)
        )
        candidates = []
        for row in range(
            max(0, center_row[0] - max_row_delta),
            min(costs.shape[0], center_row[0] + max_row_delta + 1),
        ):
            for column in range(
                max(0, center_row[1]),
                min(costs.shape[1], max_column + 1),
            ):
                if not self._blocked(costs[row, column]):
                    distance = abs(row - center_row[0]) + abs(column - center_row[1])
                    candidates.append((distance, row, column))
        if not candidates:
            return None
        _, row, column = min(candidates)
        return int(row), int(column)

    def _goal_candidates(
        self,
        costs: np.ndarray,
        costmap: SemanticCostmap,
    ) -> list[tuple[int, int]]:
        target_column = self._vehicle_to_cell(
            self.config.goal_forward_m,
            0.0,
            costmap,
        )[1]
        minimum_column = self._vehicle_to_cell(
            self.config.minimum_goal_forward_m,
            0.0,
            costmap,
        )[1]
        target_column = min(costs.shape[1] - 1, max(0, target_column))
        minimum_column = min(costs.shape[1] - 1, max(0, minimum_column))
        center_row = self._vehicle_to_cell(0.0, 0.0, costmap)[0]
        lateral_delta = int(
            round(self.config.goal_lateral_tolerance_m / costmap.config.resolution)
        )
        rows = range(
            max(0, center_row - lateral_delta),
            min(costs.shape[0], center_row + lateral_delta + 1),
        )

        candidates = []
        step = max(
            1,
            int(round(self.config.goal_search_step_m / costmap.config.resolution)),
        )
        for column in range(target_column, minimum_column - 1, -step):
            row_candidates = [
                row for row in rows if not self._blocked(costs[row, column])
            ]
            row_candidates.sort(key=lambda row: (abs(row - center_row), costs[row, column]))
            candidates.extend((row, column) for row in row_candidates)
        return candidates

    def _astar(
        self,
        costs: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> tuple[np.ndarray | None, float, int]:
        if self._blocked(costs[start]) or self._blocked(costs[goal]):
            return None, math.inf, 0
        frontier = [(self._heuristic(start, goal), 0.0, start)]
        distances = {start: 0.0}
        parents: dict[tuple[int, int], tuple[int, int]] = {}
        expanded = 0
        while frontier:
            _, distance, current = heappop(frontier)
            if distance != distances.get(current):
                continue
            expanded += 1
            if current == goal:
                return self._reconstruct(parents, current), distance, expanded
            for row_delta, column_delta, step_distance in self._NEIGHBORS:
                neighbor = (current[0] + row_delta, current[1] + column_delta)
                if not self._in_bounds(neighbor, costs.shape) or self._blocked(costs[neighbor]):
                    continue
                if row_delta and column_delta:
                    if self._blocked(costs[current[0], neighbor[1]]) or self._blocked(
                        costs[neighbor[0], current[1]]
                    ):
                        continue
                next_distance = distance + step_distance * self._cell_cost(
                    costs[neighbor]
                )
                if next_distance >= distances.get(neighbor, math.inf):
                    continue
                distances[neighbor] = next_distance
                parents[neighbor] = current
                priority = next_distance + self._heuristic(neighbor, goal)
                heappush(frontier, (priority, next_distance, neighbor))
        return None, math.inf, expanded

    @staticmethod
    def _reconstruct(
        parents: dict[tuple[int, int], tuple[int, int]],
        current: tuple[int, int],
    ) -> np.ndarray:
        path = [current]
        while current in parents:
            current = parents[current]
            path.append(current)
        path.reverse()
        return np.asarray(path, dtype=np.int32)

    def _cells_to_vehicle(
        self,
        cells: np.ndarray,
        costmap: SemanticCostmap,
    ) -> np.ndarray:
        rows = cells[:, 0].astype(np.float64)
        columns = cells[:, 1].astype(np.float64)
        x = costmap.config.x_min + (columns + 0.5) * costmap.config.resolution
        y = costmap.config.y_min + (rows + 0.5) * costmap.config.resolution
        return np.column_stack((x, y))

    @staticmethod
    def _vehicle_to_cell(
        x: float,
        y: float,
        costmap: SemanticCostmap,
    ) -> tuple[int, int]:
        column = int(math.floor((x - costmap.config.x_min) / costmap.config.resolution))
        row = int(math.floor((y - costmap.config.y_min) / costmap.config.resolution))
        return row, column

    @staticmethod
    def _in_bounds(cell: tuple[int, int], shape: tuple[int, int]) -> bool:
        return 0 <= cell[0] < shape[0] and 0 <= cell[1] < shape[1]

    @staticmethod
    def _blocked(value: int) -> bool:
        return int(value) == LETHAL_COST

    def _cell_cost(self, value: int) -> float:
        value = int(value)
        if value == UNKNOWN_COST:
            return self.config.unknown_penalty
        return 1.0 + value * self.config.semantic_cost_scale

    @staticmethod
    def _heuristic(
        current: tuple[int, int],
        goal: tuple[int, int],
    ) -> float:
        return math.hypot(goal[0] - current[0], goal[1] - current[1])
