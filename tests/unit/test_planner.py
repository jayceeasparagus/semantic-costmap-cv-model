import numpy as np

from semantic_costmap.costmap import CostmapConfig, SemanticCostmap
from semantic_costmap.costmap.grid import LETHAL_COST, UNKNOWN_COST
from semantic_costmap.planning import LocalRoutePlanner, RoutePlannerConfig


def make_costmap(costs: np.ndarray) -> SemanticCostmap:
    config = CostmapConfig(
        resolution=1.0,
        x_min=0.0,
        x_max=float(costs.shape[1]),
        y_min=-2.0,
        y_max=2.0,
        ground_interpolation_iterations=0,
    )
    return SemanticCostmap(
        costs=costs.astype(np.uint8),
        class_ids=np.zeros_like(costs, dtype=np.uint8),
        evidence_count=np.ones_like(costs, dtype=np.uint16),
        obstacle_mask=costs == LETHAL_COST,
        config=config,
    )


def test_local_route_finds_a_path_around_an_obstacle() -> None:
    costs = np.zeros((4, 8), dtype=np.uint8)
    costs[0:3, 3] = LETHAL_COST
    route = LocalRoutePlanner(
        RoutePlannerConfig(
            goal_forward_m=6.0,
            minimum_goal_forward_m=5.0,
            goal_lateral_tolerance_m=2.0,
        )
    ).plan(make_costmap(costs))

    assert route.found
    assert len(route.points_vehicle) > 1
    assert not any(tuple(cell) in {(0, 3), (1, 3), (2, 3)} for cell in route.cells)
    assert route.points_vehicle[-1, 0] >= 5.0


def test_local_route_reports_failure_when_a_wall_blocks_the_grid() -> None:
    costs = np.zeros((4, 8), dtype=np.uint8)
    costs[:, 3] = LETHAL_COST
    route = LocalRoutePlanner().plan(make_costmap(costs))

    assert not route.found
    assert route.points_vehicle.shape == (0, 2)
    assert route.reason


def test_unknown_cells_are_penalized_but_traversable() -> None:
    costs = np.zeros((4, 8), dtype=np.uint8)
    costs[1, 1:6] = UNKNOWN_COST
    route = LocalRoutePlanner().plan(make_costmap(costs))

    assert route.found
    assert route.path_cost > 0.0
