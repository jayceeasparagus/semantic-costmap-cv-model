import numpy as np

from semantic_costmap.costmap import CostmapConfig, build_semantic_costmap
from semantic_costmap.costmap.grid import LETHAL_COST, UNKNOWN_COST
from semantic_costmap.fusion import PaintedPointCloud


def make_cloud(points, probabilities, confidence=None):
    points = np.asarray(points, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    class_ids = probabilities.argmax(axis=1).astype(np.uint8)
    if confidence is None:
        confidence = probabilities.max(axis=1)
    return PaintedPointCloud(
        points_camera=points.copy(),
        points_vehicle=points,
        probabilities=probabilities,
        class_ids=class_ids,
        confidence=np.asarray(confidence, dtype=np.float32),
        costs=np.zeros(len(points), dtype=np.int16),
        rows=np.zeros(len(points), dtype=np.int32),
        columns=np.zeros(len(points), dtype=np.int32),
        source_indices=np.arange(len(points)),
    )


def test_costmap_places_semantic_cost_in_metric_cell():
    config = CostmapConfig(
        resolution=1.0,
        x_min=0.0,
        x_max=4.0,
        y_min=-2.0,
        y_max=2.0,
        obstacle_z_min=5.0,
        obstacle_z_max=6.0,
    )
    cloud = make_cloud(
        [[1.2, -0.2, -1.0], [1.3, -0.1, -1.0]],
        [[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]],
    )

    result = build_semantic_costmap(cloud, config)

    assert result.costs.shape == (4, 4)
    assert result.costs[1, 1] == 220
    assert result.class_ids[1, 1] == 1
    assert result.evidence_count[1, 1] == 2
    assert result.costs[0, 0] == UNKNOWN_COST


def test_raw_lidar_obstacle_never_gets_erased_by_drivable_semantics():
    config = CostmapConfig(
        resolution=1.0,
        x_min=0.0,
        x_max=3.0,
        y_min=-1.0,
        y_max=1.0,
    )
    cloud = make_cloud(
        [[1.2, 0.2, -1.0]],
        [[1.0, 0.0, 0.0, 0.0, 0.0]],
    )
    raw_obstacle = np.array([[1.5, 0.4, 0.2]])

    result = build_semantic_costmap(cloud, config, raw_obstacle)

    assert result.costs[1, 1] == LETHAL_COST
    assert result.obstacle_mask[1, 1]


def test_background_and_low_confidence_leave_unknown_cells():
    config = CostmapConfig(
        resolution=1.0,
        x_min=0.0,
        x_max=3.0,
        y_min=-1.0,
        y_max=1.0,
        obstacle_z_min=5.0,
        obstacle_z_max=6.0,
    )
    cloud = make_cloud(
        [[0.2, 0.2, -1.0], [1.2, 0.2, -1.0]],
        [[0.0, 0.0, 0.0, 0.0, 1.0], [0.4, 0.3, 0.2, 0.1, 0.0]],
        confidence=[1.0, 0.4],
    )

    result = build_semantic_costmap(cloud, config)

    assert np.all(result.costs == UNKNOWN_COST)
