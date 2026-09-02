import numpy as np

from semantic_costmap.fusion import paint_points
from semantic_costmap.geometry import ProjectionResult
from semantic_costmap.inference import SegmentationResult


def test_paint_points_samples_probabilities_and_costs():
    class_ids = np.array([[0, 1], [2, 4]], dtype=np.uint8)
    probabilities = np.zeros((5, 2, 2), dtype=np.float32)
    for row in range(2):
        for column in range(2):
            probabilities[class_ids[row, column], row, column] = 0.9
    segmentation = SegmentationResult(
        class_ids=class_ids,
        probabilities=probabilities,
        confidence=np.full((2, 2), 0.9, dtype=np.float32),
    )
    projection = ProjectionResult(
        rows=np.array([0.1, 0.9, 1.0]),
        columns=np.array([0.2, 1.0, 0.0]),
        depth=np.array([2.0, 3.0, -1.0]),
        valid=np.array([True, True, False]),
    )
    camera_points = np.array(
        [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    vehicle_points = camera_points + np.array([1.0, 2.0, 3.0])

    painted = paint_points(
        camera_points,
        vehicle_points,
        projection,
        segmentation,
    )

    np.testing.assert_array_equal(painted.class_ids, [0, 4])
    np.testing.assert_array_equal(painted.costs, [0, -1])
    np.testing.assert_array_equal(painted.source_indices, [0, 1])
    np.testing.assert_allclose(painted.confidence, [0.9, 0.9])
    np.testing.assert_allclose(painted.points_vehicle, vehicle_points[:2])


def test_paint_points_rejects_mismatched_frames():
    segmentation = SegmentationResult(
        class_ids=np.zeros((1, 1), dtype=np.uint8),
        probabilities=np.ones((5, 1, 1), dtype=np.float32),
        confidence=np.ones((1, 1), dtype=np.float32),
    )
    projection = ProjectionResult(
        rows=np.array([0.0]),
        columns=np.array([0.0]),
        depth=np.array([1.0]),
        valid=np.array([True]),
    )

    try:
        paint_points(np.zeros((1, 3)), np.zeros((2, 3)), projection, segmentation)
    except ValueError as error:
        assert "equal shape" in str(error)
    else:
        raise AssertionError("mismatched point frames should fail")
