import numpy as np

from semantic_costmap.geometry.projection import project_camera_points


def test_project_camera_points_uses_a2d2_axes():
    matrix = np.array(
        [[100.0, 0.0, 50.0], [0.0, 120.0, 40.0], [0.0, 0.0, 1.0]]
    )
    points = np.array(
        [
            [10.0, 0.0, 0.0],
            [10.0, 1.0, 2.0],
            [10.0, -1.0, -2.0],
            [-1.0, 0.0, 0.0],
        ]
    )

    result = project_camera_points(points, matrix, (100, 80))

    np.testing.assert_allclose(result.columns[:3], [50.0, 40.0, 60.0])
    np.testing.assert_allclose(result.rows[:3], [40.0, 16.0, 64.0])
    np.testing.assert_array_equal(result.valid, [True, True, True, False])


def test_projection_rejects_pixels_outside_image():
    matrix = np.array(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
    )
    result = project_camera_points(
        np.array([[1.0, 2.0, 0.0]]),
        matrix,
        (100, 80),
    )
    assert not result.valid[0]
