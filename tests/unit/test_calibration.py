import numpy as np

from semantic_costmap.geometry.calibration import (
    transform_between_views,
    transform_points,
    transform_to_vehicle,
    view_axes,
)


def test_view_axes_are_orthonormal():
    view = {
        "origin": [1.0, 2.0, 3.0],
        "x-axis": [2.0, 0.0, 0.0],
        "y-axis": [0.2, 3.0, 0.0],
    }
    axes = view_axes(view)
    np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-12)
    assert np.linalg.det(axes) > 0.0


def test_transform_between_views_round_trip():
    source = {
        "origin": [1.0, 0.0, 0.0],
        "x-axis": [1.0, 0.0, 0.0],
        "y-axis": [0.0, 1.0, 0.0],
    }
    target = {
        "origin": [0.0, 2.0, 0.0],
        "x-axis": [0.0, 1.0, 0.0],
        "y-axis": [-1.0, 0.0, 0.0],
    }
    points = np.array([[2.0, 1.0, 0.5], [0.0, -2.0, 1.0]])
    target_points = transform_points(
        points,
        transform_between_views(source, target),
    )
    recovered = transform_points(
        target_points,
        transform_between_views(target, source),
    )
    np.testing.assert_allclose(recovered, points, atol=1e-12)

    expected_source_origin = transform_to_vehicle(source)[:3, 3]
    np.testing.assert_allclose(expected_source_origin, [1.0, 0.0, 0.0])
