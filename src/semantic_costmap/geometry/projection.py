"""Projection from the A2D2 camera coordinate frame into image pixels."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProjectionResult:
    rows: np.ndarray
    columns: np.ndarray
    depth: np.ndarray
    valid: np.ndarray


def project_camera_points(
    points: np.ndarray,
    camera_matrix: np.ndarray,
    resolution: tuple[int, int],
    minimum_depth: float = 0.1,
) -> ProjectionResult:
    """Project A2D2 camera-frame points without using stored row/column data.

    A2D2 camera coordinates use +x forward, +y left, and +z up. The image
    coordinates therefore use ``u = cx - fx*y/x`` and
    ``v = cy - fy*z/x``.
    """

    points = np.asarray(points, dtype=np.float64)
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if camera_matrix.shape != (3, 3):
        raise ValueError("camera_matrix must have shape (3, 3)")
    if len(resolution) != 2:
        raise ValueError("resolution must contain width and height")

    width, height = resolution
    depth = points[:, 0]
    columns = np.full(len(points), np.nan, dtype=np.float64)
    rows = np.full(len(points), np.nan, dtype=np.float64)

    in_front = np.isfinite(points).all(axis=1) & (depth > minimum_depth)
    columns[in_front] = (
        camera_matrix[0, 2]
        - camera_matrix[0, 0] * points[in_front, 1] / depth[in_front]
    )
    rows[in_front] = (
        camera_matrix[1, 2]
        - camera_matrix[1, 1] * points[in_front, 2] / depth[in_front]
    )
    valid = (
        in_front
        & (columns >= 0.0)
        & (columns < width)
        & (rows >= 0.0)
        & (rows < height)
    )
    return ProjectionResult(
        rows=rows,
        columns=columns,
        depth=depth.copy(),
        valid=valid,
    )
