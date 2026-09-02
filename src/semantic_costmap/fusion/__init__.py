"""Camera-LiDAR semantic fusion."""

from semantic_costmap.fusion.point_painting import (
    PaintedPointCloud,
    paint_points,
    save_painted_cloud,
)

__all__ = ["PaintedPointCloud", "paint_points", "save_painted_cloud"]
