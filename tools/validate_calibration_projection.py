#!/usr/bin/env python3
"""Validate independently computed LiDAR-to-image coordinates on A2D2."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from semantic_costmap.geometry import load_a2d2_calibration, project_camera_points


DEFAULT_IMAGE = Path(
    "data/raw/a2d2_sample/camera/"
    "20180807145028_camera_frontcenter_000000091.png"
)
DEFAULT_LIDAR = Path(
    "data/raw/a2d2_sample/lidar/"
    "20180807145028_lidar_frontcenter_000000091.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute A2D2 image pixels from calibration and 3D points."
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--lidar", type=Path, default=DEFAULT_LIDAR)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("configs/a2d2_cams_lidars.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/calibration/projection_validation.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = Image.open(args.image).convert("RGB")
    lidar = np.load(args.lidar)
    calibration = load_a2d2_calibration(args.calibration)

    if image.size != calibration.resolution:
        raise ValueError(
            f"image size {image.size} does not match calibration "
            f"{calibration.resolution}"
        )

    projection = project_camera_points(
        lidar["points"],
        calibration.camera_matrix,
        calibration.resolution,
    )
    reference_valid = (
        (lidar["col"] >= 0)
        & (lidar["col"] < image.width)
        & (lidar["row"] >= 0)
        & (lidar["row"] < image.height)
    )
    compared = projection.valid & reference_valid
    errors = np.hypot(
        projection.columns[compared] - lidar["col"][compared],
        projection.rows[compared] - lidar["row"][compared],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(14, 8))
    axis.imshow(image)
    axis.scatter(
        lidar["col"][compared],
        lidar["row"][compared],
        s=4,
        c="red",
        label="A2D2 reference",
    )
    axis.scatter(
        projection.columns[compared],
        projection.rows[compared],
        s=1,
        c="cyan",
        label="computed from calibration",
    )
    axis.set_xlim(0, image.width)
    axis.set_ylim(image.height, 0)
    axis.set_axis_off()
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(figure)

    print("Projection input: calibrated 3D camera-frame points")
    print("Reference row/col used only for validation")
    print(f"Compared points: {compared.sum()}")
    print(f"Median pixel error: {np.median(errors):.6g}")
    print(f"95th percentile pixel error: {np.percentile(errors, 95):.6g}")
    print(f"Maximum pixel error: {errors.max():.6g}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
