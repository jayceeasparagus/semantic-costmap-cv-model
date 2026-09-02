#!/usr/bin/env python3
"""Demonstrate pose-aware accumulation with explicit synthetic poses."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from semantic_costmap.costmap import CostmapConfig, SemanticCostmap, costmap_to_rgb
from semantic_costmap.mapping import GlobalMapConfig, Pose2D, PoseAwareAccumulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-costmap",
        type=Path,
        default=Path("outputs/costmap/semantic_costmap.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/accumulation"),
    )
    return parser.parse_args()


def load_local_costmap(path: Path) -> SemanticCostmap:
    data = np.load(path)
    config = CostmapConfig(
        resolution=float(data["resolution"]),
        x_min=float(data["x_min"]),
        x_max=float(data["x_max"]),
        y_min=float(data["y_min"]),
        y_max=float(data["y_max"]),
        semantic_z_min=float(data["semantic_z_min"]),
        semantic_z_max=float(data["semantic_z_max"]),
        obstacle_z_min=float(data["obstacle_z_min"]),
        obstacle_z_max=float(data["obstacle_z_max"]),
        minimum_confidence=float(data["minimum_confidence"]),
        minimum_points_per_cell=int(data["minimum_points_per_cell"]),
    )
    return SemanticCostmap(
        costs=data["costs"],
        class_ids=data["class_ids"],
        evidence_count=data["evidence_count"],
        obstacle_mask=data["obstacle_mask"],
        config=config,
    )


def main() -> None:
    args = parse_args()
    local = load_local_costmap(args.local_costmap)
    accumulator = PoseAwareAccumulator(
        GlobalMapConfig(
            resolution=local.config.resolution,
            x_min=-20.0,
            x_max=80.0,
            y_min=-40.0,
            y_max=40.0,
        )
    )
    poses = [
        Pose2D(0.0, 0.0, 0.0),
        Pose2D(5.0, 0.0, 0.03),
        Pose2D(10.0, 1.0, 0.06),
        Pose2D(15.0, 2.0, 0.09),
    ]
    for timestamp, pose in enumerate(poses):
        accumulator.update_local_costmap(local, pose, float(timestamp))

    grid = accumulator.grid(timestamp=float(len(poses) + 3))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = args.output_dir / "accumulated_costmap.npz"
    preview_path = args.output_dir / "accumulated_costmap_preview.png"
    np.savez_compressed(
        arrays_path,
        costs=grid,
        resolution=accumulator.config.resolution,
        x_min=accumulator.config.x_min,
        y_min=accumulator.config.y_min,
    )
    Image.fromarray(costmap_to_rgb(grid), mode="RGB").save(preview_path)
    print("Pose source: explicit synthetic demonstration poses")
    print(f"Accumulated observations: {len(poses)}")
    print(f"Known global cells: {(grid != 255).sum()}")
    print(f"Saved: {arrays_path}")
    print(f"Saved: {preview_path}")


if __name__ == "__main__":
    main()
