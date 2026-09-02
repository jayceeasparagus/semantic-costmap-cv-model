#!/usr/bin/env python3
"""Generate an offline vehicle-relative costmap from painted A2D2 points."""

import argparse
from pathlib import Path

import numpy as np

from semantic_costmap.costmap import CostmapConfig, build_semantic_costmap, save_costmap
from semantic_costmap.fusion import PaintedPointCloud


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--painted-points",
        type=Path,
        default=Path("outputs/fusion/painted_points.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/costmap"),
    )
    parser.add_argument("--resolution", type=float, default=0.20)
    parser.add_argument("--forward-range", type=float, default=50.0)
    parser.add_argument("--side-range", type=float, default=20.0)
    parser.add_argument("--minimum-confidence", type=float, default=0.50)
    parser.add_argument("--disable-raytracing", action="store_true")
    parser.add_argument("--raytrace-max-range", type=float, default=50.0)
    parser.add_argument("--disable-ground-interpolation", action="store_true")
    parser.add_argument("--ground-interpolation-iterations", type=int, default=2)
    parser.add_argument("--ground-interpolation-min-neighbors", type=int, default=3)
    return parser.parse_args()


def load_painted_cloud(path: Path) -> PaintedPointCloud:
    data = np.load(path)
    return PaintedPointCloud(
        points_camera=data["points_camera"],
        points_vehicle=data["points_vehicle"],
        probabilities=data["probabilities"],
        class_ids=data["class_ids"],
        confidence=data["confidence"],
        costs=data["costs"],
        rows=data["rows"],
        columns=data["columns"],
        source_indices=data["source_indices"],
    )


def main() -> None:
    args = parse_args()
    painted = load_painted_cloud(args.painted_points)
    config = CostmapConfig(
        resolution=args.resolution,
        x_max=args.forward_range,
        y_min=-args.side_range,
        y_max=args.side_range,
        minimum_confidence=args.minimum_confidence,
        raytrace_free_space=not args.disable_raytracing,
        raytrace_max_range=args.raytrace_max_range,
        ground_interpolation_iterations=(
            0
            if args.disable_ground_interpolation
            else args.ground_interpolation_iterations
        ),
        ground_interpolation_min_neighbors=(
            args.ground_interpolation_min_neighbors
        ),
    )
    costmap = build_semantic_costmap(
        painted,
        config,
        raw_points_vehicle=painted.points_vehicle,
    )
    paths = save_costmap(args.output_dir, costmap)

    known = costmap.costs != 255
    print(f"Grid size: {config.width} x {config.height} cells")
    print(f"Resolution: {config.resolution:.2f} m/cell")
    print(f"Known cells: {known.sum()}")
    print(f"Lethal obstacle cells: {costmap.obstacle_mask.sum()}")
    raytraced = (costmap.costs == 0) & (costmap.evidence_count == 0)
    print(f"Raytraced/interpolated free cells: {raytraced.sum()}")
    if known.any():
        print(f"Known cost range: {costmap.costs[known].min()}-{costmap.costs[known].max()}")
    for name, path in paths.items():
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
