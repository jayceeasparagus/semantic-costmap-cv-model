#!/usr/bin/env python3
"""Build timestamp-aligned, bus-derived odometry poses for A2D2 playback."""

import argparse
from pathlib import Path

from semantic_costmap.odometry import build_odometry, load_bus_frames, write_pose_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bus-json", type=Path, required=True)
    parser.add_argument(
        "--camera-metadata-dir",
        type=Path,
        help="Optional directory of A2D2 camera JSON files with cam_tstamp.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--initial-yaw-deg", type=float, default=0.0)
    parser.add_argument("--initial-x", type=float, default=0.0)
    parser.add_argument("--initial-y", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = load_bus_frames(args.bus_json)
    records = build_odometry(
        frames,
        args.camera_metadata_dir,
        initial_yaw=args.initial_yaw_deg * 3.141592653589793 / 180.0,
        initial_x=args.initial_x,
        initial_y=args.initial_y,
    )
    write_pose_csv(records, args.output_csv)
    print(f"Frames written: {len(records)}")
    print("Pose source: A2D2 bus-derived odometry (not SLAM ground truth)")
    print(f"Output: {args.output_csv}")
    print(
        f"Final pose: x={records[-1].x:.2f} m, "
        f"y={records[-1].y:.2f} m, yaw={records[-1].yaw:.3f} rad"
    )


if __name__ == "__main__":
    main()
