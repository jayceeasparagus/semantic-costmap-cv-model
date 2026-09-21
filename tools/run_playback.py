#!/usr/bin/env python3
"""Process an A2D2 sequence and save debug playback plus latency benchmarks."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from semantic_costmap.config import DEFAULT_CHECKPOINT_PATH
from semantic_costmap.costmap import (
    confidence_to_rgb,
    costmap_to_rgb,
    semantic_map_to_rgb,
)
from semantic_costmap.mapping import GlobalMapConfig, PoseAwareAccumulator
from semantic_costmap.pipeline import SemanticCostmapPipeline
from semantic_costmap.playback import (
    discover_frame_pairs,
    load_pose_csv,
    render_trajectory,
    render_frame,
    summarize_timings,
)
from semantic_costmap.planning import LocalRoutePlanner, RoutePlannerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/raw/a2d2_playback/camera"),
    )
    parser.add_argument(
        "--lidar-dir",
        type=Path,
        default=Path("data/raw/a2d2_playback/lidar"),
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("configs/a2d2_cams_lidars.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/playback"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every Nth paired frame before applying --max-frames",
    )
    parser.add_argument("--gif-fps", type=float, default=2.5)
    parser.add_argument(
        "--poses-csv",
        type=Path,
        help="CSV with frame_id,timestamp,x,y,yaw map-to-base poses",
    )
    parser.add_argument("--global-map-range", type=float, default=100.0)
    parser.add_argument("--dynamic-decay-seconds", type=float, default=2.0)
    parser.add_argument("--map-gap-fill-iterations", type=int, default=1)
    parser.add_argument("--map-gap-fill-min-neighbors", type=int, default=5)
    parser.add_argument(
        "--plan-route",
        action="store_true",
        help="Optionally run the experimental frame-level route planner",
    )
    parser.add_argument("--route-goal-forward", type=float, default=12.0)
    parser.add_argument("--route-min-forward", type=float, default=5.0)
    parser.add_argument("--route-lateral-tolerance", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = discover_frame_pairs(args.image_dir, args.lidar_dir)
    if not pairs:
        raise FileNotFoundError("no paired A2D2 frames were found")
    if args.stride < 1:
        raise ValueError("stride must be at least one")
    if args.gif_fps <= 0.0:
        raise ValueError("gif-fps must be positive")
    pairs = pairs[:: args.stride]
    if args.max_frames > 0:
        pairs = pairs[: args.max_frames]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_directory = args.output_dir / "frames"
    frame_directory.mkdir(parents=True, exist_ok=True)

    pipeline = SemanticCostmapPipeline(
        args.checkpoint,
        args.calibration,
        device=args.device,
    )
    pose_records = load_pose_csv(args.poses_csv) if args.poses_csv else None
    accumulator = None
    if pose_records is not None:
        accumulator = PoseAwareAccumulator(
            GlobalMapConfig(
                resolution=pipeline.costmap_config.resolution,
                x_min=-args.global_map_range,
                x_max=args.global_map_range,
                y_min=-args.global_map_range,
                y_max=args.global_map_range,
                dynamic_decay_seconds=args.dynamic_decay_seconds,
                map_gap_fill_iterations=args.map_gap_fill_iterations,
                map_gap_fill_min_neighbors=args.map_gap_fill_min_neighbors,
            )
        )
    route_planner = None
    if args.plan_route:
        route_planner = LocalRoutePlanner(
            RoutePlannerConfig(
                goal_forward_m=args.route_goal_forward,
                minimum_goal_forward_m=args.route_min_forward,
                goal_lateral_tolerance_m=args.route_lateral_tolerance,
            )
        )

    rendered_frames = []
    frame_records = []
    last_pose_timestamp = None
    for index, pair in enumerate(pairs):
        result = pipeline.process(pair.image_path, pair.lidar_path)
        if accumulator is not None:
            if pair.frame_id not in pose_records:
                raise ValueError(f"pose CSV has no pose for frame {pair.frame_id}")
            pose_record = pose_records[pair.frame_id]
            accumulator.update_local_costmap(
                result.costmap,
                pose_record.pose,
                pose_record.timestamp,
            )
            last_pose_timestamp = pose_record.timestamp
        frame_timings = dict(result.timings_ms)
        route = None
        if route_planner is not None:
            route_start = time.perf_counter()
            route = route_planner.plan(result.costmap)
            route_ms = (time.perf_counter() - route_start) * 1000.0
            frame_timings["route"] = route_ms
            frame_timings["total"] = result.timings_ms["total"] + route_ms
        rendered = render_frame(result, pair.frame_id, route)
        rendered.save(frame_directory / f"frame_{index:03d}.png")
        rendered_frames.append(rendered)
        frame_record = {
            "frame_id": pair.frame_id,
            "timings_ms": frame_timings,
        }
        if route is not None:
            frame_record["route"] = route.to_dict()
        frame_records.append(frame_record)
        print(
            f"[{index + 1}/{len(pairs)}] {pair.frame_id}: "
            f"{result.timings_ms['total']:.1f} ms"
        )

    gif_path = args.output_dir / "semantic_costmap_playback.gif"
    rendered_frames[0].save(
        gif_path,
        save_all=True,
        append_images=rendered_frames[1:],
        duration=int(round(1000.0 / args.gif_fps)),
        loop=0,
        optimize=False,
    )
    summary = summarize_timings(
        [record["timings_ms"] for record in frame_records]
    )
    benchmark = {
        "device": str(pipeline.segmenter.device),
        "checkpoint_epoch": pipeline.segmenter.epoch,
        "summary": summary,
        "frames": frame_records,
    }
    if args.plan_route:
        route_records = [record["route"] for record in frame_records]
        benchmark["route_planning"] = {
            "frames_with_route": sum(
                record["found"] for record in route_records
            ),
            "frame_count": len(route_records),
            "mean_path_points": float(
                np.mean([record["point_count"] for record in route_records])
            ),
        }
    if accumulator is not None:
        accumulated = accumulator.grid(last_pose_timestamp)
        semantic_classes = accumulator.semantic_grid(last_pose_timestamp)
        semantic_known = accumulator.semantic_known_mask(last_pose_timestamp)
        accumulated_arrays = args.output_dir / "accumulated_costmap.npz"
        accumulated_preview = args.output_dir / "accumulated_costmap_preview.png"
        semantic_preview = args.output_dir / "accumulated_semantic_map.png"
        confidence_preview = args.output_dir / "accumulated_confidence.png"
        confidence = accumulator.confidence_grid(last_pose_timestamp)
        np.savez_compressed(
            accumulated_arrays,
            costs=accumulated,
            semantic_class_ids=semantic_classes,
            semantic_known_mask=semantic_known,
            confidence=confidence,
            resolution=accumulator.config.resolution,
            x_min=accumulator.config.x_min,
            y_min=accumulator.config.y_min,
        )
        Image.fromarray(costmap_to_rgb(accumulated), mode="RGB").save(
            accumulated_preview
        )
        Image.fromarray(
            semantic_map_to_rgb(semantic_classes, semantic_known),
            mode="RGB",
        ).save(semantic_preview)
        Image.fromarray(
            confidence_to_rgb(confidence, semantic_known),
            mode="RGB",
        ).save(confidence_preview)
        trajectory_preview = args.output_dir / "odometry_trajectory.png"
        trajectory_image = render_trajectory(list(pose_records.values()))
        trajectory_image.save(trajectory_preview)
        benchmark["accumulation"] = {
            "pose_source": str(args.poses_csv),
            "known_cells": int((accumulated != 255).sum()),
            "preview": str(accumulated_preview),
            "semantic_preview": str(semantic_preview),
            "confidence_preview": str(confidence_preview),
            "mean_confidence": float(confidence[semantic_known].mean())
            if semantic_known.any()
            else 0.0,
            "active_dynamic_cells": int(
                np.count_nonzero(
                    (accumulator.dynamic_costs >= 0)
                    & (
                        last_pose_timestamp - accumulator.dynamic_last_seen
                        <= accumulator.config.dynamic_decay_seconds
                    )
                )
            ),
            "trajectory_preview": str(trajectory_preview),
        }
    benchmark_path = args.output_dir / "benchmark.json"
    benchmark_path.write_text(json.dumps(benchmark, indent=2) + "\n")

    print(f"Mean end-to-end rate: {summary['mean_fps']:.2f} FPS")
    for stage, values in summary["stages_ms"].items():
        print(
            f"{stage}: mean {values['mean']:.1f} ms, "
            f"p95 {values['p95']:.1f} ms"
        )
    print(f"Saved playback: {gif_path}")
    print(f"Saved benchmark: {benchmark_path}")
    if accumulator is not None:
        print(f"Saved accumulated map: {accumulated_arrays}")
        print(f"Saved accumulated preview: {accumulated_preview}")
        print(f"Saved semantic map: {semantic_preview}")
        print(f"Saved confidence map: {confidence_preview}")
        print(f"Saved trajectory preview: {trajectory_preview}")


if __name__ == "__main__":
    main()
