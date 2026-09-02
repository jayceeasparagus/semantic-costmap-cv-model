#!/usr/bin/env python3
"""Process an A2D2 sequence and save debug playback plus latency benchmarks."""

import argparse
import json
from pathlib import Path

from semantic_costmap.config import DEFAULT_CHECKPOINT_PATH
from semantic_costmap.pipeline import SemanticCostmapPipeline
from semantic_costmap.playback import (
    discover_frame_pairs,
    render_frame,
    summarize_timings,
)


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = discover_frame_pairs(args.image_dir, args.lidar_dir)
    if not pairs:
        raise FileNotFoundError("no paired A2D2 frames were found")
    pairs = pairs[: args.max_frames]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_directory = args.output_dir / "frames"
    frame_directory.mkdir(parents=True, exist_ok=True)

    pipeline = SemanticCostmapPipeline(
        args.checkpoint,
        args.calibration,
        device=args.device,
    )
    rendered_frames = []
    frame_records = []
    for index, pair in enumerate(pairs):
        result = pipeline.process(pair.image_path, pair.lidar_path)
        rendered = render_frame(result, pair.frame_id)
        rendered.save(frame_directory / f"frame_{index:03d}.png")
        rendered_frames.append(rendered)
        frame_records.append(
            {"frame_id": pair.frame_id, "timings_ms": result.timings_ms}
        )
        print(
            f"[{index + 1}/{len(pairs)}] {pair.frame_id}: "
            f"{result.timings_ms['total']:.1f} ms"
        )

    gif_path = args.output_dir / "semantic_costmap_playback.gif"
    rendered_frames[0].save(
        gif_path,
        save_all=True,
        append_images=rendered_frames[1:],
        duration=400,
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


if __name__ == "__main__":
    main()
