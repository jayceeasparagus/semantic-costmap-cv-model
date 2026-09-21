#!/usr/bin/env python3
"""Simple command-line entry point for semantic mapping and video overlays."""

import argparse
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image

from semantic_costmap.config import DEFAULT_CHECKPOINT_PATH
from semantic_costmap.inference import SemanticSegmenter, colorize_class_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", type=Path)
    input_group.add_argument("--camera-dir", type=Path)
    parser.add_argument("--lidar-dir", type=Path)
    parser.add_argument("--poses-csv", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("configs/a2d2_cams_lidars.json"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/semantic_map"))
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--gif-fps", type=float, default=10.0)
    parser.add_argument("--global-map-range", type=float, default=200.0)
    parser.add_argument(
        "--fps",
        type=float,
        help="Output FPS for camera-only video mode; defaults to source FPS",
    )
    return parser.parse_args()


def run_sequence(args: argparse.Namespace) -> None:
    if args.lidar_dir is None:
        raise ValueError("--lidar-dir is required when using --camera-dir")
    if not args.camera_dir.is_dir():
        raise FileNotFoundError(args.camera_dir)
    if not args.lidar_dir.is_dir():
        raise FileNotFoundError(args.lidar_dir)

    command = [
        sys.executable,
        str(Path(__file__).with_name("run_playback.py")),
        "--image-dir",
        str(args.camera_dir),
        "--lidar-dir",
        str(args.lidar_dir),
        "--checkpoint",
        str(args.checkpoint),
        "--calibration",
        str(args.calibration),
        "--device",
        args.device,
        "--stride",
        str(args.stride),
        "--max-frames",
        str(args.max_frames),
        "--gif-fps",
        str(args.gif_fps),
        "--global-map-range",
        str(args.global_map_range),
        "--output-dir",
        str(args.output_dir),
    ]
    if args.poses_csv is not None:
        command.extend(("--poses-csv", str(args.poses_csv)))
    subprocess.run(command, check=True)


def run_video(args: argparse.Namespace) -> None:
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    output_fps = args.fps or (source_fps if source_fps > 0.0 else 10.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.video.stem}_semantic.mp4"
    segmenter = SemanticSegmenter(args.checkpoint, args.device)
    writer = None
    frame_index = 0
    processed = 0

    try:
        while True:
            success, frame_bgr = capture.read()
            if not success:
                break
            if frame_index % args.stride != 0:
                frame_index += 1
                continue
            if args.max_frames > 0 and processed >= args.max_frames:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb, mode="RGB")
            result = segmenter.predict(image)
            mask = Image.fromarray(colorize_class_ids(result.class_ids), mode="RGB")
            overlay = Image.blend(image, mask, alpha=0.45)

            confidence = (result.confidence * 255.0).clip(0, 255).astype(np.uint8)
            confidence_bgr = cv2.applyColorMap(confidence, cv2.COLORMAP_TURBO)
            overlay_bgr = cv2.cvtColor(np.asarray(overlay), cv2.COLOR_RGB2BGR)
            panel = np.hstack((overlay_bgr, confidence_bgr))

            if writer is None:
                height, width = panel.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    output_fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create video: {output_path}")
            writer.write(panel)
            processed += 1
            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if processed == 0:
        raise RuntimeError("The video contained no frames to process")
    print(f"Processed {processed} video frames")
    print("Saved video:", output_path)


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("stride must be at least one")
    if args.video is not None:
        if args.poses_csv is not None or args.lidar_dir is not None:
            raise ValueError("video mode accepts camera input only")
        run_video(args)
    else:
        run_sequence(args)


if __name__ == "__main__":
    main()
