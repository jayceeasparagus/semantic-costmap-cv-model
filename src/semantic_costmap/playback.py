"""A2D2 frame pairing, playback rendering, and benchmark summaries."""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw

from semantic_costmap.config import class_colors
from semantic_costmap.costmap import costmap_to_rgb
from semantic_costmap.pipeline import FrameResult


FRAME_PATTERN = re.compile(r"_(\d{9})\.(?:png|npz)$")


@dataclass(frozen=True)
class FramePair:
    frame_id: str
    image_path: Path
    lidar_path: Path


def _frame_id(path: Path) -> str | None:
    match = FRAME_PATTERN.search(path.name)
    return match.group(1) if match else None


def discover_frame_pairs(
    image_directory: str | Path,
    lidar_directory: str | Path,
) -> list[FramePair]:
    """Match RGB and LiDAR files by the numeric A2D2 frame identifier."""

    image_directory = Path(image_directory)
    lidar_directory = Path(lidar_directory)
    images = {
        frame_id: path
        for path in sorted(image_directory.glob("*.png"))
        if (frame_id := _frame_id(path)) is not None
    }
    lidars = {
        frame_id: path
        for path in sorted(lidar_directory.glob("*.npz"))
        if (frame_id := _frame_id(path)) is not None
    }
    return [
        FramePair(frame_id, images[frame_id], lidars[frame_id])
        for frame_id in sorted(images.keys() & lidars.keys())
    ]


def render_frame(result: FrameResult, frame_id: str) -> Image.Image:
    """Create a four-panel RGB/segmentation/fusion/costmap debug frame."""

    panel_size = (480, 302)
    rgb = result.image.resize(panel_size, Image.Resampling.BILINEAR)
    palette = np.asarray(class_colors(), dtype=np.uint8)
    segmentation = Image.fromarray(
        palette[result.segmentation.class_ids],
        mode="RGB",
    ).resize(panel_size, Image.Resampling.NEAREST)
    semantic_overlay = Image.blend(rgb, segmentation, 0.45)

    painted_overlay = rgb.copy()
    draw = ImageDraw.Draw(painted_overlay)
    scale_x = panel_size[0] / result.image.width
    scale_y = panel_size[1] / result.image.height
    for row, column, class_id in zip(
        result.painted_points.rows,
        result.painted_points.columns,
        result.painted_points.class_ids,
    ):
        x = int(column * scale_x)
        y = int(row * scale_y)
        color = tuple(int(value) for value in palette[class_id])
        draw.point((x, y), fill=color)

    costmap = Image.fromarray(
        costmap_to_rgb(result.costmap.costs),
        mode="RGB",
    ).resize(panel_size, Image.Resampling.NEAREST)

    header_height = 36
    canvas = Image.new(
        "RGB",
        (panel_size[0] * 2, panel_size[1] * 2 + header_height),
        "white",
    )
    canvas.paste(rgb, (0, header_height))
    canvas.paste(semantic_overlay, (panel_size[0], header_height))
    canvas.paste(painted_overlay, (0, header_height + panel_size[1]))
    canvas.paste(costmap, (panel_size[0], header_height + panel_size[1]))
    labels = "RGB | semantic overlay | painted LiDAR | vehicle costmap"
    ImageDraw.Draw(canvas).text(
        (12, 10),
        f"Frame {frame_id} - {labels}",
        fill="black",
    )
    return canvas


def summarize_timings(frame_timings: list[dict[str, float]]) -> dict:
    if not frame_timings:
        raise ValueError("at least one frame timing is required")

    summary = {"frame_count": len(frame_timings), "stages_ms": {}}
    for stage in frame_timings[0]:
        values = np.asarray([item[stage] for item in frame_timings])
        summary["stages_ms"][stage] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
        }
    mean_total = summary["stages_ms"]["total"]["mean"]
    summary["mean_fps"] = 1000.0 / mean_total
    return summary
