"""A2D2 frame pairing, playback rendering, and pose-aware summaries."""

import csv
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw

from semantic_costmap.config import class_colors
from semantic_costmap.costmap import costmap_to_rgb
from semantic_costmap.mapping import Pose2D
from semantic_costmap.pipeline import FrameResult


FRAME_PATTERN = re.compile(r"_(\d{9})\.(?:png|npz)$")


@dataclass(frozen=True)
class FramePair:
    frame_id: str
    image_path: Path
    lidar_path: Path


@dataclass(frozen=True)
class PoseRecord:
    frame_id: str
    pose: Pose2D
    timestamp: float


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


def load_pose_csv(path: str | Path) -> dict[str, PoseRecord]:
    """Load timestamped map-to-base planar poses keyed by A2D2 frame ID."""

    records = {}
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"frame_id", "timestamp", "x", "y", "yaw"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(
                "pose CSV is missing columns: " + ", ".join(sorted(missing))
            )
        for row in reader:
            frame_id = row["frame_id"].strip().zfill(9)
            if frame_id in records:
                raise ValueError(f"duplicate pose for frame {frame_id}")
            records[frame_id] = PoseRecord(
                frame_id=frame_id,
                pose=Pose2D(
                    x=float(row["x"]),
                    y=float(row["y"]),
                    yaw=float(row["yaw"]),
                ),
                timestamp=float(row["timestamp"]),
            )
    if not records:
        raise ValueError("pose CSV contains no poses")
    return records


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
        costmap_to_rgb(
            result.costmap.costs,
            result.costmap.observed_free_mask,
        ),
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


def render_trajectory(records: list[PoseRecord]) -> Image.Image:
    """Render the vehicle path used to place local maps globally."""

    if not records:
        raise ValueError("at least one pose record is required")

    width, height, margin = 800, 600, 50
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    points = [(record.pose.x, record.pose.y) for record in records]
    x_values, y_values = zip(*points)
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)

    def to_pixel(point: tuple[float, float]) -> tuple[int, int]:
        x, y = point
        pixel_x = margin + int((x - x_min) / x_span * (width - 2 * margin))
        pixel_y = height - margin - int((y - y_min) / y_span * (height - 2 * margin))
        return pixel_x, pixel_y

    pixel_points = [to_pixel(point) for point in points]
    if len(pixel_points) > 1:
        draw.line(pixel_points, fill=(30, 100, 220), width=4)
    draw.ellipse((*tuple(value - 6 for value in pixel_points[0]),
                  *tuple(value + 6 for value in pixel_points[0])), fill=(20, 150, 40))
    draw.ellipse((*tuple(value - 6 for value in pixel_points[-1]),
                  *tuple(value + 6 for value in pixel_points[-1])), fill=(210, 50, 40))
    draw.text((margin, 15), "Pose trajectory used for map accumulation", fill="black")
    draw.text((margin, height - 30), f"x: {x_min:.1f} to {x_max:.1f} m | y: {y_min:.1f} to {y_max:.1f} m", fill="black")
    return image


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
