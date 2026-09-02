"""Reusable semantic segmentation inference helpers."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from semantic_costmap.config import (
    IMAGE_HEIGHT,
    IMAGE_MEAN,
    IMAGE_STD,
    IMAGE_WIDTH,
    NUM_CLASSES,
    class_colors,
)
from semantic_costmap.models import SemanticUNet


@dataclass(frozen=True)
class SegmentationResult:
    """Prediction arrays at the original image resolution."""

    class_ids: np.ndarray
    probabilities: np.ndarray
    confidence: np.ndarray


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access it")

    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")

    return torch.device(requested)


def colorize_class_ids(class_ids: np.ndarray) -> np.ndarray:
    """Convert an HxW class-ID mask into an HxWx3 RGB image."""

    if class_ids.ndim != 2:
        raise ValueError("class_ids must be a two-dimensional array")
    if class_ids.size and (
        class_ids.min() < 0 or class_ids.max() >= NUM_CLASSES
    ):
        raise ValueError("class_ids contains an unsupported class")

    palette = np.asarray(class_colors(), dtype=np.uint8)
    return palette[class_ids]


class SemanticSegmenter:
    """Load the trained U-Net and run deterministic RGB inference."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
    ) -> None:
        self.device = resolve_device(device)
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(self.checkpoint_path)

        self.model = SemanticUNet(
            num_classes=NUM_CLASSES,
            base_channels=32,
        )
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.epoch = checkpoint.get("epoch")
        self.best_navigation_miou = checkpoint.get("best_navigation_miou")

    @staticmethod
    def _preprocess(image: Image.Image) -> torch.Tensor:
        resized = image.convert("RGB").resize(
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            Image.Resampling.BILINEAR,
        )
        values = np.asarray(resized, dtype=np.float32) / 255.0
        mean = np.asarray(IMAGE_MEAN, dtype=np.float32)
        std = np.asarray(IMAGE_STD, dtype=np.float32)
        values = (values - mean) / std
        tensor = torch.from_numpy(values.transpose(2, 0, 1))
        return tensor.unsqueeze(0).float()

    def predict(self, image: Image.Image) -> SegmentationResult:
        """Return classes, probabilities, and confidence at source resolution."""

        rgb_image = image.convert("RGB")
        tensor = self._preprocess(rgb_image).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)
            probabilities = functional.interpolate(
                probabilities,
                size=(rgb_image.height, rgb_image.width),
                mode="bilinear",
                align_corners=False,
            )[0]
            confidence, class_ids = probabilities.max(dim=0)

        return SegmentationResult(
            class_ids=class_ids.cpu().numpy().astype(np.uint8),
            probabilities=probabilities.cpu().numpy().astype(np.float32),
            confidence=confidence.cpu().numpy().astype(np.float32),
        )
