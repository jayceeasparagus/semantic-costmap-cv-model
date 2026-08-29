"""PyTorch dataset for prepared A2D2 RGB images and semantic masks."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .manifest import SampleRecord, read_manifest


class A2D2SegmentationDataset(Dataset):
    """Load paired samples from a JSON-lines manifest."""

    def __init__(
        self,
        manifest: str | Path | list[SampleRecord],
        root: str | Path = ".",
        image_size: tuple[int, int] = (320, 512),
        horizontal_flip_probability: float = 0.0,
    ) -> None:
        self.records = (
            read_manifest(manifest)
            if isinstance(manifest, (str, Path))
            else list(manifest)
        )
        self.root = Path(root)
        self.image_size = image_size
        self.horizontal_flip_probability = horizontal_flip_probability

        if not self.records:
            raise ValueError("Dataset manifest is empty")
        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be between 0 and 1")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image = Image.open(self.root / record.image).convert("RGB")
        mask = Image.open(self.root / record.mask).convert("L")

        height, width = self.image_size
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        mask = mask.resize((width, height), Image.Resampling.NEAREST)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = np.asarray(mask, dtype=np.int64)

        if random.random() < self.horizontal_flip_probability:
            image_array = np.flip(image_array, axis=1).copy()
            mask_array = np.flip(mask_array, axis=1).copy()

        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array.copy())
        return image_tensor, mask_tensor
