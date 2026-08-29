"""Convert A2D2 RGB label images into compact semantic training masks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .taxonomy import SemanticTaxonomy


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a six-digit RGB color, got {hex_color!r}")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def rgb_to_keys(rgb: np.ndarray) -> np.ndarray:
    """Pack an RGB image into one integer key per pixel."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB array, got shape {rgb.shape}")
    values = rgb.astype(np.uint32, copy=False)
    return (values[..., 0] << 16) | (values[..., 1] << 8) | values[..., 2]


class A2D2LabelConverter:
    """Fast color-to-training-ID conversion backed by a 24-bit lookup table."""

    def __init__(
        self,
        color_to_class: dict[str, str],
        taxonomy: SemanticTaxonomy,
    ) -> None:
        taxonomy.validate(color_to_class.values())
        self.taxonomy = taxonomy
        self.color_to_class = dict(color_to_class)
        self._lookup = np.full(1 << 24, taxonomy.ignore_id, dtype=np.uint8)

        for hex_color, source_class in self.color_to_class.items():
            red, green, blue = hex_to_rgb(hex_color)
            key = (red << 16) | (green << 8) | blue
            self._lookup[key] = taxonomy.train_id_for(source_class)

    @classmethod
    def from_files(
        cls,
        class_list_path: str | Path,
        taxonomy_path: str | Path,
    ) -> "A2D2LabelConverter":
        color_to_class = json.loads(
            Path(class_list_path).read_text(encoding="utf-8")
        )
        taxonomy = SemanticTaxonomy.from_yaml(taxonomy_path)
        return cls(color_to_class, taxonomy)

    def convert_array(self, label_rgb: np.ndarray) -> np.ndarray:
        """Return an HxW uint8 mask; unknown colors become the ignore ID."""
        return self._lookup[rgb_to_keys(label_rgb)]

    def convert_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> np.ndarray:
        input_path = Path(input_path)
        output_path = Path(output_path)
        label_rgb = np.asarray(Image.open(input_path).convert("RGB"))
        train_ids = self.convert_array(label_rgb)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(train_ids, mode="L").save(output_path)
        return train_ids
