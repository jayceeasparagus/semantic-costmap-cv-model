"""Shared model preprocessing and semantic navigation configuration."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SemanticClass:
    """One model class and its navigation interpretation."""

    class_id: int
    name: str
    cost: int | None
    color: tuple[int, int, int]


IMAGE_HEIGHT = 320
IMAGE_WIDTH = 512
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
IGNORE_ID = 255

SEMANTIC_CLASSES = (
    SemanticClass(0, "drivable", 0, (0, 200, 0)),
    SemanticClass(1, "non_drivable", 220, (255, 128, 0)),
    SemanticClass(2, "static_obstacle", 254, (255, 0, 0)),
    SemanticClass(3, "dynamic_obstacle", 254, (255, 0, 255)),
    SemanticClass(4, "background", None, (0, 0, 0)),
)

NUM_CLASSES = len(SEMANTIC_CLASSES)
BACKGROUND_CLASS_ID = 4
DEFAULT_CHECKPOINT_PATH = Path(
    "outputs/checkpoints/epoch29_restore/best_semantic_unet.pt"
)


def class_names() -> tuple[str, ...]:
    return tuple(item.name for item in SEMANTIC_CLASSES)


def class_colors() -> tuple[tuple[int, int, int], ...]:
    return tuple(item.color for item in SEMANTIC_CLASSES)


def class_costs(background_value: int = -1) -> tuple[int, ...]:
    """Return integer costs, replacing the no-cost background marker."""

    return tuple(
        background_value if item.cost is None else item.cost
        for item in SEMANTIC_CLASSES
    )
