from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


SAMPLE_ROOT = Path("data/raw/a2d2_sample")

IMAGE_PATH = next((SAMPLE_ROOT / "camera").glob("*.png"))
MASK_PATH = Path(
    "outputs/a2d2_sample/label_train_ids.png"
)

OUTPUT_PATH = Path(
    "outputs/a2d2_sample/label_train_ids_color.png"
)


PROJECT_COLORS = {
    0: (0, 180, 0),       # traversable: green
    1: (255, 180, 0),     # caution: yellow
    2: (255, 0, 0),       # non-traversable: red
    3: (180, 0, 180),     # static obstacle: purple
    4: (0, 0, 255),       # dynamic obstacle: blue
    255: (30, 30, 30),    # ignored/background: dark gray
}


def main() -> None:
    image = Image.open(IMAGE_PATH).convert("RGB")
    mask = np.array(Image.open(MASK_PATH))

    color_mask = np.zeros(
        (*mask.shape, 3),
        dtype=np.uint8,
    )

    for train_id, color in PROJECT_COLORS.items():
        color_mask[mask == train_id] = color

    figure, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image)
    axes[0].set_title("RGB image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Numeric training IDs")
    axes[1].axis("off")

    axes[2].imshow(color_mask)
    axes[2].set_title("Project categories")
    axes[2].axis("off")

    figure.tight_layout()
    figure.savefig(
        OUTPUT_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()