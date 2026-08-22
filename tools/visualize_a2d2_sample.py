from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


SAMPLE_ROOT = Path("data/raw/a2d2_sample")

IMAGE_PATH = next((SAMPLE_ROOT / "camera").glob("*.png"))
LABEL_PATH = next((SAMPLE_ROOT / "label").glob("*.png"))


def main() -> None:
    image = Image.open(IMAGE_PATH)
    label = Image.open(LABEL_PATH)

    figure, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].imshow(image)
    axes[0].set_title("RGB camera image")
    axes[0].axis("off")

    axes[1].imshow(label)
    axes[1].set_title("A2D2 semantic label colors")
    axes[1].axis("off")

    figure.tight_layout()
    figure.savefig(
        "outputs/a2d2_sample_pair.png",
        dpi=150,
        bbox_inches="tight",
    )

    print("Saved visualization to outputs/a2d2_sample_pair.png")


if __name__ == "__main__":
    main()