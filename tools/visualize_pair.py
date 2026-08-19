from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


DATASET_ROOT = Path(
    "data/raw/rellis3d_sample/Rellis_3D_image_example"
)

IMAGE_DIR = DATASET_ROOT / "pylon_camera_node"
LABEL_DIR = DATASET_ROOT / "pylon_camera_node_label_id"


def main() -> None:
    image_path = sorted(IMAGE_DIR.glob("*.jpg"))[0]
    label_path = LABEL_DIR / f"{image_path.stem}.png"

    image = Image.open(image_path)
    label = np.array(Image.open(label_path))

    figure, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].imshow(image)
    axes[0].set_title("RGB image")
    axes[0].axis("off")

    axes[1].imshow(label, cmap="tab20")
    axes[1].set_title("Semantic label IDs")
    axes[1].axis("off")

    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()