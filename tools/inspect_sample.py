from pathlib import Path

import numpy as np
from PIL import Image


DATASET_ROOT = Path(
    "data/raw/rellis3d_sample/Rellis_3D_image_example"
)

IMAGE_DIR = DATASET_ROOT / "pylon_camera_node"
LABEL_DIR = DATASET_ROOT / "pylon_camera_node_label_id"


def main() -> None:
    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))

    if not image_paths:
        raise FileNotFoundError(
            f"No JPG images found in {IMAGE_DIR}"
        )

    print(f"Found {len(image_paths)} image(s)\n")

    for image_path in image_paths:
        label_path = LABEL_DIR / f"{image_path.stem}.png"

        print(f"Image: {image_path.name}")
        print(f"Expected label: {label_path.name}")

        if not label_path.exists():
            print("ERROR: matching label was not found\n")
            continue

        image = Image.open(image_path)
        label = Image.open(label_path)
        label_array = np.array(label)

        print(f"Image size: {image.size}")
        print(f"Image mode: {image.mode}")
        print(f"Label size: {label.size}")
        print(f"Label mode: {label.mode}")
        print(f"Unique label IDs: {np.unique(label_array)}")
        print()


if __name__ == "__main__":
    main()