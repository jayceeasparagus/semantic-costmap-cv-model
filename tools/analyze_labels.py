from pathlib import Path

import numpy as np
from PIL import Image


DATASET_ROOT = Path(
    "data/raw/rellis3d_sample/Rellis_3D_image_example"
)

LABEL_DIR = DATASET_ROOT / "pylon_camera_node_label_id"


def main() -> None:
    label_paths = sorted(LABEL_DIR.glob("*.png"))

    for label_path in label_paths:
        label = np.array(Image.open(label_path))
        values, counts = np.unique(label, return_counts=True)

        total_pixels = label.size

        print(f"\n{label_path.name}")
        print("-" * len(label_path.name))

        for value, count in zip(values, counts):
            percentage = 100 * count / total_pixels
            print(
                f"Class ID {value:>2}: "
                f"{count:>8} pixels "
                f"({percentage:>6.2f}%)"
            )


if __name__ == "__main__":
    main()