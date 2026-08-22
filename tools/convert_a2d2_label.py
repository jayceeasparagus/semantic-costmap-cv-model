import json
from pathlib import Path
import numpy as np
import yaml
from PIL import Image

LABEL_PATH = Path("data/raw/a2d2_sample/label/" "20180807145028_label_frontcenter_000000091.png")

CLASS_LIST_PATH = Path("configs/a2d2_class_list.json")
MAPPING_PATH = Path("configs/a2d2_costmap_mapping.yaml")
OUTPUT_PATH = Path("outputs/a2d2_sample/label_train_ids.png")

CATEGORY_TO_ID = {"traversable": 0, "caution": 1, "non_traversable": 2, 
                  "static_obstacle": 3, "dynamic_obstacle": 4,}

IGNORE_ID = 255

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")

    return (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16),)

def main():
    class_list = json.loads(CLASS_LIST_PATH.read_text())

    mapping = yaml.safe_load(MAPPING_PATH.read_text())

    class_to_category = {}

    for category, class_names in mapping["class_groups"].items():
        for class_name in class_names:
            if class_name in class_to_category:
                raise ValueError(f"Class appears in multiple groups: {class_name}")

            class_to_category[class_name] = category

    label_rgb = np.array(Image.open(LABEL_PATH).convert("RGB"))

    train_ids = np.full(label_rgb.shape[:2], IGNORE_ID, dtype=np.uint8,)

    for hex_color, class_name in class_list.items():
        rgb_color = hex_to_rgb(hex_color)

        pixel_mask = np.all(label_rgb == rgb_color, axis=2,)

        if not np.any(pixel_mask):
            continue

        category = class_to_category.get(class_name)

        if category is None:
            raise ValueError(f"Class is not present in mapping: {class_name}")

        if category == "background_ignore":
            continue

        if category not in CATEGORY_TO_ID:
            raise ValueError(f"Unsupported category: {category}")

        train_ids[pixel_mask] = CATEGORY_TO_ID[category]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True,)

    Image.fromarray(train_ids).save(OUTPUT_PATH)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Output shape: {train_ids.shape}")
    print("Training IDs present:", np.unique(train_ids),)

if __name__ == "__main__":
    main()