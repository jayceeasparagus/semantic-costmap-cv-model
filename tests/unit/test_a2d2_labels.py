from pathlib import Path

import numpy as np

from offroad_vision.data.a2d2_labels import A2D2LabelConverter

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_rgb_labels_convert_to_training_ids() -> None:
    converter = A2D2LabelConverter.from_files(
        PROJECT_ROOT / "configs/a2d2_class_list.json",
        PROJECT_ROOT / "configs/a2d2_semantic_mapping_v2.yaml",
    )
    label = np.array(
        [
            [
                [255, 0, 255],
                [135, 206, 255],
                [72, 209, 204],
                [1, 2, 3],
            ]
        ],
        dtype=np.uint8,
    )

    converted = converter.convert_array(label)

    assert converted.dtype == np.uint8
    assert converted.shape == (1, 4)
    assert converted.tolist() == [[0, 5, 255, 255]]
