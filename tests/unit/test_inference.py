import numpy as np
import pytest

from semantic_costmap.inference import colorize_class_ids, resolve_device


def test_colorize_class_ids() -> None:
    class_ids = np.asarray([[0, 1], [2, 4]], dtype=np.uint8)
    colors = colorize_class_ids(class_ids)

    assert colors.shape == (2, 2, 3)
    assert colors[0, 0].tolist() == [0, 200, 0]
    assert colors[1, 1].tolist() == [0, 0, 0]


def test_colorize_rejects_unknown_ids() -> None:
    with pytest.raises(ValueError):
        colorize_class_ids(np.asarray([[5]], dtype=np.uint8))


def test_resolve_cpu_device() -> None:
    assert resolve_device("cpu").type == "cpu"
