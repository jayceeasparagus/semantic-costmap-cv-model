from types import SimpleNamespace

import numpy as np
from sensor_msgs.msg import Image

from semantic_costmap_ros.conversions import (
    image_message_to_pil,
    transform_message_to_matrix,
)


def test_transform_message_to_matrix_handles_translation():
    message = SimpleNamespace(
        translation=SimpleNamespace(x=1.0, y=2.0, z=3.0),
        rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    matrix = transform_message_to_matrix(message)
    np.testing.assert_allclose(matrix[:3, :3], np.eye(3))
    np.testing.assert_allclose(matrix[:3, 3], [1.0, 2.0, 3.0])


def test_rgb_image_conversion_respects_row_step():
    message = Image()
    message.width = 2
    message.height = 1
    message.encoding = "rgb8"
    message.step = 8
    message.data = bytes([255, 0, 0, 0, 255, 0, 9, 9])
    image = np.asarray(image_message_to_pil(message))
    np.testing.assert_array_equal(
        image,
        np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8),
    )
