"""ROS message conversions that can be tested without loading the model."""

import numpy as np
from PIL import Image as PilImage
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2


def transform_message_to_matrix(transform_message) -> np.ndarray:
    """Convert geometry_msgs/Transform into a 4x4 homogeneous matrix."""

    translation = transform_message.translation
    rotation = transform_message.rotation
    quaternion = np.array(
        [rotation.x, rotation.y, rotation.z, rotation.w],
        dtype=np.float64,
    )
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        raise ValueError("transform quaternion has zero length")
    x, y, z, w = quaternion / norm
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    matrix[:3, 3] = [translation.x, translation.y, translation.z]
    return matrix


def image_message_to_pil(message: Image) -> PilImage.Image:
    """Convert common uncompressed ROS RGB encodings without cv_bridge."""

    if message.encoding not in {"rgb8", "bgr8"}:
        raise ValueError("camera image encoding must be rgb8 or bgr8")
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
    image = rows[:, : message.width * 3].reshape(message.height, message.width, 3)
    if message.encoding == "bgr8":
        image = image[:, :, ::-1]
    return PilImage.fromarray(image.copy(), mode="RGB")


def pointcloud_message_to_xyz(message: PointCloud2) -> np.ndarray:
    records = point_cloud2.read_points(
        message,
        field_names=("x", "y", "z"),
        skip_nans=True,
    )
    if isinstance(records, np.ndarray) and records.dtype.names:
        return np.column_stack((records["x"], records["y"], records["z"]))
    return np.asarray(list(records), dtype=np.float64).reshape(-1, 3)
