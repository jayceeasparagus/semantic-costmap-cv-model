"""Replay synchronized A2D2 camera, LiDAR, and bus-derived odometry data."""

from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Time as RosTime
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from semantic_costmap.geometry import load_a2d2_calibration
from semantic_costmap.geometry.calibration import transform_to_vehicle
from semantic_costmap.odometry import build_odometry, load_bus_frames
from semantic_costmap.playback import discover_frame_pairs


class A2D2ReplayNode(Node):
    """Publish a small synchronized sequence for ROS and SLAM Toolbox."""

    def __init__(self) -> None:
        super().__init__("a2d2_replay")
        self.declare_parameter("image_dir", "data/raw/a2d2_playback/camera")
        self.declare_parameter("lidar_dir", "data/raw/a2d2_playback/lidar")
        self.declare_parameter("bus_json", "")
        self.declare_parameter("calibration", "configs/a2d2_cams_lidars.json")
        self.declare_parameter("publish_rate", 5.0)
        self.declare_parameter("loop", False)
        self.declare_parameter("timestamp_mode", "bus")
        self.declare_parameter("camera_frame", "front_center_camera")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("odom_frame", "odom")

        image_dir = Path(str(self.get_parameter("image_dir").value))
        lidar_dir = Path(str(self.get_parameter("lidar_dir").value))
        calibration_path = Path(str(self.get_parameter("calibration").value))
        bus_path = str(self.get_parameter("bus_json").value)
        self.timestamp_mode = str(self.get_parameter("timestamp_mode").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.pairs = discover_frame_pairs(image_dir, lidar_dir)
        if not self.pairs:
            raise ValueError("A2D2 replay found no paired camera/LiDAR files")

        self.calibration = load_a2d2_calibration(calibration_path)
        if bus_path:
            bus_frames = load_bus_frames(bus_path)
            poses = build_odometry(bus_frames)
            self.poses = {record.frame_id: record for record in poses}
        else:
            raise ValueError("bus_json is required for synchronized odometry replay")
        missing = [pair.frame_id for pair in self.pairs if pair.frame_id not in self.poses]
        if missing:
            raise ValueError("bus JSON has no pose for frame(s): " + ", ".join(missing[:5]))

        self.image_publisher = self.create_publisher(Image, "image", 2)
        self.camera_info_publisher = self.create_publisher(CameraInfo, "camera_info", 2)
        self.points_publisher = self.create_publisher(PointCloud2, "points", 2)
        self.odom_publisher = self.create_publisher(Odometry, "odom", 2)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self._publish_sensor_transform()
        self.index = 0
        period = 1.0 / max(float(self.get_parameter("publish_rate").value), 0.1)
        self.timer = self.create_timer(period, self._publish_next)
        self.get_logger().info(f"Replaying {len(self.pairs)} synchronized A2D2 frames")

    def _publish_sensor_transform(self) -> None:
        matrix = transform_to_vehicle(self.calibration.view)
        message = TransformStamped()
        message.header.frame_id = self.base_frame
        message.child_frame_id = self.camera_frame
        message.transform.translation.x = float(matrix[0, 3])
        message.transform.translation.y = float(matrix[1, 3])
        message.transform.translation.z = float(matrix[2, 3])
        x, y, z, w = _rotation_to_quaternion(matrix[:3, :3])
        message.transform.rotation.x = x
        message.transform.rotation.y = y
        message.transform.rotation.z = z
        message.transform.rotation.w = w
        self.static_tf_broadcaster.sendTransform(message)

    def _stamp(self, timestamp_s: float) -> RosTime:
        if self.timestamp_mode == "now":
            return self.get_clock().now().to_msg()
        seconds = int(timestamp_s)
        nanoseconds = int(round((timestamp_s - seconds) * 1e9))
        if nanoseconds >= 1_000_000_000:
            seconds += 1
            nanoseconds -= 1_000_000_000
        return RosTime(sec=seconds, nanosec=nanoseconds)

    def _publish_next(self) -> None:
        if self.index >= len(self.pairs):
            if self.loop:
                self.index = 0
            else:
                self.timer.cancel()
                self.get_logger().info("A2D2 replay complete")
                return

        pair = self.pairs[self.index]
        pose = self.poses[pair.frame_id]
        stamp = self._stamp(pose.timestamp)
        image = np.asarray(PILImage.open(pair.image_path).convert("RGB"))
        lidar = np.load(pair.lidar_path)
        self._publish_image(image, stamp)
        self._publish_camera_info(stamp)
        self._publish_points(lidar, stamp)
        self._publish_odometry(pose, stamp)
        self.index += 1

    def _publish_image(self, image: np.ndarray, stamp: RosTime) -> None:
        message = Image()
        message.header = Header(stamp=stamp, frame_id=self.camera_frame)
        message.height, message.width = image.shape[:2]
        message.encoding = "rgb8"
        message.is_bigendian = False
        message.step = int(message.width * 3)
        message.data = image.astype(np.uint8, copy=False).tobytes()
        self.image_publisher.publish(message)

    def _publish_camera_info(self, stamp: RosTime) -> None:
        message = CameraInfo()
        message.header = Header(stamp=stamp, frame_id=self.camera_frame)
        message.width, message.height = self.calibration.resolution
        message.distortion_model = "plumb_bob"
        message.d = [0.0] * 5
        message.k = self.calibration.camera_matrix.reshape(-1).tolist()
        message.r = np.eye(3).reshape(-1).tolist()
        message.p = [
            float(self.calibration.camera_matrix[0, 0]), 0.0,
            float(self.calibration.camera_matrix[0, 2]), 0.0,
            0.0, float(self.calibration.camera_matrix[1, 1]),
            float(self.calibration.camera_matrix[1, 2]), 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        self.camera_info_publisher.publish(message)

    def _publish_points(self, lidar, stamp: RosTime) -> None:
        points = np.asarray(lidar["points"], dtype=np.float32)
        header = Header(stamp=stamp, frame_id=self.camera_frame)
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        rows = [tuple(float(value) for value in point) for point in points]
        self.points_publisher.publish(point_cloud2.create_cloud(header, fields, rows))

    def _publish_odometry(self, pose, stamp: RosTime) -> None:
        message = Odometry()
        message.header = Header(stamp=stamp, frame_id=self.odom_frame)
        message.child_frame_id = self.base_frame
        message.pose.pose.position.x = pose.x
        message.pose.pose.position.y = pose.y
        message.pose.pose.orientation.z = float(np.sin(pose.yaw / 2.0))
        message.pose.pose.orientation.w = float(np.cos(pose.yaw / 2.0))
        self.odom_publisher.publish(message)

        transform = TransformStamped()
        transform.header = message.header
        transform.child_frame_id = self.base_frame
        transform.transform.translation.x = pose.x
        transform.transform.translation.y = pose.y
        transform.transform.rotation = message.pose.pose.orientation
        self.tf_broadcaster.sendTransform(transform)


def _rotation_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        return (
            float((rotation[2, 1] - rotation[1, 2]) / scale),
            float((rotation[0, 2] - rotation[2, 0]) / scale),
            float((rotation[1, 0] - rotation[0, 1]) / scale),
            float(0.25 * scale),
        )
    diagonal = np.diag(rotation)
    index = int(np.argmax(diagonal))
    values = np.zeros(4, dtype=np.float64)
    values[index] = np.sqrt(max(0.0, 1.0 + 2.0 * diagonal[index] - trace)) / 2.0
    denominator = max(4.0 * values[index], 1e-12)
    if index == 0:
        values[1] = (rotation[0, 1] + rotation[1, 0]) / denominator
        values[2] = (rotation[0, 2] + rotation[2, 0]) / denominator
        values[3] = (rotation[2, 1] - rotation[1, 2]) / denominator
    elif index == 1:
        values[0] = (rotation[0, 1] + rotation[1, 0]) / denominator
        values[2] = (rotation[1, 2] + rotation[2, 1]) / denominator
        values[3] = (rotation[0, 2] - rotation[2, 0]) / denominator
    else:
        values[0] = (rotation[0, 2] + rotation[2, 0]) / denominator
        values[1] = (rotation[1, 2] + rotation[2, 1]) / denominator
        values[3] = (rotation[1, 0] - rotation[0, 1]) / denominator
    return tuple(float(value) for value in values)


def main() -> None:
    rclpy.init()
    node = A2D2ReplayNode()
    try:
        rclpy.spin(node)
    except ExternalShutdownException:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
