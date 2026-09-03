"""Convert a 3D PointCloud2 into the planar LaserScan used by SLAM Toolbox."""

import math

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener

from semantic_costmap.geometry import transform_points
from semantic_costmap_ros.conversions import transform_message_to_matrix


def points_to_ranges(
    points: np.ndarray,
    *,
    min_height: float,
    max_height: float,
    angle_min: float,
    angle_max: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
) -> np.ndarray:
    """Return the nearest valid XY range in each angular bin."""

    count = int(math.ceil((angle_max - angle_min) / angle_increment))
    ranges = np.full(count, np.inf, dtype=np.float32)
    if points.size == 0:
        return ranges

    distances = np.hypot(points[:, 0], points[:, 1])
    angles = np.arctan2(points[:, 1], points[:, 0])
    valid = (
        np.isfinite(points).all(axis=1)
        & (points[:, 2] >= min_height)
        & (points[:, 2] <= max_height)
        & (distances >= range_min)
        & (distances <= range_max)
        & (angles >= angle_min)
        & (angles < angle_max)
    )
    bins = ((angles[valid] - angle_min) / angle_increment).astype(np.int64)
    np.minimum.at(ranges, bins, distances[valid].astype(np.float32))
    return ranges


class PointCloudToLaserScanNode(Node):
    def __init__(self) -> None:
        super().__init__("pointcloud_to_laserscan")
        defaults = {
            "target_frame": "base_link",
            "transform_tolerance": 0.1,
            "min_height": -1.0,
            "max_height": 1.5,
            "angle_min": -math.pi,
            "angle_max": math.pi,
            "angle_increment": math.radians(0.5),
            "scan_time": 0.2,
            "range_min": 0.3,
            "range_max": 50.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.target_frame = str(self.get_parameter("target_frame").value)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.publisher = self.create_publisher(LaserScan, "scan", 2)
        self.subscription = self.create_subscription(
            PointCloud2, "cloud_in", self._cloud_callback, 2
        )

    def _value(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def _cloud_callback(self, message: PointCloud2) -> None:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                message.header.frame_id,
                Time.from_msg(message.header.stamp),
                timeout=Duration(seconds=self._value("transform_tolerance")),
            )
        except TransformException as error:
            self.get_logger().warning(f"Point cloud transform unavailable: {error}")
            return

        records = point_cloud2.read_points(
            message, field_names=("x", "y", "z"), skip_nans=True
        )
        if isinstance(records, np.ndarray) and records.dtype.names:
            points = np.column_stack((records["x"], records["y"], records["z"]))
        else:
            points = np.asarray(list(records), dtype=np.float64).reshape(-1, 3)
        points = transform_points(
            points, transform_message_to_matrix(transform.transform)
        )

        angle_min = self._value("angle_min")
        angle_max = self._value("angle_max")
        angle_increment = self._value("angle_increment")
        ranges = points_to_ranges(
            points,
            min_height=self._value("min_height"),
            max_height=self._value("max_height"),
            angle_min=angle_min,
            angle_max=angle_max,
            angle_increment=angle_increment,
            range_min=self._value("range_min"),
            range_max=self._value("range_max"),
        )

        scan = LaserScan()
        scan.header = message.header
        scan.header.frame_id = self.target_frame
        scan.angle_min = angle_min
        scan.angle_max = angle_min + (len(ranges) - 1) * angle_increment
        scan.angle_increment = angle_increment
        scan.time_increment = 0.0
        scan.scan_time = self._value("scan_time")
        scan.range_min = self._value("range_min")
        scan.range_max = self._value("range_max")
        scan.ranges = ranges.tolist()
        self.publisher.publish(scan)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PointCloudToLaserScanNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
