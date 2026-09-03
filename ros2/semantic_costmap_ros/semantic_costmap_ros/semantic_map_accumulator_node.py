"""Accumulate painted semantic points using the map-to-sensor TF pose."""

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformException, TransformListener

from semantic_costmap.geometry import transform_points
from semantic_costmap.mapping import GlobalMapConfig, PoseAwareAccumulator
from semantic_costmap_ros.conversions import transform_message_to_matrix


def painted_cloud_arrays(message: PointCloud2):
    fields = ("x", "y", "z", "class_id", "confidence", "cost")
    records = point_cloud2.read_points(message, field_names=fields, skip_nans=True)
    if isinstance(records, np.ndarray) and records.dtype.names:
        points = np.column_stack((records["x"], records["y"], records["z"]))
        classes = records["class_id"].astype(np.uint8)
        costs = records["cost"].astype(np.int16)
        return points, classes, costs
    array = np.asarray(list(records), dtype=np.float64).reshape(-1, 6)
    return array[:, :3], array[:, 3].astype(np.uint8), array[:, 5].astype(np.int16)


class SemanticMapAccumulatorNode(Node):
    def __init__(self) -> None:
        super().__init__("semantic_map_accumulator")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("resolution", 0.20)
        self.declare_parameter("map_range", 50.0)
        self.declare_parameter("dynamic_decay_seconds", 2.0)
        self.map_frame = self.get_parameter("map_frame").value
        self.odom_frame = self.get_parameter("odom_frame").value
        map_range = float(self.get_parameter("map_range").value)
        self.accumulator = PoseAwareAccumulator(
            GlobalMapConfig(
                resolution=float(self.get_parameter("resolution").value),
                x_min=-map_range,
                x_max=map_range,
                y_min=-map_range,
                y_max=map_range,
                dynamic_decay_seconds=float(
                    self.get_parameter("dynamic_decay_seconds").value
                ),
            )
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.subscription = self.create_subscription(
            PointCloud2,
            "painted_points",
            self._cloud_callback,
            2,
        )
        self.publisher = self.create_publisher(
            OccupancyGrid,
            "semantic_global_costmap",
            2,
        )
        self.pending_cloud: PointCloud2 | None = None
        self.retry_timer = self.create_timer(0.5, self._retry_pending_cloud)

    def _cloud_callback(self, message: PointCloud2) -> None:
        try:
            sensor_time = Time.from_msg(message.header.stamp)
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    message.header.frame_id,
                    sensor_time,
                    timeout=Duration(seconds=0.1),
                )
            except TransformException:
                # Perception arrives after SLAM has processed the scan. Apply
                # the newest map->odom correction to the sensor-time
                # odom->sensor pose instead of pretending the cloud is new.
                try:
                    transform = self.tf_buffer.lookup_transform_full(
                        self.map_frame,
                        Time(),
                        message.header.frame_id,
                        sensor_time,
                        self.odom_frame,
                        timeout=Duration(seconds=0.1),
                    )
                except TransformException:
                    # Very short replays can begin before TF listeners cache
                    # the first odometry sample. Seed the map with the latest
                    # available SLAM pose; continuous runs normally use one of
                    # the timestamp-preserving paths above.
                    transform = self.tf_buffer.lookup_transform(
                        self.map_frame,
                        message.header.frame_id,
                        Time(),
                        timeout=Duration(seconds=0.1),
                    )
            points, classes, costs = painted_cloud_arrays(message)
            points_map = transform_points(
                points,
                transform_message_to_matrix(transform.transform),
            )
        except (TransformException, ValueError) as error:
            self.pending_cloud = message
            self.get_logger().warning(f"Map accumulation skipped: {error}")
            return

        timestamp = Time.from_msg(message.header.stamp).nanoseconds / 1e9
        self.accumulator.update_map_points(
            points_map,
            classes,
            costs,
            timestamp,
        )
        # The map contains historical evidence, but this grid publication is
        # current. A fresh stamp prevents Nav2 from aging out a delayed update.
        self._publish_grid(timestamp, self.get_clock().now().to_msg())
        if self.pending_cloud is message:
            self.pending_cloud = None

    def _retry_pending_cloud(self) -> None:
        if self.pending_cloud is not None:
            self._cloud_callback(self.pending_cloud)

    def _publish_grid(self, timestamp: float, stamp) -> None:
        costs = self.accumulator.grid(timestamp)
        message = OccupancyGrid()
        message.header = Header(stamp=stamp, frame_id=self.map_frame)
        message.info.resolution = self.accumulator.config.resolution
        message.info.width = self.accumulator.config.width
        message.info.height = self.accumulator.config.height
        message.info.origin.position.x = self.accumulator.config.x_min
        message.info.origin.position.y = self.accumulator.config.y_min
        message.info.origin.orientation.w = 1.0
        occupancy = np.full(costs.shape, -1, dtype=np.int8)
        known = costs != 255
        occupancy[known] = np.rint(
            costs[known].astype(np.float32) * 100.0 / 254.0
        ).astype(np.int8)
        message.data = occupancy.ravel().tolist()
        self.publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SemanticMapAccumulatorNode()
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
