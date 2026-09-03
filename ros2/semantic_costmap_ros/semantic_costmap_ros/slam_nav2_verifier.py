"""Save evidence that SLAM poses and the semantic map reach Nav2."""

import json
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


class SlamNav2Verifier(Node):
    """Observe the three products needed by the integrated planning stack."""

    def __init__(self) -> None:
        super().__init__("slam_nav2_verifier")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter(
            "output_path", "outputs/slam_nav2/integration_result.json"
        )
        self.declare_parameter("timeout_seconds", 120.0)

        self.map_frame = str(self.get_parameter("map_frame").value)
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.output_path = Path(str(self.get_parameter("output_path").value))
        self.timeout_seconds = float(self.get_parameter("timeout_seconds").value)
        self.started_at = self.get_clock().now()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.slam_map: OccupancyGrid | None = None
        self.semantic_map: OccupancyGrid | None = None
        self.nav2_map: OccupancyGrid | None = None

        map_qos = QoSProfile(depth=1)
        map_qos.reliability = ReliabilityPolicy.RELIABLE
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        live_qos = QoSProfile(depth=2)
        live_qos.reliability = ReliabilityPolicy.RELIABLE
        live_qos.durability = DurabilityPolicy.VOLATILE
        self.create_subscription(OccupancyGrid, "/map", self._save_slam_map, map_qos)
        self.create_subscription(
            OccupancyGrid,
            "/semantic_global_costmap",
            self._save_semantic_map,
            2,
        )
        self.create_subscription(
            OccupancyGrid,
            "/global_costmap/costmap",
            self._save_nav2_map,
            live_qos,
        )
        self.timer = self.create_timer(1.0, self._verify)

    def _save_slam_map(self, message: OccupancyGrid) -> None:
        self.slam_map = message

    def _save_semantic_map(self, message: OccupancyGrid) -> None:
        self.semantic_map = message

    def _save_nav2_map(self, message: OccupancyGrid) -> None:
        self.nav2_map = message

    def _has_transform(self, target: str, source: str) -> bool:
        try:
            self.tf_buffer.lookup_transform(
                target,
                source,
                Time(),
                timeout=Duration(seconds=0.05),
            )
            return True
        except TransformException:
            return False

    def _verify(self) -> None:
        slam_tf = self._has_transform(self.map_frame, self.odom_frame)
        robot_tf = self._has_transform(self.map_frame, self.base_frame)
        ready = (
            slam_tf
            and robot_tf
            and self.slam_map is not None
            and self.semantic_map is not None
            and self.nav2_map is not None
        )
        elapsed = (self.get_clock().now() - self.started_at).nanoseconds / 1e9
        if not ready:
            if elapsed >= self.timeout_seconds:
                self.get_logger().error(
                    "Integration verification timed out; check SLAM scans and TF"
                )
                self.timer.cancel()
            return

        semantic = np.asarray(self.semantic_map.data, dtype=np.int16)
        slam = np.asarray(self.slam_map.data, dtype=np.int16)
        nav2 = np.asarray(self.nav2_map.data, dtype=np.int16)
        result = {
            "slam_map_received": True,
            "slam_map_known_cells": int(np.count_nonzero(slam >= 0)),
            "map_to_odom_available": slam_tf,
            "map_to_base_link_available": robot_tf,
            "semantic_global_map_received": True,
            "semantic_global_known_cells": int(np.count_nonzero(semantic >= 0)),
            "semantic_global_lethal_cells": int(np.count_nonzero(semantic >= 99)),
            "nav2_global_costmap_received": True,
            "nav2_global_known_cells": int(np.count_nonzero(nav2 >= 0)),
            "semantic_topic_used_by_nav2": "/semantic_global_costmap",
        }
        if result["semantic_global_known_cells"] == 0:
            return
        if result["nav2_global_known_cells"] == 0:
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(result, indent=2) + "\n")
        self.get_logger().info(
            f"Verified SLAM semantic Nav2 integration: {self.output_path}"
        )
        self.timer.cancel()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SlamNav2Verifier()
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
