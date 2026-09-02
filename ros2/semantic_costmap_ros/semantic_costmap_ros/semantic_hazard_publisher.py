"""Publish a deterministic semantic hazard grid for the Nav2 demo."""

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from std_msgs.msg import Header


class SemanticHazardPublisher(Node):
    def __init__(self) -> None:
        super().__init__("semantic_hazard_publisher")
        self.declare_parameter("topic", "/semantic_costmap")
        self.declare_parameter("width", 50)
        self.declare_parameter("height", 30)
        self.declare_parameter("resolution", 0.4)
        self.declare_parameter("origin_x", -10.0)
        self.declare_parameter("origin_y", -6.0)
        self.declare_parameter("hazard", False)
        self.declare_parameter("publish_rate", 2.0)
        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.resolution = float(self.get_parameter("resolution").value)
        self.origin_x = float(self.get_parameter("origin_x").value)
        self.origin_y = float(self.get_parameter("origin_y").value)
        self.publisher = self.create_publisher(
            OccupancyGrid,
            str(self.get_parameter("topic").value),
            2,
        )
        period = 1.0 / max(float(self.get_parameter("publish_rate").value), 0.1)
        self.timer = self.create_timer(period, self.publish_grid)

    def publish_grid(self) -> None:
        data = np.zeros((self.height, self.width), dtype=np.int8)
        if bool(self.get_parameter("hazard").value):
            barrier_column = self.width // 2
            data[2:-1, barrier_column - 1:barrier_column + 1] = 100
        message = OccupancyGrid()
        message.header = Header(
            stamp=self.get_clock().now().to_msg(),
            frame_id="map",
        )
        message.info.resolution = self.resolution
        message.info.width = self.width
        message.info.height = self.height
        message.info.origin.position.x = self.origin_x
        message.info.origin.position.y = self.origin_y
        message.info.origin.orientation.w = 1.0
        message.data = data.ravel().tolist()
        self.publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SemanticHazardPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
