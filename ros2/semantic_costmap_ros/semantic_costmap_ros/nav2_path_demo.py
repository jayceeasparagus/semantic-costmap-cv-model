"""Request Nav2 paths with and without a semantic hazard and save evidence."""

import json
import math
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import OccupancyGrid, Path as RosPath
from PIL import Image, ImageDraw
from rcl_interfaces.msg import Parameter as ParameterMessage
from rcl_interfaces.srv import SetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter


class Nav2PathDemo(Node):
    def __init__(self) -> None:
        super().__init__("nav2_path_demo")
        self.declare_parameter("output_dir", "outputs/nav2_demo")
        self.declare_parameter("start_x", -7.0)
        self.declare_parameter("start_y", -3.0)
        self.declare_parameter("goal_x", 7.0)
        self.declare_parameter("goal_y", -3.0)
        self.declare_parameter("goal_tolerance", 0.6)
        self.output_dir = Path(str(self.get_parameter("output_dir").value))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start = (float(self.get_parameter("start_x").value), float(self.get_parameter("start_y").value))
        self.goal = (float(self.get_parameter("goal_x").value), float(self.get_parameter("goal_y").value))
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.semantic_grid: OccupancyGrid | None = None
        self.create_subscription(OccupancyGrid, "/semantic_costmap", self._grid_callback, 2)
        self.action_client = ActionClient(self, ComputePathToPose, "/compute_path_to_pose")
        self.parameter_client = self.create_client(
            SetParameters, "/semantic_hazard_publisher/set_parameters"
        )

    def _grid_callback(self, message: OccupancyGrid) -> None:
        self.semantic_grid = message

    def pose(self, x: float, y: float) -> PoseStamped:
        message = PoseStamped()
        message.header.frame_id = "map"
        message.header.stamp = self.get_clock().now().to_msg()
        message.pose.position.x = x
        message.pose.position.y = y
        message.pose.orientation.w = 1.0
        return message

    def request_path(self) -> RosPath:
        if not self.action_client.wait_for_server(timeout_sec=30.0):
            raise RuntimeError("Nav2 ComputePathToPose action server did not start")
        goal = ComputePathToPose.Goal()
        goal.start = self.pose(*self.start)
        goal.goal = self.pose(*self.goal)
        goal.use_start = True
        goal.planner_id = "GridBased"
        goal_handle_future = self.action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, goal_handle_future, timeout_sec=30.0)
        goal_handle = goal_handle_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError("Nav2 rejected ComputePathToPose goal")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)
        result = result_future.result()
        if result is None or not result.result.path.poses:
            raise RuntimeError("Nav2 returned an empty path")
        return result.result.path

    def set_hazard(self, enabled: bool) -> None:
        if not self.parameter_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("semantic hazard parameter service did not start")
        request = SetParameters.Request()
        request.parameters = [
            Parameter("hazard", Parameter.Type.BOOL, enabled).to_parameter_msg()
        ]
        future = self.parameter_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        response = future.result()
        if response is None or not response.results[0].successful:
            reason = response.results[0].reason if response else "no response"
            raise RuntimeError(f"could not set semantic hazard: {reason}")

    def wait_for_grid_update(self, previous_stamp) -> None:
        for _ in range(30):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.semantic_grid is not None and self.semantic_grid.header.stamp != previous_stamp:
                return
        raise RuntimeError("semantic hazard grid did not update")

    def settle_costmap(self) -> None:
        # The semantic layer consumes the grid on its callback/update cycle.
        # Give Nav2 a few cycles after the publisher changes state.
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.1)

    def run(self) -> None:
        self.set_hazard(False)
        baseline = self.request_path()
        old_stamp = self.semantic_grid.header.stamp if self.semantic_grid else None
        self.set_hazard(True)
        self.wait_for_grid_update(old_stamp)
        self.settle_costmap()
        hazard_path = self.request_path()
        self._validate_path(hazard_path)
        self._save_path(baseline, "path_without_semantics.json")
        self._save_path(hazard_path, "path_with_semantics.json")
        self._save_overlay(baseline, hazard_path)
        result = {
            "baseline_points": len(baseline.poses),
            "hazard_points": len(hazard_path.poses),
            "baseline_length_m": self._length(baseline),
            "hazard_length_m": self._length(hazard_path),
            "semantic_hazard_changed_route": self._length(hazard_path) > self._length(baseline) + 0.5,
        }
        if not result["semantic_hazard_changed_route"]:
            raise RuntimeError("semantic hazard did not produce a longer alternate route")
        (self.output_dir / "planning_result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))

    def _validate_path(self, path: RosPath) -> None:
        endpoint = path.poses[-1].pose.position
        distance = math.hypot(endpoint.x - self.goal[0], endpoint.y - self.goal[1])
        if distance > self.goal_tolerance:
            raise RuntimeError(f"path endpoint is {distance:.2f} m from goal")
        if self.semantic_grid is None:
            raise RuntimeError("no semantic grid received")
        grid = self.semantic_grid
        for pose in path.poses:
            point = pose.pose.position
            column = int((point.x - grid.info.origin.position.x) / grid.info.resolution)
            row = int((point.y - grid.info.origin.position.y) / grid.info.resolution)
            if 0 <= row < grid.info.height and 0 <= column < grid.info.width:
                if grid.data[row * grid.info.width + column] >= 100:
                    raise RuntimeError("planned path enters a lethal semantic cell")

    @staticmethod
    def _length(path: RosPath) -> float:
        return sum(
            math.hypot(
                current.pose.position.x - previous.pose.position.x,
                current.pose.position.y - previous.pose.position.y,
            )
            for previous, current in zip(path.poses, path.poses[1:])
        )

    def _save_path(self, path: RosPath, filename: str) -> None:
        payload = [
            {
                "x": pose.pose.position.x,
                "y": pose.pose.position.y,
            }
            for pose in path.poses
        ]
        (self.output_dir / filename).write_text(json.dumps(payload, indent=2) + "\n")

    def _save_overlay(self, baseline: RosPath, hazard: RosPath) -> None:
        grid = self.semantic_grid
        if grid is None:
            raise RuntimeError("no semantic grid available for overlay")
        scale = 12
        image = Image.new("RGB", (grid.info.width * scale, grid.info.height * scale), "white")
        draw = ImageDraw.Draw(image)
        for row in range(grid.info.height):
            for column in range(grid.info.width):
                value = grid.data[row * grid.info.width + column]
                color = (235, 235, 235) if value < 100 else (210, 55, 55)
                draw.rectangle(
                    (column * scale, (grid.info.height - row - 1) * scale,
                     (column + 1) * scale - 1, (grid.info.height - row) * scale - 1),
                    fill=color,
                )
        self._draw_path(draw, baseline, grid, scale, (40, 90, 220))
        self._draw_path(draw, hazard, grid, scale, (20, 150, 40))
        self._draw_point(draw, self.start, grid, scale, "black")
        self._draw_point(draw, self.goal, grid, scale, "yellow")
        image.save(self.output_dir / "path_overlay.png")

    @staticmethod
    def _draw_path(draw, path, grid, scale, color):
        points = []
        for pose in path.poses:
            point = pose.pose.position
            points.append((
                int((point.x - grid.info.origin.position.x) / grid.info.resolution * scale),
                int((grid.info.height - (point.y - grid.info.origin.position.y) / grid.info.resolution) * scale),
            ))
        if len(points) > 1:
            draw.line(points, fill=color, width=3)

    @staticmethod
    def _draw_point(draw, point, grid, scale, color):
        x = int((point[0] - grid.info.origin.position.x) / grid.info.resolution * scale)
        y = int((grid.info.height - (point[1] - grid.info.origin.position.y) / grid.info.resolution) * scale)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Nav2PathDemo()
    try:
        node.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
