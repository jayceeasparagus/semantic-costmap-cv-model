"""Plan from the live SLAM pose to a distant low-cost cell and save an overlay."""

import json
import math
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import OccupancyGrid, Path as RosPath
from PIL import Image, ImageDraw
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


class Nav2GoalDemo(Node):
    """Request one route using the current SLAM pose and the live costmap."""

    def __init__(self) -> None:
        super().__init__("nav2_goal_demo")
        self.declare_parameter("output_dir", "outputs/slam_nav2")
        self.declare_parameter("goal_search_min_m", 8.0)
        self.declare_parameter("goal_search_max_m", 30.0)
        self.declare_parameter("goal_search_width_m", 5.0)
        self.output_dir = Path(str(self.get_parameter("output_dir").value))
        self.min_distance = float(self.get_parameter("goal_search_min_m").value)
        self.max_distance = float(self.get_parameter("goal_search_max_m").value)
        self.search_width = float(self.get_parameter("goal_search_width_m").value)
        self.grid: OccupancyGrid | None = None
        self.planning = False
        self.goal_candidates = []
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(OccupancyGrid, "/global_costmap/costmap", self._grid_callback, 2)
        self.action_client = ActionClient(self, ComputePathToPose, "/compute_path_to_pose")
        self.timer = self.create_timer(1.0, self._try_plan)

    def _grid_callback(self, message: OccupancyGrid) -> None:
        self.grid = message

    def _try_plan(self) -> None:
        if self.planning or self.grid is None:
            return
        try:
            transform = self.tf_buffer.lookup_transform(
                self.grid.header.frame_id, "base_link", Time(), timeout=Duration(seconds=0.1)
            )
        except TransformException:
            return
        robot_pose = (float(transform.transform.translation.x), float(transform.transform.translation.y))
        yaw = self._yaw(transform.transform.rotation)
        # A short replay can leave the live Nav2 raster one cell away from
        # the exact SLAM pose.  Snap the displayed start to a nearby legal
        # low-cost cell so the planner receives valid endpoints.
        start = self._nearest_low_cost(robot_pose)
        if start is None:
            start = robot_pose
        self.goal_candidates = self._find_low_cost_goals(start, yaw)
        if not self.goal_candidates:
            self.get_logger().warning("No distant low-cost goal cell found yet")
            return
        self.planning = True
        self.timer.cancel()
        self._send_request(start, self.goal_candidates.pop(0))

    def _nearest_low_cost(self, point):
        grid = self.grid
        if grid is None:
            return None
        best, best_distance = None, float("inf")
        for row in range(2, max(2, grid.info.height - 2)):
            for column in range(2, max(2, grid.info.width - 2)):
                value = grid.data[row * grid.info.width + column]
                if value < 0 or value >= 100:
                    continue
                x = grid.info.origin.position.x + (column + 0.5) * grid.info.resolution
                y = grid.info.origin.position.y + (row + 0.5) * grid.info.resolution
                distance = math.hypot(x - point[0], y - point[1])
                if distance < best_distance:
                    best, best_distance = (x, y), distance
        return best

    def _find_low_cost_goals(self, start, yaw):
        grid = self.grid
        if grid is None:
            return None
        candidates = []
        for row in range(2, max(2, grid.info.height - 2)):
            for column in range(2, max(2, grid.info.width - 2)):
                value = grid.data[row * grid.info.width + column]
                # Inflation raises the numeric cost of nearby but still
                # traversable cells.  Only lethal and unknown cells are
                # rejected when selecting the demonstration goal.
                if value < 0 or value >= 100:
                    continue
                x = grid.info.origin.position.x + (column + 0.5) * grid.info.resolution
                y = grid.info.origin.position.y + (row + 0.5) * grid.info.resolution
                dx, dy = x - start[0], y - start[1]
                forward = dx * math.cos(yaw) + dy * math.sin(yaw)
                lateral = abs(-dx * math.sin(yaw) + dy * math.cos(yaw))
                distance = math.hypot(dx, dy)
                if self.min_distance <= forward <= self.max_distance and lateral <= self.search_width:
                    candidates.append((distance, (x, y)))
        candidates.sort(reverse=True, key=lambda item: item[0])
        return [point for _, point in candidates[:20]]

    def _send_request(self, start, goal) -> None:
        if not self.action_client.wait_for_server(timeout_sec=20.0):
            self.get_logger().error("Nav2 ComputePathToPose action server unavailable")
            self.destroy_node()
            return
        request = ComputePathToPose.Goal()
        request.start, request.goal = self._pose(*start), self._pose(*goal)
        request.use_start, request.planner_id = True, "GridBased"
        future = self.action_client.send_goal_async(request)
        future.add_done_callback(lambda result: self._goal_response(result, start, goal))

    def _goal_response(self, future, start, goal) -> None:
        handle = future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error("Nav2 rejected the path request")
            self.destroy_node()
            return
        result_future = handle.get_result_async()
        result_future.add_done_callback(lambda result: self._save_result(result, start, goal))

    def _save_result(self, future, start, goal) -> None:
        path = future.result().result.path
        if not path.poses:
            if self.goal_candidates:
                self.get_logger().warning("Candidate was unreachable; trying another green goal")
                self._send_request(start, self.goal_candidates.pop(0))
                return
            self.get_logger().error("Nav2 returned an empty path for all green candidates")
            self.destroy_node()
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._save_overlay(path, start, goal)
        payload = {
            "start": {"x": start[0], "y": start[1]},
            "goal": {"x": goal[0], "y": goal[1]},
            "path_points": len(path.poses),
            "path_length_m": self._path_length(path),
            "goal_selection": "farthest low-cost cell ahead of current SLAM pose",
        }
        (self.output_dir / "nav2_goal_result.json").write_text(json.dumps(payload, indent=2) + "\n")
        self.get_logger().info(f"Nav2 path saved to {self.output_dir / 'nav2_path_overlay.png'}")
        self.destroy_node()

    def _pose(self, x, y):
        pose = PoseStamped()
        pose.header.frame_id = self.grid.header.frame_id if self.grid else "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y, pose.pose.orientation.w = x, y, 1.0
        return pose

    def _save_overlay(self, path, start, goal):
        grid = self.grid
        if grid is None:
            return
        scale = 5
        image = Image.new("RGB", (grid.info.width * scale, grid.info.height * scale), "#555555")
        draw = ImageDraw.Draw(image)
        for row in range(grid.info.height):
            for column in range(grid.info.width):
                value = grid.data[row * grid.info.width + column]
                color = "#eeeeee" if value < 0 else ("#00cc00" if value < 100 else "#d63838")
                draw.rectangle((column * scale, (grid.info.height - row - 1) * scale, (column + 1) * scale - 1, (grid.info.height - row) * scale - 1), fill=color)
        points = [self._pixel(p.pose.position.x, p.pose.position.y, grid, scale) for p in path.poses]
        if len(points) > 1:
            draw.line(points, fill="#164de3", width=max(2, scale))
        self._draw_point(draw, start, grid, scale, "black")
        self._draw_point(draw, goal, grid, scale, "yellow")
        image.save(self.output_dir / "nav2_path_overlay.png")

    @staticmethod
    def _pixel(x, y, grid, scale):
        return (int((x - grid.info.origin.position.x) / grid.info.resolution * scale), int((grid.info.height - (y - grid.info.origin.position.y) / grid.info.resolution) * scale))

    def _draw_point(self, draw, point, grid, scale, color):
        x, y = self._pixel(point[0], point[1], grid, scale)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)

    @staticmethod
    def _path_length(path):
        return sum(math.hypot(b.pose.position.x - a.pose.position.x, b.pose.position.y - a.pose.position.y) for a, b in zip(path.poses, path.poses[1:]))

    @staticmethod
    def _yaw(rotation):
        return math.atan2(2.0 * rotation.w * rotation.z, 1.0 - 2.0 * rotation.z * rotation.z)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Nav2GoalDemo()
    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
