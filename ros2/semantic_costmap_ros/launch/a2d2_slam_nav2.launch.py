"""Run A2D2 replay, SLAM semantic accumulation, and Nav2 planning together."""

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("semantic_costmap_ros")
    slam_launch = PathJoinSubstitution(
        [package_share, "launch", "a2d2_slam.launch.py"]
    )
    nav2_config = PathJoinSubstitution(
        [package_share, "config", "a2d2_slam_nav2.yaml"]
    )

    return LaunchDescription(
        [
            IncludeLaunchDescription(PythonLaunchDescriptionSource(slam_launch)),
            Node(
                package="nav2_planner",
                executable="planner_server",
                name="planner_server",
                output="screen",
                parameters=[nav2_config],
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_navigation",
                output="screen",
                parameters=[
                    {
                        "autostart": True,
                        "node_names": ["planner_server"],
                    }
                ],
            ),
            Node(
                package="semantic_costmap_ros",
                executable="slam_nav2_verifier",
                name="slam_nav2_verifier",
                output="screen",
                parameters=[nav2_config],
            ),
        ]
    )
