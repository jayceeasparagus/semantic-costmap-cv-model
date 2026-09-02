from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = PathJoinSubstitution(
        [FindPackageShare("semantic_costmap_ros"), "config", "a2d2_slam.yaml"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("autostart", default_value="true"),
            Node(
                package="semantic_costmap_ros",
                executable="a2d2_replay",
                name="a2d2_replay",
                output="screen",
                parameters=[config],
            ),
            Node(
                package="pointcloud_to_laserscan",
                executable="pointcloud_to_laserscan_node",
                name="pointcloud_to_laserscan",
                output="screen",
                remappings=[("cloud_in", "/points"), ("scan", "/scan")],
                parameters=[config],
            ),
            Node(
                package="slam_toolbox",
                executable="async_slam_toolbox_node",
                name="slam_toolbox",
                output="screen",
                parameters=[config],
            ),
            Node(
                package="semantic_costmap_ros",
                executable="semantic_costmap_node",
                name="semantic_costmap_node",
                output="screen",
                parameters=[config],
            ),
            Node(
                package="semantic_costmap_ros",
                executable="semantic_map_accumulator",
                name="semantic_map_accumulator",
                output="screen",
                parameters=[config],
            ),
        ]
    )
