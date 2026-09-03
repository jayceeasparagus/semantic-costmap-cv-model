from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare("semantic_costmap_ros"), "config", "a2d2_slam.yaml"]
    )
    config = LaunchConfiguration("params_file")
    slam_launch = PathJoinSubstitution(
        [FindPackageShare("slam_toolbox"), "launch", "online_async_launch.py"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("autostart", default_value="true"),
            DeclareLaunchArgument("params_file", default_value=default_config),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(slam_launch),
                launch_arguments={
                    "slam_params_file": config,
                    "use_sim_time": "false",
                    "autostart": LaunchConfiguration("autostart"),
                }.items(),
            ),
            Node(
                package="semantic_costmap_ros",
                executable="pointcloud_to_laserscan",
                name="pointcloud_to_laserscan",
                output="screen",
                remappings=[("cloud_in", "/points"), ("scan", "/scan")],
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
            # A2D2 is a short finite replay. Give SLAM, TF, and perception
            # subscribers time to initialize before the first sensor frame.
            TimerAction(
                period=3.0,
                actions=[
                    Node(
                        package="semantic_costmap_ros",
                        executable="a2d2_replay",
                        name="a2d2_replay",
                        output="screen",
                        parameters=[config],
                    )
                ],
            ),
        ]
    )
