from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    package_share = FindPackageShare("semantic_costmap_ros")
    nav2_config = PathJoinSubstitution([package_share, "config", "nav2_demo.yaml"])
    map_config = PathJoinSubstitution([package_share, "config", "nav2_demo_map.yaml"])
    return LaunchDescription(
        [
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="map_to_base_link",
                arguments=["0", "0", "0", "0", "0", "0", "map", "base_link"],
                output="screen",
            ),
            Node(
                package="nav2_map_server",
                executable="map_server",
                name="map_server",
                output="screen",
                parameters=[{"yaml_filename": map_config}],
            ),
            ExecuteProcess(
                cmd=[
                    "python3",
                    "-m",
                    "semantic_costmap_ros.semantic_hazard_publisher",
                    "--ros-args",
                    "-r",
                    "__node:=semantic_hazard_publisher",
                    "--params-file",
                    nav2_config,
                ],
                output="screen",
            ),
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
                        "node_names": ["map_server", "planner_server"],
                    }
                ],
            ),
            ExecuteProcess(
                cmd=[
                    "python3",
                    "-m",
                    "semantic_costmap_ros.nav2_path_demo",
                    "--ros-args",
                    "-r",
                    "__node:=nav2_path_demo",
                ],
                output="screen",
            ),
        ]
    )
