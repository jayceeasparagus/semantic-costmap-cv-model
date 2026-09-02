from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    parameters = PathJoinSubstitution(
        [FindPackageShare("semantic_costmap_ros"), "config", "semantic_costmap.yaml"]
    )
    return LaunchDescription(
        [
            Node(
                package="semantic_costmap_ros",
                executable="semantic_costmap_node",
                name="semantic_costmap_node",
                output="screen",
                parameters=[parameters],
            )
        ]
    )
