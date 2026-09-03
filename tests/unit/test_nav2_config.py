from pathlib import Path

import yaml


CONFIG_PATH = Path(
    "ros2/semantic_costmap_ros/config/nav2_semantic_layers.yaml"
)
SLAM_NAV2_CONFIG_PATH = Path(
    "ros2/semantic_costmap_ros/config/a2d2_slam_nav2.yaml"
)
A2D2_SLAM_CONFIG_PATH = Path(
    "ros2/semantic_costmap_ros/config/a2d2_slam.yaml"
)


def test_inflation_follows_semantic_layer_in_local_and_global_costmaps():
    config = yaml.safe_load(CONFIG_PATH.read_text())

    for costmap_name in ("local_costmap", "global_costmap"):
        parameters = config[costmap_name][costmap_name]["ros__parameters"]
        plugins = parameters["plugins"]
        assert plugins.index("semantic_layer") < plugins.index("inflation_layer")
        assert parameters["inflation_layer"]["inflation_radius"] > 0.0
        assert (
            parameters["inflation_layer"]["plugin"]
            == "nav2_costmap_2d::InflationLayer"
        )


def test_semantic_topics_select_local_and_persistent_maps():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    local = config["local_costmap"]["local_costmap"]["ros__parameters"]
    global_map = config["global_costmap"]["global_costmap"]["ros__parameters"]

    assert local["semantic_layer"]["topic"] == "/semantic_costmap"
    assert global_map["semantic_layer"]["topic"] == "/semantic_global_costmap"


def test_live_nav2_uses_slam_map_and_persistent_semantic_grid():
    config = yaml.safe_load(SLAM_NAV2_CONFIG_PATH.read_text())
    parameters = config["global_costmap"]["global_costmap"]["ros__parameters"]

    assert parameters["global_frame"] == "map"
    assert parameters["robot_base_frame"] == "base_link"
    assert parameters["plugins"] == [
        "static_layer",
        "semantic_layer",
        "inflation_layer",
    ]
    assert parameters["semantic_layer"]["topic"] == "/semantic_global_costmap"
    assert parameters["plugins"].index("semantic_layer") < parameters["plugins"].index(
        "inflation_layer"
    )


def test_a2d2_live_replay_uses_current_ros_time_and_slam_tf_frames():
    config = yaml.safe_load(A2D2_SLAM_CONFIG_PATH.read_text())
    replay = config["a2d2_replay"]["ros__parameters"]
    slam = config["slam_toolbox"]["ros__parameters"]

    assert replay["timestamp_mode"] == "now"
    assert replay["odom_frame"] == slam["odom_frame"] == "odom"
    assert replay["base_frame"] == slam["base_frame"] == "base_link"
    assert slam["map_frame"] == "map"
