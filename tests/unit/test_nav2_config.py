from pathlib import Path

import yaml


CONFIG_PATH = Path(
    "ros2/semantic_costmap_ros/config/nav2_semantic_layers.yaml"
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
