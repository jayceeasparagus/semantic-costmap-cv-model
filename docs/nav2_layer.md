# Nav2 semantic costmap layer

`semantic_costmap_layer` is a C++ `nav2_costmap_2d::Layer` plugin. It subscribes
to the perception node's `nav_msgs/OccupancyGrid`, transforms each grid cell
into the Nav2 costmap frame with TF, converts occupancy values back to Nav2's
0-254 range, and merges them with the master grid using the maximum cost.

Unknown semantic cells are skipped. If the Nav2 master cell is unknown, valid
semantic evidence initializes it; otherwise the layer never lowers an existing
static-map or obstacle-layer cost.

Add the plugin before inflation in a Nav2 costmap configuration:

```yaml
local_costmap:
  local_costmap:
    ros__parameters:
      plugins: ["obstacle_layer", "semantic_layer", "inflation_layer"]
      semantic_layer:
        plugin: "semantic_costmap_layer/SemanticCostmapLayer"
        enabled: true
        topic: /semantic_costmap
        maximum_age: 1.0
```

Build and test with:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --base-paths ros2 --symlink-install
colcon test --base-paths ros2 --packages-select semantic_costmap_layer
colcon test-result --verbose
```
