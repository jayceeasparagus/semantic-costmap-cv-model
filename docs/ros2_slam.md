# ROS 2 replay and SLAM Toolbox

The A2D2 replay node publishes the synchronized interfaces expected by the
perception stack:

| Topic | Message | Source |
|---|---|---|
| `/image` | `sensor_msgs/Image` | A2D2 RGB frame |
| `/camera_info` | `sensor_msgs/CameraInfo` | A2D2 camera calibration |
| `/points` | `sensor_msgs/PointCloud2` | A2D2 camera-frame points |
| `/odom` | `nav_msgs/Odometry` | bus-derived speed/yaw-rate integration |
| TF `odom -> base_link` | dynamic transform | same bus-derived pose |
| TF `base_link -> front_center_camera` | static transform | A2D2 calibration |

`pointcloud_to_laserscan` height-filters the cloud and supplies `/scan` to
SLAM Toolbox. SLAM Toolbox owns scan matching, loop closure, and the
`map -> odom` transform. The semantic accumulator then looks up the complete
`map -> odom -> base_link` chain when it receives painted points.

## Run in the ROS container

Use a larger contiguous A2D2 sequence for meaningful motion. The eight-frame
sample is an interface smoke test and may not produce a useful SLAM map.

```bash
source /opt/ros/jazzy/setup.bash
source /opt/semantic_ros/setup.bash
ros2 launch semantic_costmap_ros a2d2_slam.launch.py
```

The default config expects the bus file and replay data under
`data/raw/a2d2_playback`. Override those parameters for a larger sequence.
The replay's `timestamp_mode: bus` preserves A2D2 timestamps. Use
`timestamp_mode: now` when testing against a wall-clock-only ROS graph.

This is an integration path, not a claim that the small local sample has
enough travel for robust loop closure. For a headless check, verify that the
replay publishes all four message topics, `/scan` is nonempty, and SLAM Toolbox
publishes `map -> odom`.
