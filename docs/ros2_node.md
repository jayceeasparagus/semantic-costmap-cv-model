# ROS 2 perception node

`semantic_costmap_ros` wraps the tested Python pipeline in one ROS 2 node. It
subscribes to:

- `image` (`sensor_msgs/Image`, `rgb8` or `bgr8`);
- `camera_info` (`sensor_msgs/CameraInfo`);
- `points` (`sensor_msgs/PointCloud2`).

The node uses TF to transform LiDAR points into both the camera optical frame
and `base_link`. It rejects image/cloud pairs farther apart than the configured
time tolerance, projects with the `CameraInfo` matrix, and publishes:

- `semantic_mask` (`sensor_msgs/Image`);
- `painted_points` (`sensor_msgs/PointCloud2`);
- `semantic_costmap` (`nav_msgs/OccupancyGrid`).

Build and run from the repository root:

```bash
source /opt/ros/jazzy/setup.bash
python3 -m pip install -e .
colcon build --base-paths ros2 --symlink-install
source install/setup.bash
ros2 launch semantic_costmap_ros semantic_costmap.launch.py
```

Remap the three input topics as needed. The point-cloud frame must have valid TF
transforms to the `CameraInfo.header.frame_id` and configured `base_frame`.
The published `OccupancyGrid` uses `-1` for unknown and scales native costs into
the message's standard 0-100 range. The Nav2 plugin converts them back to native
costmap values.
