# Pose and SLAM accumulation

SLAM remains an upstream pose provider. The project does not alter SLAM
Toolbox's occupancy map or implement loop closure. Instead,
`semantic_map_accumulator` consumes the standard TF chain from the painted point
frame into `map`. With SLAM Toolbox this is normally composed from
`map -> odom -> base_link`.

Each painted point is transformed at its sensor timestamp and inserted into a
global metric grid. Static classes use maximum-cost accumulation so later
drivable predictions cannot erase an obstacle. Dynamic-obstacle cells live in a
separate layer and expire after `dynamic_decay_seconds`; when they expire, any
underlying static cost becomes visible again.

The accumulator subscribes to `painted_points` and publishes
`semantic_global_costmap`. Configure the Nav2 semantic layer to consume either
the local `semantic_costmap` or persistent `semantic_global_costmap`, depending
on whether the robot has a valid map-frame pose.

An offline headless check can use explicit poses without ROS:

```bash
python tools/demo_pose_accumulation.py
```

This demo validates coordinate placement only. The integrated ROS launch uses
A2D2 bus odometry for `odom -> base_link`, converts LiDAR into laser scans, and
runs SLAM Toolbox as the owner of `map -> odom`:

```bash
ros2 launch semantic_costmap_ros a2d2_slam_nav2.launch.py
```

The transform lookup performed for every painted cloud therefore includes the
SLAM correction instead of treating odometry as a global pose. The resulting
`semantic_global_costmap` is persistent in `map` and is consumed by the Nav2
global semantic layer. The launch writes a verification record to
`outputs/slam_nav2/integration_result.json` after observing the SLAM map, both
required TF transforms, the semantic map, and Nav2's global costmap.
