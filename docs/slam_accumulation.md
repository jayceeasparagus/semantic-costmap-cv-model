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

An offline headless check uses explicit synthetic poses because the small A2D2
sample does not provide the ROS TF stream:

```bash
python tools/demo_pose_accumulation.py
```

This demo validates coordinate placement only. A live run should source SLAM
Toolbox or localization transforms and can be inspected in RViz2 by displaying
the published occupancy grid and painted point cloud.
