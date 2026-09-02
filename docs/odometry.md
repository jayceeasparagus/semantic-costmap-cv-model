# A2D2 bus-derived odometry

A2D2 stores bus data as a list of camera-frame records. Each record includes a
camera timestamp and short signal histories. The project uses vehicle speed and
the z-axis angular velocity to make a relative planar trajectory:

```text
v = vehicle_speed / 3.6
w = angular_velocity_omega_z * pi / 180
x += v cos(yaw + w dt / 2) dt
y += v sin(yaw + w dt / 2) dt
yaw += w dt
```

This is a useful deterministic replay pose source. It is not a replacement for
SLAM: integration drifts, and the initial yaw is supplied by the user. A live
ROS system should instead use SLAM Toolbox's `map -> odom -> base_link` TF.

## Generate poses

The bus file is public A2D2 data and should remain outside Git. For example:

```bash
python tools/build_a2d2_poses.py \
  --bus-json data/raw/a2d2_playback/bus/20180807145028_bus_signals.json \
  --output-csv outputs/poses/20180807_bus_odometry.csv
```

If matching camera metadata JSON files have been downloaded, add:

```bash
--camera-metadata-dir data/raw/a2d2_playback/camera_metadata
```

The output columns are `frame_id,timestamp,x,y,yaw`, which is the same format
accepted by `tools/run_playback.py`.

## Replay and accumulate

```bash
python tools/run_playback.py \
  --poses-csv outputs/poses/20180807_bus_odometry.csv \
  --max-frames 8 \
  --device cpu
```

The playback saves a GIF, the persistent map arrays and preview, and a
trajectory preview. For a meaningful map, use a larger contiguous A2D2
sequence and download the corresponding camera, LiDAR, and bus records.
