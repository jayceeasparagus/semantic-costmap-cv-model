# Demonstration checklist

This sequence verifies the complete project with inspectable artifacts. Run it
from the repository root after placing the local A2D2 samples and epoch-29
checkpoint in their documented locations.

```bash
source .venv/bin/activate

python tools/run_inference.py --device cpu
python tools/validate_calibration_projection.py
python tools/paint_semantic_points.py --device cpu
python tools/generate_costmap.py
python tools/run_playback.py --device cpu --max-frames 8
python tools/demo_pose_accumulation.py
```

To use measured poses during playback, create a CSV with the columns
`frame_id,timestamp,x,y,yaw` and run:

```bash
python tools/run_playback.py --device cpu --poses-csv poses.csv
```

Expected outputs:

| Stage | Main artifact |
|---|---|
| segmentation | `outputs/inference/*_overlay.png` |
| calibration | `outputs/calibration/projection_validation.png` |
| point painting | `outputs/fusion/painted_points_overlay.png` |
| local grid | `outputs/costmap/semantic_costmap_preview.png` |
| playback | `outputs/playback/semantic_costmap_playback.gif` |
| benchmark | `outputs/playback/benchmark.json` |
| accumulation | `outputs/accumulation/accumulated_costmap_preview.png` |

Run all automated checks afterward:

```bash
tools/run_checks.sh
```

For a ROS demonstration, build and source the workspace, launch the nodes, and
show `/semantic_mask`, `/painted_points`, `/semantic_costmap`, and
`/semantic_global_costmap` in RViz2. A live or recorded sensor source must
provide `image`, `camera_info`, `points`, and the required TF chain.
