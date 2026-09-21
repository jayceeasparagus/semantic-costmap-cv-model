# Demonstration checklist

This sequence verifies the primary semantic-mapping workflow with inspectable artifacts. Run it
from the repository root after placing the local A2D2 samples and epoch-29
checkpoint in their documented locations.

```bash
source .venv/bin/activate

python tools/run_inference.py --device cpu
python tools/validate_calibration_projection.py
python tools/paint_semantic_points.py --device cpu
python tools/generate_costmap.py
python tools/run_playback.py --device cpu --stride 5 --max-frames 60 --gif-fps 10
python tools/demo_pose_accumulation.py
```

To use measured poses during playback, create a CSV with the columns
`frame_id,timestamp,x,y,yaw` and run:

```bash
python tools/run_playback.py --device cpu --stride 5 --max-frames 60 \
  --gif-fps 10 --poses-csv poses.csv
```

Expected outputs:

| Stage | Main artifact |
|---|---|
| segmentation | `outputs/inference/*_overlay.png` |
| calibration | `outputs/calibration/projection_validation.png` |
| point painting | `outputs/fusion/painted_points_overlay.png` |
| frame-level semantic grid | `outputs/costmap/semantic_costmap_preview.png` |
| playback | `outputs/playback/semantic_costmap_playback.gif` |
| benchmark | `outputs/playback/benchmark.json` |
| persistent map | `outputs/accumulation/accumulated_costmap_preview.png` |
| map confidence | `outputs/accumulation/accumulated_confidence.png` |

Run all automated checks afterward:

```bash
tools/run_checks.sh
```

The route-planning code is an optional experiment and is not part of the main
mapping demonstration.
