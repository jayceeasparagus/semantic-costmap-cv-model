# Testing

Run every available local check from the repository virtual environment:

```bash
source .venv/bin/activate
tools/run_checks.sh
```

The test layers are:

- focused Python unit tests for model shape, calibration, projection, fusion,
  costmap aggregation, playback pairing, and temporal accumulation;
- one integration test that runs the local epoch-29 checkpoint and A2D2 sample
  through the complete offline pipeline;
- ROS 2 pytest checks for message conversion;
- GoogleTest checks for Nav2 cost conversion and conservative merging;
- a headless colcon build, plugin registration check, and node smoke tests.

The integration test skips in a clean clone because datasets and checkpoints
are intentionally not committed. GitHub Actions runs the portable unit suite on
Python 3.12. ROS packages are validated locally against ROS 2 Jazzy because the
standard Python CI image does not contain Nav2.
