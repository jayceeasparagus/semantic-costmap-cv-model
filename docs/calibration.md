# Camera-LiDAR calibration

A2D2's camera-aligned LiDAR files store each 3D point in the selected camera's
coordinate frame. They also include `row` and `col` convenience values. The
pipeline does not use those convenience values during normal projection.

For the front-center camera, the calibrated coordinate convention is:

- `+x`: forward/depth;
- `+y`: left;
- `+z`: up.

Using the undistorted camera matrix, a point is projected with:

```text
column = cx - fx * y / x
row    = cy - fy * z / x
```

Points behind the camera or outside the calibrated image resolution are
discarded. `geometry/calibration.py` also implements the A2D2 view transforms
needed to convert points between a sensor view and the vehicle frame.

Run the independent projection check with:

```bash
python tools/validate_calibration_projection.py
```

On the included local sample, all 9,261 points match A2D2's stored validation
coordinates to floating-point precision (95th-percentile error below
`3e-13` pixels). This demonstrates that projection comes from the 3D points and
calibration rather than from copied pixel coordinates.
