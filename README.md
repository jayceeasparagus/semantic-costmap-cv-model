# Semantic Costmap Perception Pipeline

A robotics perception pipeline that combines RGB semantic segmentation with LiDAR geometry to generate navigation-compatible semantic costmaps.

## Goal

Train an RGB segmentation model, project its semantic probabilities onto calibrated LiDAR points, rasterize the labeled points into a bird's-eye-view grid, and integrate the resulting semantic layer with ROS 2 and Nav2.

## Planned flow

```text
RGB image -> semantic segmentation -> class probabilities
                                           |
LiDAR + calibration -----------------------+
                                           |
                                           v
                               semantically painted points
                                           |
                                           v
                              bird's-eye semantic cost grid
                                           |
                                           v
                              ROS 2 / Nav2 layered costmap