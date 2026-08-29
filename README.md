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
                              ROS 2 / Nav2 layered costmap```

SLAM or localization supplies robot poses so observations can be placed consistently over time. This project integrates an existing pose provider rather than implementing SLAM itself.

## Initial scope

- Six-class RGB semantic segmentation
- Camera-LiDAR calibration and projection
- Confidence-aware semantic point painting
- Bird's-eye-view semantic cost generation
- Temporal accumulation using timestamped poses
- ROS 2 replay and perception nodes
- Custom Nav2 costmap layer
- RViz visualization and system benchmarks

## Design documents

- [`docs/architecture.md`](docs/architecture.md) defines the complete system flow, interfaces, coordinate frames, safety rules, and delivery stages.
- [`configs/semantic_classes.yaml`](configs/semantic_classes.yaml) is the source of truth for model classes, A2D2 label grouping, and navigation costs.
