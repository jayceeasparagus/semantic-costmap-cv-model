# Repository guide

## Why there are both scripts and a Python package

Files in `tools/` are commands a user runs. Files in
`src/semantic_costmap/` contain reusable logic shared by those commands, tests,
and later ROS 2 nodes. This avoids copying model loading, class definitions, and
geometry code between scripts.

## `pyproject.toml`

`pyproject.toml` describes this repository as an installable Python package and
stores the pytest configuration. It is a text configuration file, not a
runtime service. The editable install below makes package imports work while
still using the source files directly:

```bash
python -m pip install -e ".[dev]"
```

## `requirements.txt`

This is a simpler dependency list for environments that do not use an editable
install. PyTorch is included because local inference and tests now depend on it.
Training environments may still select a CUDA-specific PyTorch build.

## `configs/a2d2_class_list.json`

This is A2D2's color legend. It maps every source label color to an A2D2 class
name; it does not define the U-Net architecture.

## `data/` and `outputs/`

Datasets, generated images, costmaps, and model checkpoints are ignored by Git.
They can be large or machine-specific. Ignoring them does not remove them from
the computer.

## Tests

Focused tests protect model compatibility, semantic class definitions,
projection math, and costmap behavior. Run all non-ROS Python tests with:

```bash
python -m pytest
```

## Optional ROS 2 and Docker

The offline Python pipeline is the primary project path. ROS 2 can wrap the
stable perception and mapping pieces in topics and transforms, while Docker
records a reproducible CPU environment. Nav2, SLAM, and the small route
planner are optional integrations; they are not required to run or evaluate
the semantic map.
