#!/usr/bin/env bash
set -e

source /opt/ros/jazzy/setup.bash
source /opt/semantic_ros/setup.bash
source /opt/semantic_venv/bin/activate

exec "$@"
