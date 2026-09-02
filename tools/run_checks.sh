#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python -m pytest -q
python -m compileall -q src tools

if [[ -f /opt/ros/jazzy/setup.bash ]]; then
  set +u
  source /opt/ros/jazzy/setup.bash
  set -u
  colcon --log-base log_ros build \
    --base-paths ros2 \
    --symlink-install \
    --build-base build_ros \
    --install-base install_ros
  colcon --log-base log_ros_all_tests test \
    --base-paths ros2 \
    --build-base build_ros \
    --install-base install_ros
  colcon test-result --test-result-base build_ros --verbose
fi

git diff --check
echo "All available checks passed."
