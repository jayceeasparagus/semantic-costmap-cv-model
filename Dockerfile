FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    python3-numpy \
    python3-pil \
    python3-yaml \
    ros-jazzy-navigation2 \
    ros-jazzy-nav2-bringup \
    ros-jazzy-pointcloud-to-laserscan \
    ros-jazzy-slam-toolbox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

RUN python3 -m venv --system-site-packages /opt/semantic_venv \
    && /opt/semantic_venv/bin/python -m pip install --upgrade pip \
    && /opt/semantic_venv/bin/python -m pip install \
       --index-url https://download.pytorch.org/whl/cpu torch \
    && /opt/semantic_venv/bin/python -m pip install -e . --no-deps \
    && . /opt/ros/jazzy/setup.sh \
    && colcon build --base-paths ros2 --install-base /opt/semantic_ros

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
