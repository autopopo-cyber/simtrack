#!/bin/bash
# frontier_exploration_ros2 启动器（MRTSP 全局排序探索，替 firefly）。
source /opt/ros/jazzy/setup.bash
source ~/exploration_ws/install/setup.bash
exec ros2 launch frontier_exploration_ros2 frontier_explorer.launch.py
