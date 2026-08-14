#!/bin/bash
# nav2 启动器。若 configs/nav2_fast_params.yaml 存在则用它（提速版），否则默认。
source /opt/ros/jazzy/setup.bash
PARAMS=~/simtrack/configs/nav2_fast_params.yaml
if [ -f "$PARAMS" ]; then
  exec ros2 launch nav2_bringup navigation_launch.py params_file:="$PARAMS"
else
  exec ros2 launch nav2_bringup navigation_launch.py
fi
