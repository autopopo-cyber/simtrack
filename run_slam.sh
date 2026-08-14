#!/bin/bash
# slam_toolbox 启动器。若 configs/slam_tuned_params.yaml 存在则用它（调过回环参数：
# loop_search_maximum_distance 3→7 破雪球、降阈值、更密关键帧），否则默认。
source /opt/ros/jazzy/setup.bash
PARAMS=~/simtrack/configs/slam_tuned_params.yaml
if [ -f "$PARAMS" ]; then
  exec ros2 launch slam_toolbox online_sync_launch.py slam_params_file:="$PARAMS"
else
  exec ros2 launch slam_toolbox online_sync_launch.py
fi
