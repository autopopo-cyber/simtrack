# API 速查 — 环境变量 / 话题 / 接口 / 命令

> 会话恢复速查。最后更新 2026-08-14。全部在远程 Linux（qin@100.64.63.98）上跑。

## 一、sim_bridge 环境变量（实验的全部开关）

| Env | 默认 | 含义 |
|---|---|---|
| `MAZE` | loop20 | 迷宫名 → `confirmed/maze_<name>.png`。可用: loop20 / rooms5x5 / **rooms10x10** / **rooms10x10n80**(0.8m门) / rooms10x10n60 |
| `ODOM_DRIFT_PCT` | 0 | 漂移总开关（百分比,5=5%）。>0 才有 /true_pose |
| `ODOM_DRIFT_YAW_BIAS_DEG` | 0.5 | 陀螺零偏幅值（实际符号由seed抽签） |
| `ODOM_DRIFT_NOISE_V/W` | 0.01 | 速度/角速度噪声 std |
| `ODOM_DRIFT_SEED` | 42 | 抽签种子（同seed同漂移，可复现；seed42→scale 1.027/yaw -0.05°/s） |
| `CORRECT_PERIOD_S` | 30 | 周期重定位周期（0=关）。每30s对参考图匹配重置漂移 |
| `CORRECT_REF` | true | **true**=对真迷宫图(仿真特权,仅上限参考) / **map**=对slam自建/map(诚实,真机可复现) |
| `LIDAR_RANGE` | 15.0 | 量程m。**L2设计点=10**（10%@反射）；真实墙面可达15-30 |
| `LIDAR_NOISE_M` | 0.0 | 测距噪声std。**L2设计点=0.03** |
| `GOAL_X`/`GOAL_Y`/`GOAL_WEIGHT` | - | firefly_explorer 的目标导向模式（两值同给才启用） |

## 二、ROS 话题 / TF

| 名字 | 类型 | QoS | 说明 |
|---|---|---|---|
| `/scan` | LaserScan | BEST_EFFORT/VOLATILE | 10Hz, 360rays, frame=laser。**永远真值扫描+噪声** |
| `/odom` | Odometry | 默认 | **漂移位姿**（pose错,twist真值速度）; odom→base_footprint TF 同步发 |
| `/true_pose` | PoseStamped | 默认 | 真值诊断（frame刻意="true"防误用）。漂移开才发 |
| `/clock` | Clock | BEST_EFFORT | 仿真时钟；**bridge用wall-clock跑自己**（避免鸡生蛋） |
| `/cmd_vel` | Twist | 默认 | Nav2→狗 |
| `/map` | OccupancyGrid | RELIABLE/**TRANSIENT_LOCAL**(latched) | slam发; 订阅端QoS必须匹配否则收不到 |
| TF 链 | | | map→odom(slam) → base_footprint(bridge,漂移) → base_link/laser(静态) |

## 三、SimBackend 关键接口（sim_server.py）

```python
SimBackend(maze_path, start=(x,y,yaw), lidar_rays=360, lidar_fov_deg=360,
           lidar_range=15.0, range_noise_m=0.0, timestep=0.005,
           use_mujoco_viewer=False, px_per_m=50)
.set_cmd_vel(vx_body, vyaw)   # 机身系速度指令
.step()                        # 一个物理步
.get_true_pose() -> (x,y,yaw)  # 真值
.get_scan() -> (ranges, angles)  # 机身系; inf=未命中; 噪声在内
.scan_match(ranges,angles, ix,iy,iyaw) -> ((x,y,yaw), score)  # 对真迷宫图(特权)
._match_score(r,a,x,y,yaw) -> int  # 命中墙像素数
._check_collision / _is_free     # 0.8×0.4胶囊5点采样
```

## 四、maze_gen CLI

```bash
python -m simtrack.maze_gen <name> [seed]   # 生成到 confirmed/：PNG+标注图+meta.json
# gen_rooms_grid(rows, cols, room_m, door_w, extra_prob, wall_jitter, seed)
# n80/n60 与 rooms10x10 同seed42→同拓扑→goal_runner航点表通用（唯一变量=门宽）
```

## 五、标准实验流程（远程）

```bash
# 0) 全清（铁律：kill-session杀不死launch子树！）
pkill -9 -f "simtrack.sim_bridge|simtrack.goal_runner|slam_toolbox|nav2_bringup|record_traj|monitor_progress|component_container"
sleep 2; pgrep -af "slam_toolbox|nav2_bringup|sim_bridge|goal_runner"   # 必须为空
tmux kill-session -t sim 2>/dev/null
tmux new-session -d -s sim -n bridge; tmux new-window -t sim -n slam; tmux new-window -t sim -n nav2; tmux new-window -t sim -n drive; tmux new-window -t sim -n mon

# 1) bridge（顺序铁律：bridge先起=它是/clock主人，nav2没时钟会激活超时弃疗）
tmux send-keys -t sim:0 "cd ~/simtrack && source /opt/ros/jazzy/setup.bash && \
  MAZE=rooms10x10n80 ODOM_DRIFT_PCT=5 ODOM_DRIFT_YAW_BIAS_DEG=0.4 ODOM_DRIFT_SEED=42 \
  CORRECT_PERIOD_S=30 CORRECT_REF=map LIDAR_RANGE=10 LIDAR_NOISE_M=0.03 \
  /usr/bin/python3 -m simtrack.sim_bridge" Enter     # ← 必须 /usr/bin/python3（pitfalls#2）
# 2) slam → 3) nav2(等~16s lifecycle active) → 4) goal_runner → 5) monitor
tmux send-keys -t sim:1 "bash ~/simtrack/run_slam.sh" Enter
tmux send-keys -t sim:2 "bash ~/simtrack/run_nav2.sh" Enter   # sleep 16
tmux send-keys -t sim:3 "cd ~/simtrack && source /opt/ros/jazzy/setup.bash && /usr/bin/python3 -m simtrack.goal_runner" Enter
tmux send-keys -t sim:4 "cd ~/simtrack && source /opt/ros/jazzy/setup.bash && (/usr/bin/python3 monitor_progress.py > _mon_stdout.log 2>&1 &)" Enter

# 采集：先确认狗在动再录！（tail _progress.log 看位姿变化）
/usr/bin/python3 record_traj.py 900 _traj.csv
# 抓日志（tmux 80列折行，数字被切，必须 -A3 上下文）：
tmux capture-pane -p -t sim:0 -S -4000 | grep -A3 "重定位" | tail -120
# 本地分析：.venv/Scripts/python.exe scripts/analyze_drift.py _traj.csv
#   （或手算: odom_err=hypot(odom-true), 每60s分段看趋势=有界性）
```

## 六、Windows 本地开发循环

```bash
# 本地无ROS：只能 py_compile + maze_gen + 分析
python -m py_compile simtrack/<file>.py
.venv/Scripts/python.exe -m simtrack.maze_gen <name>
# 部署：paramiko（密码"1"，见 scripts/*_remote.py 的连接模式）
.venv/Scripts/python.exe -c "import paramiko; ssh=...; sftp.put(local, remote)"
```
