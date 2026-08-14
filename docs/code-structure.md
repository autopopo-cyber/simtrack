# 代码结构 — 仓库地图

> 会话恢复速查。最后更新 2026-08-14。**先看"活跃文件"区，其余是历史/资产**。

## 一、活跃文件（ROS2 栈，改这里）

| 文件 | 行数 | 职责 | 关键点 |
|---|---|---|---|
| `simtrack/sim_server.py` | ~360 | MuJoCo 仿真核心（无ROS） | `SimBackend`: get_scan(解析射线+噪声) / scan_match(特权匹配) / _check_collision(0.8×0.4胶囊) / 真值位姿。**env 无关，构造参数控制一切** |
| `simtrack/sim_bridge.py` | ~440 | ROS2 桥（栈的心脏） | OdometryDrift(漂移模型) / _correction_cb(30s周期重定位,CORRECT_REF选参考图) / _scan_match_map+_match_score_map(对自建图匹配,no-flip约定) / 发 /scan /odom /clock /true_pose /tf |
| `simtrack/goal_runner.py` | ~310 | 航点驱动（当前主力） | 21房间航点表(seed42硬编码!) / `_route_step` A*推进(free=1,unknown=8,墙=∞,数组外=unknown) / `_fan_step` 扇形兜底 / `_step_target` 直线步进 / 快进续跑 / 4次失败跳过航点 |
| `simtrack/maze_gen.py` | ~350 | 迷宫生成器（离线） | gen_rooms_grid(DFS树+extra_prob环路+wall_jitter破对称) / MAZES dict / meta.json sidecar / `python -m simtrack.maze_gen <name> [seed]` |
| `simtrack/firefly_explorer.py` | ~440 | 自研frontier探索（已被替代） | 保留：_detect_frontiers 的"开-未知"过滤逻辑 + 动态min_size + goal_mode |
| `configs/slam_tuned_params.yaml` | 79 | slam_toolbox 调参版 | loop_search 3→7(破雪球), minimum_travel_distance 0.3, 阈值降低 |
| `configs/nav2_fast_params.yaml` | 470 | Nav2 提速版 | MPPI iteration 1→3, vx_max 0.7, robot_radius 0.22, allow_unknown true |
| `run_slam.sh` / `run_nav2.sh` / `run_frontier.sh` | — | 远程启动器 | configs/ 有则用调参版；run_frontier source ~/exploration_ws |

**脚本**（远程用，`/usr/bin/python3` 跑！见 pitfalls #2）：
`record_traj.py`(三轨迹CSV,slam列有bug) `monitor_progress.py`(15s一行progress) `analyze_drift.py`(本地分析,matplotlib/PIL) `costmap_probe.py`(查格子代价) `probe_map_convention.py`(行约定探针) `dump_map_remote.py` `check_nav_remote.py` `scan_check.py` `send_goal.py`

## 二、历史文件（纯 Python 原型时期，被超越，勿改）

| 文件 | 说明 |
|---|---|
| `test_scripts/algo3_headless.py` (2826行) | 旧主战场单体：自制栅格/A*/DWA/纯追踪/frontier/二维码。天花板2m漂移 |
| `test_scripts/algo0~9_*.py` | 演化链（bounce→arc→firefly→各种frontier变体） |
| `simtrack/scan_matching.py` | 旧匹配器（对膨胀墙掩膜，**对自洽漂移瞎**）—algo3_headless用 |
| `simtrack/scan_match.py` (未跟踪) | likelihood-field匹配器（AMCL思想），**从未接线** |
| `simtrack/odometry.py` | 旧漂移模型（偏置随机游走，比新的更细）+QR修正 |
| `simtrack/algorithms/{dwa,vo,base}.py` `nav.py` `map.py` `waypoints.py` `lidar.py` `simulation.py` `runner.py` `trackgen.py` `obstacles*.py` `obstacle_tracker.py` `models/` | 旧组件库 |
| `_src/` | 最早期（依赖外部 unitree_rl_gym / tangent_arc_planner） |

**被超越原因一句话**：踩坑§17——手写定位栈无位姿图回环，漂移天花板≈2m；slam_toolbox+Nav2 是调了五年的标准件。
完整清单见 docs/2026-08-13-ros2-mujoco-pipeline.md §八。

## 三、资产与数据

| 目录/文件 | 内容 |
|---|---|
| `confirmed/` | 迷宫高度图 PNG + meta.json + 标注图（loop20/rooms5x5/rooms10x10/**n80/n60窄门**） |
| `scans/` `runs/` `assets/landmarks/` | 旧实验资产、ArUco 码 |
| `_*.csv/_*.png` 根目录散落 | 历次实验轨迹数据（_mapref=自建图修正轮, _l2=L2真实参数轮） |
| `.zcode/plans/` | 会话计划存档 |

## 四、Git 模型

```
master ── 3890da0 (现在, 已与 ros2-mujoco-pipeline 快进合并统一)
   │
   ├ tag v2.0-ros2-slam-nav ← 里程碑: 50×50到终点/漂移结题0.08m/0.8m窄门任务级
   ├ (历史) 2bcda70 = 纯Python时期终点
   └ ros2-mujoco-pipeline 分支保留为历史指针(与master同位)
推送到 github.com/autopopo-cyber/simtrack（TUN代理不稳, push失败就重试循环,一般<12次内过）
```

## 五、下次接手的第一件事清单

1. 读 `docs/PRD.md`（验收状态）→ 本文 → `docs/pitfalls-ros2.md`
2. 远程栈大概率还在 tmux "sim" 里跑（或已死）——`ssh qin@100.64.63.98` 密码"1"，`tmux attach -t sim`
3. 改代码流程：本地改 → `python -m py_compile` → paramiko sftp 传单文件 → 重启对应窗口（pkill+pgrep确认！）
4. 跑实验流程：开录制(record_traj)前**确认狗在动**；抓 tmux 日志要 `grep -A3`（80列折行）
