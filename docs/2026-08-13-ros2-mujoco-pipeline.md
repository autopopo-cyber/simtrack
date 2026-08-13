# ROS2 + MuJoCo 导航管线 —— 2026-08-13

> 里程碑：用自己写的 MuJoCo 仿真后端完全替代 Gazebo，接通 slam_toolbox + Nav2 标准栈。
> 全链路验证通过：建图 + 自主导航（5/7 航点到达）。

## 一、背景与决策

旧方案（Python 从零写 SLAM）遇到架构天花板：scan-matching 对自洽漂移盲（identity=1.0@2m）、
手写 costmap/inflation/恢复行为成本高。调研 ROS 生态后确认 slam_toolbox（位姿图+回环）+
Nav2（costmap+规划+控制+BT）已解决这些问题，且调了五年。

**决策**：转 ROS2。simtrack 从"独立 Python 导航原型"降级为"MuJoCo 仿真后端"。
算法资产（frontier/gate 探索、障碍跟踪器）后续移植为 ROS2 节点。

产品全貌（来自 robot-system 设计文档）：遥控开进房间 → 信号丢触发自主遍历 →
体素建图+图像采集 → 回起点 → 4090 出 3DGS。运控交给宇树 RL 模型（或 rl_sar 自定义 policy）。

## 二、环境

| 项 | 值 |
|---|---|
| 远程开发机 | `qin@100.64.63.98`（密码 `1`），SSH 免密已配 |
| OS | Ubuntu 24.04 LTS, x86_64, 62GB RAM, RTX 2080 Ti |
| ROS2 | Jazzy（`/opt/ros/jazzy`，apt 清华源装了 slam_toolbox 2.8.5 + nav2 1.3.12） |
| MuJoCo | 3.8.0（系统 python3 已装） |
| DockerHub | 被墙（但 apt 国内源通，Jazzy 原生跑不装 Docker） |
| tshell | 8765 端口有旧实例但 TUN 代理导致 502；当前用 SSH 操作 |

**不装 Docker**——宿主机已有 Jazzy + MuJoCo，原生跑最快。部署到 RK3588 时再考虑容器化。

## 三、架构

```
maze_gen.py ──→ maze20.png（20×20m 迷宫高度图，原点左下角）
     │
sim_server.py ──→ SimBackend（MuJoCo 场景 + 解析射线 + 碰撞检测）
     │              ├── init/step/set_cmd_vel/get_scan/get_true_pose
     │              └── 碰撞检测：5 点足印检查，碰墙停（不穿墙）
     │
sim_bridge.py ──→ ROS2 节点，替代 Gazebo
     │              ├── 发布 /scan (LaserScan, frame=laser, 360 rays)
     │              ├── 发布 /odom (Odometry, odom→base_footprint)
     │              ├── 发布 TF: odom→base_footprint→base_link→(隐含), base_footprint→laser
     │              ├── 订阅 /cmd_vel (Twist) → SimBackend.set_cmd_vel
     │              ├── 物理 100Hz, 发布 10Hz, wall-clock 时间戳（不用 sim_time）
     │              └── 发 /clock（但当前不用 use_sim_time，wall clock 对齐）
     │
slam_toolbox  ──→ online_sync 模式，吃 /scan 建 /map + 发 map→odom TF
     │
Nav2         ──→ navigation_launch: costmap + NavFn planner + DWB controller + BT
     │              ├── 需要 base_link（sim_bridge 发 base_footprint→base_link 静态 TF）
     │              └── 自动激活（干净启动时）
     │
/cmd_vel ←── Nav2 controller 输出 → sim_bridge → MuJoCo 狗移动（闭环）
```

### Frame 约定（踩坑记录）

| 包 | 期望的 base frame | sim_bridge 发的 |
|---|---|---|
| slam_toolbox | `base_footprint` | `odom → base_footprint` ✅ |
| Nav2 (local_costmap) | `base_link` | `base_footprint → base_link`（静态零偏移 TF）✅ |

**必须同时有 base_footprint 和 base_link**——这是 sim_bridge 里两个静态 TF 的原因。

### 迷宫坐标系（maze_gen.py）

```
世界系：x→右, y→上, 原点(0,0) = 迷宫左下角
图像系：col = x * 50, row = (20 - y) * 50（标准图像 y 翻转）
起点：(1.5, 1.5) 朝 +x
分辨率：50px/m（2cm/px），墙厚 0.3m
结构：20×20m 外框 + 中央 8×8m 方块(6,6~14,14) + 左下竖墙(3,3~3,10) + 右上竖墙(17,10~17,17)
走廊宽 6m（中央方块到外边界），天然回路（绕中央方块走一圈 = SLAM 回环）
```

## 四、自己写的三个文件

| 文件 | 行数 | 职责 |
|---|---|---|
| `simtrack/maze_gen.py` | ~90 | 生成迷宫高度图 PNG（数据驱动墙段定义，可扩展） |
| `simtrack/sim_server.py` | ~250 | SimBackend：MuJoCo 场景 + 解析式 heightfield 射线 + 碰撞检测 + 干净 API |
| `simtrack/sim_bridge.py` | ~190 | ROS2 节点：替代 Gazebo，全 topic 自控，wall-clock 时间戳 |

远程 `~/simtrack/` 还有辅助脚本（不在 git 里）：`explore_loop.py`（绕圈探索建图）、
`waypoints.py`/`run_loop.sh`（航点导航测试）、`turtlebot3_headless.launch.py`（Gazebo 验证用）。

## 五、验证结果

### 5.1 Gazebo 标准栈验证（先验证 ROS 原生算法能跑）

- Gazebo Harmonic headless（`-r -s` 无 GUI）+ turtlebot3 + slam_toolbox 建图 ✅
- Nav2 NavigateToPose "Goal succeeded" ✅（tb3_simulation_launch.py, headless+use_rviz:=False）
- 踩了 5 个坑（Gazebo GUI 崩/TwistStamped 类型/TF38s偏移/AMCL初始位姿竞态/Nav2 lifecycle）全是配置问题

### 5.2 MuJoCo 自有管线验证（核心目标）

| 测试 | 结果 |
|---|---|
| sim_server 自测（射线+位姿+碰撞） | ✅ 本地+远程一致 |
| sim_bridge topic 发布 | ✅ /scan 10Hz, /odom, /tf, /clock |
| slam_toolbox 吃 MuJoCo 激光建图 | ✅ 19.7×19.8m, 74.7% free, 回环正确 |
| Nav2 单目标导航 | ✅ (1.5,1.5)→(5.0,1.5) SUCCEEDED |
| Nav2 整圈航点导航 | **5/7 SUCCEEDED**（详见下表） |

### 5.3 航点导航详情

| # | 航点 | 状态 | 备注 |
|---|---|---|---|
| 1 | (15, 3) 底部右行 | ✅ | 底部走廊 13.5m |
| 2 | (18.5, 5) 右下走廊 | ✅ | 进入右侧走廊 |
| 3 | (18.5, 18) 右上角 | ❌ ABORTED | 右上隔断墙附近 inflation 封住目标 |
| 4 | (10, 18) 顶部中央 | ✅ | 横穿顶部走廊 |
| 5 | (2, 18) 左上角 | ❌ ABORTED | 角落 inflation |
| 6 | (2, 2) 左下角 | ✅ | 下行左侧走廊 |
| 7 | (10, 3) 底部中央 | ✅ | 回到起点附近 |

**2 个失败**：目标点离墙太近（角落），costmap inflation 封住。
**不急着修**（观察中）——修法：目标往走廊中央移 / 减 inflation_radius / 加 xy_goal_tolerance。

## 六、运行指南（远程机器）

```bash
# 0. SSH 进去
ssh qin@100.64.63.98

# 1. 启动三件套（sim_bridge + slam_toolbox + Nav2）
source /opt/ros/jazzy/setup.bash
cd ~/simtrack
tmux new-session -d -s sim -n bridge "source /opt/ros/jazzy/setup.bash; cd ~/simtrack; python3 -m simtrack.sim_bridge 2>&1"
sleep 4
tmux new-window -t sim -n slam "source /opt/ros/jazzy/setup.bash; ros2 launch slam_toolbox online_sync_launch.py 2>&1"
sleep 5
tmux new-window -t sim -n nav2 "source /opt/ros/jazzy/setup.bash; ros2 launch nav2_bringup navigation_launch.py 2>&1"

# 2. 等 ~25s 让 Nav2 自动激活，验证
ros2 lifecycle get /bt_navigator   # 应为 active [3]

# 3. 发导航目标
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: map}, pose: {position: {x: 10.0, y: 1.5, z: 0.0}, orientation: {w: 1.0}}}}"

# 4. 航点整圈（可选）
nohup bash ~/simtrack/run_loop.sh > /dev/null 2>&1 &
cat ~/simtrack/loop_result.txt   # 查结果
```

### 干净重启（清残留进程）

```bash
# tmux kill + Python 清理脚本（避免 pkill 误杀 SSH）
tmux kill-server 2>/dev/null
python3 -c "
import os,signal,subprocess
t=['simtrack','slam_toolbox','navigation_launch','component_container']
me=os.getpid();par=os.getppid()
for l in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()[1:]:
    p=l.strip().split(None,1)
    if len(p)<2:continue
    pid=int(p[0])
    if pid in(me,par):continue
    if any(x in p[1] for x in t):
        try:os.kill(pid,signal.SIGKILL)
        except:pass
"
sleep 2; ros2 daemon stop; sleep 1; ros2 daemon start
```

## 七、踩坑全记录（给后续复用）

1. **Gazebo GUI over SSH 崩** → headless launch（`-s` server only，跳过 `-g` GUI）
2. **cmd_vel Twist vs TwistStamped** → Gazebo Harmonic 桥要 TwistStamped，Nav2 发 Twist → 改 bridge YAML 一行
3. **TF 时间戳落后 /clock 38 秒** → 根因是反复重启的孤儿进程发陈旧 TF，干净启动即 33ms
4. **AMCL 初始位姿竞态** → costmap 超时前发 `/initialpose`（或用 slam_toolbox mapping 模式不需要初始位姿）
5. **Nav2 lifecycle 卡 inactive** → 干净启动自动激活；手动 `ros2 lifecycle set ... activate`
6. **base_footprint vs base_frame** → slam_toolbox 要 `base_footprint`，Nav2 要 `base_link` → 加静态 TF `base_footprint→base_link`
7. **sim_bridge 不用 use_sim_time** → 用 wall-clock 时间戳，避免 /clock 同步问题（sim_time 版导致 slam "Failed to compute odom pose"）
8. **碰撞检测** → sim_server 加 5 点足印检查，碰墙停（之前 contype=0 无碰撞，探索脚本把狗开出迷宫 80m）

## 八、已否决 / 超越的旧方案

- `simtrack/scan_matching.py` / `scan_match.py` —— Python 手写 scan-to-map，被 slam_toolbox 取代
- Python 手写 A*/DWA/纯追踪/costmap —— 被 Nav2 取代
- `--odom` / `--match` / `--qr-spacing` 等 algo3_headless 参数 —— 旧原型调试用，新管线不需要
- algo3_headless.py 本身 —— 保留作 A/B 基线，不改动
