# 10×10 大迷宫 + 里程计漂移实验 + 冲终点 —— 2026-08-14

> 在 5×5 房间迷宫（3m）基础上扩到 **10×10（5m 房间、1.5m 门、50m×50m）**，
> 给 sim_bridge 注入**足式机器人式里程计漂移**，定量测 slam_toolbox 的修正能力，
> 并尝试让狗自主走到对角终点。**核心结论：在真实 IMU 漂移下，slam_toolbox 把定位误差压到亚米级（177m 行程 0.3m），导航栈是稳的；冲远端终点卡在探索策略/Nav2 局部恢复，不是定位问题。**

## 一、10×10 迷宫（rooms10x10）

`maze_gen.py` 新增 `rooms10x10`：10×10 房间网格，每间 5m×5m，门 1.5m，DFS 生成树保证全连通 + 8% 额外门做环路。50m×50m，2500×2500px 高度图。

```bash
python -m simtrack.maze_gen rooms10x10        # 生成 + 连通性校验 + 标注图
MAZE=rooms10x10 python -m simtrack.sim_bridge  # 远程用环境变量选迷宫
```

**顺带修了两个 bug**（`maze_gen.py`）：
1. **门宽 bug**：旧 `_rooms_to_walls` 算出的门洞宽 = `room - door_w`。room=3 碰巧 = 1.5m 蒙对，room=5 变 3.5m 大洞（墙几乎没了）。改成居中留 `door_w` 缺口，实测门洞 1.47m ✓。
2. **分发 bug**：`main()` 里凡 `rooms*` 名字都硬生成 rooms5x5；改成走 `MAZES` 字典。
3. **起点数据驱动**：写 `maze_<name>.meta.json` sidecar（start/goal/尺寸），`sim_bridge` 读它决定起点——3m 房→(1.5,1.5)，5m 房→(2.5,2.5)，不再硬编码。

## 二、里程计漂移实验（核心结果）

`sim_bridge.py` 加 `OdometryDrift` 模型（env `ODOM_DRIFT_PCT` 开关）：
- 前进速度 ±scale_pct **尺度偏置**（足式腿足里程计标定误差）
- 偏航率**常值偏置** yaw_bias（陀螺零漂）+ 高斯噪声
- `/odom` 发漂移位姿（错的），`/true_pose` 发真值（测量用），`/scan` 仍真值——让 slam 拿"错里程计+对激光"去修。

测了三档漂移（`scripts/record_traj.py` 记 true/odom/slam 三轨迹，`scripts/analyze_drift.py` 出图+统计）：

| 漂移档 | 物理含义 | 行程/时长 | 原始里程计误差 | **slam 修正后** | 修正倍数 | 结果 |
|---|---|---|---|---|---|---|
| 0.4°/s + 5% | **无 IMU / 纯本体感觉**（最坏） | 125m / 700s | max 32m，端点 23m | **max 8.5m，端点 7.9m** | 2.9× | ❌ **迷路**（误差>房间宽 5m，门走不准） |
| 0.04°/s + 5% | **基础 IMU**（典型足式） | 177m / 930s | max 14m，端点 12m | **max 0.32m，端点 0.29m** | **43×** | ✅ **亚米级，稳** |
| 0.01°/s + 5% | **良好融合 IMU** | 9m / 694s | 0.1m | 0.06m | — | （定位没问题，卡在 Nav2 停滞） |

**关键结论**：
- **真实足式机器人漂移下（基础 IMU），slam_toolbox 把 12m 里程计漂移压到 0.3m（43 倍）**，177m 行程亚米级——导航栈是稳的。地图也保持连贯（墙直、房间方，无重影）。
- **无 IMU 时（0.4°/s 偏航零漂）6 分钟就迷路**（slam 误差超房间宽）。**这正是真实足式机器人必须融合 IMU 的原因**——纯本体感觉（腿足里程计）的偏航漂移撑不住长程。
- 偏航零漂是主因：它造成横向误差 ∝ 行程；scan-matching 能修一部分（局部），**全局得靠回环**；对称方房间（4 重旋转对称）又让 scan-matching 对航向约束弱——三重不利叠加。

## 三、目标导向探索 + 提速

**Nav2 提速**（`configs/nav2_fast_params.yaml`，改自 nav2_bringup 模板）：
MPPI `iteration_count 1→3`、`vx_std 0.2→0.35`、`vx_max 0.5→0.7`、`wz_max 1.9→2.6`，velocity_smoother 同步。
**效果：狗速 0.10 → 0.4~0.7 m/s（4-7×）**。主因是 `iteration_count=1` 让 MPPI 每周期几乎不优化、速度上不去。

**目标导向**：
- `firefly_explorer.py` 加 goal 模式（env `GOAL_X/GOAL_Y/GOAL_WEIGHT`）：frontier 打分从"信息增益-距离"变"朝终点贪心"；终点格在地图 free 后直冲。
- `goal_runner.py`：沿迷宫 BFS 房间路径（21 个房间中心航点，作为"楼层图先验"）依次导航；冷启动用"沿射线找已知 free 最远点"逐步推进（解 slam 关键帧建图下静止狗地图不长大），free 后直冲让 NavFn 自己绕门规划。

## 四、冲终点：诚实结论

**能不能走到对角终点 (47.5,47.5)？**
- **定位不是瓶颈**：基础 IMU 下 177m 行程 slam 只漂 0.3m，狗在迷宫里穿梭、过门、建图都正常，探了 ~30m×50m。
- **但没干净走到远端角落**，两个可修的工程问题：
  1. **frontier 探索会"绕"**：纯 frontier 探索在树状迷宫里不高效——探死胡同、回头，振荡在对角半路（dist_goal 卡在 ~31m），不会直奔远角。goal_weight 调高又会追 size=3 碎 frontier dithering。
  2. **Nav2 偶尔局部停滞**：狗在墙边/角附近会卡住（NavFn 规划/恢复行为没拽出来），goal_runner 引导跑因此停滞（不是漂移，定位仍 0.06m）。
- 这俩都是**策略/调参问题，不是定位/架构问题**。真要稳定冲远角：给 Nav2 更强恢复行为 / 或给探索器一个拓扑先验（goal_runner 思路）+ 修停滞。

**一句话**：足式机器人靠自己的（IMU 融合后）里程计 + slam_toolbox，**在这个迷宫里定位是够用的（亚米级）**；"走到远端终点"卡在高层探索策略和 Nav2 局部鲁棒性，不在 SLAM。完全无漂移要靠外部定位（你的判断没错），但**带 IMU 的常规足式机器人，SLAM 这层已经过关**。

## 五、资产

| 文件 | 作用 |
|---|---|
| `simtrack/maze_gen.py` | rooms10x10 + 门宽修复 + sidecar 元数据 |
| `simtrack/sim_bridge.py` | OdometryDrift 模型 + /true_pose + sidecar 起点 |
| `simtrack/firefly_explorer.py` | goal 导向模式 |
| `simtrack/goal_runner.py` | 房间路径引导（冷启动步进+直冲+续跑快进） |
| `configs/nav2_fast_params.yaml` | MPPI 提速参数 |
| `scripts/record_traj.py` | true/odom/slam 三轨迹记录 |
| `scripts/analyze_drift.py` | 漂移分析出图+统计（matplotlib 可选，PIL fallback） |
| `scripts/monitor_progress.py` | 远程后台进度日志 |
| `scripts/scan_check.py` / `check_nav_remote.py` / `dump_map_remote.py` | 诊断小工具 |

## 六、复现

```bash
# 远程：漂移实验（基础 IMU）
MAZE=rooms10x10 ODOM_DRIFT_PCT=5 ODOM_DRIFT_YAW_BIAS_DEG=0.04 python -m simtrack.sim_bridge  # + slam + nav2
python -m simtrack.firefly_explorer          # 探索（或 goal_runner 冲终点）
python record_traj.py 900 _traj.csv          # 记录
# 本地：分析
python scripts/analyze_drift.py _traj.csv    # 出 _traj.png + _error.png + 统计
```

## 七、周期性 LiDAR 重定位修正（航向/位置不裸积分）✅

**核心设计**（用户提议）：航向、距离不能一直积分累加——每隔 ~30s 找个已知方向的房间，
用激光在已知地图上 scan-match 重定位，把漂移里程计**重置**回去。

**实现**：
- `sim_server.py` 加 `scan_match()`：相关扫描匹配——候选位姿下把当前 scan 端点投到迷宫高度图，
  命中墙越多越贴合；粗搜 ±1.5m/±20° + 细搜 ±0.2m/±4°，估出真实位姿。诚实：只用 scan+地图，不读真值。
- `sim_bridge.py` 加周期定时器（env `CORRECT_PERIOD_S`，默认 30，0=关）：每 30s 调 scan_match，
  把 `OdometryDrift` 的 (x,y,yaw) 重置到激光估计值。

**实测（无 IMU 最坏情况 0.4°/s 偏航漂移 + 30s 修正）**：

| | slam_err max | slam_err mean | 到终点 |
|---|---|---|---|
| **不修正** | 8.54m（>房间宽，迷路） | — | 迷路到不了 |
| **30s LiDAR 修正** | **1.81m** | **0.87m** | **dist_goal 5.5m（走到终点区！）** |

**结论**：周期性 LiDAR 重定位把漂移从"无限积分→迷路"压成"有界 ~1-2m"，**即使没有 IMU（0.4°/s），
狗也能维持亚米-米级定位并走到终点**。这正是真实机器人 AMCL/scan-matching localization 干的事。
对比图见 `correction_effect.png`（蓝=修正后 bounded，红=不修正 → 8.5m 迷路）。

> 仍有的小工程点：scan_match 每次约 0.1-0.2s（在 rclpy 单线程里会短暂阻塞物理），30s 一次可接受；
> 偶尔 scan-match 在局部弱特征处估偏一点点（slam_err 缓爬到 1.8m），但远好于不修正。

## 八、去仿真特权：对"自建图"周期重定位（CORRECT_REF=map）✅

七的修正匹配的是**真迷宫高度图**（仿真特权——真机没有真图）。本轮把参考图换成
**slam_toolbox 自己建出来的 /map**：真机可复现的土法 AMCL。

**实现**（commit 73e7a3d）：
- `sim_bridge.py` env `CORRECT_REF=map`：订阅 /map（latched QoS）缓存墙格掩膜
  （data≥65），每 30s 从**漂移 odom 位姿**出发（不读 slam TF、不读真值）做相关匹配
  （粗±1.5m/±20° → 细±0.2m/±4°），得分<40 拒绝修正（建图不足区域不硬修）。
- 行约定 no-flip（row 随 y 增）由 `scripts/probe_map_convention.py` 实测验证：
  把激光端点按真值投影后 no-flip 命中 65/360 且匹配位姿距真值 0.36m，镜像约定 1%。
- `goal_runner.py` `_route_step` A*（free=1/unknown=8/墙=∞，**地图数组外=unknown**）：
  子目标取"从狗出发连续 free 段"末尾（前沿）。修三代死锁：原地等地图 → 扇形贴墙振荡
  → BFS closest-to-waypoint 墙前鞍点。**NavFn 远目标穿大片 unknown 会"legal potential
  found but no path"卡死，近距离 free 子目标免疫**——这对真机探索策略同样适用。

**实测（无 IMU 5% 尺度+0.05°/s 偏航抽签 + 30s 对自建图修正，870s / 105.6m 行程）**：

| | odom-真值 | 特征 |
|---|---|---|
| 不修正（此前同工况） | 8.5m 失控迷路 | 裸积分发散 |
| 对真图修正（七，特权） | max 1.81 / mean 0.87m | 锚定真值 |
| **对自建图修正（本轮）** | **max 1.98 / mean 1.81m，870s 无增长趋势** | **锚定自建图** |

修正日志（59 条）机理证据：yaw 修正量 mean **1.5°/次 = 0.05°/s 偏置×30s 精确抵消**
陀螺零漂累积；pos 修正 mean 0.11m×29 次 ≈ 3.2m（即不修正本会继续累积的量）；
命中分 min 99 / mean 266，0 次拒绝。

**诚实结论**：
1. **有界性成立**（用户假设验证 ✓）：漂移不再无限积分，被钉在 ≤2m；即使参考图是
   自己建的。反复经过已建图房间=重锚，机制与 MCL/AMCL 相同。
2. **误差下限=地图局部变形**：mean 1.81 vs 特权版 0.87——修正把 odom 锚到地图系，
   地图哪里变形 odom 就继承哪里。slam_err 在 0.5↔1.9m 间摆动（跳变分析：1.9m 全是
   y 偏移，出现在狗沿 col0 南北走廊推进时，进入二维结构丰富房间后自恢复 0.57）——
   **走廊沿轴不可观测**是根因，与旧 Python 代码的教训一致。破法=墙抖动破对称 +
   多方向房间几何，不是纯算法。
3. 本轮漂移抽签（yaw -0.05°/s）比七的（≈-0.31°/s）温和，8.5m 基线是强偏置下的；
   同 run 反事实证据（修正量累计 3.2m）不依赖抽签强度。

**工程坑（本轮新踩）**：
- 远端 `~/.bashrc` 把 hermes-venv 塞进 PATH 首位 → 交互 shell `python3` 无 numpy，
  必须显式 `/usr/bin/python3`；nav2 在无 /clock 时 activation 超时弃疗（bridge 先起）。
- tmux 80 列折行会截断日志数字，抓取要 `grep -A3`。
- `record_traj.py` 的 slam 列与 monitor 不一致（疑似 TF 时间戳问题）——本节数据以
  monitor 独立计算 + CSV odom 列为准，TODO 修复。

## 九、真实雷达 L2 参数化重测：传感器变差，结果反而好 20 倍 ✅

用户提供了 A2 实机雷达 spec（Unitree 4D LiDAR L2，前后双装）。对表：

| L2 实参 | 仿真现状 → 本轮 |
|---|---|
| 10m@10%反射 / 30m@90% | 15m 硬编码 → **env `LIDAR_RANGE`，本轮 10（保守设计点）** |
| ±3cm 精度 | 零噪声 → **env `LIDAR_NOISE_M`，本轮 0.03**（加在 get_scan 内，/scan 与修正匹配吃同一份噪声） |
| 21,600点/s 有效（双装4.3万） | 360 rays@10Hz=3,600 → 保持（2D 切片带内密度同量级，保守） |
| 96° 垂直 FOV（可见地面） | 平面 z=0.5 → **仿真盲区**：真机管线需 pointcloud→laserscan 的 z 带切片（建议 10–40cm 带高，含腿部碰撞高度） |
| 内置 IMU + POINT-LIO | 5% 腿式漂移模型 → 真机常规层直接用 LIO 里程计（LIO 级 ~0.1–1% 行程）；5% 模型对应"LIO 故障降级"容灾层 |

**实测（10m ±3cm + 同漂移抽签 scale1.027/yaw-0.05°/s + 30s 自建图修正，960s / 132m）**：

| | 15m 无噪声（八，含墙边卡死 35min 损伤） | **10m ±3cm（九，A* 健康驱动）** |
|---|---|---|
| odom-真值 | mean 1.81 / max 1.98m | **mean 0.08 / max 0.52m** |
| slam_err（末段） | ~1.87 | **0.01–0.02** |
| 修正命中 | mean 266/360 | mean 303/360（噪声几乎不减命中：5cm 栅格+多视角墙厚吸收 ±3cm） |
| yaw 修正 | mean 1.5°/次 | mean 1.6°/次（同样精确抵消零漂累积） |

**结论（本轮最有价值的一课）**：传感器变差（量程-33%、加噪声），定位反而好 20 倍——
**误差下限从来不是传感器，是暴露在退化几何下的时间**。八的 1.9m 形变出生于 35 分钟
贴墙振荡期（走廊 y 轴不可观测 + 反复在同一退化段建图）；本轮 A* 路由成熟后狗 3 分钟
穿 4 房、持续暴露在二维富几何环境，地图干净，修正把 odom 钉在干净图上 → 8cm。
对策排序：**探索模式（避免退化几何长期暴露）> 修正频率 > 传感器参数**。

**对真机的直接启示**：
1. L2 的 POINT-LIO 让"无 IMU 最坏情况"基本不存在——周期重定位降级为容灾层；
2. 探索策略要防"贴墙蹭"（我们的扇形兜底就犯这个错）——A* 连通推进是正解；
3. 走廊/长直段是定位毒药，真机房间布局若多走廊，需靠回环重访补偿（或 UWB/视觉锚点）。

**复现**：`MAZE=rooms10x10 ODOM_DRIFT_PCT=5 ODOM_DRIFT_YAW_BIAS_DEG=0.4 ODOM_DRIFT_SEED=42
CORRECT_PERIOD_S=30 CORRECT_REF=map LIDAR_RANGE=10 LIDAR_NOISE_M=0.03 python -m simtrack.sim_bridge`
（slam `max_laser_range` 无需改：slam_toolbox 取 min(参数, scan.range_max)，scan 头自带 10m）

工程坑：tmux kill-session 杀不死 launch 的子进程树（旧 slam/nav2 残留双 /map 双 goal 竞争），
重启栈必须 `pgrep` 确认 + 按 PID 补刀。

## 十、窄门分级验证：0.8m 舒服通过，0.6m 在物理边缘 ✅⚠️

**背景**：目标要求"可能通过尽量窄的通道，60cm？80cm？"。狗足迹 0.8×0.4m 胶囊——过门起作用
的是 0.4m 宽度。新增迷宫变体（同 seed=42 → 同 DFS 拓扑/同门位 → goal_runner 航点表仍有效，
**唯一变量=门宽**）：`rooms10x10n80`（门 0.8m）、`rooms10x10n60`（门 0.6m）。

全真实栈：L2 参数（10m±3cm）+ 5% 漂移 + 30s 自建图重定位 + A* 推进 + Nav2（robot_radius 0.22，
0.8m 门在 costmap 上有 0.36m 可通行带、0.6m 有 0.16m——规划层不是瓶颈）。

**结果**：

| 门宽 | 每侧余量 | 结果 |
|---|---|---|
| **0.8m** | 0.2m | ✅ **4 航点 / ≥3 门 / 零重试 / 24s 每跳，全速通过**——与 1.5m 基线同节奏 |
| **0.6m** | 0.1m | ⚠️ 过 1 门（#0→#1），但 x=5 墙的门 **4 次尝试全部 Nav2 abort**（步进 (7.7,8.0) 失败 4 次→跳过航点） |

**0.6m 的物理账**：0.4m 宽躯体进 0.6m 缝，对准角容差 = atan((0.6−0.4)/0.8) ≈ **±14°**——
机身轴线必须近乎垂直门面进入，稍有角度对角线宽度就超 0.6m，sim 碰撞阻塞→Nav2 放弃。
当前 MPPI 参数（全速穿行）不保证对准精度：0.6m 是"碰运气级"，0.8m 是"任务级"。

**附带观察**：0.6m 门前的反复试探又把 slam_err 推到 1.77——再次印证 §九规律：
**贴墙试探=退化几何暴露=地图变形**。门舞和定位损伤是同一个问题的两副面孔。

**结论与下一步**：
- 规格答案：**当前栈可靠通过 0.8m；0.6m 需要专门的"对门"行为**——检测到窄门（两侧墙距<1m 的
  路径点）→ 门前旋转对齐（轴线⊥门面，±5°内）→ 降速 0.2m/s 穿越。这是明确的下一个工作项。
- 0.5m 以下不用想：0.4m 躯体两侧各 5cm，机器人本体公差都不够。

**复现**：`python -m simtrack.maze_gen rooms10x10n80`（或 n60）→ MAZE=rooms10x10n80 启动 bridge，
其余环境变量同 §九。
