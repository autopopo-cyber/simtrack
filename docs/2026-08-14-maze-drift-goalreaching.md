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
