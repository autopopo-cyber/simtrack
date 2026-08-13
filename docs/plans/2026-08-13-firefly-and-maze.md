# 下一步计划：firefly_explorer + 复杂迷宫 —— 2026-08-13

> 前置：ROS2 MuJoCo 管线已验证（见 `docs/2026-08-13-ros2-mujoco-pipeline.md`）。
> 当前状态：sim_bridge → slam_toolbox → Nav2 全通，5/7 航点导航成功。
> 用户指示：先 2（firefly_explorer 自主探索）后 3（复杂迷宫），航点失败先观察不急修。

## ✅ Phase 1 完成（firefly_explorer，2026-08-13 实测）

`simtrack/firefly_explorer.py` 已实现并远程验证通过。在 20×20m 迷宫里从 (1.5,1.5) 出发
**全自主探索**，无人工航点：

| # | frontier 目标 | 距离 | 结果 |
|---|---|---|---|
| 1 | (1.8,15.8) | 15.3m | ✅ 到达 |
| 2 | (3.3,9.7) | 6.3m | ❌ Nav2 ABORTED（角落 inflation，见下方"已知问题"）→ 拉黑，自动选下一个 |
| 3 | (18.8,13.5) | 17.0m | ✅ 到达 |
| 4 | (17.6,16.6) | 3.3m | ✅ 到达 |
| 5 | (0.8,6.0) | 19.8m | ❌ Nav2 ABORTED（同上）→ 拉黑 |
| 6 | (0.4,9.6) | 18.5m | ✅ 到达 |

- **4 次到达 / 2 次优雅失败**，最长单次导航 18.5m。机器人自主横穿整张地图。
- frontier 数量：**234 → 23**（约 90% 探索完成，仍在下降）。
- 失败处理：`_result_cb` 收到 status=6(ABORTED) 立刻拉黑该区域半径 2.5m 并重选下一个
  frontier，**无卡死、无崩溃**。看门狗 90s 超时作为兜底（本次实测 Nav2 自己 19s 就 abort 了，没等到）。

### 关键设计点（已验证）

- frontier 检测：纯 numpy（`_dilate4` 自写膨胀 + `_cluster8` BFS 连通域），**不依赖 cv2/scipy**，
  避免远程依赖问题。移植 algo3 的 `_open_frontier`（free 格的未知邻居需开阔）——参数
  `wall_clear_cells`，slam_toolbox 0.05m/格下默认 1（轻过滤），0 = 纯 Yamauchi。
- 聚类：动态 min_size（簇 <40 时降到 2，防早期稀疏前沿被全过滤卡死——algo3 的老坑）。
- 打分：`size - dist_weight·distance`（信息增益优先，略偏近）。
- **竞态守卫**：单调 `_goal_seq`。看门狗取消+重发后，旧目标的 result 回调按 seq 丢弃，
  不会污染新目标的状态/统计（firefly 跑长循环必须防这个）。
- 目标序号/坐标全部在回调闭包里捕获，不读 `self.current_goal`（避免过期值）。

### ⚠️ 远程运行坑：python3 被 hermes-venv 抢走（无 numpy）

远程 `~/simtrack` 的 tmux 会话里 `python3` = `/home/qin/hermes-venv/bin/python3`（一个没装 numpy
的 venv），不是 `/usr/bin/python3`。直接 `python3 -m simtrack.firefly_explorer` 会
`ModuleNotFoundError: No module named 'numpy'`。sim_bridge 不受影响是因为它只依赖 rclpy
（ROS PYTHONPATH 提供）。

**解法**：显式用 `/usr/bin/python3`（system + `~/.local` 都有 numpy，rclpy 经 ROS PYTHONPATH 可用）。
远程已放启动器 `~/simtrack/run_firefly.sh`：
```bash
#!/bin/bash
source /opt/ros/jazzy/setup.bash
cd ~/simtrack
exec /usr/bin/python3 -m simtrack.firefly_explorer
```
启动：`tmux new-window -t sim -n firefly "bash ~/simtrack/run_firefly.sh"`

### 已知问题（不阻塞，与航点角落失败同源，观察中）

Nav2 全局规划器在某些位置报 `Failed to create a plan from potential when a legal
potential was found` → abort。根因：机器人停在未知边缘/角落时，全局 costmap 把周围 unknown
当 lethal，NavFn 无法从起点扩散。firefly 靠拉黑+重选绕过，探索仍能推进。彻底解决要么：
(a) 全局 costmap 允许穿越 unknown（`track_unknown_space`/`unknown_cost_value` 调参），或
(b) frontier 目标向已知区偏移一点（别停在未知边缘）。留待复杂迷宫阶段一起调。



## Phase 1：firefly_explorer（自主探索节点）

### 目标

让机器人**自己找路探索**，而不是人给航点。这是 robot-system 设计文档里的"萤火V3引擎"。

### 架构

```
/map (OccupancyGrid from slam_toolbox)
  │
firefly_explorer.py (ROS2 node)
  ├── 1. 提取 frontiers（free/unknown 边界的连通段）
  ├── 2. 过滤窄缝（宽度 < 机器人直径的 frontier 跳过）
  ├── 3. 打分选最优（距离最近 + 未知区域面积最大）
  ├── 4. 发 NavigateToPose goal 到 frontier 中心
  ├── 5. 等结果（成功→找下一个；失败/超时→拉黑该区域，找下一个）
  └── 6. 无新 frontier → 探索完成，回报
  │
NavigateToPose action → Nav2 规划+控制 → /cmd_vel → sim_bridge → MuJoCo
```

### 与旧 algo3_headless.py 的对应

| algo3_headless 概念 | firefly_explorer 实现 | 备注 |
|---|---|---|
| frontier / find_gates | OccupancyGrid 形态学操作找 free/unknown 边界 | 标准做法，numpy/cv2 |
| 门宽度判断（PASS_CLEAR） | frontier 长度 × 分辨率 ≥ 机器人直径 | 防窄缝陷阱 |
| A* 寻路 | **Nav2 planner**（不用自己写） | 大大简化 |
| DWA 局部控制 | **Nav2 controller**（DWB/MPPI） | 不用自己写 |
| 经验墙（HIT_CONFIRMED） | costmap obstacle layer 自动标记 | ROS 内建 |
| 障碍跟踪器 ObstacleTracker | 可选：移植为独立 ROS2 节点 | 后续加 |

**核心**：只写 frontier 检测 + 选点逻辑（~200 行），导航执行全交给 Nav2。

### 实现步骤

1. **创建 `simtrack/firefly_explorer.py`**（ROS2 Python 节点）
   - 订阅 `/map`（nav_msgs/OccupancyGrid）
   - 回调里做 frontier 检测（cv2 形态学梯度 或 numpy 差分）
   - 维护已访问/已拉黑区域集合
   - 用 ActionClient 发 NavigateToPose
   - 状态机：IDLE → PLANNING → NAVIGATING → (SUCCESS|FAIL) → IDLE

2. **frontier 检测算法**（numpy 向量化）
   ```python
   # free=0, unknown=-1, occupied=100
   free_mask = (grid == 0)
   unknown_mask = (grid == -1)
   # frontier = free 格子中，至少有一个 unknown 邻居的
   from scipy.ndimage import binary_dilation
   frontier = free_mask & binary_dilation(unknown_mask, iterations=1)
   # 连通域聚类 → 每个 cluster = 一个候选 frontier
   ```

3. **frontier 打分**
   - 距离：robot 到 frontier 中心的 Manhattan/A* 距离（近优先）
   - 信息增益：frontier 周围 unknown 格数量（大优先）
   - 综合分 = α/距离 + β×信息增益

4. **测试**
   - 在 20×20m 迷宫里跑，机器人从 (1.5,1.5) 出发
   - 期望：自动发现走廊尽头 → 导航过去 → 发现新 frontier → 继续探索 → 绕完整圈
   - 成功标准：地图覆盖率 >90%，无人工干预

### 预计工作量

- firefly_explorer.py：~200-300 行（frontier 检测 + 状态机 + action client）
- 测试调试：地图边界 / Nav2 规划失败处理 / 死锁防护
- **不写**：规划器、控制器、costmap（Nav2 全包了）

---

## Phase 2：复杂迷宫

### 目标

更真实的迷宫环境——不只是"绕一个方块"，而是有房间、死胡同、窄缝、多路径。
用于压测 Nav2 在复杂场景下的表现。

### ✅ 2.0 完成：rooms5x5 传统房间迷宫（2026-08-13）

`maze_gen.py` 重构为多迷宫（`MAZES` 字典 + CLI `python -m simtrack.maze_gen rooms5x5 [seed]`），
新增 `gen_rooms_grid()`：

- 5×5 房间，每间 3×3m（总 15×15m），门宽 1.5m（居中于 3m 墙段）。
- 随机 DFS 生成树 → **保证全连通 + 起点(0,0)→终点(4,4) 有路**；`extra_prob=0.08` 额外开门
  造少量环路 → 留下死胡同（seed=7 得 4 个 1 门死胡同）。
- 每个房间 1~4 扇门（生成树保证 ≥1）。BFS 校验连通性 + 打印起点→终点路径。
- 输出高度图 `confirmed/maze_rooms5x5.png` + 人眼核对图 `_annot.png`（标 S/G/门数）。

**sim 适配（关键 bug 修复）**：
- `sim_server.py` 原 `px_per_m = hf_w // 20` 硬编码 20m 宽 → rooms5x5(15m/750px) 会算成 37px/m，
  射线尺度全错。改为参数 `px_per_m=50`（maze_gen 统一）。
- `sim_bridge.py` 加 `MAZE` 环境变量选迷宫（`confirmed/maze_<name>.png`），启动：
  `MAZE=rooms5x5 bash ~/simtrack/run_sim.sh`。
- 迷宫文件命名统一为 `maze_<name>.png`（旧 `maze20.png` → `maze_loop20.png`）。

**实测（rooms5x5，firefly 自主探索）**：
- Nav2 **能穿过 1.5m 门**（"Reached the goal!"/"Goal succeeded"），机器人在房间网格里逐间探索。
- firefly 连续到达多个房间：(7.0,9.7)→(7.2,7.6)→(7.1,13.8)→(4.2,7.2)→(10.9,14.0)…，单段最长 9.6m。
- 修了一个 busy-loop bug：机器人在某 frontier 正上方时(distance≈0)，Nav2 秒成功→重选同点→
  0.5s 内刷 15 行。修复：`_pick_best` 跳过离机器人 < `min_goal_dist`(0.35m) 的 frontier。

### maze_gen.py 扩展方案

当前 maze_gen.py 是**数据驱动**的（墙段列表），扩展只需加更多 WALLS 条目。

#### 2a. 房间迷宫（中等复杂度）

```
20×20m，4-6 个房间，门洞连接：
┌────┬────┬────┐
│    │    │    │
│ R1 │ R2 │ R3 │
│    │ 门 │    │
├─门─┼────┼─门─┤
│    │    │    │
│ R4 │ R5 │ R6 │
│    │    │    │
└────┴────┴────┘
每个房间 6×6m，门洞 1.5m 宽
```

#### 2b. 死胡同迷宫（高复杂度）

在走廊上加分支死胡同——测试 Nav2 的回退/重规划能力。
frontier 探索器需要能识别死胡同（frontier 到头了）并折返。

#### 2c. 窄缝测试

0.8-1.2m 宽的缝隙——Nav2 inflation 默认 0.55m，1.2m 缝勉强可过，0.8m 过不去。
用于验证"窄缝探测"概念（旧 algo3_headless 的 PASS_CLEAR 逻辑在 ROS 层的等价）。

### 实现方式

```python
# maze_gen.py 增加预设方案
MAZES = {
    "loop20": {  # 当前版本
        "size": (20, 20),
        "walls": [...],
    },
    "rooms20": {  # 房间迷宫
        "size": (20, 20),
        "walls": [...],
    },
    "deadends30": {  # 死胡同迷宫
        "size": (30, 20),
        "walls": [...],
    },
}
```

`SimBackend(maze="rooms20")` 选择不同迷宫。

### 预计工作量

- maze_gen.py 扩展（加 WALLS 定义 + 预设选择）：~100 行
- 主要是设计墙段坐标（可程序化生成：房间分割 → 门洞裁剪）

---

## 优先级与时间线

| 优先级 | 任务 | 预计 | 依赖 |
|---|---|---|---|
| **P0** | firefly_explorer.py | 1-2 天 | 当前管线（已就绪） |
| **P1** | 房间迷宫（rooms20） | 半天 | maze_gen.py 扩展 |
| **P2** | 死胡同迷宫 + 窄缝测试 | 半天 | P1 |
| **观察** | 航点角落失败（inflation 调参） | 随时 | 不阻塞 |

## 后续方向（本次不做，记录方向）

- **接 rl_sar**：MuJoCo 仿真已在，换 cmd_vel → rl_sar 关节控制即可上自定义步态
- **接 unitree_ros2**：真机部署时把 sim_bridge 换成 unitree_ros2 + LiDAR 驱动
- **障碍跟踪器移植**：ObstacleTracker 作为独立 ROS2 节点，给 Nav2 costmap 喂预测速度
- **二维码地标**：作为绝对修正源（slam_toolbox localization 模式的辅助）
