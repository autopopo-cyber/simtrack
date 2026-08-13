# 下一步计划：firefly_explorer + 复杂迷宫 —— 2026-08-13

> 前置：ROS2 MuJoCo 管线已验证（见 `docs/2026-08-13-ros2-mujoco-pipeline.md`）。
> 当前状态：sim_bridge → slam_toolbox → Nav2 全通，5/7 航点导航成功。
> 用户指示：先 2（firefly_explorer 自主探索）后 3（复杂迷宫），航点失败先观察不急修。

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
