# Firefly V4 功能规格文档

`test_scripts/algo3_v4.py` · commit `a403d03` · 737行

---

## 1. 架构概览

```
                    ┌──────────────────────────────┐
                    │        ODA 循环 (1Hz)         │
                    │                              │
  scan(LIDAR) ─────→│  find_gates() 栅格前沿搜索   │
  10Hz              │  merge_gates() 连通域聚类     │
                    │  pick_gate("far") 选最远门    │
                    │  astar_to() 三级跳A*         │
                    │  gen_yellow_waypoints() 每1m │
                    │                              │
                    └──────────┬───────────────────┘
                               │ yellow_wps
                               ▼
                    ┌──────────────────────────────┐
                    │       移动执行 (每帧)         │
                    │                              │
                    │  target_wp = yellow_wps[0]   │
                    │  Mover.step(tx, ty) 5m/s     │
                    │  到达1m内 → pop下一个黄球     │
                    │                              │
                    └──────────────────────────────┘
```

**核心原则**
- 决策帧内门/路径一秒不变（主人确认）
- A* 中间格只走 FREE（已扫描），目标允许 UNKNOWN（前沿门）
- 走歪了没关系，A* 下一秒纠正
- 碰撞检测用机器人半径 0.5m 预判圈

---

## 2. 数据结构

### 2.1 SLAM 字典地图

```python
UNKNOWN, FREE, WALL = 0, 1, 2
grid = {}              # {(vx, vy): state}  0.1m 细格，无限大，稀疏存储
_wd  = {}              # wall_dist 缓存: (vx, vy) → 曼哈顿距离到最近墙
_cnt = {FREE: 0, WALL: 0}  # 增量计数器，避免扫全字典
```

**VOXEL = 0.1m**。所有坐标以细格为单位。

**gget(vx, vy)**: 查 grid，不存在的 key 返回 UNKNOWN。O(1) 字典查找。

**gset(vx, vy, val)**: 写入 grid。
- 相同值不重复写
- 旧值非 UNKNOWN 时递减 `_cnt[old]`
- 新值递增 `_cnt[val]`
- **写 WALL 时清空 `_wd` 缓存**（墙变了，所有距离要重算）

⚠️ **坑**: load_state() 恢复时直接用 `grid[k] = v` 绕过 gset()，避免了每个 WALL 都触发 `_wd.clear()`。但这样 `_wd` 不会为旧数据预热，首次 `wall_dist()` 会 O(21×21) 扫描。

### 2.2 门格式

所有门统一为四元组: `(cg, wd, vx, vy)`
- `cg`: A* 距离（从机器人到门的代价）
- `wd`: 离最近墙的曼哈顿距离（越大越安全）
- `vx, vy`: 细格坐标（0.1m 分辨率）

`pick_gate("far")` 取列表最后一个（已按 cg 升序排序）。

### 2.3 路标里程碑

```python
milestones = [(vx, vy), ...]   # 走过的轨迹，细格坐标
```

撒点条件: `wall_dist(vx, vy) > CLEARANCE`（不在墙边才记录）

---

## 3. 函数规格

### 3.1 感知层

#### `scan(bx, by)`

| 属性 | 说明 |
|------|------|
| 输入 | `bx, by` 世界坐标 |
| 输出 | 无（副作用: 修改 grid, polygon, gate_cells) |
| 调用频率 | 每 LIDAR_TICK=20 步（10Hz） |
| 复杂度 | O(LIDAR_RAYS × LIDAR_STEPS) = 120×150 = 18,000 步 |

**算法**:
1. 120 条射线均匀分布 (0→2π)
2. 每条射线最多走 LIDAR_STEPS=150 步 (15m/0.1m)
3. 每步检查 `is_obstacle_world(wx, wy)`:
   - 是障碍 → 标记 WALL（命中格+前一格），停止射线
   - 不是障碍 → 标记 FREE，检查是否邻接 UNKNOWN（是则加入 gate_cells）
4. 射线命中点排序后建多边形边界

**副作用**:
- 修改 `grid`（填 FREE/WALL）
- 修改 `polygon`（多边形边界，V4 已不使用但仍在维护）
- 修改 `gate_cells`（栅格前沿，V4 已不使用但仍在维护）

⚠️ **坑**: polygon 和 gate_cells 在 V4 中已废弃（ODA 用 find_gates 替代 polygon_gates），但 scan() 仍然维护它们，浪费约 20% 扫描时间。

#### `is_obstacle_world(wx, wy)`

返回 True 如果:
1. hfield 像素 ≠ ROAD_PIX (128)（赛道墙）
2. 距离任意障碍物 < OBS_CLEAR (1.5m)

⚠️ **坑**: `sample_hf()` 使用 SCALE=2.0 坐标转换。hfield 的 MuJoCo 坐标 `(mx, my)` 与世界坐标 `(wx, wy)` 的关系: `mx = wx/SCALE`。像素坐标: `px = mx×40, py = 1999 - my×40`。

#### `blocked(wx, wy)`

| 属性 | 说明 |
|------|------|
| 输入 | 世界坐标 |
| 输出 | bool |
| 用途 | 移动前预判碰撞 |

以机器人半径 ROBOT_R=5 格（0.5m）画圆，逐格检查 `is_obstacle_world()`。

### 3.2 规划层

#### `find_gates(fvx, fvy)` → `(gates, came_from)`

| 属性 | 说明 |
|------|------|
| 输入 | 机器人细格坐标 |
| 输出 | `gates`: 门列表 [(cg,wd,vx,vy),...], `came_from`: A* 回溯字典 |
| 复杂度 | O(ASTAR_MAX_EXPAND × 4) = 120,000 节点 |

**算法**:
1. 以机器人为起点，A* 跳步展开（三级跳）
2. 每步检查: `gget(cx,cy) == FREE`? 邻接 UNKNOWN? `wall_dist > CLEARANCE`?
3. 满足 → 加入 `gates` 列表
4. 收集满 MAX_GATES=200 或超过 MAX_GATE_DIST=3000 格 → 停止
5. 调用 `merge_gates()` 聚类合并

**展开规则**: 四方向跳步，墙距离代价 `penalty = max(0, 20-wd) × 3`（贴墙格加罚，远离墙不加罚）。

⚠️ **关键**: 门的三个条件：
1. `gget(cx,cy) == FREE` — 门在已探索区域
2. 邻接 UNKNOWN — 门是前沿
3. `wall_dist > CLEARANCE` (=5格=0.5m) — 门离墙安全

这保证了 A* 能可靠到达门（门在 FREE 里，A* 能在 FREE 里展开）。

#### `merge_gates(raw_gates, came_from)` → `clusters`

| 属性 | 说明 |
|------|------|
| 输入 | 原始门列表 + A* 回溯字典 |
| 输出 | 聚类后的门列表 [(cg,wd,vx,vy),...] |
| 算法 | 4邻域 BFS 连通域 → 簇 → 每簇选离几何中心最近的门 |

**为什么需要聚类**: 相邻的栅格前沿在语义上是同一扇门（如走廊尽头），不应被当作 100 个独立目标。

**聚类规则**:
- 4邻域连通（上下左右）
- 每簇取离几何中心最近的门体素
- 门体素必须在 `came_from` 中（否则 A* 不可达）

#### `polygon_gates(bx, by)` → `gates`

⚠️ **V4 已废弃**。保留函数但 ODA 不再调用。门来自多边形边缘中点 → 门在 UNKNOWN → A* 死锁。

**废弃原因**: 多边形边缘中点落在 UNKNOWN 区域。`astar_to` 的 `walkable(goal)` 要求 FREE。即使 V4 放宽为目标只禁 WALL，从起点到 UNKNOWN 目标的路径往往只有 1-2 步（门紧贴探索边界），A* 几乎找不到有效路径。

#### `pick_gate(gates, mode, stuck)` → `gate` or None

| mode | 行为 |
|------|------|
| `"far"` (默认) | 取最后一个（A* 最远） |
| `"near"` | 取第一个（A* 最近） |
| `"mix"` | ≥50个门取最远，否则取最近 |
| `stuck=True` | 强制取最近（不管 mode） |

#### `astar_to(fvx, fvy, tfx, tfy)` → `path` or None

| 属性 | 说明 |
|------|------|
| 输入 | 起点(fvx,fvy)、目标(tfx,tfy) 细格坐标 |
| 输出 | 世界坐标路径 [(wx,wy),...] 从起点后第一个到目标 |
| 返回 None | 起点不可达、目标在墙里、A* 耗尽步数 |

**边界检查**:
- `walkable(fvx, fvy) == False` → None（起点必须 FREE + 离墙安全）
- `gget(tfx, tfy) == WALL` → None（目标不能是墙）
- ⚠️ **目标允许 UNKNOWN**（与 V3 不同）— 这是前沿门可达的关键

**跳步规则** (`jump_steps`):
- `wall_dist ≥ 10格(1m)` → 一次跳 10 格
- `wall_dist ≥ 3格(0.3m)` → 一次跳 3 格
- `wall_dist < 3格` → 一次走 1 格

每跳检查 `walkable(nx, ny)`，不通则退回。**UNKNOWN 格被 `walkable` 挡在路径中间**—这是 "A* 只走 FREE" 的实现。

⚠️ **坑**: `fine_path()` 从 came_from 回溯路径，返回世界坐标 `((px+0.5)*VOXEL, (py+0.5)*VOXEL)`。如果目标 UNKNOWN，路径终点在 UNKNOWN 格的中心—没关系，机器人走到后扫描，UNKNOWN 变 FREE。

#### `gen_yellow_waypoints(raw_path)` → `yellow_wps`

| 属性 | 说明 |
|------|------|
| 输入 | 世界坐标路径 [(wx,wy),...] |
| 输出 | 世界坐标路点 [(wx,wy),...] 间距 ≈1m |

**算法**: 沿路径累加欧氏距离，每超过 1m 就记录一个点。强制包含终点。

⚠️ **边角**: 如果路径 < 1m，至少返回起点+终点两个黄球。

### 3.3 行动层

#### `Mover.step(tx, ty, step)`

| 属性 | 说明 |
|------|------|
| 朝向 | YAW_RATE=6.0 rad/s 转动 |
| 速度 | 恒速 SPEED=5.0 m/s |
| 碰撞 | 预判 `blocked(nx, ny)` → bounce (30°~90° 随机转向) |
| 卡住 | STUCK_TIMEOUT=300 步 (1.5s) 移动 < 0.5m → 大角度弹跳 (90°~180°) |

**Bounce 机制**: 碰撞后设 `force` 计数器（0.3s 强制保持反弹方向），避免立刻回弹。

⚠️ **坑**: 碰撞检测用机器人当前位置+速度向量预判下一帧位置，但 MuJoCo timestep 0.005s 极短，速度 5m/s → 每帧只移动 0.025m。实际碰撞检测窗口非常窄。

### 3.4 迷失恢复

**触发条件**: `no_gate_count > MAX_NO_GATE (5)` + 有里程碑

**恢复流程**: 从新到旧遍历里程碑，找到第一个 A* 可达的 → 路径回溯，只取终点作为目标黄球。

**fallback**: 遍历完仍无路 → bounce 大角度转向。

---

## 4. 可视化

| 球类型 | 颜色 | 大小 | 高度 | 数量 | 用途 |
|--------|------|------|------|------|------|
| 里程碑 (mstone) | 蓝 rgba(0.3,0.6,1.0) | 0.2 | z=1.5 | 1000 | 走过轨迹 |
| 门球 (gate) | 金 rgba(1.0,0.8,0.2) | 0.25 | z=2.0 | 50 | 当前目标门 |
| 黄球 (wp) | 亮黄 rgba(1.0,1.0,0.0) | 0.15 | z=0.8 | 200 | A*路径路点 |
| 终点 | 绿 rgba(0.2,1.0,0.2) | 1.5 | z=2.0 | 1 | 固定(3,95) |
| 障碍物 | 红 rgba(0.9,0.2,0.2) | 半径1.0 | z=2.0 | 随机 | 圆柱 |

所有球使用 MuJoCo mocap（动捕球）— 位置可随时修改，无物理碰撞。初始位置 `(0,0,-10)` 藏在地下，激活后移到目标位置。

⚠️ **坑**: MuJoCo mocap body 名字必须与 init 列表一致，否则 `self.m.body(name)` 抛 KeyError。200 个 wp_0..wp_199 逐一在 `waypoint_bodies` 中注册。

---

## 5. 常量参考

| 常量 | 值 | 含义 |
|------|-----|------|
| VOXEL | 0.1 | 细格精度 (m) |
| ROBOT_R | 5 | 机器人半径 (格) = 0.5m |
| LIDAR_RANGE | 15.0 | 激光最大距离 (m) |
| LIDAR_RAYS | 120 | 射线数量 |
| LIDAR_TICK | 20 | 扫描间隔 (步) = 10Hz |
| PLAN_INTERVAL | 200 | 规划间隔 (步) = 1Hz |
| SPEED | 5.0 | 移动速度 (m/s) |
| YAW_RATE | 6.0 | 最大转动速度 (rad/s) |
| SAFE_R | 0.5 | 安全半径 (m) |
| JUMP_1M / JUMP_03 / JUMP_NEAR | 10 / 3 / 1 | A* 跳步 |
| WALL_SCAN_RADIUS | 10 | wall_dist 扫描半径 (格) |
| WALL_BUFFER_CELLS | 20 | 墙缓冲区 (格) = 2m |
| WALL_PENALTY | 3 | 贴墙代价倍数 |
| MAX_GATES | 200 | 最多收集门数 |
| ASTAR_MAX_EXPAND | 30000 | A* 最大展开节点 |
| MAX_NO_GATE | 5 | 无门后触发迷失恢复 |
| STUCK_TIMEOUT | 300 | 卡住超时 (步) = 1.5s |
| ARRIVE_THRESH | 1.0 | 到达黄球阈值 (m) |
| RENDER_SKIP | 100 | 渲染跳帧 = 5× |
| FINISH | (3.0, 95.0) | 终点世界坐标 |

---

## 6. 已知问题 & 坑点

### 6.1 性能
1. `find_gates()` 每次展开到 30000 节点——85 个门用 30000 展开
2. `find_gates` 和后续 `astar_to` 分别做 A*——可复用 came_from
3. scan() 维护已废弃的 polygon/gate_cells — 浪费 ~20% 扫描时间
4. `wall_dist()` 首次调用 O(21×21) 扫描，写 WALL 时缓存全清

### 6.2 边界条件
1. **起点必须在 FREE**: (3,3) 硬编码，地图加载后确保此位置是道路
2. **地图外**: `sample_hf` 越界返回 -1 → `is_obstacle_world` 判为障碍
3. **种子随机**: `FIXED_SEED = random.randint(0, 999999)` 每次启动不同障碍物
4. **终点检测**: 距 FINISH < 3m 停止，不是精确到达

### 6.3 逻辑风险
1. `gate_cells` 收集但从未被清理——无限增长
2. milestones 可能是细格坐标也可能是世界坐标——初始化时存的是 `(int(d.qpos[0]/VOXEL), ...)`，但 add_milestone 接受世界坐标
3. `yellow_wps` 在非 PLAN_INTERVAL 帧可能和渲染不同步
4. 迷失恢复只取路标终点，不重建黄球路径——机器人需要走多步才到

### 6.4 V3→V4 迁移教训
- **polygon_gates 死锁**: 门在 UNKNOWN → A* 不允许 UNKNOWN → 永远无路径
- **多边形边中点 vs 栅格前沿**: 前者是几何中点在未探索区，后者是已探索区的前沿格
- **上下文腐烂**: 五次补丁五次幻觉，"感觉改了" 不如 37 个单元测试

---

## 7. 文件布局

```
test_scripts/
├── algo3_firefly.py        # V3 (polygon_gates 门系统)
├── algo3_v4.py             # V4 (find_gates 门系统 + 黄球渲染) ← 当前
├── algo4_midline.py         # V4存档 (扇形采样实验, 等复活)
├── test_algo3_v4.py         # V4单元测试, 37/37全过
└── *.json                   # 各种workflow配置 (ComfyUI, 跟仿真无关)

../DESIGN_V4.md              # 本文档
../DESIGN.md                 # V3设计文档 (已过时)
../confirmed/track_clean.png  # 地图 (不变, MD5: 57271e30...)
../scans/scan_dict.npz       # 扫描存档 (自动生成)
```

---

## 8. 测试

`test_algo3_v4.py` — 无需 MuJoCo viewer，37 个独立测试，7 个模块全部通过：

| # | 模块 | 测试数 | 关键验证 |
|---|------|--------|---------|
| 1 | 激光扫描 scan() | 4 | FREE/WALL 填充, 15m 范围 |
| 2 | wall_dist + walkable | 3 | 缓存命中, UNKNOWN 不可走 |
| 3 | find_gates + merge_gates | 12 | 88个门全在FREE邻接UNKNOWN |
| 4 | A* astar_to() | 4 | 到门/起点WALL/目标WALL/目标UNKNOWN |
| 5 | gen_yellow_waypoints() | 4 | 空列表/单点/间距≈1m |
| 6 | blocked() 避障 | 2 | 空地/障碍物 |
| 7 | 全流程模拟 | 7 | scan→gates→A*→黄球→移动→重扫 |

**V3 → V4 唯一改动**: `gates = polygon_gates(bx, by)` → `gates, _ = find_gates(vx, vy)`
