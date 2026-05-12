# Firefly V4 设计文档

## 一句话

`algo3_v4.py` = V3 的完整复制，**只改了一行**：门系统从 `polygon_gates` 换成 `find_gates + merge_gates`。单元测试 37/37 全绿。

## 架构：ODA 循环

```
每 200 步 (1Hz):
  scan()                          观察：激光填充字典SLAM
  gates, _ = find_gates(vx, vy)   决策：栅格级前沿搜索→门列表
  gate = pick_gate(gates, "far")  决策：选最远的门作为目标
  raw = astar_to(vx, vy, gx, gy)  规划：三级跳A*到门
  yellow_wps = gen_yellow_waypoints(raw)  渲染：每1m一个黄球
  target_wp = yellow_wps[0]       行动：取第一个黄球

每帧 (0.005s):
  mv.step(tx, ty)                 恒速 5m/s 朝 target_wp 冲
  到达 → pop 下一个黄球
```

**关键约束**（主人确认）：
- 决策帧内门/路径一秒不变
- A* 中间格只走 FREE（已扫描），目标允许 UNKNOWN（前沿门）
- 走歪了没关系，A* 下一秒纠正

## 核心修改：门系统

### V3 的问题：polygon_gates 死锁

```
polygon_gates:
  LIDAR 120射线 → 多边形边界
  相邻命中点间距 >3m → GATE 边
  合并相邻 GATE 边 → 取边中点作为门
  过滤：跳过 FREE（已探索）、跳过 WALL（墙内）
  → 门全在 UNKNOWN 区域

astar_to:
  walkable(goal) → gget == FREE → UNKNOWN ≠ FREE → 返回 None
  → 没有路径 → 没有黄球 → 没有 [GATE] 打印
  → no_gate_count 累计 → 回退里程碑 → 机器人乱撞
```

这就是为什么 V3 日志全是 `[BACK]`、没有一条 `[GATE]`。

### V4 的修复：find_gates + merge_gates

```
find_gates:
  从机器人位置展开 A* 跳步搜索
  每步检查：当前格 FREE? + 邻接 UNKNOWN? + 离墙安全?
  → 收集所有“前沿门”体素 (在 FREE 区域，邻接 UNKNOWN)

merge_gates:
  4邻域连通域聚类
  每簇取离几何中心最近的体素作为门
  → 门在 FREE，邻接 UNKNOWN
  → A* 可直接到达
```

**两者的门格式完全相同**：`(cg, wd, vx, vy)` = (A*距离, 离墙距离, 细格x, 细格y)

区别只在门的**位置**：polygon 门在 UNKNOWN，find_gates 门在 FREE。

### 实际改动（一行）

```python
# algo3_firefly.py 第 670 行：
- gates = polygon_gates(bx, by)
+ gates, _ = find_gates(vx, vy)   # find_gates返回(gates, came_from)
```

## 代码关键点

### SLAM 字典 (第 63-80 行)

```python
UNKNOWN, FREE, WALL = 0, 1, 2
grid = {}              # {(vx,vy): state}  0.1m 细格，无限大
_wd = {}               # wall_dist 缓存：O(1) 查离墙距离
_cnt = {FREE:0, WALL:0}  # 增量计数，避免扫全字典
```

写 WALL 时清 `_wd` 缓存（因为墙变了，距离要重算）。

### 激光扫描 scan() (第 134-161 行)

120 条射线，每条最多 150 步（15m / 0.1m），碰到障碍物标记 WALL，经过处标记 FREE。同时维护 `gate_cells`（FREE 邻接 UNKNOWN 的格）。

### 三级跳 A* (第 211-222 行)

```python
# 跳步规则：
JUMP_1M = 10    # wall_dist ≥ 1m → 一次跳 10 格
JUMP_03 = 3     # wall_dist ≥ 0.3m → 一次跳 3 格  
JUMP_NEAR = 1   # wall_dist < 0.3m → 一次走 1 格
```

搜索时 `walkable(nx, ny)` 挡着 → 只从 FREE 格展开，UNKNOWN/WALL 不通。

### astar_to 目标检查 (第 420-423 行)

```python
if not walkable(fvx, fvy): return None   # 起点必须在 FREE
if gget(tfx, tfy) == WALL: return None   # 目标不能是墙
# UNKNOWN 目标允许 → 前沿门可以到达
```

与 V3 的区别：V3 要求 `walkable(tfx, tfy)`（= FREE），V4 只禁 WALL。

### 黄球渲染 (第 469-483 行 + XML 第 494-497 行)

```python
# BallManager:
def add_waypoint(self, wx, wy):
    self.d.mocap_pos[body.mocapid] = [wx, wy, 0.8]

# XML:
<geom type="sphere" size="0.15" rgba="1.0 1.0 0.0 0.9"/>  # 亮黄，小
```

200 个球（MAX_WAYPOINT_BALLS=200），z=0.8 高度，每帧渲染。与里程碑蓝球（z=1.5）、门球（z=2.0）分层清晰。

### 门查找 find_gates (第 252-291 行)

A* 跳步搜索 + 前沿检测 + 墙距离惩罚：

```python
if gget(cx, cy) == FREE:
    has_unk = any(gget(cx+dx, cy+dy) == UNKNOWN 
                  for dy in (-1,0,1) for dx in (-1,0,1))
    if has_unk and wall_dist(cx, cy) > CLEARANCE:
        gates.append((cg, wall_dist(cx, cy), cx, cy))
```

收集 MAX_GATES=200 个或超过 MAX_GATE_DIST=3000 格后停止。

### 门合并 merge_gates (第 296-354 行)

4邻域 BFS 连通域聚类，每簇取离几何中心最近的体素（且必须在 `came_from` 中，即 A* 可达）。按 A* 距离排序→`far` 模式取最后一个。

### 移动 Mover (第 520-557 行)

```python
def step(self, tx, ty, step):
    # 计算目标朝向，转动受限 (YAW_RATE=6.0 rad/s)
    # 恒速 SPEED=5.0 m/s
    # 碰到障碍 → bounce (30°~90° 随机转向)
    # 卡住超时 (STUCK_TIMEOUT=300步=1.5s) → 大角度反弹
```

### 迷失恢复 (第 700-718 行)

无门超过 MAX_NO_GATE=5 次 → 从里程碑列表倒序搜索最近的 A* 可达路标 → 回溯。

## 单元测试 (test_algo3_v4.py)

37 个测试，7 个模块：

| 模块 | 测试数 | 验证内容 |
|------|--------|---------|
| 1. 激光扫描 | 4 | FREE/WALL 填充，15m范围 |
| 2. wall_dist + walkable | 3 | 缓存，边界条件 |
| 3. 门查找 | 12 | 88个门，全在FREE邻接UNKNOWN |
| 4. A* | 4 | 到门/起点WALL/目标WALL/目标UNKNOWN |
| 5. 黄球生成 | 4 | 空路径/单点/间距≈1m |
| 6. 避障 blocked() | 2 | 空地/障碍物 |
| 7. 全流程模拟 | 7 | scan→gates→A*→黄球→移动→重新scan |

PASS=37/37, FAIL=0/37。不需要 MuJoCo viewer。

## 文件清单

```
test_scripts/
├── algo3_firefly.py       # V3 (polygon_gates, 无黄球渲染)
├── algo3_v4.py            # V4 (find_gates, 黄球渲染)  ← 当前
├── algo4_midline.py        # V4存档 (扇形采样, 等复活)
├── test_algo3_v4.py        # V4单元测试, 37/37全过
└── ...
```

## 已知限制

1. `find_gates` 每1秒重新搜索 → 可缓存门坐标减少计算
2. 门合并后 lost `came_from` → `astar_to` 重新寻路 → 浪费计算
3. `find_gates` 内部 A* 和 `astar_to` 各自展开 → 可复用 came_from
4. 起点始终固定 (3,3) → 地图四边墙内不可达
5. 黄球只在决策帧更新 → 帧间不重绘（但路径一秒不变，无所谓）

## 经验教训

- **上下文腐烂是真实的**：五次修改五次幻觉，永远在没有 MuJoCo 的单元测试里验证
- **单元测试是唯一真相**：37 个独立测试比 5 轮"我感觉改了"靠谱一万倍
- **一行代码的改变不要用十次补丁**：`polygon_gates → find_gates` 是最小修改，不需要拆成 5 个 commit
- **先测再跑**：test_algo3_v4.py 先证明所有模块正常，再进 MuJoCo 仿真
