# Firefly V3 设计文档

## 核心理念

**LIDAR 只管画图，Move 只管走路。门不是终点——门是"值得过去看一眼"的候选点。**

萤火虫不贪心、不预判、不看远。每一步只闪一下（LIDAR 扫描），朝最亮的光斑飞过去（Move），到了再闪、再看。**越简单的循环越不犯错。**

## 数据流

```
┌─────────────────────────────────────────────────────┐
│                    LIDAR 10Hz                        │
│  scan(bx, by)                                        │
│  ├─ 120射线, 步长0.1m, 最大15m                       │
│  ├─ 每步: explored[y,x]=True                        │
│  ├─ 命中障碍: obstacles[y,x]=True                   │
│  │            obstacles[prev_y,prev_x]=True  ← 膨胀   │
│  └─ 只写不读 (纯感知)                                │
├─────────────────────────────────────────────────────┤
│                    Move 1Hz                          │
│  mv.step(tx, ty)                                     │
│  ├─ 朝 path[path_idx] 直线移动 (speed=5m/s)          │
│  ├─ blocked() → bounce (随机转45-120°)               │
│  ├─ stuck检测 → 300步不动 → bounce(90-180°)          │
│  └─ 只读不写 (纯执行)                                │
├─────────────────────────────────────────────────────┤
│              决策循环 (主循环 200Hz)                   │
│  ┌─ path跑完/不存在?                                 │
│  │   └─ find_gates(vx,vy)  → 收集前20个门           │
│  │      pick_gate(远近混用) → gate_path() 回溯       │
│  ├─ path存在但走歪了?                                │
│  │   └─ wander>600 → 三层保底恢复                    │
│  ├─ 该放路标了? (Manhattan≥30格)                     │
│  │   └─ wall_dist>CLEARANCE → milestones.append     │
│  └─ 没门了?                                         │
│      └─ no_gate_count>3 → backtrack                 │
└─────────────────────────────────────────────────────┘
```

## 核心数据结构

```
obstacles[500,500] bool    ← LIDAR扫描积累的障碍物 (31KB)
explored[500,500] bool     ← LIDAR扫过的区域 (31KB)
gt[500,500] bool           ← 地面真值, 启动时预计算, O(1)查表
milestones[]               ← 路标链 [(vx,vy), ...] 运行时3m间隔
path[]                     ← 当前A*路径 [(wx,wy), ...]
path_idx                   ← 当前路径消费位置
```

**按位压缩后总计 ~62KB。500×500 = 250K格，每格 0.1m×0.1m。**

## 完整决策流

```
每个 timestep (0.005s):
  │
  ├─ LIDAR_TICK? (每20步=10Hz)
  │   └─ scan(bx, by) → 标记 obstacles + explored
  │
  ├─ 路标放置?
  │   └─ Manhattan距上标 ≥ 30格 (3m) + wall_dist > 5格 (0.5m)
  │       └─ milestones.append + save_state()
  │
  ├─ path 需要重新规划? (path==None 或 path_idx到头)
  │   └─ find_gates(vx,vy) → 收集前20个门
  │      gate = pick_gate(远近混用)
  │      STUCK?    → gates[0]    最近门, 逃命
  │      FAR模式?  → gates[-1]   最远门, 铺前线
  │      NEAR模式? → gates[len//2] 中位, 均匀覆盖
  │      gate_path() → 回溯选中门的路径 → path
  │      ⚠️ 无门? → no_gate_count++ → backtrack
  │
  ├─ 走歪了? (path存在但3秒离目标越来越远)
  │   └─ Layer 1: line_clear 到最近5个路标 → 瞬回
  │      Layer 2: path=None → 触发 find_gates → A*找路
  │      Layer 3: backtrack A* → 走复杂走廊回退
  │      Layer 4: bounce → 随机转, LIDAR续扫
  │
  ├─ path 存在且正常 → 消费
  │   └─ 到目标<1m? → path_idx++
  │      距离增大5%? → wander++  (累积600=3秒)
  │      距离减小/稳定? → wander-- (最少0)
  │      → mv.step(tx, ty)
  │
  └─ 否则
      └─ mv._bounce(90, 180)
```

## 关键决策点与调试要点

### 1. 门选择 (pick_gate)

| 参数 | 行为 | 使用场景 |
|------|------|---------|
| `stuck=True` | gates[0] (最近) | no_gate_count>0 时触发 |
| `mode="near"` | gates[len//2] (中位) | 默认, 均匀扫全图 |
| `mode="far"` | gates[-1] (最远) | 先铺前线, 快速扩张 |

调试日志: `[GATE] gates=N stuck=T/F` — 看收集了多少门和当前状态

### 2. 迷失恢复

触发条件: `wander > 600` (连续3秒离目标越来越远)

```
[LOST] →路标(x,y)    ← Layer 1 成功: line_clear 到路标
[LOST] 重新规划       ← Layer 2 触发: path=None, 下次循环走 find_gates
```

### 3. 回溯

触发条件: `no_gate_count > 3` (连续4次 find_gates 返回空)

```
[BACK] →路标(x,y)    ← 回溯到上一个路标
[BACK] →起点(x,y)    ← 上一个路标也到不了, 回溯到起点
bounce               ← 起点也到不了, 放弃 → 随机转
```

### 4. 安全间隙

所有涉及位置决策的环节都强制执行安全间隙:

| 环节 | 检测 | 阈值 |
|------|------|------|
| LIDAR扫描 | 命中→膨胀1格 | 0.1m |
| A*展开邻居 | `wall_dist(nx,ny) > ROBOT_R` | 0.5m (5格) |
| 路标放置 | `wall_dist(vx,vy) > CLEARANCE` | 0.5m (5格) |
| 门过滤 | 门本身 `wall_dist > CLEARANCE` | 0.5m |
| A*代价 | 离墙<2m (20格) 罚分×3 | 2m缓冲区 |
| blocked() | 半径5格圆检测 gt[] | 物理碰撞 |

## 运行参数速查

```
VOXEL      = 0.1m    # 格子分辨率
W3         = 500     # 50m / 0.1m
SPEED      = 5.0     # 正常速度 m/s
SPEED_MAX  = 8.0     # 远距离加速上限
LIDAR_RANGE= 15.0    # 激光探测半径 m
LIDAR_RAYS = 120     # 每圈射线数
LIDAR_TICK = 20      # 扫描频率 (200Hz/20=10Hz)
SAFE_R     = 0.5     # 机器人物理半径 m
ROBOT_R    = 5       # 半径格数 (0.5/0.1)
CLEARANCE  = 5       # 路标安全距离格数
MILESTONE_STEP = 30  # 路标间隔格数 (3m)
FIXED_SEED = 42      # 固定随机种子

EXPLORE_MODE = "near"  # 探索模式: "near" | "far"
```

## 调试命令

```bash
# 监控关键指标
grep -E '\[(GATE|BACK|LOST|WAYPOINT|BOUNCE)\]' | tail -50

# 检查探索覆盖率
python3 -c "
import numpy as np
state = np.load('scans/scan_vox.npy')
print(f'obstacles={state[0].sum():,}  explored={state[1].sum():,}  /250K')
print(f'覆盖率: {state[1].sum()/250000*100:.1f}%')
"
```

## 架构铁律

> **看和走解耦** — LIDAR只往表里写，Move只从表里读
> **门不是终点** — path走完自然重新评估，不预设路线
> **越简单越不犯错** — 不加预测、不加贪心、不加全局规划
> **偏离=有鬼** — 目标是直线可达的，歪了就是出问题了
