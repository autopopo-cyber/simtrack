# 机器狗 50m 迷宫 — 效率探索 + 已知地图快速寻路 + 绑架恢复 实现计划

> **For Hermes:** 主人已授权自主执行（"开干别问我了，我睡了"）。按本计划 task-by-task 执行，每步验证后 commit。
> 设计规格: `docs/superpowers/specs/2026-08-05-nav-efficiency-design.md` (commit f7c434b)

**Goal:** ①探索未知地图效率更优 ②已知地图+障碍变化快速寻路（走中间）③随机位置快速到起点/终点

**Architecture:** 双层地图（static 旧地图背景 + live 实时障碍层，射线清除）+ Voronoi 代价走中间 + 两阶段流程 + 随机采样绑架恢复

**Tech Stack:** Python + MuJoCo EGL + numpy, 主文件 `test_scripts/algo3_headless.py`

---

## 环境速查

- Python: `/home/qin/mujoco-venv/bin/python`（MuJoCo 3.8.0，无 pytest）
- 分支: `exp/dog-navigation`；测试跑法: `python3 algo3_headless.py --seed N --render-every 0 --timeout T`
- 后台跑: tshell `curl -H "X-Token: orangepi" -d '{"cmd":"..."}' http://127.0.0.1:8765/start`
- 成绩单: `scans/baseline_seed<N>.json`；地图保存: `scans/scan_dict.npz`（现有 save_state/load_state，per-seed）
- 障碍: `obs_world = gen_obstacles(FIXED_SEED)`，`is_obstacle_world()` 是**世界真值**（仅碰撞/渲染，规划禁用）

---

## Task 1: 双层地图基础设施 — static_grid + gget_plan()

**Objective:** grid（live 层）保持现状；新增 static_grid 存旧地图；规划用 gget_plan() 叠加视图。

**Files:** Modify `test_scripts/algo3_headless.py`（SLAM 字典区 ~L92-110）

**Step 1:** 在 `grid` 定义后加：
```python
static_grid = {}   # 旧地图背景（只读）：WALL=永久的墙；加载地图时填充
KNOWN_MAP_MODE = False  # True=阶段2（加载旧地图），规划叠加 static_grid

def gget_plan(vx, vy):
    """规划用叠加视图：static 的墙永远 WALL；live 障碍/自由优先；其余 UNKNOWN"""
    s = static_grid.get((vx, vy), UNKNOWN)
    if s == WALL:
        return WALL
    l = grid.get((vx, vy), UNKNOWN)
    if l != UNKNOWN:
        return l
    return s
```

**Step 2:** 语法检查 `python3 -c "import ast; ast.parse(...)"`

**Step 3:** Commit `feat: 双层地图 static_grid + gget_plan 叠加视图`

---

## Task 2: scan() 射线清除

**Objective:** LIDAR 扫描时射线穿过的格强制 FREE（清掉旧障碍），命中格标 WALL。

**Files:** Modify `test_scripts/algo3_headless.py:171-186`（scan 函数）

**Step 1:** 改造 scan()：
```python
def scan(bx, by):
    for a in np.linspace(0, 2*math.pi, LIDAR_RAYS):
        cos_a, sin_a = math.cos(a), math.sin(a)
        prev_vx, prev_vy = int(bx/VOXEL), int(by/VOXEL)
        for step_i in range(1, LIDAR_STEPS+1):
            wx, wy = bx + cos_a*step_i*VOXEL, by + sin_a*step_i*VOXEL
            vx, vy = int(wx/VOXEL), int(wy/VOXEL)
            if is_obstacle_world(wx, wy):
                gset(vx, vy, WALL)          # 命中格 = 当前障碍
                if gget(prev_vx, prev_vy) == UNKNOWN:
                    gset(prev_vx, prev_vy, FREE)
                break
            # 射线清除：穿过格强制 FREE（旧障碍被"照"掉），static 墙不受影响（gset 只写 live）
            if gget(vx, vy) != FREE:
                gset(vx, vy, FREE)
            prev_vx, prev_vy = vx, vy
```

**Step 2:** 验证：跑 `--seed 7 --timeout 30`，确认无异常、FREE/WALL 计数正常

**Step 3:** Commit `feat: scan 射线清除 — 旧障碍位置被新扫描覆盖`

---

## Task 3: 保存/加载地图（--save-map / --load-map）

**Objective:** 阶段1 结束时保存地图到指定文件；阶段2 启动时加载为 static_grid。

**Files:** Modify `test_scripts/algo3_headless.py`（args L81-87 + 文件读写 L584-608）

**Step 1:** 加参数：
```python
ap.add_argument("--save-map", type=str, default="", help="跑完保存地图到文件 (npz)")
ap.add_argument("--load-map", type=str, default="", help="加载旧地图为 static_grid (npz)")
```

**Step 2:** 写 `save_map(path)` / `load_map(path)`：
```python
def save_map(path):
    """保存当前 grid（含墙+障碍）到文件"""
    if not grid: return
    xs = sorted(set(k[0] for k in grid)); ys = sorted(set(k[1] for k in grid))
    minx, maxx = xs[0], xs[-1]; miny, maxy = ys[0], ys[-1]
    arr = np.full((maxy-miny+1, maxx-minx+1), UNKNOWN, dtype=np.int8)
    for (vx, vy), val in grid.items():
        arr[vy-miny, vx-minx] = val
    np.savez(path, grid=arr, offset=(minx, miny), seed=FIXED_SEED)
    print(f"  [MAP] saved {len(grid)} cells → {path}", flush=True)

def load_map(path):
    """加载地图为 static_grid，开启 KNOWN_MAP_MODE"""
    global static_grid, KNOWN_MAP_MODE
    if not os.path.exists(path): return False
    data = np.load(path, allow_pickle=True)
    arr = data["grid"]; ox, oy = data["offset"]
    static_grid.clear()
    for vy in range(arr.shape[0]):
        for vx in range(arr.shape[1]):
            if arr[vy, vx] != UNKNOWN:
                static_grid[(vx+ox, vy+oy)] = int(arr[vy, vx])
    KNOWN_MAP_MODE = True
    print(f"  [MAP] loaded {len(static_grid)} cells from {path} (KNOWN_MAP_MODE)", flush=True)
    return True
```

**Step 3:** 主循环初始化处调用（加载在 build 前，保存在主循环结束后）：
- 加载：`if args.load_map: load_map(args.load_map)`
- 保存：收尾 `if args.save_map: save_map(args.save_map)`

**Step 4:** 切换规划到 gget_plan：把 find_gates/astar_to/traversable/wall_dist 里的 `gget` 改为 `gget_plan`（KNOWN_MAP_MODE 下有效；探索模式 gget_plan==gget 无差异）

**Step 5:** Commit `feat: 地图保存/加载 + 规划用 gget_plan 叠加视图`

---

## Task 4: 走中间 Voronoi 代价

**Objective:** A* 代价加 1/d² 项，路径贴通道中线。

**Files:** Modify `test_scripts/algo3_headless.py`（find_gates L299-312 / astar_to L407-415）

**Step 1:** 加常量 `VORONOI_C = 2.0`（L45 附近）

**Step 2:** find_gates 代价行加项：
```python
# 原: ng = cg + js + penalty
d = max(1, wall_dist(nx, ny))
ng = cg + js + penalty + VORONOI_C / (d*d)
```
（KNOWN_MAP_MODE 下生效；探索模式不启用避免影响现有探索——用 `if KNOWN_MAP_MODE:` 条件）

**Step 3:** astar_to 同样处理（注意 astar 代价当前无 penalty，只加 Voronoi 项）

**Step 4:** Commit `feat: 走中间 Voronoi 代价 (1/d²)`

---

## Task 5: 两阶段验收测试

**Objective:** 阶段1 建图保存 → 阶段2 换障碍 seed 用旧地图快速导航。

**Step 1:** 阶段1（探索建图，约 700s）：
```
python3 algo3_headless.py --seed 7 --render-every 0 --timeout 700 --save-map scans/map_seed7.npz
```
期望：arrived=True，保存 map_seed7.npz

**Step 2:** 阶段2（换 seed 99 用旧地图，目标 < 首次 30%≈210s）：
```
python3 algo3_headless.py --seed 99 --render-every 0 --timeout 400 --load-map scans/map_seed7.npz
```
期望：arrived=True，碰撞=0，不重新探索（cov 不涨），用时显著少于首次

**Step 3:** 记录成绩到 docs，Commit `test: 两阶段验收 (seed7建图→seed99寻路)`

---

## Task 6: 探索效率优化（信息增益 + 目标保持）

**Objective:** 探索阶段更快（seed7 目标 < 500s，当前 649s）

**Files:** Modify `test_scripts/algo3_headless.py`（pick_gate / find_gates）

**Step 1:** pick_gate score 加信息增益项：门附近 UNKNOWN 格数（可新扫面积）：
```python
# 在 score 循环里，对每个门算 gain = 门周围 5x5 内 UNKNOWN 数
gain = sum(1 for dy in range(-2,3) for dx in range(-2,3)
           if gget(gx+dx, gy+dy) == UNKNOWN)
score = 0.45*advance + 0.15*(1.0/d) + 0.15*(size/50.0) + 0.25*(gain/25.0)
```
（探索模式启用；KNOWN_MAP_MODE 下保持终点导向）

**Step 2:** 验证：跑 seed7 短测（120s）对比步数/推进，调权重

**Step 3:** 全量跑 seed7 验证 < 500s，Commit `perf: 探索信息增益评分`

---

## Task 7: 绑架随机位置（--random-start --target）

**Objective:** 随机位置（避开墙）→ 快速到起点/终点

**Files:** Modify `test_scripts/algo3_headless.py`（args + 初始化 + FINISH）

**Step 1:** 加参数：
```python
ap.add_argument("--random-start", type=int, default=0, help="1=从随机位置(避开墙)出发")
ap.add_argument("--target", type=str, default="finish", help="目标: start|finish")
```

**Step 2:** 随机采样函数：
```python
def random_road_pos(seed, min_dist_from_start=5.0):
    rng = random.Random(seed)
    for _ in range(5000):
        wx, wy = rng.uniform(0.5, 49.5), rng.uniform(0.5, 49.5)
        if sample_hf(wx, wy) != ROAD_PIX: continue          # 不在墙里
        if _obs_hits_wall(wx, wy, 0.7): continue            # 不在墙边
        if any(math.hypot(wx-ox, wy-oy) < OBS_CLEAR+0.3 for ox,oy in obs_world): continue  # 不在障碍里
        if math.hypot(wx-2.5, wy-2.5) < min_dist_from_start: continue  # 别太近起点（无意义）
        return wx, wy
    return 2.5, 2.5  # fallback
```

**Step 3:** 初始化：`if args.random_start: d.qpos[0], d.qpos[1] = random_road_pos(...)`；`FINISH = (2.5,2.5) if args.target=="start" else (2.5,47.5)`

**Step 4:** 验收：`--random-start 1 --load-map scans/map_seed7.npz --timeout 60` → 到达目标 < 60s，碰撞=0

**Step 5:** Commit `feat: 绑架随机位置快速到达起点/终点`

---

## Task 8: 标牌贴墙（响应主人"别堵路"）

**Objective:** 标牌从通道中心移到墙脚贴墙（低矮），不挡路，视觉识别仍工作

**Files:** Modify `test_scripts/landmarks.py`

**Step 1:** 标牌位置：通道下墙脚（y=墙线+0.15），低矮 0.3m 高 × 1m 宽：
```python
# landmark_positions: wz = 0.2 (中心高), h = 0.4 (总高 0.4m, 覆盖 0-0.4m)
# wy = ch*5.0 + 0.15 (贴墙脚, 面向通道内)
```

**Step 2:** 验证 EGL 渲染：独立脚本测相机 1m 高平视能否看到墙脚标牌（上轮发现墙脚可能被 hfield 挡，若不可见则标牌放墙顶斜挂或通道边 0.5m 处——变通）

**Step 3:** 跑 seed7 短测确认：标牌不挡导航（对比无标牌时推进速度）

**Step 4:** Commit `feat: 标牌贴墙放置不堵路`

---

## Task 9: B 阶段 — 运行中障碍变化（--obs-reseed）

**Objective:** 运行中途障碍物变化，机器狗边跑边适应（A 完成后）

**Files:** Modify `test_scripts/algo3_headless.py`

**Step 1:** 加参数 `--obs-reseed <step>`：到指定步数时 `obs_world = gen_obstacles(new_seed)`（重新生成），is_obstacle_world 用新障碍；live grid 射线清除自动处理旧障碍

**Step 2:** 验证：`--load-map scans/map_seed7.npz --obs-reseed 5000`，障碍变化后机器人检测、重规划、到达

**Step 3:** Commit `feat: B阶段 运行中障碍变化`

---

## Task 10: 文档 + 技能 + wiki 化

**Objective:** 记录成绩、更新 docs、mystory、skill references

**Step 1:** 更新 `docs/2026-08-05-dog50-maze-pitfalls.md`（新增"双层地图/走中间/绑架"章节 + 成绩表）

**Step 2:** mystory 写 observations（双层地图 Decision / 射线清除 Bug / Voronoi 走中间 Lesson）

**Step 3:** 更新 skill references（robot-navigation-algorithms）

**Step 4:** Commit `docs: 双层地图+走中间+绑架 成绩记录`

---

## 验收总表

| # | 目标 | 验收 |
|---|---|---|
| 1 | 探索效率 | seed7 到达 < 500s（当前 649s） |
| 2 | 已知地图寻路 | 换 seed 从起点→终点 < 首次 30%，碰撞=0 |
| 3 | 绑架恢复 | 随机位置→目标 < 60s，碰撞=0 |
| 4 | 走中间 | 路径 wall_dist 平均 > 1m（可视化验证） |
| 5 | 标牌贴墙 | 标牌不挡路，机器人正常推进 |
| 6 | B 阶段 | 运行中障碍变化，自适应到达 |
