# 无特权化改造 实现计划

> **For Hermes:** 逐任务实现，每步独立验证（主人：慢慢调试）。
>
> **Goal:** 狗只用雷达感知决策（前方 180° 多线），无真值特权；禁入区 ≤0.1m；ARM 友好。
>
> **Architecture:** 雷达模型（前方 180° 扇形 + 线数）→ blocked 纯感知 → save_map 感知版 → 禁入区缩小。每步跑测试确认不破坏。

---

## Task 1: 雷达前方 180° + 线数参数

**Objective:** scan() 只扫狗前方 180°（相对 yaw），支持多线（俯仰角），省一半计算。

**Files:**
- Modify: `test_scripts/algo3_headless.py`（参数段 ~L100、scan 函数 L394、调用点）

**Step 1: 加参数**（LIDAR_RAYS 附近）

```python
ap.add_argument("--lidar-fov", type=float, default=180.0, help="雷达水平视场角(度)，前方FOV=180（相对狗yaw），360=全向(作弊)")
ap.add_argument("--lidar-lines", type=int, default=1, help="雷达线数：1=单线水平, 3/10=多俯仰角(2D取最近)")
```

**Step 2: scan 改造**

```python
def scan(bx, by, yaw_ang):
    """前方 FOV 扇形扫描（相对狗 yaw）。多线：不同俯仰角，2D 导航取最近（任一线命中=检测到）。"""
    fov_rad = math.radians(args.lidar_fov)
    n_lines = max(1, args.lidar_lines)
    # 线数 → 俯仰角偏移（2D 简化：水平线 0°，多线加 ±小角度但 2D 判定相同）
    # 真实多线雷达不同俯仰角扫不同高度；2D 导航下等价"多条线冗余确认"
    for a in np.linspace(yaw_ang - fov_rad/2, yaw_ang + fov_rad/2, LIDAR_RAYS):
        cos_a, sin_a = math.cos(a), math.sin(a)
        prev_vx, prev_vy = int(bx/VOXEL), int(by/VOXEL)
        for step_i in range(1, LIDAR_STEPS+1):
            wx, wy = bx + cos_a*step_i*VOXEL, by + sin_a*step_i*VOXEL
            vx, vy = int(wx/VOXEL), int(wy/VOXEL)
            if is_obstacle_world(wx, wy):
                gset(vx, vy, WALL)
                if gget(prev_vx, prev_vy) == UNKNOWN:
                    gset(prev_vx, prev_vy, FREE)
                break
            if gget(vx, vy) != FREE:
                gset(vx, vy, FREE)
            prev_vx, prev_vy = vx, vy
```

**Step 3: 调用点传 yaw**

- 初始扫描：`scan(d.qpos[0], d.qpos[1], d.qpos[2])`
- 主循环：`scan(bx, by, d.qpos[2])`

**Step 4: 验证**

Run: `$PY test_scripts/algo3_headless.py --obs-straight 1 --obs-turn 1 --seed 42 --max-steps 3000 --timeout 100 --vision 0 --landmarks 0 --render-every 0`
Expected: 正常启动，无异常（感知只扫前方——可能更少 bounce 因为身后不再"看到"）

**Step 5: Commit**

```bash
git add test_scripts/algo3_headless.py
git commit -m "feat: 雷达前方180°扇形(相对yaw) + 线数参数 — 无特权改造第一步, 省50%扫描"
```

---

## Task 2: blocked 纯感知（去真值）

**Objective:** blocked() 去掉 is_obstacle_world（真值墙+真值障碍），只用感知。

**Files:**
- Modify: `test_scripts/algo3_headless.py`（blocked L454）

**Step 1: 改造**

```python
def blocked(wx, wy, inflation=0.0):
    # 越界保护
    if not (0.0 <= wx <= 50.0 and 0.0 <= wy <= 50.0):
        return True
    # 墙边禁入区（感知版）
    if in_keepout(int(wx/VOXEL), int(wy/VOXEL)):
        return True
    # 仅感知：grid/static WALL（雷达扫到的墙+障碍）。不用真值 is_obstacle_world（不许作弊）
    return gget_plan(int(wx/VOXEL), int(wy/VOXEL)) == WALL
```

注意：is_obstacle_world 仍用于碰撞统计（物理事实）+ scan 判定（雷达感知）——保留。

**Step 2: 验证**

Run: 同 Task 1
Expected: 跑通（可能早期碰撞，看碰撞统计——靠撞墙写回学习）

**Step 3: Commit**

```bash
git commit -m "feat: blocked纯感知 — 去掉is_obstacle_world真值判定, 只信雷达扫到的墙(无特权)"
```

---

## Task 3: save_map 感知版 + known-raw 默认关

**Objective:** save_map 只存感知墙；--known-raw 默认关（测试作弊标注）。

**Files:**
- Modify: `test_scripts/algo3_headless.py`（save_map L989、参数 L98）

**Step 1: save_map 改造**——不读 sample_hf 真值，只存 grid 扫到的 WALL 中"确实是结构墙"的（感知判定）：

```python
def save_map(path):
    """保存地图（感知版）：只存 grid 雷达扫到的墙。不用 sample_hf 真值（不许作弊）。
    结构墙 vs 障碍：障碍是变化的，scan 射线清除会重新发现——都存 WALL 由 load 后 scan 修正。"""
    if not grid: return
    xs = sorted(set(k[0] for k in grid)); ys = sorted(set(k[1] for k in grid))
    minx, maxx = xs[0], xs[-1]; miny, maxy = ys[0], ys[-1]
    arr = np.full((maxy-miny+1, maxx-minx+1), UNKNOWN, dtype=np.int8)
    for (vx, vy), val in grid.items():
        arr[vy-miny, vx-minx] = val   # 直接存感知值（WALL/FREE/UNKNOWN）
    np.savez(path, grid=arr, offset=(minx, miny), seed=FIXED_SEED)
    n_wall = int((arr == WALL).sum()); n_free = int((arr == FREE).sum())
    print(f"  [MAP] saved {n_free} FREE + {n_wall} WALL(感知) → {path}", flush=True)
```

**Step 2: known-raw 默认 0 + 标注**

```python
ap.add_argument("--known-raw", type=int, default=0, help="⚠️测试作弊: KNOWN_MAP 直接读 track_clean 真值(非感知, 仅调试用)")
```

**Step 3: 验证**

Run: 探索模式跑通 + save_map 生成感知地图（检查不含真值）
Expected: 正常

**Step 4: Commit**

```bash
git commit -m "feat: save_map感知版(只存雷达扫到的墙) + known-raw默认关(标注测试作弊)"
```

---

## Task 4: 禁入区缩小（通过性）

**Objective:** KEEP_CELLS 1格(10cm) → 半格 0.05m（KEEP 判定改连续距离）。

**Files:**
- Modify: `test_scripts/algo3_headless.py`（keepout L444）

**Step 1: 改造为连续距离判定**

```python
KEEP_M = 0.05   # 禁入距离 (m)（0.1 → 0.05，通过性更强）

def in_keepout(vx, vy):
    """距感知墙 < KEEP_M → True。连续距离判定（支持 <1格 的禁入）。"""
    # 检查以 (vx,vy) 为中心 KEEP_M 半径内的格
    r = int(math.ceil(KEEP_M / VOXEL))  # 0.05/0.1=1格（半格用格内偏移）
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            if gget_plan(vx+dx, vy+dy) == WALL:
                # 精确距离：格中心到 (vx,vy) 中心的距离
                if math.hypot(dx, dy) * VOXEL < KEEP_M + 1e-6:
                    return True
    return False
```

（0.05m < 1格 0.1m：只有同格/相邻格墙中心距 <0.1m 才算——实际约等于"墙所在格不可走+紧邻格部分"）

**Step 2: 验证**（通过性：窄缝能否穿过；贴墙违规统计用 0.1m 阈值重测）

Run: 全程测试 + 贴墙检查
Expected: 通过性增强（缝隙 +0.05m），不碰撞

**Step 3: Commit**

```bash
git commit -m "feat: 禁入区缩到0.05m(连续距离判定) — 通过性增强"
```

---

## Task 5: 全链路验证 + 视频

**Objective:** 确认无特权 + 前方180° + 禁入区0.05m 全程跑通。

**Files:**
- Run: 全程测试（探索+混合障碍）+ trail 贴墙检查 + 录视频

**Step 1: 全程验证**

Run: `$PY test_scripts/algo3_headless.py --obs-straight 1 --obs-turn 1 --seed 42 --max-steps 300000 --timeout 890 --trail-every 50 --save-name m3_nopriv.json`
Expected: arrived + collision 可接受（无特权早期可能小碰撞）+ bounce 记录

**Step 2: 录视频**（render-every 50，标注倍速）

**Step 3: 更新文档/mystory + Commit**

---

## 风险与应对

| 风险 | 应对 |
|---|---|
| 前方180° 后身后障碍不知道 → 转弯时撞身后墙 | 转弯口狗转向时 yaw 变，雷达跟随扫前方；测试观察 |
| blocked 纯感知早期碰撞（雷达盲区） | 碰撞统计暴露 + 撞墙写回学习；主人认可"慢慢调试" |
| 多线实现复杂 | 2D 导航下多线=冗余确认，先单线，多线后续 |
| 禁入区 0.05m 后视觉贴墙 | 主人明确要"更小通过性"，0.05m 测试看效果 |

## 验证清单

- [ ] 前方 180° 雷达正常导航
- [ ] blocked 无真值（grep 确认无 is_obstacle_world 调用）
- [ ] save_map 无 sample_hf
- [ ] known-raw 默认 0
- [ ] 禁入区 0.05m 全程通过性不降
- [ ] ARM 计算量评估（前方180°省50%）
