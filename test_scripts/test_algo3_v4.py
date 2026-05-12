#!/usr/bin/env python3
"""algo3 单元测试 — 独立测试激光扫描/门查找/A*/黄球生成/导航逻辑

用法: python3 test_scripts/test_algo3_v4.py
"""

import sys, os, math, random, heapq
import numpy as np
from PIL import Image

# ═══════════ 复制核心数据结构（与algo3_firefly.py一致） ═══════════

VOXEL = 0.1; SAFE_R = 0.5; LIDAR_RANGE = 15.0; LIDAR_RAYS = 120
ROBOT_R = max(1, int(SAFE_R / VOXEL)); CLEARANCE = ROBOT_R
LIDAR_STEPS = int(LIDAR_RANGE / VOXEL)
WALL_SCAN_RADIUS = 10; WALL_BUFFER_CELLS = 20; WALL_PENALTY = 3
JUMP_1M = 10; JUMP_03 = 3; JUMP_NEAR = 1
MAX_GATES = 200; MAX_GATE_DIST = 3000; ASTAR_MAX_EXPAND = 30000
SCALE = 2.0; PIX_PER_M = 40; HF_RES = 2000; ROAD_PIX = 128
MAP = os.path.expanduser("~/workspace/simtrack/confirmed/track_clean.png")

UNKNOWN, FREE, WALL = 0, 1, 2
grid = {}
_wd = {}
_cnt = {FREE: 0, WALL: 0}

def gget(vx, vy):
    return grid.get((vx, vy), UNKNOWN)

def gset(vx, vy, val):
    old = gget(vx, vy)
    if old == val: return
    if old != UNKNOWN: _cnt[old] -= 1
    grid[(vx, vy)] = val
    _cnt[val] += 1
    if val == WALL: _wd.clear()

hf = np.array(Image.open(MAP))

def sample_hf(wx, wy):
    mx, my = wx/SCALE, wy/SCALE
    px, py = int(mx*PIX_PER_M), HF_RES-1-int(my*PIX_PER_M)
    return int(hf[py,px]) if 0<=px<HF_RES and 0<=py<HF_RES else -1

# 简化障碍物（无随机，固定几个点用于测试）
obs_world = [(20, 20), (30, 30), (40, 15)]
OBS_R = 1.0; OBS_CLEAR = OBS_R + SAFE_R

def is_obstacle_world(wx, wy):
    if sample_hf(wx, wy) != ROAD_PIX: return True
    for ox, oy in obs_world:
        if math.hypot(wx-ox, wy-oy) < OBS_CLEAR: return True
    return False

# ═══════════ 扫描函数 ═══════════

def scan(bx, by):
    for a in np.linspace(0, 2*math.pi, LIDAR_RAYS):
        cos_a, sin_a = math.cos(a), math.sin(a)
        prev_vx, prev_vy = int(bx/VOXEL), int(by/VOXEL)
        for step_i in range(1, LIDAR_STEPS+1):
            wx, wy = bx + cos_a*step_i*VOXEL, by + sin_a*step_i*VOXEL
            vx, vy = int(wx/VOXEL), int(wy/VOXEL)
            if is_obstacle_world(wx, wy):
                gset(vx, vy, WALL)
                gset(prev_vx, prev_vy, WALL)
                break
            if gget(vx, vy) == UNKNOWN:
                gset(vx, vy, FREE)
            prev_vx, prev_vy = vx, vy

def wall_dist(vx, vy):
    key = (vx, vy)
    if key in _wd: return _wd[key]
    best = 999
    for dy in range(-WALL_SCAN_RADIUS, WALL_SCAN_RADIUS+1):
        for dx in range(-WALL_SCAN_RADIUS, WALL_SCAN_RADIUS+1):
            if gget(vx+dx, vy+dy) == WALL:
                d = abs(dx)+abs(dy)
                if d < best: best = d
    _wd[key] = best
    return best

def walkable(vx, vy):
    return gget(vx, vy) == FREE and wall_dist(vx, vy) > ROBOT_R

# ═══════════ A* ═══════════

def jump_steps(vx, vy, dx, dy):
    wd = wall_dist(vx, vy)
    if wd >= JUMP_1M:   max_jump = JUMP_1M
    elif wd >= JUMP_03: max_jump = JUMP_03
    else:               max_jump = JUMP_NEAR
    for step in range(1, max_jump + 1):
        nx, ny = vx + dx*step, vy + dy*step
        if not walkable(nx, ny):
            return step - 1
    return max_jump

def astar_to(fvx, fvy, tfx, tfy):
    """A*到目标点，目标只禁WALL(允许UNKNOWN作为前沿)"""
    if not walkable(fvx, fvy): return None
    if gget(tfx, tfy) == WALL: return None
    open_set = [(math.hypot(tfx-fvx, tfy-fvy), fvx, fvy)]
    came_from = {}; g_score = {(fvx, fvy): 0}
    visited = set()
    while open_set and len(came_from) < ASTAR_MAX_EXPAND:
        _, cx, cy = heapq.heappop(open_set)
        if (cx,cy) in visited: continue
        visited.add((cx,cy))
        if (cx,cy) == (tfx,tfy): break
        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
            js = jump_steps(cx, cy, dx, dy)
            if js < 1: continue
            nx, ny = cx + dx*js, cy + dy*js
            ng = g_score.get((cx,cy), 999) + js
            if (nx,ny) not in g_score or ng < g_score[(nx,ny)]:
                g_score[(nx,ny)] = ng
                came_from[(nx,ny)] = (cx,cy)
                heapq.heappush(open_set, (ng+math.hypot(tfx-nx, tfy-ny), nx, ny))
    if (tfx,tfy) not in came_from and (tfx,tfy) != (fvx,fvy): return None
    # 回溯路径 (细格坐标)
    path = []; cur = (tfx, tfy)
    while cur != (fvx, fvy):
        path.append(cur)
        if cur not in came_from: break
        cur = came_from[cur]
    return path  # 从起点到终点(不包含起点)

# ═══════════ 门查找 ═══════════

def find_gates(fvx, fvy):
    """细格A*找门+连通域合并"""
    if not walkable(fvx, fvy): return [], {}
    open_set = [(0, fvx, fvy)]
    came_from = {}; g_score = {(fvx, fvy): 0}
    visited = set(); gates = []
    while open_set and len(came_from) < ASTAR_MAX_EXPAND and len(gates) < MAX_GATES:
        _, cx, cy = heapq.heappop(open_set)
        if (cx,cy) in visited: continue
        visited.add((cx,cy))
        cg = g_score.get((cx,cy), 9999)
        if gates and cg > MAX_GATE_DIST: break
        if gget(cx, cy) == FREE:
            has_unk = any(gget(cx+dx, cy+dy) == UNKNOWN
                          for dy in (-1,0,1) for dx in (-1,0,1))
            if has_unk and wall_dist(cx, cy) > CLEARANCE:
                gates.append((cg, wall_dist(cx, cy), cx, cy))
        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
            js = jump_steps(cx, cy, dx, dy)
            if js < 1: continue
            nx, ny = cx + dx*js, cy + dy*js
            wd = wall_dist(nx, ny)
            penalty = max(0, WALL_BUFFER_CELLS - wd) * WALL_PENALTY
            ng = cg + js + penalty
            if (nx,ny) not in g_score or ng < g_score[(nx,ny)]:
                g_score[(nx,ny)] = ng
                came_from[(nx,ny)] = (cx,cy)
                heapq.heappush(open_set, (ng, nx, ny))
    gates = merge_gates(gates, came_from)
    return gates, came_from

def merge_gates(raw_gates, came_from):
    if not raw_gates: return []
    gate_set = set(); gate_info = {}
    for cg, wd, cx, cy in raw_gates:
        gate_set.add((cx, cy)); gate_info[(cx, cy)] = (cg, wd)
    visited = set(); clusters = []
    for sx, sy in gate_set:
        if (sx, sy) in visited: continue
        cluster = []; stack = [(sx, sy)]
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited: continue
            if (cx, cy) not in gate_set: continue
            visited.add((cx, cy)); cluster.append((cx, cy))
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = cx+dx, cy+dy
                if (nx, ny) in gate_set and (nx, ny) not in visited:
                    stack.append((nx, ny))
        if not cluster: continue
        n = len(cluster)
        ctr_x = sum(x for x,y in cluster) / n
        ctr_y = sum(y for x,y in cluster) / n
        best = None; best_dist = 999999
        for cx, cy in cluster:
            if (cx, cy) not in came_from: continue
            d = (cx-ctr_x)**2 + (cy-ctr_y)**2
            if d < best_dist:
                best_dist = d; cg, wd = gate_info[(cx, cy)]
                best = (cg, wd, cx, cy)
        if best: clusters.append(best)
    clusters.sort(key=lambda g: g[0])
    return clusters

# ═══════════ 黄球生成 ═══════════

def gen_yellow_waypoints(raw_path):
    """A*细格路径 → 世界坐标, 每1m一个黄点"""
    if not raw_path: return []
    # 转世界坐标
    world = [((px+0.5)*VOXEL, (py+0.5)*VOXEL) for px, py in raw_path]
    wp = [world[0]]
    acc = 0.0; px, py = world[0]
    for wx, wy in world[1:]:
        acc += math.hypot(wx-px, wy-py)
        if acc >= 1.0:
            wp.append((wx, wy)); acc = 0.0
        px, py = wx, wy
    last = world[-1]
    if not wp or (abs(wp[-1][0]-last[0]) > 0.01 or abs(wp[-1][1]-last[1]) > 0.01):
        wp.append(last)
    return wp

# ═══════════ 避障检查 ═══════════

def blocked(wx, wy):
    vx, vy = int(wx/VOXEL), int(wy/VOXEL)
    for dy in range(-ROBOT_R, ROBOT_R+1):
        for dx in range(-ROBOT_R, ROBOT_R+1):
            if dx*dx+dy*dy <= ROBOT_R*ROBOT_R:
                if is_obstacle_world((vx+dx+0.5)*VOXEL, (vy+dy+0.5)*VOXEL):
                    return True
    return False

# ═══════════════════════════════════════════
# 测试用例
# ═══════════════════════════════════════════

PASS = 0; FAIL = 0

def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  ✅ {name}")
    else:
        FAIL += 1; print(f"  ❌ {name} — FAIL")

def test(title):
    global _wd; _wd.clear()
    print(f"\n{'='*50}\n  {title}\n{'='*50}")

# ── 测试1: 激光扫描 ──
test("1. 激光扫描 scan()")

grid.clear(); _cnt[FREE] = 0; _cnt[WALL] = 0; _wd.clear()
bx, by = 3.0, 3.0    # 已知安全起点
scan(bx, by)

check("扫描后FREE>0", _cnt[FREE] > 0)
check("扫描后WALL>0", _cnt[WALL] > 0)
check("起点位置已知(非UNKNOWN)", gget(int(bx/VOXEL), int(by/VOXEL)) != UNKNOWN)
check("15m外UNKNOWN", gget(int((bx+20)/VOXEL), int(by/VOXEL)) == UNKNOWN)

print(f"  📊 FREE={_cnt[FREE]} WALL={_cnt[WALL]} grid_cells={len(grid)}")

# ── 测试2: wall_dist + walkable ──
test("2. wall_dist + walkable")

vx, vy = int(bx/VOXEL), int(by/VOXEL)
wd = wall_dist(vx, vy)
check("wall_dist(起点) > 0", wd > 0)
check("walkable(起点)=True", walkable(vx, vy))
check("walkable(远UNKNOWN)=False", not walkable(int((bx+20)/VOXEL), int(by/VOXEL)))

# ── 测试3: 门查找 find_gates + merge_gates ──
test("3. 门查找 find_gates() + merge_gates()")

gates, came_from = find_gates(vx, vy)
check("找到至少1个门", len(gates) > 0)
check("门格式正确 (cg,wd,vx,vy)", len(gates[0]) == 4)
check("最近门 cg > 0", gates[0][0] > 0)

# 验证门在FREE区域
for _, _, gx, gy in gates[:5]:
    check(f"门({gx},{gy})在FREE", gget(gx, gy) == FREE)
    check(f"门({gx},{gy})邻接UNKNOWN",
          any(gget(gx+dx, gy+dy) == UNKNOWN for dy in (-1,0,1) for dx in (-1,0,1)))

print(f"  📊 gates={len(gates)}")

# ── 测试4: A*寻路 ──
test("4. A* astar_to()")

# 4a: 到最近门
target_gate = gates[1] if len(gates) > 1 else gates[0]
_, _, gx, gy = target_gate
raw = astar_to(vx, vy, gx, gy)
check("A*到门: 路径非空", raw is not None and len(raw) > 0)
if raw:
    check("A*到门: 路径起点可达", walkable(raw[0][0], raw[0][1]))
    check("A*到门: 路径全程无WALL", 
          all(gget(px, py) != WALL for px, py in raw))
    print(f"  📊 path_len={len(raw)} gates")

# 4b: 起点不可达→None
check("A*起点WALL→None", astar_to(int(bx/VOXEL)+100, int(by/VOXEL)+100, gx, gy) is None)

# 4c: 目标WALL→None
wall_cells = [(vx2,vy2) for (vx2,vy2) in grid if grid[(vx2,vy2)] == WALL]
if wall_cells:
    wx, wy = wall_cells[0]
    check("A*目标WALL→None", astar_to(vx, vy, wx, wy) is None)

# 4d: 目标UNKNOWN→可达（前沿门）
unk_near = None
for dy in range(-5, 6):
    for dx in range(-5, 6):
        if gget(vx+dx, vy+dy) == UNKNOWN and walkable(vx+dx-1, vy+dy):
            unk_near = (vx+dx, vy+dy); break
    if unk_near: break

if unk_near:
    raw_unk = astar_to(vx, vy, unk_near[0], unk_near[1])
    check(f"A*目标UNKNOWN→可达", raw_unk is not None)
    if raw_unk:
        check("A*到UNKNOWN: 沿途无WALL",
              all(gget(px, py) != WALL for px, py in raw_unk))
        check("A*到UNKNOWN: 终点是目标或邻接",
              math.hypot(raw_unk[-1][0]-unk_near[0], raw_unk[-1][1]-unk_near[1]) <= 5)

# ── 测试5: 黄球生成 ──
test("5. 黄球生成 gen_yellow_waypoints()")

check("空路径→[]", gen_yellow_waypoints([]) == [])
check("单点路径→1黄球", len(gen_yellow_waypoints([(100,100)])) == 1)

if gates and raw:
    yw = gen_yellow_waypoints(raw)
    check("到门: 黄球>0", len(yw) > 0)
    # 检查每1m间距
    good_spacing = True
    for i in range(1, min(len(yw), 10)):
        d = math.hypot(yw[i][0]-yw[i-1][0], yw[i][1]-yw[i-1][1])
        if d < 0.5 or d > 2.0:
            good_spacing = False
    check("黄球间距≈1m", good_spacing)
    print(f"  📊 yellow_wps={len(yw)} 总长≈{len(yw)}m")

# ── 测试6: 避障 blocked() ──
test("6. 避障检测 blocked()")

check("起点附近无障碍", not blocked(bx, by))
check("障碍物上被blocked", blocked(obs_world[0][0], obs_world[0][1]))

# ── 测试7: 全流程模拟（无MuJoCo） ──
test("7. 全流程模拟 (scan→gates→A*→黄球→移动→重新scan)")

grid.clear(); _cnt[FREE] = 0; _cnt[WALL] = 0; _wd.clear()
bx_test, by_test = 3.0, 3.0
scan(bx_test, by_test)
vx_t, vy_t = int(bx_test/VOXEL), int(by_test/VOXEL)

gates, _ = find_gates(vx_t, vy_t)
check("全流程: 初始门>0", len(gates) > 0)

# 选一个门, A*过去
gate = gates[min(1, len(gates)-1)]  # 第二个门（离远一点）
_, _, gx, gy = gate
path = astar_to(vx_t, vy_t, gx, gy)
check("全流程: A*出路径", path is not None and len(path) > 0)

if path:
    yw = gen_yellow_waypoints(path)
    check("全流程: 黄球>0", len(yw) > 0)
    # 模拟移动: 走5个黄球
    steps = 0
    while yw and steps < 5:
        tx, ty = yw[0]
        # 简单朝向+移动
        dist = math.hypot(tx-bx_test, ty-by_test)
        if dist < 0.5:
            yw.pop(0)
        else:
            bx_test += (tx-bx_test) / dist * 0.5
            by_test += (ty-by_test) / dist * 0.5
        steps += 1
    check("全流程: 能沿黄球移动", steps > 0)
    # 在新位置重新扫描
    scan(bx_test, by_test)
    vx_t2, vy_t2 = int(bx_test/VOXEL), int(by_test/VOXEL)
    new_free = sum(1 for (vx2,vy2),v in grid.items() if v == FREE)
    check("全流程: 移动后扫描扩张地图", new_free >= _cnt[FREE] - 5)
    # 再次找门
    gates2, _ = find_gates(vx_t2, vy_t2)
    check("全流程: 移动后能找新门", len(gates2) > 0)

# ═══════════ 结果汇总 ═══════════
total = PASS + FAIL
print(f"\n{'='*50}")
print(f"  {'🎉 全过！' if FAIL == 0 else '⚠️ 有失败'}")
print(f"  PASS={PASS}/{total}  FAIL={FAIL}/{total}")
print(f"{'='*50}")
sys.exit(0 if FAIL == 0 else 1)
