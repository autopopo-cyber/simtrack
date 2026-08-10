#!/usr/bin/env python3
"""离线复现楔住点：加载 scan_dict.npz 感知图，从 (49.3,6.1) 跑 find_gates→fine_path→
模拟 pure pursuit + blocked/d_clear，看狗到底被什么挡住。"""
import math, sys, os, heapq
import numpy as np

d = np.load("scans/scan_dict.npz", allow_pickle=True)
arr = d["grid"]; ox, oy = [int(v) for v in d["offset"]]
UNKNOWN, FREE, WALL = 0, 1, 2
VOXEL = 0.1
ROBOT_R = 2
KEEP_M = 0.05
WALL_BUFFER_CELLS = 20; WALL_PENALTY = 3; UNKNOWN_PENALTY = 8
JUMP_1M = 20; JUMP_03 = 6; JUMP_NEAR = 1
MAX_GATES = 200; MAX_GATE_DIST = 500; ASTAR_MAX_EXPAND = 250000
LOOKAHEAD = 4.0; STOP_MARGIN = 0.4
bad_gates = set()
grid = {}
H, W = arr.shape
for vy in range(H):
    for vx in range(W):
        if arr[vy, vx] != UNKNOWN:
            grid[(vx+ox, vy+oy)] = int(arr[vy, vx])
def gget(vx, vy): return grid.get((vx, vy), UNKNOWN)
def gget_plan(vx, vy): return grid.get((vx, vy), UNKNOWN)

_wd = {}
def wall_dist(vx, vy):
    key = (vx, vy)
    if key in _wd: return _wd[key]
    best = 999
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if gget_plan(vx+dx, vy+dy) == WALL:
                dd = abs(dx)+abs(dy)
                if dd < best: best = dd
    if best < 999:
        _wd[key] = best
        return best
    _wd[key] = JUMP_1M + 1
    return JUMP_1M + 1

def in_keepout(vx, vy):
    r = int(math.ceil((KEEP_M + 0.5*VOXEL) / VOXEL))
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            if gget_plan(vx+dx, vy+dy) == WALL:
                dist_surf = (math.hypot(dx, dy) - 0.5) * VOXEL
                if dist_surf < KEEP_M + 1e-6:
                    return True
    return False

def blocked(wx, wy):
    if not (0.0 <= wx <= 50.0 and 0.0 <= wy <= 50.0): return True
    if in_keepout(int(wx/VOXEL), int(wy/VOXEL)): return True
    return gget_plan(int(wx/VOXEL), int(wy/VOXEL)) == WALL

def forward_clear(bx, by, yaw_ang):
    for k in range(1, int(LOOKAHEAD / 0.05) + 1):
        px = bx + math.cos(yaw_ang) * 0.05 * k
        py = by + math.sin(yaw_ang) * 0.05 * k
        if blocked(px, py):
            return 0.05 * k
    return LOOKAHEAD

def jump_steps(vx, vy, dx, dy):
    wd = wall_dist(vx, vy)
    if wd >= JUMP_1M:   max_jump = JUMP_1M
    elif wd >= JUMP_03: max_jump = JUMP_03
    else:               max_jump = JUMP_NEAR
    for step in range(1, max_jump + 1):
        nx, ny = vx + dx*step, vy + dy*step
        if gget_plan(nx, ny) == WALL:
            return step - 1
    return max_jump

def _nearest_walkable(vx, vy, max_r=8, min_dist=0):
    if gget_plan(vx, vy) != WALL and (min_dist == 0 or wall_dist(vx, vy) >= min_dist):
        return vx, vy
    seen = {(vx, vy)}
    q = [(vx, vy, 0)]
    while q:
        cx, cy, dist = q.pop(0)
        if dist >= max_r: continue
        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
            nx, ny = cx+dx, cy+dy
            if (nx,ny) in seen: continue
            seen.add((nx,ny))
            if gget_plan(nx, ny) != WALL and (min_dist == 0 or wall_dist(nx, ny) >= min_dist):
                return nx, ny
            q.append((nx, ny, dist+1))
    return None

def find_gates(fvx, fvy):
    if gget_plan(fvx, fvy) == WALL: return [], {}
    start = _nearest_walkable(fvx, fvy)
    if start is None: return [], {}
    fvx, fvy = start
    open_set = [(0, fvx, fvy)]
    came_from = {}; g_score = {(fvx, fvy): 0}
    visited = set(); gates = []
    while open_set and len(came_from) < ASTAR_MAX_EXPAND and len(gates) < MAX_GATES:
        _, cx, cy = heapq.heappop(open_set)
        if (cx,cy) in visited: continue
        visited.add((cx,cy))
        cg = g_score.get((cx,cy), 9999)
        if gates and cg > MAX_GATE_DIST: break
        if cg > MAX_GATE_DIST: continue
        if gget_plan(cx, cy) == FREE:
            has_unk = any(gget_plan(cx+dx, cy+dy) == UNKNOWN
                          for dy in (-1,0,1) for dx in (-1,0,1))
            if has_unk and (cx, cy) != (fvx, fvy) and wall_dist(cx, cy) > ROBOT_R - 1 and (cx, cy) not in bad_gates:
                gates.append((cg, cx, cy))
        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
            js = jump_steps(cx, cy, dx, dy)
            if js < 1: continue
            nx, ny = cx + dx*js, cy + dy*js
            wd = wall_dist(nx, ny)
            penalty = max(0, WALL_BUFFER_CELLS - wd) * WALL_PENALTY
            if gget_plan(nx, ny) == UNKNOWN:
                penalty += UNKNOWN_PENALTY
            ng = cg + js + penalty
            if (nx,ny) not in g_score or ng < g_score[(nx,ny)]:
                g_score[(nx,ny)] = ng
                came_from[(nx,ny)] = (cx,cy)
                heapq.heappush(open_set, (ng, nx, ny))
    return gates, came_from

def fine_path(sx, sy, gx, gy, came_from):
    path = []; cur = (gx, gy)
    while cur != (sx, sy):
        path.append(cur)
        if cur not in came_from: break
        cur = came_from[cur]
    path.reverse()
    return [((px+0.5)*VOXEL, (py+0.5)*VOXEL) for px, py in path]

# 从楔住点出发
bx, by = 49.3, 6.1
vx, vy = int(bx/VOXEL), int(by/VOXEL)
print(f"狗格 ({vx},{vy}) gget_plan={gget_plan(vx,vy)} wall_dist={wall_dist(vx,vy)} keepout={in_keepout(vx,vy)}")
gates, came_from = find_gates(vx, vy)
print(f"gates={len(gates)} (未聚类)")
# 找 (451,71) 附近的门
near = [g for g in gates if abs(g[1]-451) < 10 and abs(g[2]-71) < 10]
print(f"门(45.1,7.1)附近原始门格: {len(near)} 个", near[:5])
if gates:
    # 假设目标是最近的门
    gates.sort()
    cg, gx, gy = gates[0]
    print(f"最近门 ({(gx+0.5)*VOXEL:.1f},{(gy+0.5)*VOXEL:.1f}) cg={cg}")
    path = fine_path(vx, vy, gx, gy, came_from)
    print(f"path={len(path)} 点:")
    for p in path[:12]:
        print(f"   ({p[0]:.2f},{p[1]:.2f}) blocked={blocked(p[0],p[1])}")
    # 模拟 pure pursuit 第一步：lookahead 3m
    tx, ty = path[0]
    for (lx, ly) in path:
        if math.hypot(lx-bx, ly-by) >= 3.0:
            tx, ty = lx, ly
            break
    tgt_yaw = math.atan2(ty-by, tx-bx)
    print(f"lookahead 目标 ({tx:.2f},{ty:.2f}) yaw={math.degrees(tgt_yaw):.0f}°")
    # d_clear 全向扫描
    print("d_clear 剖面 (每15°):")
    for deg in range(0, 360, 15):
        a = math.radians(deg)
        print(f"  {deg:3d}°: {forward_clear(bx, by, a):.2f}m", end="")
        if deg % 45 == 0: print(f"   <- {'目标' if abs(deg-math.degrees(tgt_yaw))<8 else ''}")
        else: print()
