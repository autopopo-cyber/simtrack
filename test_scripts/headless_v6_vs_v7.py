#!/usr/bin/env python3
"""headless对比V6连通性 vs V7递归多边形"""
import sys, os, math, random
from collections import defaultdict
import numpy as np
from PIL import Image
import mujoco

MAP = os.path.expanduser("~/workspace/simtrack/confirmed/track_clean.png")
SCALE = 2.0; HF_RES = 2000; PIX_PER_M = 40; ROAD_PIX = 128
SAFE_R = 0.5
LIDAR_RANGE = 15.0; LIDAR_RAYS = 120
LIDAR_STEPS = int(LIDAR_RANGE / 0.1)
DECIDE_RADIUS = 10.0
GAP_YELLOW_M = 1.0
HIT_BACKOFF = 0.2
OBSERVE_TICK = 20
START_POS = (3.0, 3.0)
INIT_SCAN_FRAMES = 400
V6_VOXEL = 0.1
V6_CONNECT_R = 0.5
FIXED_SEED = 424242

random.seed(FIXED_SEED); np.random.seed(FIXED_SEED)

hf = np.array(Image.open(MAP))

def gen_centerline():
    pts = []; y0 = 2.5
    for seg in range(10):
        y = y0+seg*5.0; x0, x1 = (5.0,45.0) if seg%2==0 else (45.0,5.0)
        for j in range(10): pts.append((x0+(j/9.0)*(x1-x0), y))
    for mx, my in [(46.5,3.75),(47.5,5.0),(46.5,6.25)]:
        for gy in range(5): pts.append((mx, my+gy*10.0))
    for mx, my in [(3.5,8.75),(2.5,10.0),(3.5,11.25)]:
        for gy in range(4): pts.append((mx, my+gy*10.0))
    return pts

def gen_obstacles(seed):
    rng = random.Random(seed); cl = gen_centerline()
    obs_world = []; idx = 0
    for _ in range(12):
        cx, cy = cl[idx % len(cl)]
        ox = cx + rng.uniform(-1.5, 1.5)
        oy = cy + rng.uniform(-0.5, 0.5)
        obs_world.append((ox, oy)); idx += rng.randint(10, 20)
    return obs_world

obs_world = gen_obstacles(FIXED_SEED)

def sample_hf(wx, wy):
    mx, my = wx/SCALE, wy/SCALE
    px, py = int(mx*PIX_PER_M), HF_RES-1-int(my*PIX_PER_M)
    return int(hf[py,px]) if 0<=px<HF_RES and 0<=py<HF_RES else -1

def is_obstacle_world(wx, wy):
    v = sample_hf(wx, wy)
    if v == -1 or v < ROAD_PIX: return True
    for ox, oy in obs_world:
        if math.hypot(wx-ox, wy-oy) < 1.0 + SAFE_R: return True
    return False

wall_points = set()
def _snap(v): return round(v, 1)

def observe(bx, by):
    for a in np.linspace(0, 2*math.pi, LIDAR_RAYS):
        cos_a, sin_a = math.cos(a), math.sin(a)
        for step_i in range(1, LIDAR_STEPS + 1):
            wx = bx + cos_a * step_i * 0.1
            wy = by + sin_a * step_i * 0.1
            if is_obstacle_world(wx, wy):
                hx = bx + cos_a * (step_i * 0.1 - HIT_BACKOFF)
                hy = by + sin_a * (step_i * 0.1 - HIT_BACKOFF)
                wall_points.add((_snap(hx), _snap(hy)))
                break

OBS_XML = "".join(
    f'<body name="obs{j}" pos="{x:.1f} {y:.1f} 2.0">'
    f'<geom type="cylinder" size="1.0 2.0" rgba="0.9 0.2 0.2 0.9"/></body>'
    for j, (x, y) in enumerate(obs_world))

xml = f"""<mujoco>
  <compiler angle="radian"/><option timestep="0.005"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset><hfield name="track" size="50.0 50.0 4.0 2.0" file="{MAP}"/></asset>
  <worldbody>
    <light pos="50 50 80" dir="0 0 -1"/>
    {OBS_XML}
    <geom type="hfield" hfield="track" pos="50 50 0.0" rgba="0.25 0.30 0.35 1.0" friction="0 0 0"/>
    <body name="bot" pos="{START_POS[0]} {START_POS[1]} 0.5">
      <joint type="slide" axis="1 0 0" damping="0"/>
      <joint type="slide" axis="0 1 0" damping="0"/>
      <geom type="cylinder" size="0.5 0.5" rgba="1 0.3 0 1" friction="0 0 0"/>
    </body>
  </worldbody>
</mujoco>"""

m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
d.qpos[0] = START_POS[0]; d.qpos[1] = START_POS[1]
mujoco.mj_forward(m, d)

for _ in range(INIT_SCAN_FRAMES):
    bx, by = d.qpos[0], d.qpos[1]
    if _ % OBSERVE_TICK == 0:
        observe(bx, by)
    mujoco.mj_step(m, d)

bx, by = d.qpos[0], d.qpos[1]
nearby = [(wx, wy) for wx, wy in wall_points if abs(wx-bx) <= DECIDE_RADIUS and abs(wy-by) <= DECIDE_RADIUS]
print(f"wall_points={len(wall_points)} nearby={len(nearby)}")

# ═══════════════════════ V7 polygon ═══════════════════════
def polygon_boundary_v7(points, bx, by):
    n = len(points)
    if n < 3: return [], 0
    robx, roby = bx, by
    polar = [(math.atan2(wy-roby, wx-robx), math.hypot(wx-robx, wy-roby), wx, wy) for wx, wy in points]
    polar.sort()
    calls = [0]
    def _subdivide(ax, ay, bx_w, by_w, depth=0):
        calls[0] += 1
        d = math.hypot(bx_w-ax, by_w-ay)
        if d <= GAP_YELLOW_M or depth > 60: return [(ax, ay), (bx_w, by_w)]
        ang_a = math.atan2(ay-roby, ax-robx)
        ang_b = math.atan2(by_w-roby, bx_w-robx)
        if ang_b < ang_a: ang_b += 2*math.pi
        best_pt, best_dist = None, float('inf')
        for ang, dist, wx, wy in polar:
            a = ang if ang >= ang_a else ang + 2*math.pi
            if ang_a < a < ang_b and dist < best_dist:
                if not (abs(wx-ax)<0.05 and abs(wy-ay)<0.05) and not (abs(wx-bx_w)<0.05 and abs(wy-by_w)<0.05):
                    best_dist = dist; best_pt = (wx, wy)
        if not best_pt: return [(ax, ay), (bx_w, by_w)]
        cx, cy = best_pt
        left = _subdivide(ax, ay, cx, cy, depth+1)
        right = _subdivide(cx, cy, bx_w, by_w, depth+1)
        return left[:-1] + right
    _, _, wx0, wy0 = polar[0]
    _, _, wx1, wy1 = polar[n//2]
    _, _, wx2, wy2 = polar[-1]
    init = [(wx0, wy0), (wx1, wy1), (wx2, wy2)]
    poly = []
    for k in range(3):
        ax, ay = init[k]; bx_w, by_w = init[(k+1)%3]
        seg = _subdivide(ax, ay, bx_w, by_w)
        poly.extend(seg if k==0 else seg[1:])
    lines = []
    for k in range(len(poly)):
        fx, fy = poly[k]; tx, ty = poly[(k+1)%len(poly)]
        d = math.hypot(tx-fx, ty-fy)
        lines.append((fx, fy, tx, ty, 'yellow' if d > GAP_YELLOW_M else 'blue'))
    return lines, calls[0]

lines_v7, calls = polygon_boundary_v7(nearby, bx, by)
blues_v7 = sum(1 for _,_,_,_,c in lines_v7 if c=='blue')
yellows_v7 = sum(1 for _,_,_,_,c in lines_v7 if c=='yellow')
print(f"V7: poly顶点={blues_v7+yellows_v7} blues={blues_v7} yellows={yellows_v7} calls={calls}")

# ═══════════════════════ V6 connectivity ═══════════════════════
def _point_key(p):
    return (int(p[0]/V6_VOXEL), int(p[1]/V6_VOXEL))

def _build_adj(points):
    grid = defaultdict(list)
    for p in points: grid[_point_key(p)].append(p)
    adj = defaultdict(set)
    for p in points:
        kx, ky = _point_key(p)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for q in grid.get((kx+dx, ky+dy), []):
                    if q is not p and math.hypot(p[0]-q[0], p[1]-q[1]) < V6_CONNECT_R:
                        adj[p].add(q); adj[q].add(p)
    return adj

def _find_comps(points, adj):
    parent = {p: p for p in points}
    def find(p):
        while parent[p] != p: parent[p] = parent[parent[p]]; p = parent[p]
        return p
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    for p in points:
        for q in adj.get(p, set()): union(p, q)
    comps = defaultdict(list)
    for p in points: comps[find(p)].append(p)
    return list(comps.values())

def _trace_comp(comp, adj):
    lines = []; visited = set()
    endpoints = [p for p in comp if len(list(adj.get(p, set()) & set(comp))) <= 1]
    if not endpoints: endpoints = [comp[0], comp[0]]
    for start in endpoints:
        if start in visited: continue
        stack = [start]; parent = {start: None}; ordered = []
        while stack:
            cur = stack.pop(); ordered.append(cur)
            for nb in adj.get(cur, set()) & set(comp):
                if nb not in parent: parent[nb] = cur; stack.append(nb)
        seen_edges = set()
        for cur in ordered:
            for nb in adj.get(cur, set()) & set(comp):
                e = (min(cur, nb), max(cur, nb))
                if e not in seen_edges:
                    seen_edges.add(e)
                    lines.append((cur[0], cur[1], nb[0], nb[1], 'blue'))
        visited.update(ordered)
    return lines, endpoints

def _bridge(all_endpoints):
    candidates = []
    for i, eps_i in enumerate(all_endpoints):
        for ep_i in eps_i:
            for j, eps_j in enumerate(all_endpoints):
                if i >= j: continue
                for ep_j in eps_j:
                    candidates.append((math.hypot(ep_i[0]-ep_j[0], ep_i[1]-ep_j[1]), ep_i, ep_j, i, j))
    candidates.sort()
    used = set(); result = []
    for d, ep_a, ep_b, ci, cj in candidates:
        if ep_a in used and ep_b in used: continue
        result.append((ep_a[0], ep_a[1], ep_b[0], ep_b[1], 'yellow' if d > GAP_YELLOW_M else 'blue'))
        used.add(ep_a); used.add(ep_b)
    return result

adj = _build_adj(nearby)
comps = _find_comps(nearby, adj)
all_lines = []; all_eps = []
for comp in comps:
    cl, eps = _trace_comp(comp, adj)
    all_lines.extend(cl); all_eps.append(eps)
all_lines.extend(_bridge(all_eps))

blues_v6 = sum(1 for _,_,_,_,c in all_lines if c=='blue')
yellows_v6 = sum(1 for _,_,_,_,c in all_lines if c=='yellow')
print(f"V6: comps={len(comps)} lines={blues_v6+yellows_v6} blues={blues_v6} yellows={yellows_v6}")
