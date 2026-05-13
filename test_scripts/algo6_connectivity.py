#!/usr/bin/env python3
"""V6: 体素连通性 → 聚类 → 端点桥接 → 闭环
无种子三角形，全覆盖所有墙体点。
"""

import sys, os, math, random
from collections import defaultdict
import numpy as np
from PIL import Image
import mujoco, mujoco.viewer

# ═══════════════════════════════════════════
# 参数（同V7）
# ═══════════════════════════════════════════

MAP = os.path.expanduser("~/workspace/simtrack/confirmed/track_clean.png")
SCALE = 2.0; HF_RES = 2000; PIX_PER_M = 40; ROAD_PIX = 128
SAFE_R = 0.5
LIDAR_RANGE = 15.0; LIDAR_RAYS = 120
LIDAR_STEPS = int(LIDAR_RANGE / 0.1)
DECIDE_RADIUS = 10.0
GAP_YELLOW_M = 1.0
HIT_BACKOFF = 0.2
OBSERVE_TICK = 20; RENDER_SKIP = 20
FIXED_SEED = random.randint(0, 999999)
START_POS = (3.0, 3.0)
INIT_SCAN_FRAMES = 400

V6_VOXEL = 0.1       # 体素精度
V6_CONNECT_R = 0.2   # 相邻阈值

# ═══════════════════════════════════════════
# 地图 + 障碍物（同V7）
# ═══════════════════════════════════════════

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

# ═══════════════════════════════════════════
# Observe（同V7）
# ═══════════════════════════════════════════

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

# ═══════════════════════════════════════════
# V6 Decide: 体素连通性
# ═══════════════════════════════════════════

def _point_key(p):
    """把点映射到体素网格key"""
    return (int(p[0] / V6_VOXEL), int(p[1] / V6_VOXEL))

def _build_adjacency(points):
    """为点集建邻接表。两点距离<V6_CONNECT_R则相邻。"""
    # 用体素grid加速
    grid = defaultdict(list)
    for p in points:
        grid[_point_key(p)].append(p)

    adj = defaultdict(set)
    for p in points:
        kx, ky = _point_key(p)
        # 检查3x3邻域体素
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for q in grid.get((kx + dx, ky + dy), []):
                    if q is not p and math.hypot(p[0] - q[0], p[1] - q[1]) < V6_CONNECT_R:
                        adj[p].add(q)
                        adj[q].add(p)
    return adj

def _find_components(points, adj):
    """Union-Find找连通分量"""
    parent = {p: p for p in points}

    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for p in points:
        for q in adj.get(p, set()):
            union(p, q)

    comps = defaultdict(list)
    for p in points:
        comps[find(p)].append(p)
    return list(comps.values())

def _trace_component(comp, adj):
    """追踪连通分量：从端点出发沿链走，画蓝线。"""
    lines = []
    comp_set = set(comp)
    # 计算每个点的度
    degree = {p: len(adj.get(p, set()) & comp_set) for p in comp}
    endpoints = [p for p, d in degree.items() if d <= 1]

    if not endpoints:
        # 闭环：取任意点，走一圈
        endpoints = [comp[0]]

    visited = set()

    for start in endpoints:
        if start in visited:
            continue
        # 沿链走：每次选一个未访问邻居
        cur = start
        visited.add(cur)
        chain = [cur]
        while True:
            neighbors = [n for n in (adj.get(cur, set()) & comp_set) if n not in visited]
            if not neighbors:
                break
            # 优先选最接近当前方向的邻居
            if len(chain) >= 2:
                dx, dy = chain[-1][0] - chain[-2][0], chain[-1][1] - chain[-2][1]
                nxt = min(neighbors, key=lambda n: abs(math.atan2(
                    n[1]-cur[1], n[0]-cur[0]) - math.atan2(dy, dx)))
            else:
                nxt = neighbors[0]
            cur = nxt
            visited.add(cur)
            chain.append(cur)

        # 沿chain顺序画线
        for i in range(len(chain) - 1):
            lines.append((chain[i][0], chain[i][1], chain[i+1][0], chain[i+1][1], 'blue'))

    return lines, endpoints

def _bridge_endpoints(all_endpoints, adj, used_eps):
    """桥接所有组件端点。
    对每个未用端点，找最近的另一个组件的未用端点。
    距离>1m→黄线，≤1m→蓝线。
    返回新增线 + 更新used_eps。
    """
    # 为每个端点标注所属组件
    ep_to_comp = {}
    for i, eps in enumerate(all_endpoints):
        for ep in eps:
            ep_to_comp[ep] = i

    # 收集所有端点对，按距离排序
    candidates = []
    for i, eps_i in enumerate(all_endpoints):
        for ep_i in eps_i:
            for j, eps_j in enumerate(all_endpoints):
                if i >= j:
                    continue
                for ep_j in eps_j:
                    d = math.hypot(ep_i[0] - ep_j[0], ep_j[1] - ep_j[1])
                    candidates.append((d, ep_i, ep_j, i, j))

    candidates.sort()
    colored_lines = []

    for d, ep_a, ep_b, ci, cj in candidates:
        if ep_a in used_eps and ep_b in used_eps:
            continue
        color = 'yellow' if d > GAP_YELLOW_M else 'blue'
        colored_lines.append((ep_a[0], ep_a[1], ep_b[0], ep_b[1], color))
        used_eps.add(ep_a)
        used_eps.add(ep_b)

    return colored_lines

def decide_connectivity(bx, by):
    """V6连通性decide"""
    # 取附近点
    nearby = []
    for wx, wy in wall_points:
        if abs(wx - bx) <= DECIDE_RADIUS and abs(wy - by) <= DECIDE_RADIUS:
            nearby.append((wx, wy))

    if len(nearby) < 3:
        return []

    # 建邻接
    adj = _build_adjacency(nearby)

    # 连通分量
    comps = _find_components(nearby, adj)
    print(f"  [CONN] nearby={len(nearby)} comps={len(comps)}", flush=True)

    # 追踪每个分量 + 收集端点
    all_lines = []
    all_endpoints = []
    for comp in comps:
        comp_lines, eps = _trace_component(comp, adj)
        all_lines.extend(comp_lines)
        all_endpoints.append(eps)

    # 桥接
    used_eps = set()
    bridge_lines = _bridge_endpoints(all_endpoints, adj, used_eps)
    all_lines.extend(bridge_lines)

    blues = sum(1 for _,_,_,_,c in all_lines if c == 'blue')
    yellows = sum(1 for _,_,_,_,c in all_lines if c == 'yellow')
    print(f"  [V6] lines={len(all_lines)} blues={blues} yellows={yellows}", flush=True)

    return all_lines

# ═══════════════════════════════════════════
# 可视化（同V7）
# ═══════════════════════════════════════════

def _rotation_matrix_z_to_xy(dx, dy):
    L = math.hypot(dx, dy)
    if L < 0.001:
        return np.eye(3, dtype=np.float64)
    ux, uy = dx / L, dy / L
    return np.array([
        [ uy * uy, -ux * uy,  ux],
        [-ux * uy,  ux * ux,  uy],
        [-ux,      -uy,       0]
    ], dtype=np.float64)

def draw_lines(user_scn, lines):
    user_scn.ngeom = 0
    for fx, fy, tx, ty, color in lines:
        if user_scn.ngeom >= user_scn.maxgeom:
            break
        geom = user_scn.geoms[user_scn.ngeom]
        mid = np.array([(fx + tx) / 2, (fy + ty) / 2, 1.0], dtype=np.float64)
        d = math.hypot(tx - fx, ty - fy)
        rgba = [0.2, 0.5, 1.0, 1.0] if color == 'blue' else [1.0, 0.9, 0.1, 1.0]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.array([0.05, max(d / 2, 0.01), 0], dtype=np.float64),
            mid, np.eye(3, dtype=np.float64).flatten(),
            np.array(rgba, dtype=np.float32)
        )
        geom.mat[:] = _rotation_matrix_z_to_xy(tx - fx, ty - fy)
        user_scn.ngeom += 1

# ═══════════════════════════════════════════
# MuJoCo场景（同V7）
# ═══════════════════════════════════════════

def build_xml():
    OBS_XML = "".join(
        f'<body name="obs{j}" pos="{x:.1f} {y:.1f} 2.0">'
        f'<geom type="cylinder" size="1.0 2.0" rgba="0.9 0.2 0.2 0.9"/></body>'
        for j, (x, y) in enumerate(obs_world))
    return f"""<mujoco>
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

# ═══════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════

print(f"━━━ V6 体素连通性 ━━━ seed={FIXED_SEED} ━━━", flush=True)

xml = build_xml()
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
d.qpos[0] = START_POS[0]; d.qpos[1] = START_POS[1]
mujoco.mj_forward(m, d)

# 扫描
print(f"  [SCAN] {INIT_SCAN_FRAMES}帧...", flush=True)
for _ in range(INIT_SCAN_FRAMES):
    bx, by = d.qpos[0], d.qpos[1]
    if _ % OBSERVE_TICK == 0:
        observe(bx, by)
    mujoco.mj_step(m, d)
print(f"  [OK] wall_points={len(wall_points)}", flush=True)

# decide
bx, by = d.qpos[0], d.qpos[1]
lines = decide_connectivity(bx, by)

# 显示
print(f"=== viewer运行中 ===", flush=True)
with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
    v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.distance = 25; v.cam.elevation = -35; v.cam.azimuth = 180
    v.cam.lookat[:] = np.array([bx, by, 0.5], dtype=np.float64)

    step = 0
    while v.is_running():
        if step % RENDER_SKIP == 0:
            draw_lines(v.user_scn, lines)
            v.sync()
        mujoco.mj_step(m, d)
        step += 1
