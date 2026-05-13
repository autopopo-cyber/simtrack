#!/usr/bin/env python3
"""V7: Observe-Decide解耦 + 递归增量多边形 + user_scn原生画线

observe: 激光→距离-0.2m→0.1m精度点→set去重
decide:  递归增量多边形（3点→细分大边→墙/门分离）
可视化: v.user_scn动态capsule，长度匹配实际线段
"""

import sys, os, math, time, random
import numpy as np
from PIL import Image
import mujoco, mujoco.viewer

# ═══════════════════════════════════════════
# 参数
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

# ═══════════════════════════════════════════
# 地图 + 障碍物
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
    while idx < len(cl):
        cx, cy = cl[idx]; wx, wy = cx*SCALE, cy*SCALE
        obs_world.append((wx, wy+rng.uniform(-2.0,2.0)))
        idx += rng.randint(3,8)
    return [(x,y) for x,y in obs_world if math.hypot(x-6,y-6)>5.0]

obs_world = gen_obstacles(FIXED_SEED)
OBS_R = 1.0; OBS_CLEAR = OBS_R + SAFE_R

def sample_hf(wx, wy):
    mx, my = wx/SCALE, wy/SCALE
    px, py = int(mx*PIX_PER_M), HF_RES-1-int(my*PIX_PER_M)
    return int(hf[py,px]) if 0<=px<HF_RES and 0<=py<HF_RES else -1

def is_obstacle_world(wx, wy):
    if sample_hf(wx, wy) != ROAD_PIX: return True
    for ox, oy in obs_world:
        if math.hypot(wx-ox, wy-oy) < OBS_CLEAR: return True
    return False

# ═══════════════════════════════════════════
# Observe: 激光 → 点集去重
# ═══════════════════════════════════════════

wall_points = set()

def _snap(v):
    return round(v * 10) / 10

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

def get_nearby_points(bx, by):
    nearby = []
    for wx, wy in wall_points:
        if abs(wx - bx) <= DECIDE_RADIUS and abs(wy - by) <= DECIDE_RADIUS:
            nearby.append((wx, wy))
    return nearby

# ═══════════════════════════════════════════
# Decide: 递归增量多边形
# ═══════════════════════════════════════════

def polygon_boundary(points, bx, by):
    n = len(points)
    if n < 3:
        return []

    robx, roby = bx, by

    polar = []
    for wx, wy in points:
        polar.append((math.atan2(wy - roby, wx - robx),
                      math.hypot(wx - robx, wy - roby), wx, wy))
    polar.sort()

    _subdivide_calls = [0]  # 用list实现闭包可变引用

    def _subdivide(ax, ay, bx_w, by_w, depth=0):
        _subdivide_calls[0] += 1
        call_id = _subdivide_calls[0]
        d = math.hypot(bx_w - ax, by_w - ay)
        if d <= GAP_YELLOW_M or depth > 60:
            return [(ax, ay), (bx_w, by_w)]

        ang_a = math.atan2(ay - roby, ax - robx)
        ang_b = math.atan2(by_w - roby, bx_w - robx)
        if ang_b < ang_a:
            ang_b += 2 * math.pi

        best_pt = None
        best_dist = float('inf')
        for ang, dist, wx, wy in polar:
            a = ang if ang >= ang_a else ang + 2 * math.pi
            if ang_a < a < ang_b and dist < best_dist:
                if not (abs(wx - ax) < 0.05 and abs(wy - ay) < 0.05) and \
                   not (abs(wx - bx_w) < 0.05 and abs(wy - by_w) < 0.05):
                    best_dist = dist
                    best_pt = (wx, wy)

        if not best_pt:
            return [(ax, ay), (bx_w, by_w)]

        cx, cy = best_pt
        print(f"  [SUB #{call_id} d={depth}] ({ax:.1f},{ay:.1f})→({bx_w:.1f},{by_w:.1f}) |AB|={d:.1f}m → insert ({cx:.1f},{cy:.1f})",
              flush=True)
        left = _subdivide(ax, ay, cx, cy, depth + 1)
        right = _subdivide(cx, cy, bx_w, by_w, depth + 1)
        return left[:-1] + right

    _, _, wx0, wy0 = polar[0]
    _, _, wx1, wy1 = polar[n // 2]
    _, _, wx2, wy2 = polar[-1]
    init = [(wx0, wy0), (wx1, wy1), (wx2, wy2)]

    poly = []
    for k in range(len(init)):
        ax, ay = init[k]
        bx_w, by_w = init[(k + 1) % len(init)]
        seg = _subdivide(ax, ay, bx_w, by_w)
        poly.extend(seg if k == 0 else seg[1:])

    print(f"  [POLY] init=3→final={len(poly)}", flush=True)

    lines = []
    for k in range(len(poly)):
        fx, fy = poly[k]
        tx, ty = poly[(k + 1) % len(poly)]
        d = math.hypot(tx - fx, ty - fy)
        lines.append((fx, fy, tx, ty, 'yellow' if d > GAP_YELLOW_M else 'blue'))

    return lines


def decide(bx, by):
    points = get_nearby_points(bx, by)
    return polygon_boundary(points, bx, by)

# ═══════════════════════════════════════════
# 可视化：v.user_scn 原生画线（动态长度capsule）
# ═══════════════════════════════════════════

def _rotation_matrix_z_to_xy(dx, dy):
    """从Z轴(0,0,1)旋转到(dx,dy,0)方向的3x3矩阵"""
    L = math.hypot(dx, dy)
    if L < 0.001:
        return np.eye(3, dtype=np.float64)
    ux, uy = dx / L, dy / L
    # Rodrigues: axis = (uy, -ux, 0), angle = π/2
    # R = I + sinθ·K + (1-cosθ)·K²  with θ=π/2, sin=1, cos=0
    # K = [[0,0,uy],[0,0,-ux],[-uy,ux,0]]
    # R = I + K + K²
    return np.array([
        [ux * ux,  ux * uy,  uy],
        [ux * uy,  uy * uy, -ux],
        [-uy,      ux,       0]
    ], dtype=np.float64)

def draw_lines(user_scn, lines):
    """在user_scn中画线段。lines=[(fx,fy,tx,ty,color),...]"""
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
            mid,
            np.eye(3, dtype=np.float64).flatten(),
            np.array(rgba, dtype=np.float32)
        )
        # 旋转capsule到线段方向
        geom.mat[:] = _rotation_matrix_z_to_xy(tx - fx, ty - fy)
        user_scn.ngeom += 1

# ═══════════════════════════════════════════
# MuJoCo场景（无mocap body）
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

print(f"━━━ V7 递归增量多边形 + user_scn原生画线 ━━━ seed={FIXED_SEED} ━━━", flush=True)

xml = build_xml()
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
d.qpos[0] = START_POS[0]; d.qpos[1] = START_POS[1]
mujoco.mj_forward(m, d)

# 阶段1: 扫描
print(f"  [SCAN] 初始扫描 {INIT_SCAN_FRAMES}帧...", flush=True)
for _ in range(INIT_SCAN_FRAMES):
    bx, by = d.qpos[0], d.qpos[1]
    if _ % OBSERVE_TICK == 0:
        observe(bx, by)
    mujoco.mj_step(m, d)
print(f"  [OK] wall_points={len(wall_points)}", flush=True)

# 阶段2: 决策
bx, by = d.qpos[0], d.qpos[1]
lines = decide(bx, by)
blues = sum(1 for _,_,_,_,c in lines if c == 'blue')
yellows = sum(1 for _,_,_,_,c in lines if c == 'yellow')
nearby = len(get_nearby_points(bx, by))
print(f"  [DECIDE] nearby={nearby} blues={blues} yellows={yellows}", flush=True)

# 阶段3: 持续显示（user_scn画线）
print(f"=== viewer运行中。关闭窗口退出 ===", flush=True)

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

    print(f"done: wall_points={len(wall_points)} blues={blues} yellows={yellows}", flush=True)
