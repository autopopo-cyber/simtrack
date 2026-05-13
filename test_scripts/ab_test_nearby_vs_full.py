#!/usr/bin/env python3
"""A/B测试：同一批wall_points → nearby vs 全量 → polygon对比
纯headless，不启动viewer。用V7完全相同的参数+障碍物。
"""

import sys, os, math, random, json
import numpy as np
from PIL import Image
import mujoco

# ═══════════════════════════════════════════
# 参数（完全同V7）
# ═══════════════════════════════════════════

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
INIT_SCAN_FRAMES = 400  # 20轮（同V7原版）

# 固定seed便于复现
FIXED_SEED = 424242
random.seed(FIXED_SEED)
np.random.seed(FIXED_SEED)

# ═══════════════════════════════════════════
# 地图 + 障碍物（完全同V7）
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
# Observe（完全同V7）
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
# polygon_boundary（完全同V7，无修改）
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

    _subdivide_calls = [0]

    def _subdivide(ax, ay, bx_w, by_w, depth=0):
        _subdivide_calls[0] += 1
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

    lines = []
    for k in range(len(poly)):
        fx, fy = poly[k]
        tx, ty = poly[(k + 1) % len(poly)]
        d = math.hypot(tx - fx, ty - fy)
        lines.append((fx, fy, tx, ty, 'yellow' if d > GAP_YELLOW_M else 'blue'))

    return lines, _subdivide_calls[0]

# ═══════════════════════════════════════════
# get_nearby_points（同V7原版）
# ═══════════════════════════════════════════

def get_nearby_points(bx, by):
    nearby = []
    for wx, wy in wall_points:
        if abs(wx - bx) <= DECIDE_RADIUS and abs(wy - by) <= DECIDE_RADIUS:
            nearby.append((wx, wy))
    return nearby

# ═══════════════════════════════════════════
# 主入口：headless A/B测试
# ═══════════════════════════════════════════

print(f"━━━ A/B测试: nearby vs 全量 polygon ━━━ seed={FIXED_SEED} ━━━", flush=True)

# 构建XML（完全同V7）
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

# 阶段1: 扫描（同V7 20轮）
print(f"  [SCAN] {INIT_SCAN_FRAMES}帧 ({INIT_SCAN_FRAMES//OBSERVE_TICK}轮)...", flush=True)
for _ in range(INIT_SCAN_FRAMES):
    bx, by = d.qpos[0], d.qpos[1]
    if _ % OBSERVE_TICK == 0:
        observe(bx, by)
    mujoco.mj_step(m, d)
print(f"  [OK] wall_points={len(wall_points)}", flush=True)

bx, by = d.qpos[0], d.qpos[1]

# 测试A: nearby点（V7原版行为）
nearby_points = get_nearby_points(bx, by)
print(f"  [A-nearby] nearby点={len(nearby_points)} (DECIDE_RADIUS={DECIDE_RADIUS}m)", flush=True)
lines_a, calls_a = polygon_boundary(nearby_points, bx, by)
blues_a = sum(1 for _,_,_,_,c in lines_a if c == 'blue')
yellows_a = sum(1 for _,_,_,_,c in lines_a if c == 'yellow')
print(f"  [A-nearby] poly顶点={blues_a+yellows_a} blues={blues_a} yellows={yellows_a} calls={calls_a}", flush=True)

# 测试B: 全量点
all_points = list(wall_points)
print(f"  [B-full]   全量点={len(all_points)}", flush=True)
lines_b, calls_b = polygon_boundary(all_points, bx, by)
blues_b = sum(1 for _,_,_,_,c in lines_b if c == 'blue')
yellows_b = sum(1 for _,_,_,_,c in lines_b if c == 'yellow')
print(f"  [B-full]   poly顶点={blues_b+yellows_b} blues={blues_b} yellows={yellows_b} calls={calls_b}", flush=True)

# 对比
print()
print("=" * 60)
if blues_a == blues_b and yellows_a == yellows_b and calls_a == calls_b:
    print("✅ nearby==全量: polygon输出完全相同")
else:
    print("❌ nearby≠全量: polygon输出不同！")
    print(f"   nearby: blues={blues_a} yellows={yellows_a} calls={calls_a}")
    print(f"   全量:   blues={blues_b} yellows={yellows_b} calls={calls_b}")

    # 找黄线差异
    yellows_a_set = set()
    for fx, fy, tx, ty, c in lines_a:
        if c == 'yellow':
            yellows_a_set.add((round(fx,1), round(fy,1), round(tx,1), round(ty,1)))
    yellows_b_set = set()
    for fx, fy, tx, ty, c in lines_b:
        if c == 'yellow':
            yellows_b_set.add((round(fx,1), round(fy,1), round(tx,1), round(ty,1)))

    only_a = yellows_a_set - yellows_b_set
    only_b = yellows_b_set - yellows_a_set
    if only_a:
        print(f"   仅nearby有 {len(only_a)} 条黄线: {sorted(only_a)[:5]}")
    if only_b:
        print(f"   仅全量有 {len(only_b)} 条黄线: {sorted(only_b)[:5]}")

print("=" * 60)
