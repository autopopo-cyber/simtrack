#!/usr/bin/env python3
"""V7: Observe-Decide 解耦 + 接口化 + 多边形连线（待实现）

observe(10Hz): 激光扫描 → hit距离-0.2m → 0.1m精度点 → set去重
decide(1Hz):   取10m内墙点 → 多边形边界提取 → 蓝线/黄线连线
               算法: 待填（角度排序 / Alpha Shape / Concave Hull）

机器人起始(3,3)，不移动。
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

LIDAR_RANGE = 15.0
LIDAR_RAYS = 120
LIDAR_STEPS = int(LIDAR_RANGE / 0.1)           # 150步

DECIDE_RADIUS = 10.0                            # 决策范围 10m
GAP_YELLOW_M = 1.0                              # >1m 黄线=门, ≤1m 蓝线=墙

MAX_LINES = 600
HIT_BACKOFF = 0.2                               # 击中后退0.2m

OBSERVE_TICK = 20                               # 10Hz
DECIDE_TICK = 200                               # 1Hz
RENDER_SKIP = 20

FIXED_SEED = random.randint(0, 999999)
START_POS = (3.0, 3.0)

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
# Observe: 激光 → 0.1m精度点集 (set去重)
# ═══════════════════════════════════════════

wall_points = set()  # {(wx, wy) 0.1m精度}

def _snap(v):
    """四舍五入到0.1m精度"""
    return round(v * 10) / 10

def observe(bx, by):
    """10Hz激光扫描。射线打到墙→距离减0.2m→0.1m精度取整→加入set。
    多次打同一位置自动去重。"""
    for a in np.linspace(0, 2*math.pi, LIDAR_RAYS):
        cos_a, sin_a = math.cos(a), math.sin(a)
        for step_i in range(1, LIDAR_STEPS + 1):
            wx = bx + cos_a * step_i * 0.1
            wy = by + sin_a * step_i * 0.1
            if is_obstacle_world(wx, wy):
                # 击中墙 → 后退0.2m → 0.1m精度
                hx = bx + cos_a * (step_i * 0.1 - HIT_BACKOFF)
                hy = by + sin_a * (step_i * 0.1 - HIT_BACKOFF)
                wall_points.add((_snap(hx), _snap(hy)))
                break

def get_nearby_points(bx, by):
    """取机器人10m范围内的墙点 → list of (wx, wy)"""
    nearby = []
    for wx, wy in wall_points:
        if abs(wx - bx) <= DECIDE_RADIUS and abs(wy - by) <= DECIDE_RADIUS:
            nearby.append((wx, wy))
    return nearby

# ═══════════════════════════════════════════
# Decide: 接口化 — 多边形边界提取（待实现）
# ═══════════════════════════════════════════

def _point_line_dist(px, py, ax, ay, bx, by):
    """点到线段AB的距离"""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

def _douglas_peucker(pts, epsilon):
    """Douglas-Peucker简化。pts=[(x,y),...]，epsilon=最大允许偏差(m)"""
    if len(pts) < 3:
        return list(pts)
    fx, fy = pts[0]; tx, ty = pts[-1]
    max_dist = 0; max_idx = 0
    for i in range(1, len(pts) - 1):
        d = _point_line_dist(pts[i][0], pts[i][1], fx, fy, tx, ty)
        if d > max_dist:
            max_dist = d; max_idx = i
    if max_dist > epsilon:
        left = _douglas_peucker(pts[:max_idx + 1], epsilon)
        right = _douglas_peucker(pts[max_idx:], epsilon)
        return left[:-1] + right
    return [pts[0], pts[-1]]

def polygon_boundary(points, bx, by):
    """递归增量多边形：三角形→对大边递归细分→墙/门自然分离。

    1. 初始化：取角度范围的三点（首/中/尾）构成三角形
    2. 递归：subdivide(边AB) → 在AB角度范围内找最近点C → 递归AC, CB
    3. 基底：边长≤1m 或 找不到更多点
    4. 最终：≤1m蓝线(墙)，>1m黄线(门)
    """
    n = len(points)
    if n < 3:
        return []

    robx, roby = bx, by  # 机器人位置（避免闭包变量名冲突）

    # 所有点按极角(-π→π)排序
    polar = []
    for wx, wy in points:
        polar.append((math.atan2(wy - roby, wx - robx),
                      math.hypot(wx - robx, wy - roby), wx, wy))
    polar.sort()

    # ── 递归细分一条边 ──
    def _subdivide(ax, ay, bx_w, by_w, depth=0):
        """递归细分边AB，返回从A到B的顶点列表 [(wx,wy),...]"""
        d = math.hypot(bx_w - ax, by_w - ay)

        # 基底：边长≤1m 或 深度超限
        if d <= GAP_YELLOW_M or depth > 60:
            return [(ax, ay), (bx_w, by_w)]

        # A和B相对机器人的极角
        ang_a = math.atan2(ay - roby, ax - robx)
        ang_b = math.atan2(by_w - roby, bx_w - robx)

        # 角度范围 [ang_a, ang_b] 逆时针
        if ang_b < ang_a:
            ang_b += 2 * math.pi

        # 在AB角度范围内找距离机器人最近的点
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

    # ── 初始化：取首/中/尾三点构成三角形 ──
    _, _, wx0, wy0 = polar[0]
    _, _, wx1, wy1 = polar[n // 2]
    _, _, wx2, wy2 = polar[-1]
    init = [(wx0, wy0), (wx1, wy1), (wx2, wy2)]

    # ── 对三角形每条边递归细分 ──
    poly = []
    for k in range(len(init)):
        ax, ay = init[k]
        bx_w, by_w = init[(k + 1) % len(init)]
        seg = _subdivide(ax, ay, bx_w, by_w)
        poly.extend(seg if k == 0 else seg[1:])

    print(f"  [POLY] recursive init=3→final={len(poly)}", flush=True)

    # ── 连线 ──
    lines = []
    for k in range(len(poly)):
        fx, fy = poly[k]
        tx, ty = poly[(k + 1) % len(poly)]
        d = math.hypot(tx - fx, ty - fy)
        color = 'yellow' if d > GAP_YELLOW_M else 'blue'
        lines.append((fx, fy, tx, ty, color))

    return lines


def decide(bx, by):
    """1Hz决策：取墙点→多边形边界提取→连线。"""
    points = get_nearby_points(bx, by)
    return polygon_boundary(points, bx, by)

# ═══════════════════════════════════════════
# 线段可视化 (复用V6)
# ═══════════════════════════════════════════

class LineManager:
    """分蓝/黄两组mocap body。蓝色用line_b*，黄色用line_y*。"""
    def __init__(self, m, d):
        self.m = m; self.d = d
        self.blue_active = 0
        self.yellow_active = 0

    def update(self, lines):
        for i in range(MAX_LINES // 2):
            self.d.mocap_pos[self.m.body(f"line_b{i}").mocapid] = [0, 0, -10]
            self.d.mocap_pos[self.m.body(f"line_y{i}").mocapid] = [0, 0, -10]
        self.blue_active = 0
        self.yellow_active = 0

        for fx, fy, tx, ty, color in lines:
            if color == 'blue':
                if self.blue_active >= MAX_LINES // 2: continue
                idx = self.blue_active; name = f"line_b{idx}"
                self.blue_active += 1
            else:
                if self.yellow_active >= MAX_LINES // 2: continue
                idx = self.yellow_active; name = f"line_y{idx}"
                self.yellow_active += 1
            self._set_line(name, fx, fy, tx, ty)

    def _set_line(self, name, fx, fy, tx, ty):
        body = self.m.body(name)
        mid = np.array([(fx + tx) / 2, (fy + ty) / 2, 1.0], dtype=np.float64)
        self.d.mocap_pos[body.mocapid] = mid
        dx, dy = tx - fx, ty - fy
        L = math.hypot(dx, dy)
        if L < 0.01:
            self.d.mocap_quat[body.mocapid] = [1, 0, 0, 0]
            return
        axis = np.array([dy / L, -dx / L, 0.0], dtype=np.float64)
        angle = math.pi / 2.0
        s = math.sin(angle / 2)
        q = np.array([math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)
        self.d.mocap_quat[body.mocapid] = q

# ═══════════════════════════════════════════
# MuJoCo场景
# ═══════════════════════════════════════════

def build_xml():
    blue_bodies = ""
    yellow_bodies = ""
    for i in range(MAX_LINES // 2):
        blue_bodies += (
            f'<body name="line_b{i}" mocap="true" pos="0 0 -10">'
            f'<geom type="capsule" size="0.05 0.5" rgba="0.2 0.5 1.0 0.9"/></body>\n'
        )
        yellow_bodies += (
            f'<body name="line_y{i}" mocap="true" pos="0 0 -10">'
            f'<geom type="capsule" size="0.05 0.5" rgba="1.0 0.9 0.1 0.9"/></body>\n'
        )
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
    {blue_bodies}
    {yellow_bodies}
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

print(f"━━━ V7 接口化 ━━━ observe点集去重 + decide空壳 ━━━ seed={FIXED_SEED} ━━━", flush=True)

xml = build_xml()
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
d.qpos[0] = START_POS[0]; d.qpos[1] = START_POS[1]
mujoco.mj_forward(m, d)
lm = LineManager(m, d)

# ── 阶段1: 初始扫描 ──
INIT_SCAN_FRAMES = 400
print(f"  [SCAN] 初始扫描 {INIT_SCAN_FRAMES}帧...", flush=True)
for _ in range(INIT_SCAN_FRAMES):
    bx, by = d.qpos[0], d.qpos[1]
    if _ % OBSERVE_TICK == 0:
        observe(bx, by)
    mujoco.mj_step(m, d)
print(f"  [OK] wall_points={len(wall_points)}", flush=True)

# ── 阶段2: 决策画图 ──
bx, by = d.qpos[0], d.qpos[1]
lines = decide(bx, by)
lm.update(lines)

nearby = len(get_nearby_points(bx, by))
print(f"  [DECIDE] nearby={nearby} blines={lm.blue_active} ylines={lm.yellow_active}", flush=True)

# ── 阶段3: 持续显示 ──
print(f"=== viewer运行中。关闭窗口退出 ===", flush=True)

with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
    v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.distance = 25; v.cam.elevation = -35; v.cam.azimuth = 180
    v.cam.lookat[:] = np.array([bx, by, 0.5], dtype=np.float64)

    step = 0
    while v.is_running():
        if step % RENDER_SKIP == 0:
            v.sync()
        mujoco.mj_step(m, d)
        step += 1

    print(f"done: wall_points={len(wall_points)} blines={lm.blue_active} ylines={lm.yellow_active}", flush=True)
