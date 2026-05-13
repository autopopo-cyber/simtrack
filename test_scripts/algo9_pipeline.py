#!/usr/bin/env python3
"""V9: scan→polygon→选门→move 完整pipeline

无随机障碍物（纯hfield赛道），验证端到端跑通。
10Hz observe → 1Hz polygon → 选朝终点最近的黄线门 → Mover朝门走
"""

import sys, os, math, time, random, heapq
from collections import defaultdict
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
DECIDE_RADIUS = 15.0
GAP_YELLOW_M = 1.0
GRID_RES = 1  # 1m grid
HIT_BACKOFF = 0.2
OBSERVE_TICK = 20; DECIDE_TICK = 200; RENDER_SKIP = 20
LINE_RADIUS = 0.08

SPEED = 2.0; SPEED_MAX = 8.0; MIN_SPEED = 1.0; SPEED_FACTOR = 2.0
YAW_RATE = 6.0
BOUNCE_FORCE_DURATION = 0.3
STUCK_TIMEOUT = 300; STUCK_DIST_THRESH = 0.5

FIXED_SEED = random.randint(0, 999999)
START_POS = (3.0, 3.0)
FINISH = (3.0, 95.0)
ARRIVE_THRESH = 1.5

# ═══════════════════════════════════════════
# 地图（无随机障碍物）
# ═══════════════════════════════════════════

hf = np.array(Image.open(MAP))
obs_world = []  # 无随机障碍

def sample_hf(wx, wy):
    mx, my = wx/SCALE, wy/SCALE
    px, py = int(mx*PIX_PER_M), HF_RES-1-int(my*PIX_PER_M)
    return int(hf[py,px]) if 0<=px<HF_RES and 0<=py<HF_RES else -1

def is_obstacle_world(wx, wy):
    return sample_hf(wx, wy) != ROAD_PIX

# ═══════════════════════════════════════════
# Observe: 激光 → 1m grid索引
# ═══════════════════════════════════════════

wall_set = set()
wall_grid = defaultdict(set)

def _snap(v):
    return round(v * 10) / 10

def _add_point(wx, wy):
    if (wx, wy) in wall_set:
        return
    wall_set.add((wx, wy))
    gx, gy = int(wx / GRID_RES), int(wy / GRID_RES)
    wall_grid[(gx, gy)].add((wx, wy))

def observe(bx, by):
    for a in np.linspace(0, 2*math.pi, LIDAR_RAYS):
        cos_a, sin_a = math.cos(a), math.sin(a)
        for step_i in range(1, LIDAR_STEPS + 1):
            wx = bx + cos_a * step_i * 0.1
            wy = by + sin_a * step_i * 0.1
            if is_obstacle_world(wx, wy):
                hx = bx + cos_a * (step_i * 0.1 - HIT_BACKOFF)
                hy = by + sin_a * (step_i * 0.1 - HIT_BACKOFF)
                _add_point(_snap(hx), _snap(hy))
                break
def get_nearby_points(bx, by):
    """用grid索引取bx,by周围DECIDE_RADIUS内的点"""
    gx0, gy0 = int((bx - DECIDE_RADIUS) / GRID_RES), int((by - DECIDE_RADIUS) / GRID_RES)
    gx1, gy1 = int((bx + DECIDE_RADIUS) / GRID_RES), int((by + DECIDE_RADIUS) / GRID_RES)
    nearby = []
    for gx in range(gx0, gx1 + 1):
        for gy in range(gy0, gy1 + 1):
            if (gx, gy) in wall_grid:
                for wx, wy in wall_grid[(gx, gy)]:
                    if abs(wx - bx) <= DECIDE_RADIUS and abs(wy - by) <= DECIDE_RADIUS:
                        nearby.append((wx, wy))
    return nearby

# ═══════════════════════════════════════════
# Grid状态机: active(≤R) / archived(>R)
# ═══════════════════════════════════════════

grid_cells = {}  # {(gx,gy): {'status': 'active'|'archived', 'gate': (fx,fy,tx,ty)|None}}

def update_grid_status(bx, by):
    """根据机器人位置更新所有非空grid的状态。"""
    for (gx, gy), pts in wall_grid.items():
        cx, cy = (gx + 0.5) * GRID_RES, (gy + 0.5) * GRID_RES
        dist = math.hypot(cx - bx, cy - by)
        new_status = 'active' if dist <= DECIDE_RADIUS else 'archived'
        if (gx, gy) not in grid_cells:
            grid_cells[(gx, gy)] = {'status': new_status, 'gate': None}
        else:
            grid_cells[(gx, gy)]['status'] = new_status

def classify_gate_line(fx, fy, tx, ty):
    """判断黄线是真门还是archived衔接。
    两端grid都archived→'gray'，否则None(保留原色)。"""
    g1 = (int(fx / GRID_RES), int(fy / GRID_RES))
    g2 = (int(tx / GRID_RES), int(ty / GRID_RES))
    s1 = grid_cells.get(g1, {}).get('status', 'active')
    s2 = grid_cells.get(g2, {}).get('status', 'active')
    if s1 == 'archived' and s2 == 'archived':
        return 'gray'
    return None  # 保留原色（真门黄线）

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

    def _subdivide(ax, ay, bx_w, by_w, depth=0):
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

    return lines


def decide(bx, by):
    """取周围DECIDE_RADIUS内墙体点做递归增量多边形（同V7行为）。"""
    points = get_nearby_points(bx, by)
    return polygon_boundary(points, bx, by)

# ═══════════════════════════════════════════
# 选门：从黄线中选朝终点最近的门
# ═══════════════════════════════════════════

def pick_gate(lines, bx, by):
    """选门：宽门+远门优先（最远门=未探索区域，V3 far模式）。
    Returns (gate_wx, gate_wy) or None."""
    best = None
    best_score = -float('inf')
    for fx, fy, tx, ty, color in lines:
        if color != 'yellow':
            continue
        width = math.hypot(tx - fx, ty - fy)
        mx, my = (fx + tx) / 2, (fy + ty) / 2
        dist = math.hypot(mx - bx, my - by)
        score = width + dist * 0.5
        if score > best_score:
            best_score = score
            best = (mx, my)
    return best

# ═══════════════════════════════════════════
# A*：基于is_obstacle_world，0.5m分辨率
# ═══════════════════════════════════════════

ASTAR_VOXEL = 0.5
ASTAR_RANGE = 30  # 搜索范围30m

def astar_to(bx, by, gx, gy):
    """A*从(bx,by)到(gx,gy)。0.5m分辨率8方向。
    Returns [(wx,wy),...] 路径点 or None。"""
    sx, sy = int(bx / ASTAR_VOXEL), int(by / ASTAR_VOXEL)
    tx, ty = int(gx / ASTAR_VOXEL), int(gy / ASTAR_VOXEL)

    def is_blocked(vx, vy):
        wx, wy = (vx + 0.5) * ASTAR_VOXEL, (vy + 0.5) * ASTAR_VOXEL
        return is_obstacle_world(wx, wy)

    if is_blocked(sx, sy) or is_blocked(tx, ty):
        return None

    open_set = [(math.hypot(tx - sx, ty - sy), sx, sy)]
    came_from = {}
    g_score = {(sx, sy): 0}
    visited = set()
    max_range_v = int(ASTAR_RANGE / ASTAR_VOXEL)

    while open_set and len(came_from) < 15000:
        _, cx, cy = heapq.heappop(open_set)
        if (cx, cy) in visited:
            continue
        visited.add((cx, cy))
        if (cx, cy) == (tx, ty):
            break

        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
            nx, ny = cx + dx, cy + dy
            if abs(nx - sx) > max_range_v or abs(ny - sy) > max_range_v:
                continue
            if is_blocked(nx, ny):
                continue
            step_cost = math.hypot(dx, dy)
            ng = g_score.get((cx, cy), 999) + step_cost
            if (nx, ny) not in g_score or ng < g_score[(nx, ny)]:
                g_score[(nx, ny)] = ng
                came_from[(nx, ny)] = (cx, cy)
                heapq.heappush(open_set, (ng + math.hypot(tx - nx, ty - ny), nx, ny))

    if (tx, ty) not in came_from and (tx, ty) != (sx, sy):
        return None

    # 回溯
    path = [((tx + 0.5) * ASTAR_VOXEL, (ty + 0.5) * ASTAR_VOXEL)]
    cur = (tx, ty)
    while cur != (sx, sy):
        if cur not in came_from:
            break
        cur = came_from[cur]
        path.append(((cur[0] + 0.5) * ASTAR_VOXEL, (cur[1] + 0.5) * ASTAR_VOXEL))
    path.reverse()
    return path[1:]  # 去掉起点（机器人位置）


def path_to_yellow_dots(path, spacing=1.0):
    """沿A*路径每spacing米取一个黄点。Returns [(wx,wy),...]"""
    if not path or len(path) < 2:
        return []
    dots = [path[0]]
    seg_start = path[0]
    dist_acc = 0.0
    for i in range(1, len(path)):
        px, py = path[i]
        step_d = math.hypot(px - seg_start[0], py - seg_start[1])
        dist_acc += step_d
        if dist_acc >= spacing:
            dots.append(path[i])
            dist_acc = 0.0
        seg_start = path[i]
    if dist_acc > 0 and dots[-1] != path[-1]:
        dots.append(path[-1])
    return dots


def yellow_dots_to_lines(dots, bx, by):
    """黄点序列→黄色线段 + 绿线(机器人→第一个黄点)。
    Returns [(fx,fy,tx,ty,color),...]"""
    extras = []
    if dots:
        extras.append((bx, by, dots[0][0], dots[0][1], 'green'))
        for i in range(len(dots) - 1):
            extras.append((dots[i][0], dots[i][1], dots[i+1][0], dots[i+1][1], 'yellow'))
    return extras

# ═══════════════════════════════════════════
# Mover（V3移植）
# ═══════════════════════════════════════════

class Mover:
    def __init__(self, m, d):
        self.m, self.d = m, d
        self.yaw = 0.0; self.speed = SPEED; self.bounce = 0
        self.force = 0; self.escaping = False
        self.stuck_t = 0; self.stuck_x = 0.0; self.stuck_y = 0.0

    def step(self, tx, ty, step):
        bx, by = self.d.qpos[0], self.d.qpos[1]
        dt = self.m.opt.timestep
        if not self.escaping:
            tgt_yaw = math.atan2(ty - by, tx - bx)
            err = (tgt_yaw - self.yaw + math.pi) % (2 * math.pi) - math.pi
            dyaw = max(-YAW_RATE * dt, min(YAW_RATE * dt, err))
            self.yaw += dyaw
            self.speed = max(MIN_SPEED, min(SPEED_MAX, math.hypot(tx - bx, ty - by) * SPEED_FACTOR))
        vx = math.cos(self.yaw) * self.speed
        vy = math.sin(self.yaw) * self.speed
        nx, ny = bx + vx * dt, by + vy * dt

        # stuck检测
        if step - self.stuck_t > STUCK_TIMEOUT:
            if math.hypot(bx - self.stuck_x, by - self.stuck_y) < STUCK_DIST_THRESH:
                self._bounce(90, 180)
            self.stuck_t = step; self.stuck_x = bx; self.stuck_y = by

        if self.force > 0:
            self.force -= 1; self.d.qvel[0] = vx; self.d.qvel[1] = vy
        elif blocked(nx, ny):
            self._bounce(45, 120)
        else:
            self.escaping = False
            self.d.qvel[0] = vx; self.d.qvel[1] = vy
        mujoco.mj_step(self.m, self.d)

    def _bounce(self, lo, hi):
        if not self.escaping:
            self.bounce += 1; self.escaping = True
            print(f"  [BOUNCE] bounce#{self.bounce} @({self.d.qpos[0]:.1f},{self.d.qpos[1]:.1f})", flush=True)
        deg = random.uniform(lo, hi) * random.choice([-1, 1])
        self.yaw += math.radians(deg)
        self.d.qvel[:] = 0
        self.force = int(BOUNCE_FORCE_DURATION / (SPEED * self.m.opt.timestep))

ROBOT_R = int(SAFE_R / 0.1)  # 5

def blocked(wx, wy):
    vx, vy = int(wx / 0.1), int(wy / 0.1)
    for dy in range(-ROBOT_R, ROBOT_R + 1):
        for dx in range(-ROBOT_R, ROBOT_R + 1):
            if dx * dx + dy * dy <= ROBOT_R * ROBOT_R:
                if is_obstacle_world((vx + dx + 0.5) * 0.1, (vy + dy + 0.5) * 0.1):
                    return True
    return False

# ═══════════════════════════════════════════
# 可视化
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

def draw_scene(user_scn, lines, robot_trail, path_dots, bx, by):
    """画完整场景：线段(蓝/黄/灰/绿) + 蓝球(轨迹) + 黄球(路径) + 绿球(机器人)"""
    user_scn.ngeom = 0

    # 线段
    for fx, fy, tx, ty, color in lines:
        if user_scn.ngeom >= user_scn.maxgeom:
            break
        geom = user_scn.geoms[user_scn.ngeom]
        mid = np.array([(fx + tx) / 2, (fy + ty) / 2, 1.0], dtype=np.float64)
        d = math.hypot(tx - fx, ty - fy)
        rgba_map = {
            'blue':   [0.2, 0.5, 1.0, 1.0],
            'yellow': [1.0, 0.9, 0.1, 1.0],
            'gray':   [0.5, 0.5, 0.5, 0.6],
            'green':  [0.2, 1.0, 0.2, 0.9],
        }
        rgba = rgba_map.get(color, [1, 1, 1, 1])
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.array([LINE_RADIUS, max(d / 2, 0.01), 0], dtype=np.float64),
            mid, np.eye(3, dtype=np.float64).flatten(),
            np.array(rgba, dtype=np.float32)
        )
        geom.mat[:] = _rotation_matrix_z_to_xy(tx - fx, ty - fy)
        user_scn.ngeom += 1

    # 蓝球（机器人轨迹，每1秒一个）
    for wx, wy in robot_trail:
        if user_scn.ngeom >= user_scn.maxgeom: break
        _add_sphere(user_scn, wx, wy, 0.15, [0.2, 0.5, 1.0, 0.8])

    # 黄球（路径点）
    for wx, wy in path_dots:
        if user_scn.ngeom >= user_scn.maxgeom: break
        _add_sphere(user_scn, wx, wy, 0.15, [1.0, 0.9, 0.1, 0.9])

    # 绿球（机器人当前位置）
    _add_sphere(user_scn, bx, by, 0.2, [0.2, 1.0, 0.2, 0.9])


def _add_sphere(user_scn, wx, wy, radius, rgba):
    if user_scn.ngeom >= user_scn.maxgeom: return
    geom = user_scn.geoms[user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom, mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0, 0], dtype=np.float64),
        np.array([wx, wy, 1.0], dtype=np.float64),
        np.eye(3, dtype=np.float64).flatten(),
        np.array(rgba, dtype=np.float32)
    )
    user_scn.ngeom += 1

# ═══════════════════════════════════════════
# MuJoCo场景
# ═══════════════════════════════════════════

def build_xml():
    FINISH_XML = f'<body mocap="true" pos="{FINISH[0]:.1f} {FINISH[1]:.1f} 2"><geom type="sphere" size="1.5" rgba="0.2 1.0 0.2 0.8"/></body>'
    return f"""<mujoco>
  <compiler angle="radian"/><option timestep="0.005"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset><hfield name="track" size="50.0 50.0 4.0 2.0" file="{MAP}"/></asset>
  <worldbody>
    <light pos="50 50 80" dir="0 0 -1"/>
    {FINISH_XML}
    <geom type="hfield" hfield="track" pos="50 50 0.0" rgba="0.25 0.30 0.35 1.0" friction="0 0 0"/>
    <body name="bot" pos="0 0 0.5">
      <joint type="slide" axis="1 0 0" damping="0"/>
      <joint type="slide" axis="0 1 0" damping="0"/>
      <geom type="cylinder" size="0.5 0.5" rgba="1 0.3 0 1" friction="0 0 0"/>
    </body>
  </worldbody>
</mujoco>"""

# ═══════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════

print(f"━━━ V9 pipeline: scan→polygon→gate→move ━━━ seed={FIXED_SEED} finish={FINISH} ━━━", flush=True)

xml = build_xml()
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
d.qpos[0] = START_POS[0]; d.qpos[1] = START_POS[1]
mujoco.mj_forward(m, d)

mv = Mover(m, d)

# 初始扫描
INIT_SCAN_FRAMES = 400
print(f"  [SCAN] 初始扫描 {INIT_SCAN_FRAMES}帧...", flush=True)
for _ in range(INIT_SCAN_FRAMES):
    bx, by = d.qpos[0], d.qpos[1]
    if _ % OBSERVE_TICK == 0:
        observe(bx, by)
    mujoco.mj_step(m, d)
print(f"  [OK] wall_set={len(wall_set)} grids={len(wall_grid)}", flush=True)

# 初始decide
bx, by = d.qpos[0], d.qpos[1]
lines = decide(bx, by)
gate = pick_gate(lines, bx, by)
dots = []
dot_idx = 0
robot_trail = [(bx, by)]
path_dots = []
if gate:
    path = astar_to(bx, by, gate[0], gate[1])
    dots = path_to_yellow_dots(path) if path else []
    print(f"  [GATE] →({gate[0]:.1f},{gate[1]:.1f}) path={len(path) if path else 0} dots={len(dots)}", flush=True)
    blues = sum(1 for _,_,_,_,c in lines if c == 'blue')
    yellows = sum(1 for _,_,_,_,c in lines if c == 'yellow')
    print(f"  [DECIDE] wall_set={len(wall_set)} blues={blues} yellows={yellows}", flush=True)

print(f"=== Pipeline running. 关闭窗口退出 ===", flush=True)

with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
    v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.distance = 25; v.cam.elevation = -35; v.cam.azimuth = 180

    step = 0
    while v.is_running():
        bx, by = d.qpos[0], d.qpos[1]
        v.cam.lookat[:] = np.array([bx, by, 0.5], dtype=np.float64)

        # observe: 10Hz
        if step % OBSERVE_TICK == 0:
            observe(bx, by)

        # decide + gate + A*: 1Hz
        if step % DECIDE_TICK == 0:
            lines = decide(bx, by)
            gate = pick_gate(lines, bx, by)
            if gate:
                path = astar_to(bx, by, gate[0], gate[1])
                dots = path_to_yellow_dots(path) if path else []
                path_dots = dots[:]
                dot_idx = 0
            else:
                dots = []
                path_dots = []

            # 每1秒记录机器人位置（蓝球轨迹）
            if step > 0:
                robot_trail.append((bx, by))

            blues = sum(1 for _,_,_,_,c in lines if c == 'blue')
            yellows = sum(1 for _,_,_,_,c in lines if c == 'yellow')
            print(f"  [D] step={step} wall_set={len(wall_set)} blues={blues} yellows={yellows}"
                  + (f" gate=({gate[0]:.1f},{gate[1]:.1f}) dots={len(dots)}" if gate else " gate=None"),
                  flush=True)

        # move 沿黄点走
        if dots and dot_idx < len(dots):
            tx, ty = dots[dot_idx]
            mv.step(tx, ty, step)
            if math.hypot(bx - tx, by - ty) < ARRIVE_THRESH:
                dot_idx += 1
        elif gate:
            mv.step(gate[0], gate[1], step)

        # 重新读取move后的真实位置
        bx, by = d.qpos[0], d.qpos[1]

        # 终点检测
        if math.hypot(bx - FINISH[0], by - FINISH[1]) < 3.0:
            print(f"\n  ★ ARRIVED! @({bx:.1f},{by:.1f}) step={step}", flush=True)
            break

        # render: 多边形 + 黄点线段 + 球
        all_lines = list(lines)
        extras = yellow_dots_to_lines(dots[dot_idx:], bx, by) if dots and dot_idx < len(dots) else []
        all_lines.extend(extras)
        if step % RENDER_SKIP == 0:
            draw_scene(v.user_scn, all_lines, robot_trail, path_dots, bx, by)
            v.sync()

        step += 1
    print(f"done: step={step} wall_set={len(wall_set)} bounce={mv.bounce}", flush=True)
