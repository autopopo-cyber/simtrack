#!/usr/bin/env python3
"""V6: Observe-Decide解耦 + 墙线可视化

observe(10Hz): 激光扫描 → 0.1m体素 → 增量记录墙（只记WALL）
decide(1Hz):   读10m内墙体素 → 8邻域BFS聚类 → BFS生成树蓝线 → 端点桥接
               → 蓝线(内部)/黄线(>1m桥)/蓝线(<1m桥) → 闭合包围线

机器人起始(3,3)，不移动，只观察+画线。
"""

import sys, os, math, time, random, heapq, json
import numpy as np
from PIL import Image
from collections import deque
import mujoco, mujoco.viewer

# ═══════════════════════════════════════════
# 参数
# ═══════════════════════════════════════════

MAP = os.path.expanduser("~/workspace/simtrack/confirmed/track_clean.png")
SCALE = 2.0; HF_RES = 2000; PIX_PER_M = 40; ROAD_PIX = 128
SAFE_R = 0.5

VOXEL = 0.1
LIDAR_RANGE = 15.0
LIDAR_RAYS = 120
LIDAR_STEPS = int(LIDAR_RANGE / VOXEL)       # 150

DECIDE_RADIUS = 10.0                          # 决策范围 10m
DECIDE_RADIUS_V = int(DECIDE_RADIUS / VOXEL)  # 100 格
CLUSTER_DIST = 0.2                            # 聚类阈值 0.2m=2格
GAP_YELLOW_M = 1.0                            # >1m 黄线
GAP_YELLOW_V = int(GAP_YELLOW_M / VOXEL)      # 10 格

MAX_LINES = 800                               # 最大线段数

OBSERVE_TICK = 20                             # 10Hz at 0.005s timestep
DECIDE_TICK = 200                             # 1Hz = 每200帧
RENDER_SKIP = 20                              # viewer sync间隔

FIXED_SEED = random.randint(0, 999999)
START_POS = (3.0, 3.0)

# ═══════════════════════════════════════════
# 地图 + 障碍物（复用V3）
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
    rng = random.Random(seed)
    cl = gen_centerline(); obs_world = []; idx = 0
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
# SLAM: 只记WALL
# ═══════════════════════════════════════════

grid = {}  # {(vx,vy): 1}

def observe(bx, by):
    """10Hz激光扫描，增量添加墙。只记WALL。"""
    for a in np.linspace(0, 2*math.pi, LIDAR_RAYS):
        cos_a, sin_a = math.cos(a), math.sin(a)
        for step_i in range(1, LIDAR_STEPS + 1):
            wx = bx + cos_a * step_i * VOXEL
            wy = by + sin_a * step_i * VOXEL
            if is_obstacle_world(wx, wy):
                vx, vy = int(wx / VOXEL), int(wy / VOXEL)
                grid[(vx, vy)] = 1
                # 也标记前一个体素（墙面内侧）
                if step_i > 1:
                    pwx = bx + cos_a * (step_i - 1) * VOXEL
                    pwy = by + sin_a * (step_i - 1) * VOXEL
                    pvx, pvy = int(pwx / VOXEL), int(pwy / VOXEL)
                    grid[(pvx, pvy)] = 1
                break

# ═══════════════════════════════════════════
# Decide: 聚类 → 连线
# ═══════════════════════════════════════════

def decide(bx, by):
    """1Hz：读取10m内墙体素 → BFS聚类 → 生成树蓝线 → 端点桥接。
    返回: [(from_wx, from_wy, to_wx, to_wy, 'blue'|'yellow'), ...]
    坐标是世界坐标（米）。
    """
    bx_v, by_v = int(bx / VOXEL), int(by / VOXEL)
    r = DECIDE_RADIUS_V

    # 1. 收集10m范围内所有墙体素
    wall_set = set()
    for vx in range(bx_v - r, bx_v + r + 1):
        for vy in range(by_v - r, by_v + r + 1):
            if (vx, vy) in grid:
                wall_set.add((vx, vy))

    if len(wall_set) < 2:
        return []

    # 2. 8邻域BFS聚类
    clusters = []
    visited = set()
    for seed in wall_set:
        if seed in visited:
            continue
        cluster = []
        q = deque([seed])
        visited.add(seed)
        while q:
            cvx, cvy = q.popleft()
            cluster.append((cvx, cvy))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    n = (cvx + dx, cvy + dy)
                    if n in wall_set and n not in visited:
                        visited.add(n)
                        q.append(n)
        clusters.append(cluster)

    if not clusters:
        return []

    # 3. 每个cluster内：BFS生成树 → 蓝线
    # 同时记录每个体素的度（树上的度）
    lines = []                # 返回值
    cluster_sets = [set(c) for c in clusters]
    degrees = {}              # {(vx,vy,cluster_id): degree}

    for ci, cluster in enumerate(clusters):
        cset = cluster_sets[ci]
        if len(cluster) <= 1:
            if len(cluster) == 1:
                degrees[(cluster[0][0], cluster[0][1], ci)] = 0
            continue

        tree_visited = set()
        seed = cluster[0]
        q = deque([(seed, None)])  # (voxel, parent)
        tree_visited.add(seed)
        deg = {}

        while q:
            (cvx, cvy), parent = q.popleft()
            d = 0
            if parent:
                d += 1  # parent edge
                # 画蓝线
                pwvx, pwvy = parent
                from_wx = (cvx + 0.5) * VOXEL
                from_wy = (cvy + 0.5) * VOXEL
                to_wx = (pwvx + 0.5) * VOXEL
                to_wy = (pwvy + 0.5) * VOXEL
                lines.append((from_wx, from_wy, to_wx, to_wy, 'blue'))

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    n = (cvx + dx, cvy + dy)
                    if n in cset and n not in tree_visited:
                        tree_visited.add(n)
                        q.append((n, (cvx, cvy)))
                        d += 1

            deg[(cvx, cvy)] = d

        for (cvx, cvy), d in deg.items():
            degrees[(cvx, cvy, ci)] = d

    # 4. 找叶子端点（树上度<=1的体素）
    all_endpoints = []  # [(vx, vy, cluster_id)]
    for ci, cluster in enumerate(clusters):
        for vx, vy in cluster:
            d = degrees.get((vx, vy, ci), 0)
            if d <= 1:
                all_endpoints.append((vx, vy, ci))

    # 5. 端点桥接不同cluster
    if len(clusters) > 1 and all_endpoints:
        # 对每个端点找最近的其他cluster的最近点
        for ep_vx, ep_vy, from_cid in all_endpoints:
            best_dist = float('inf')
            best_to = None
            for to_cid in range(len(clusters)):
                if to_cid == from_cid:
                    continue
                for tvx, tvy in clusters[to_cid]:
                    d = math.hypot(ep_vx - tvx, ep_vy - tvy) * VOXEL  # 米
                    if d < best_dist:
                        best_dist = d
                        best_to = (tvx, tvy)

            if best_to and best_dist < 999:
                tvx, tvy = best_to
                color = 'yellow' if best_dist > GAP_YELLOW_M else 'blue'
                from_wx = (ep_vx + 0.5) * VOXEL
                from_wy = (ep_vy + 0.5) * VOXEL
                to_wx = (tvx + 0.5) * VOXEL
                to_wy = (tvy + 0.5) * VOXEL
                lines.append((from_wx, from_wy, to_wx, to_wy, color))

    return lines

# ═══════════════════════════════════════════
# 线段可视化: mocap body + capsule
# ═══════════════════════════════════════════

class LineManager:
    """用mocap body + capsule geom画线段。
    capsule默认沿Z轴，旋转到线段方向(XY平面)。
    """
    def __init__(self, m, d):
        self.m = m; self.d = d
        self.bodies = []        # body名字列表
        self.active = 0

    def update(self, lines):
        """更新线段显示。lines: [(from_wx, from_wy, to_wx, to_wy, color), ...]"""
        for i in range(self.active):
            name = self.bodies[i]
            self.d.mocap_pos[self.m.body(name).mocapid] = [0, 0, -10]
        self.active = 0

        for i, (fx, fy, tx, ty, color) in enumerate(lines):
            if i >= MAX_LINES:
                break
            self._set_line(i, fx, fy, tx, ty, color)
            self.active += 1

    def _set_line(self, idx, fx, fy, tx, ty, color):
        name = f"line_{idx}"
        body = self.m.body(name)
        mid = np.array([(fx + tx) / 2, (fy + ty) / 2, 1.0], dtype=np.float64)
        self.d.mocap_pos[body.mocapid] = mid

        # capsule沿Z轴 → 旋转到线段方向（XY平面）
        dx, dy = tx - fx, ty - fy
        L = math.hypot(dx, dy)
        if L < 0.01:
            self.d.mocap_quat[body.mocapid] = [1, 0, 0, 0]
            return

        # 从Z轴(0,0,1)旋转到(dx/L, dy/L, 0)
        # 绕(dy/L, -dx/L, 0)旋转π/2
        axis = np.array([dy / L, -dx / L, 0.0], dtype=np.float64)
        angle = math.pi / 2.0
        # axis-angle to quat
        s = math.sin(angle / 2)
        q = np.array([math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float64)
        self.d.mocap_quat[body.mocapid] = q

        # 颜色: blue=(0.2,0.5,1.0,0.9), yellow=(1.0,0.9,0.1,0.9)
        rgba_key = 'blue' if color == 'blue' else 'yellow'
        # MuJoCo没有直接设geom rgba的简单API（mocap body的geom），但XML里设置后用material?
        # 实际上mocap body的geom颜色在XML中固定，无法运行时改变。
        # 解决方案：蓝色和黄色分别用不同的body组。
        # 简化：都用同一颜色，先验证算法。
        # 我们用mocap_quat已经设置好了，颜色在XML中预设蓝色。

# ═══════════════════════════════════════════
# MuJoCo场景
# ═══════════════════════════════════════════

def build_xml():
    """构建MuJoCo XML，包含线段mocap body。"""
    # 线段body：蓝色组和黄色组
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
# LineManager (支持蓝/黄分色)
# ═══════════════════════════════════════════

class LineManagerV2:
    """分蓝/黄两组mocap body。蓝色用line_b*，黄色用line_y*。"""
    def __init__(self, m, d):
        self.m = m; self.d = d
        self.blue_active = 0
        self.yellow_active = 0

    def update(self, lines):
        # 隐藏全部
        for i in range(MAX_LINES // 2):
            self.d.mocap_pos[self.m.body(f"line_b{i}").mocapid] = [0, 0, -10]
            self.d.mocap_pos[self.m.body(f"line_y{i}").mocapid] = [0, 0, -10]
        self.blue_active = 0
        self.yellow_active = 0

        for fx, fy, tx, ty, color in lines:
            if color == 'blue':
                if self.blue_active >= MAX_LINES // 2:
                    continue
                idx = self.blue_active
                name = f"line_b{idx}"
                self.blue_active += 1
            else:
                if self.yellow_active >= MAX_LINES // 2:
                    continue
                idx = self.yellow_active
                name = f"line_y{idx}"
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
# 主入口
# ═══════════════════════════════════════════

print(f"━━━ V6 Observe-Decide 解耦 ━━━ seed={FIXED_SEED} ━━━", flush=True)

xml = build_xml()
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)

d.qpos[0] = START_POS[0]; d.qpos[1] = START_POS[1]
mujoco.mj_forward(m, d)

lm = LineManagerV2(m, d)

# ── 阶段1: 初始扫描（转一圈） ──
INIT_SCAN_FRAMES = 400  # 400帧=20次observe=2秒
print(f"  [SCAN] 初始扫描 {INIT_SCAN_FRAMES}帧...", flush=True)
for _ in range(INIT_SCAN_FRAMES):
    bx, by = d.qpos[0], d.qpos[1]
    if _ % OBSERVE_TICK == 0:
        observe(bx, by)
    mujoco.mj_step(m, d)
print(f"  [OK] WALL={len(grid)}", flush=True)

# ── 阶段2: 决策画图 ──
bx, by = d.qpos[0], d.qpos[1]
lines = decide(bx, by)
lm.update(lines)

# 数一下10m内墙体素
bx_v, by_v = int(bx/VOXEL), int(by/VOXEL)
nearby = sum(1 for vx in range(bx_v-DECIDE_RADIUS_V, bx_v+DECIDE_RADIUS_V+1)
             for vy in range(by_v-DECIDE_RADIUS_V, by_v+DECIDE_RADIUS_V+1)
             if (vx, vy) in grid)
print(f"  [DECIDE] nearby={nearby} blines={lm.blue_active} ylines={lm.yellow_active}", flush=True)

# ── 阶段3: 持续显示 ──
print(f"=== viewer运行中，观察连线效果。关闭窗口退出 ===", flush=True)

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

    print(f"done: wall={len(grid)} blines={lm.blue_active} ylines={lm.yellow_active}", flush=True)
