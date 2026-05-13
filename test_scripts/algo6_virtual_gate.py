#!/usr/bin/env python3
"""萤火 V6 — 解耦观察/决策 + 虚拟门 + 三角矩形机器人

observe (10Hz): 激光→0.1m体素墙, 只记新增
decide  (1Hz): 邻接连通→蓝墙线/黄门线→虚拟门锚点→闭环修剪
render: 三角形+矩形机器人 + 蓝/黄封闭边界
"""

import math, os, random
import numpy as np
from PIL import Image
import mujoco, mujoco.viewer

# ═══════════════════════ 参数 ═══════════════════════
_HOME = os.path.expanduser("~")
MAP = os.path.join(_HOME, "workspace/simtrack/confirmed/track_clean.png")
SCALE = 2.0; HF_RES = 2000; PIX_PER_M = 40; ROAD_PIX = 128
VOXEL = 0.1
LIDAR_RANGE = 15.0; LIDAR_RAYS = 120; LIDAR_STEPS = int(LIDAR_RANGE/VOXEL)
HIT_BACKOFF = 0.2
DECIDE_RADIUS = 10.0          # 决策查询半径
CONNECT_R = 0.2               # 邻接聚类阈值
GAP_YELLOW_M = 1.0            # 黄/蓝分界
VIRTUAL_CONE_DEG = 90         # 虚拟门前方锥形角度
VIRTUAL_RANGE = 15.0          # 虚拟锚点距离
OBSERVE_TICK = 20             # 10Hz = timestep 0.005 × 20
DECIDE_TICK = 200             # 1Hz
RENDER_SKIP = 20
LINE_RADIUS = 0.06
FIXED_SEED = random.randint(0, 999999)
FINISH = (3.0, 95.0)

# ═══════════════════ 全局状态 ═══════════════════
wall_voxels = set()            # {(vx,vy), ...} 所有已观测墙体素
wall_set_raw = set()           # {(wx,wy), ...} 世界坐标激光命中点(去重用)
boundary_lines = []            # 当前帧边界线 [(fx,fy,tx,ty,color), ...]
virtual_anchors = []           # 当前虚拟锚点

# ═══════════════════ 地图 ═══════════════════
hf = np.array(Image.open(MAP))

def sample_hf(wx, wy):
    mx, my = wx/SCALE, wy/SCALE
    px, py = int(mx*PIX_PER_M), HF_RES-1-int(my*PIX_PER_M)
    return int(hf[py,px]) if 0<=px<HF_RES and 0<=py<HF_RES else -1

def is_obstacle_world(wx, wy):
    return sample_hf(wx, wy) != ROAD_PIX

# ═══════════════════ 观察 (10Hz) ═══════════════════

def observe(bx, by):
    """激光扫描, 只记录新增墙体素和命中点。"""
    new_walls = 0
    for a in np.linspace(0, 2*math.pi, LIDAR_RAYS):
        cos_a, sin_a = math.cos(a), math.sin(a)
        for step_i in range(1, LIDAR_STEPS+1):
            wx = bx + cos_a * step_i * VOXEL
            wy = by + sin_a * step_i * VOXEL
            vx, vy = int(wx/VOXEL), int(wy/VOXEL)
            if is_obstacle_world(wx, wy):
                if (vx, vy) not in wall_voxels:
                    wall_voxels.add((vx, vy))
                    new_walls += 1
                # 激光命中点 (后退0.2m)
                hx = bx + cos_a * (step_i*VOXEL - HIT_BACKOFF)
                hy = by + sin_a * (step_i*VOXEL - HIT_BACKOFF)
                wall_set_raw.add((round(hx, 1), round(hy, 1)))
                break
    return new_walls

# ═══════════════════ 连通性聚类 ═══════════════════

def _cluster_points(points):
    """Union-Find: 距离<CONNECT_R的点归一组。"""
    if len(points) < 2:
        return [points]
    n = len(points)
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj: parent[ri] = rj

    for i in range(n):
        xi, yi = points[i]
        for j in range(i+1, n):
            xj, yj = points[j]
            if abs(xi-xj) <= CONNECT_R and abs(yi-yj) <= CONNECT_R:
                if math.hypot(xi-xj, yi-yj) < CONNECT_R:
                    union(i, j)

    comps = {}
    for i, p in enumerate(points):
        root = find(i)
        comps.setdefault(root, []).append(p)
    return list(comps.values())

def _trace_chain(comp):
    """沿链追迹: 端点出发每次选最近邻。"""
    if len(comp) <= 2:
        return [comp]
    comp_set = set(comp)
    def nb(p):
        return [q for q in comp_set if q != p
                and math.hypot(p[0]-q[0], p[1]-q[1]) < CONNECT_R]
    deg = {p: len(nb(p)) for p in comp}
    starts = [p for p, d in deg.items() if d <= 1]
    if not starts:
        starts = [comp[0]]

    visited = set()
    chains = []
    for start in starts:
        if start in visited:
            continue
        chain = [start]
        visited.add(start)
        cur = start
        while True:
            nbs = [n for n in nb(cur) if n not in visited]
            if not nbs:
                break
            if len(chain) >= 2:
                dx = chain[-1][0] - chain[-2][0]
                dy = chain[-1][1] - chain[-2][1]
                nxt = min(nbs, key=lambda n:
                    abs(math.atan2(n[1]-cur[1], n[0]-cur[0]) - math.atan2(dy, dx)))
            else:
                nxt = min(nbs, key=lambda n: math.hypot(n[0]-cur[0], n[1]-cur[1]))
            cur = nxt
            visited.add(cur)
            chain.append(cur)
        chains.append(chain)
    return chains

# ═══════════════════ 虚拟门 ═══════════════════

def _virtual_anchors(bx, by, heading, nearby_world):
    """在前方锥形找锚点: 有墙用墙, 没墙虚拟。
    Returns [(wx,wy), (wx,wy)] — 左右两个锚点世界坐标。"""
    half = math.radians(VIRTUAL_CONE_DEG / 2)
    left_ang = heading - half
    right_ang = heading + half

    def best_at(angle):
        best, best_d = None, VIRTUAL_RANGE
        for wx, wy in nearby_world:
            dx, dy = wx-bx, wy-by
            d = math.hypot(dx, dy)
            if d > VIRTUAL_RANGE or d < 0.5:
                continue
            a = math.atan2(dy, dx)
            diff = abs((a - angle + math.pi) % (2*math.pi) - math.pi)
            if diff < math.radians(20) and d < best_d:
                best, best_d = (wx, wy), d
        return best  # None if no wall found

    la = best_at(left_ang)
    ra = best_at(right_ang)

    # 没墙→虚拟点
    if la is None:
        la = (bx + math.cos(left_ang)*VIRTUAL_RANGE,
              by + math.sin(left_ang)*VIRTUAL_RANGE)
    if ra is None:
        ra = (bx + math.cos(right_ang)*VIRTUAL_RANGE,
              by + math.sin(right_ang)*VIRTUAL_RANGE)
    return [la, ra]

# ═══════════════════ 决策 (1Hz) ═══════════════════

def decide(bx, by, heading):
    """1Hz: 查询周围墙体素→聚类→追链→端点对接→虚拟门→闭环修剪。
    Returns [(fx,fy,tx,ty,color), ...]"""
    global virtual_anchors

    # 1. 查询 DECIDE_RADIUS 内的墙体素 → 世界坐标
    nearby = []
    for vx, vy in wall_voxels:
        wx = (vx + 0.5) * VOXEL
        wy = (vy + 0.5) * VOXEL
        if abs(wx-bx) <= DECIDE_RADIUS and abs(wy-by) <= DECIDE_RADIUS:
            nearby.append((round(wx, 1), round(wy, 1)))

    if len(nearby) < 2:
        return []

    # 2. 虚拟门锚点 (前方锥形，有墙用墙没墙虚拟)
    va = _virtual_anchors(bx, by, heading, set(nearby))
    virtual_anchors = va  # 记下来供绘图
    # 虚拟锚点加入聚类（保证即使空旷也有门）
    nearby_with_va = nearby + va

    # 3. 聚类 → 追链 → 蓝线
    comps = _cluster_points(nearby_with_va)
    all_endpoints = []
    lines = []

    for comp in comps:
        chains = _trace_chain(comp)
        for chain in chains:
            if len(chain) < 2:
                continue
            # 链内蓝线
            for i in range(len(chain)-1):
                lines.append((*chain[i], *chain[i+1], 'blue'))
            all_endpoints.append(chain[0])
            all_endpoints.append(chain[-1])

    # 4. 端点间互连 (不同链的端点)
    for i in range(len(all_endpoints)):
        for j in range(i+1, len(all_endpoints)):
            d = math.hypot(all_endpoints[i][0]-all_endpoints[j][0],
                          all_endpoints[i][1]-all_endpoints[j][1])
            if d < 5.0:  # 5m内才考虑
                color = 'blue' if d <= GAP_YELLOW_M else 'yellow'
                lines.append((*all_endpoints[i], *all_endpoints[j], color))

    # 5. 虚拟锚点标记色 (保证虚拟门线是黄色)
    #    凡是涉及虚拟锚点的>1m连线强制黄
    va_set = set(va)
    for i, (fx, fy, tx, ty, c) in enumerate(lines):
        if (fx, fy) in va_set or (tx, ty) in va_set:
            d = math.hypot(tx-fx, ty-fy)
            if d > GAP_YELLOW_M:
                lines[i] = (fx, fy, tx, ty, 'yellow')

    return lines

# ═══════════════════ 绘制 ═══════════════════

def _rot_z_to_xy(dx, dy):
    L = math.hypot(dx, dy)
    if L < 0.001:
        return np.eye(3, dtype=np.float64)
    ux, uy = dx/L, dy/L
    return np.array([[uy*uy, -ux*uy, ux],
                     [-ux*uy, ux*ux, uy],
                     [-ux, -uy, 0]], dtype=np.float64)

def draw_all(user_scn, lines, bx, by, heading):
    """每帧绘制: 边界线 + 机器人形状 + 虚拟锚点。"""
    user_scn.ngeom = 0

    for fx, fy, tx, ty, color in lines:
        if user_scn.ngeom >= user_scn.maxgeom:
            break
        geom = user_scn.geoms[user_scn.ngeom]
        mid = np.array([(fx+tx)/2, (fy+ty)/2, 1.0], dtype=np.float64)
        d = math.hypot(tx-fx, ty-fy)
        rgba = {'blue': [0.2, 0.5, 1.0, 1.0],
                'yellow': [1.0, 0.9, 0.1, 1.0]}.get(color, [1,1,1,1])
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.array([LINE_RADIUS, max(d/2, 0.01), 0], dtype=np.float64),
            mid, np.eye(3, dtype=np.float64).flatten(),
            np.array(rgba, dtype=np.float32))
        geom.mat[:] = _rot_z_to_xy(tx-fx, ty-fy)
        user_scn.ngeom += 1

    # 虚拟锚点 → 黄色小球
    for ax, ay in virtual_anchors:
        if user_scn.ngeom >= user_scn.maxgeom:
            break
        geom = user_scn.geoms[user_scn.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.15, 0, 0], dtype=np.float64),
            np.array([ax, ay, 1.2], dtype=np.float64),
            np.eye(3, dtype=np.float64).flatten(),
            np.array([1.0, 0.8, 0.0, 0.9], dtype=np.float32))
        user_scn.ngeom += 1

    # 三角形+矩形机器人
    _draw_robot_shape(user_scn, bx, by, heading)

def _draw_robot_shape(user_scn, bx, by, heading):
    """画三角形+矩形复合机器人。尖端=heading方向。"""
    if user_scn.ngeom + 5 >= user_scn.maxgeom:
        return

    cos_h, sin_h = math.cos(heading), math.sin(heading)
    # 矩形身体: 0.5m宽×0.8m长, 中心在机器人后方0.2m
    body_cx = bx - cos_h * 0.2
    body_cy = by - sin_h * 0.2
    body_corners = [
        (body_cx + cos_h*0.4 - sin_h*0.25, body_cy + sin_h*0.4 + cos_h*0.25),
        (body_cx + cos_h*0.4 + sin_h*0.25, body_cy + sin_h*0.4 - cos_h*0.25),
        (body_cx - cos_h*0.4 + sin_h*0.25, body_cy - sin_h*0.4 - cos_h*0.25),
        (body_cx - cos_h*0.4 - sin_h*0.25, body_cy - sin_h*0.4 + cos_h*0.25),
    ]
    # 画矩形4条边
    for i in range(4):
        if user_scn.ngeom >= user_scn.maxgeom: break
        fx, fy = body_corners[i]
        tx, ty = body_corners[(i+1)%4]
        geom = user_scn.geoms[user_scn.ngeom]
        mid = np.array([(fx+tx)/2, (fy+ty)/2, 0.6], dtype=np.float64)
        d = math.hypot(tx-fx, ty-fy)
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.array([0.03, max(d/2, 0.01), 0], dtype=np.float64),
            mid, np.eye(3, dtype=np.float64).flatten(),
            np.array([1.0, 0.4, 0.1, 1.0], dtype=np.float32))
        geom.mat[:] = _rot_z_to_xy(tx-fx, ty-fy)
        user_scn.ngeom += 1

    # 三角形尖端: 从机器人中心向前延伸0.4m, 基底宽0.3m
    tip_x = bx + cos_h * 0.4
    tip_y = by + sin_h * 0.4
    tip_left  = (bx - sin_h*0.15, by + cos_h*0.15)
    tip_right = (bx + sin_h*0.15, by - cos_h*0.15)

    for a, b in [(tip_left, tip_right), (tip_left, (tip_x, tip_y)), (tip_right, (tip_x, tip_y))]:
        if user_scn.ngeom >= user_scn.maxgeom: break
        geom = user_scn.geoms[user_scn.ngeom]
        mid = np.array([(a[0]+b[0])/2, (a[1]+b[1])/2, 0.6], dtype=np.float64)
        d = math.hypot(b[0]-a[0], b[1]-a[1])
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.array([0.03, max(d/2, 0.01), 0], dtype=np.float64),
            mid, np.eye(3, dtype=np.float64).flatten(),
            np.array([1.0, 0.6, 0.2, 1.0], dtype=np.float32))
        geom.mat[:] = _rot_z_to_xy(b[0]-a[0], b[1]-a[1])
        user_scn.ngeom += 1

# ═══════════════════ XML ═══════════════════

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
      <joint type="free"/>
      <geom type="cylinder" size="0.35 0.15" rgba="0.0 0.6 0.0 0.3" friction="0 0 0"/>
    </body>
  </worldbody>
</mujoco>"""

# ═══════════════════ 主入口 ═══════════════════

def main():
    print(f"━━━ 萤火 V6 解耦+虚拟门 ━━━ voxel={VOXEL}m cone={VIRTUAL_CONE_DEG}° ━━━", flush=True)
    print(f"  observe 10Hz | decide 1Hz | seed={FIXED_SEED}", flush=True)

    xml = build_xml()
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    d.qpos[0] = 3; d.qpos[1] = 3; d.qpos[2] = 0.5
    mujoco.mj_forward(m, d)

    step = 0; heading = 0.0  # 初始朝+x
    last_new_walls = 0

    with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        v.cam.distance = 25; v.cam.elevation = -35; v.cam.azimuth = 180

        # 初始扫描
        print("  [INIT] initial scan...", flush=True)
        for _ in range(200):
            bx, by = d.qpos[0], d.qpos[1]
            if _ % OBSERVE_TICK == 0:
                observe(bx, by)
            mujoco.mj_step(m, d)
        print(f"  [INIT] wall_voxels={len(wall_voxels)} raw_points={len(wall_set_raw)}", flush=True)

        # 首次决策
        boundary_lines[:] = decide(d.qpos[0], d.qpos[1], heading)
        blues = sum(1 for *_, c in boundary_lines if c == 'blue')
        yellows = sum(1 for *_, c in boundary_lines if c == 'yellow')
        print(f"  [DECIDE#0] blues={blues} yellows={yellows}", flush=True)

        while v.is_running():
            bx, by = d.qpos[0], d.qpos[1]
            v.cam.lookat[:] = np.array([bx, by, 0.5], dtype=np.float64)

            # 键盘移动机器人 (WASD)
            _handle_keyboard(v, d, m)

            # 观察 10Hz
            if step % OBSERVE_TICK == 0:
                last_new_walls = observe(bx, by)

            # 决策 1Hz
            if step % DECIDE_TICK == 0:
                boundary_lines[:] = decide(bx, by, heading)
                blues = sum(1 for *_, c in boundary_lines if c == 'blue')
                yellows = sum(1 for *_, c in boundary_lines if c == 'yellow')
                print(f"  [DECIDE] step={step} blues={blues} yellows={yellows} walls={len(wall_voxels)}", flush=True)

            # 渲染
            if step % RENDER_SKIP == 0:
                draw_all(v.user_scn, boundary_lines, bx, by, heading)
                v.sync()

            step += 1

    print(f"done: step={step} walls={len(wall_voxels)}", flush=True)


def _handle_keyboard(v, d, m):
    """WASD 移动机器人 (测试用)。"""
    speed = 3.0
    dt = m.opt.timestep
    # 读取键盘状态来移动机器人
    # (mujoco.viewer 的键盘处理比较原始，这里简化)
    # 实际使用: 通过 v.user_scn 或其他方式
    pass  # 保持机器人不动，纯扫描


if __name__ == '__main__':
    main()
