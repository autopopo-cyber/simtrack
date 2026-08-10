#!/usr/bin/env python3
"""V5d: 1Hz规划建凸多边形 — 8m圆切割墙体 → 墙缝封闭成门 → 右手原则冲"""
import math, random, time, numpy as np, mujoco
from PIL import Image

LIDAR_RANGE = 8.0; LIDAR_RAYS = 120; SPEED = 5.0; YAW_RATE = 6.0
ARRIVE_THRESH = 1.0; PLAN_INTERVAL = 200
BOUNCE_FORCE_DURATION = 0.3; STUCK_TIMEOUT = 300; STUCK_DIST_THRESH = 0.5
MAP = "/home/qin/workspace/simtrack/confirmed/track_clean.png"
PIX_PER_M = 40; FINISH = (3.0, 95.0)
GATE_MIN_ANGLE = 0.1  # 门最小弧度(避免碎缝)

TRACK_ARR = np.array(Image.open(MAP))
def png_is_wall(wx, wy):
    px, py = int(wx * PIX_PER_M), int(wy * PIX_PER_M)
    if px < 0 or px >= TRACK_ARR.shape[1] or py < 0 or py >= TRACK_ARR.shape[0]: return True
    v = TRACK_ARR[py, px] if TRACK_ARR.ndim == 2 else int(np.mean(TRACK_ARR[py, px, :3]))
    return v < 50

def scan_walls(bx, by):
    """激光 → 投影到8m圆上的墙点 [(angle, wx, wy), ...]"""
    pts = []
    for a in np.linspace(0, 2 * math.pi, LIDAR_RAYS):
        cs, sn = math.cos(a), math.sin(a)
        hit = False
        for d in np.arange(0.1, LIDAR_RANGE + 0.1, 0.1):
            wx, wy = bx + cs * d, by + sn * d
            if png_is_wall(wx, wy):
                pts.append((a, wx, wy))
                hit = True
                break
        if not hit:
            pts.append((a, bx + cs * LIDAR_RANGE, by + sn * LIDAR_RANGE))
    return pts

def build_polygon_gates(pts, bx, by):
    """8m圆上的墙点 → 聚类成墙段 → 墙缝=门 → 返回门中点列表"""
    n = len(pts)
    # 1. 找墙段: 连续命中墙的弧
    wall_segments = []
    i = 0
    while i < n:
        a, wx, wy = pts[i]
        # 判断该射线是否命中墙: 不是LIDAR_RANGE尽头且png_is_wall
        dist = math.hypot(wx - bx, wy - by)
        is_hit = dist < LIDAR_RANGE - 0.05 and png_is_wall(wx, wy)
        if not is_hit:
            i += 1
            continue
        # 收集连续命中
        seg = []
        while i < n:
            a, wx, wy = pts[i]
            dist = math.hypot(wx - bx, wy - by)
            if dist >= LIDAR_RANGE - 0.05 or not png_is_wall(wx, wy):
                break
            seg.append((a, wx, wy))
            i += 1
        if len(seg) >= 2:
            wall_segments.append(seg)
    if len(wall_segments) < 1:
        return []

    # 2. 墙缝=相邻墙段之间的连线的中点 = 门
    gates = []
    m = len(wall_segments)
    for i in range(m):
        j = (i + 1) % m
        seg_a = wall_segments[i]
        seg_b = wall_segments[j]
        # 取相邻端: seg_a的末点, seg_b的首点
        a1, x1, y1 = seg_a[-1]
        a2, x2, y2 = seg_b[0]
        # 角度差 > 最小门宽
        ang_gap = (a2 - a1) % (2 * math.pi)
        if ang_gap < GATE_MIN_ANGLE:
            continue
        # 门中点
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        gates.append((math.atan2(my - by, mx - bx), mx, my))
    return gates

def pick_gate_right(gates, yaw):
    """右手原则: 从yaw顺时针扫第一个门 → 返回世界坐标"""
    if not gates:
        return None
    yaw_n = yaw % (2 * math.pi)
    cand = [((yaw_n - ga) % (2 * math.pi), gx, gy) for ga, gx, gy in gates]
    cand.sort(key=lambda x: x[0])
    return cand[0][1], cand[0][2]

class Mover:
    def __init__(self, m, d):
        self.m, self.d = m, d
        self.yaw = math.pi / 2
        self.speed = SPEED
        self.bounce = 0
        self.force = 0
        self.escaping = False
        self.stuck_t = 0
        self.stuck_x = 0.0
        self.stuck_y = 0.0

    def step(self, tx, ty, step_no):
        bx, by = self.d.qpos[0], self.d.qpos[1]
        dt = self.m.opt.timestep
        if not self.escaping:
            err = (math.atan2(ty - by, tx - bx) - self.yaw + math.pi) % (2 * math.pi) - math.pi
            self.yaw += max(-YAW_RATE * dt, min(YAW_RATE * dt, err))
            self.speed = SPEED
        vx = math.cos(self.yaw) * self.speed
        vy = math.sin(self.yaw) * self.speed
        if step_no - self.stuck_t > STUCK_TIMEOUT:
            if math.hypot(bx - self.stuck_x, by - self.stuck_y) < STUCK_DIST_THRESH:
                self._bounce(90, 180)
            self.stuck_t = step_no
            self.stuck_x = bx
            self.stuck_y = by
        if self.force > 0:
            self.force -= 1
            self.d.qvel[0] = vx
            self.d.qvel[1] = vy
        elif self.escaping:
            self.escaping = False
            self.d.qvel[0] = vx
            self.d.qvel[1] = vy
        else:
            self.d.qvel[0] = vx
            self.d.qvel[1] = vy
        mujoco.mj_step(self.m, self.d)

    def _bounce(self, lo, hi):
        if not self.escaping:
            self.bounce += 1
            self.escaping = True
        self.yaw += math.radians(random.uniform(lo, hi) * random.choice([-1, 1]))
        self.d.qvel[:] = 0
        self.force = int(BOUNCE_FORCE_DURATION / (SPEED * self.m.opt.timestep))

# ═══════ 主程序 ═══════
FIXED_SEED = random.randint(0, 999999)
print(f"V5d seed={FIXED_SEED} 1Hz凸多边形+8m圆切割", flush=True)

# 障碍物
random.seed(FIXED_SEED)
obs = []
road = set()
for y in range(TRACK_ARR.shape[0]):
    for x in range(TRACK_ARR.shape[1]):
        v = TRACK_ARR[y, x] if TRACK_ARR.ndim == 2 else int(np.mean(TRACK_ARR[y, x, :3]))
        if 50 <= v <= 200: road.add((x, y))
for _ in range(200):
    if len(obs) >= 12 or not road: break
    px, py = random.sample(sorted(road), 1)[0]
    wx, wy = px / PIX_PER_M, py / PIX_PER_M
    if 2 < wx < 98 and 2 < wy < 98:
        if not any(math.hypot(wx - ox, wy - oy) < 3.0 for ox, oy in obs):
            if math.hypot(wx - 3, wy - 3) > 5 and math.hypot(wx - FINISH[0], wy - FINISH[1]) > 5:
                obs.append((wx, wy))
print(f"[OBS] {len(obs)}", flush=True)

OBS_XML = "".join(
    f'<body name="o{i}" pos="{x:.1f} {y:.1f} 2.0"><geom type="cylinder" size="1.0 2.0" rgba="0.9 0.2 0.2 0.9"/></body>'
    for i, (x, y) in enumerate(obs)
)
xml = f'<mujoco><compiler angle="radian"/><option timestep="0.005"/><visual><global offwidth="1280" offheight="720"/></visual><asset><hfield name="t" size="50.0 50.0 4.0 2.0" file="{MAP}"/></asset><worldbody><light pos="50 50 80" dir="0 0 -1"/><body mocap="true" pos="3.0 95.0 2"><geom type="sphere" size="1.5" rgba="0.2 1.0 0.2 0.8"/></body>{OBS_XML}<geom type="hfield" hfield="t" pos="50 50 0.0" rgba="0.25 0.30 0.35 1.0" friction="0 0 0"/><body name="bot" pos="3.0 3.0 0.5"><joint type="slide" axis="1 0 0" damping="0"/><joint type="slide" axis="0 1 0" damping="0"/><geom type="cylinder" size="0.5 0.5" rgba="1 0.3 0 1" friction="0 0 0"/></body></worldbody></mujoco>'

m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mv = Mover(m, d)
d.qpos[0] = 3.0
d.qpos[1] = 3.0
step = 0
t0 = time.time()
target_gate = None
MAX_STEPS = 100000

print("START", flush=True)
while step < MAX_STEPS:
    bx, by = d.qpos[0], d.qpos[1]
    if math.hypot(bx - FINISH[0], by - FINISH[1]) < 3.0:
        print(f"\nARRIVED @({bx:.1f},{by:.1f}) step={step}", flush=True)
        break
    # 1Hz规划
    if step % PLAN_INTERVAL == 0:
        pts = scan_walls(bx, by)
        gates = build_polygon_gates(pts, bx, by)
        gate = pick_gate_right(gates, mv.yaw)
        if gate:
            target_gate = gate
            print(f"  [GATE] [{step}] ->({gate[0]:.1f},{gate[1]:.1f}) gates={len(gates)}", flush=True)
        else:
            target_gate = FINISH
            print(f"  [NOGATE] [{step}] ->finish", flush=True)
    # move
    if target_gate:
        mv.step(target_gate[0], target_gate[1], step)
        if math.hypot(target_gate[0] - bx, target_gate[1] - by) < ARRIVE_THRESH:
            target_gate = None
    else:
        mv.step(FINISH[0], FINISH[1], step)
    step += 1
    if step % 2000 == 0:
        print(f"  ... step={step} @({bx:.1f},{by:.1f})", flush=True)

print(f"done: step={step} t={time.time() - t0:.1f}s bounce={mv.bounce}", flush=True)
