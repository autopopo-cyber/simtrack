#!/usr/bin/env python3
"""V5c: 激光8m → 墙段+门段 → 右手原则冲门中点
门=相邻墙段之间的连线
"""
import math, random, sys, heapq
import numpy as np
import mujoco
from PIL import Image

# ═══════ 参数 ═══════
MAP = "/home/qin/workspace/simtrack/confirmed/track_clean.png"
PIX_PER_M = 40
LIDAR_RANGE = 8.0       # 扫描半径8m
LIDAR_RAYS = 120
SPEED = 5.0; YAW_RATE = 6.0
ARRIVE_THRESH = 1.0
RENDER_SKIP = 100; PLAN_INTERVAL = 200
BOUNCE_FORCE_DURATION = 0.3; STUCK_TIMEOUT = 300; STUCK_DIST_THRESH = 0.5
FINISH = (3.0, 95.0)

# PNG障碍检测
TRACK_ARR = None; TW = TH = 0
def load_png():
    global TRACK_ARR, TW, TH
    img = Image.open(MAP); TRACK_ARR = np.array(img)
    if TRACK_ARR.ndim == 2: TH, TW = TRACK_ARR.shape
    else: TH, TW = TRACK_ARR.shape[:2]
load_png()

def png_is_wall(wx, wy):
    px, py = int(wx*PIX_PER_M), int(wy*PIX_PER_M)
    if px < 0 or px >= TW or py < 0 or py >= TH: return True
    v = TRACK_ARR[py, px] if TRACK_ARR.ndim == 2 else int(np.mean(TRACK_ARR[py, px, :3]))
    return v < 50

# ═══════ 激光 → 墙段+门段 ═══════
def scan_walls(bx, by):
    """激光8m → 返回墙段 [(start_angle, end_angle, mid_x, mid_y), ...]"""
    hits = []
    for i, a in enumerate(np.linspace(0, 2*math.pi, LIDAR_RAYS)):
        cs, sn = math.cos(a), math.sin(a)
        hit = False
        for d in np.arange(0.1, LIDAR_RANGE + 0.1, 0.1):
            wx, wy = bx + cs*d, by + sn*d
            if png_is_wall(wx, wy):
                hits.append((a, wx, wy, True)); hit = True; break
        if not hit:
            wx, wy = bx + cs*LIDAR_RANGE, by + sn*LIDAR_RANGE
            hits.append((a, wx, wy, False))
    hits.sort(key=lambda h: h[0])

    # 分组: 连续命中=墙段, 连续未命中=门段
    wall_segments = []
    i = 0; n = len(hits)
    while i < n:
        if not hits[i][3]:  # 未命中→跳过(门)
            i += 1; continue
        # 找到一段连续命中
        start = i
        while i < n and hits[i][3]: i += 1
        seg_hits = hits[start:i]
        if len(seg_hits) < 2: continue  # 太短忽略
        a_start = seg_hits[0][0]; a_end = seg_hits[-1][0]
        mid_x = sum(h[1] for h in seg_hits) / len(seg_hits)
        mid_y = sum(h[2] for h in seg_hits) / len(seg_hits)
        wall_segments.append((a_start, a_end, mid_x, mid_y))
    return wall_segments

def find_gates_from_walls(walls, bx, by, robot_yaw):
    """从墙段列表找门: 相邻墙段之间的连线→门, 右手原则选第一个"""
    if len(walls) < 1: return None
    gates = []
    n = len(walls)
    for i in range(n):
        j = (i+1) % n
        a1_end = walls[i][1]   # 第一段墙的结束角
        a2_start = walls[j][0]  # 下一段墙的开始角
        # 门中点: 两段墙端点之间的中点
        x1, y1 = walls[i][3], walls[i][2]  # 墙段中心(近似)
        # 取两端的实际端点更好——用墙弧端点
        xx1, yy1 = bx + math.cos(walls[i][1])*LIDAR_RANGE, by + math.sin(walls[i][1])*LIDAR_RANGE
        xx2, yy2 = bx + math.cos(walls[j][0])*LIDAR_RANGE, by + math.sin(walls[j][0])*LIDAR_RANGE
        gate_mx = (xx1 + xx2) / 2
        gate_my = (yy1 + yy2) / 2
        gate_ang = math.atan2(gate_my-by, gate_mx-bx)
        gates.append((gate_ang, gate_mx, gate_my))
    if not gates: return None

    # 右手原则: 从yaw顺时针找第一个门
    yaw_norm = robot_yaw % (2*math.pi)
    candidates = [( (yaw_norm - ga) % (2*math.pi), gx, gy) for ga, gx, gy in gates]
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1], candidates[0][2]

# ═══════ Mover ═══════
class Mover:
    def __init__(self, m, d):
        self.m, self.d = m, d
        self.yaw = math.pi/2  # 朝北
        self.speed = SPEED; self.bounce = 0
        self.force = 0; self.escaping = False
        self.stuck_t = 0; self.stuck_x = 0.0; self.stuck_y = 0.0

    def step(self, tx, ty, step_no):
        bx, by = self.d.qpos[0], self.d.qpos[1]
        dt = self.m.opt.timestep
        if not self.escaping:
            tgt_yaw = math.atan2(ty-by, tx-bx)
            err = (tgt_yaw - self.yaw + math.pi) % (2*math.pi) - math.pi
            dyaw = max(-YAW_RATE*dt, min(YAW_RATE*dt, err))
            self.yaw += dyaw; self.speed = SPEED
        vx = math.cos(self.yaw)*self.speed; vy = math.sin(self.yaw)*self.speed
        if step_no - self.stuck_t > STUCK_TIMEOUT:
            if math.hypot(bx-self.stuck_x, by-self.stuck_y) < STUCK_DIST_THRESH: self._bounce(90,180)
            self.stuck_t = step_no; self.stuck_x = bx; self.stuck_y = by
        if self.force > 0: self.force -= 1; self.d.qvel[0] = vx; self.d.qvel[1] = vy
        elif self.escaping: self.escaping = False; self.d.qvel[0] = vx; self.d.qvel[1] = vy
        else: self.d.qvel[0] = vx; self.d.qvel[1] = vy
        mujoco.mj_step(self.m, self.d)

    def _bounce(self, lo, hi):
        if not self.escaping: self.bounce += 1; self.escaping = True
        deg = random.uniform(lo, hi)*random.choice([-1,1])
        self.yaw += math.radians(deg); self.d.qvel[:] = 0
        self.force = int(BOUNCE_FORCE_DURATION/(SPEED*self.m.opt.timestep))

# ═══════ MuJoCo ═══════
def build_xml(obs_world):
    obs_xml = "".join(f'<body name="obs{i}" pos="{x:.1f} {y:.1f} 2.0"><geom type="cylinder" size="1.0 2.0" rgba="0.9 0.2 0.2 0.9"/></body>' for i,(x,y) in enumerate(obs_world))
    fin_xml = f'<body mocap="true" pos="{FINISH[0]:.1f} {FINISH[1]:.1f} 2"><geom type="sphere" size="1.5" rgba="0.2 1.0 0.2 0.8"/></body>'
    return f"""<mujoco>
  <compiler angle="radian"/><option timestep="0.005"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset><hfield name="track" size="50.0 50.0 4.0 2.0" file="{MAP}"/></asset>
  <worldbody>
    <light pos="50 50 80" dir="0 0 -1"/>
    {fin_xml}{obs_xml}
    <geom type="hfield" hfield="track" pos="50 50 0.0" rgba="0.25 0.30 0.35 1.0" friction="0 0 0"/>
    <body name="bot" pos="3.0 3.0 0.5">
      <joint type="slide" axis="1 0 0" damping="0"/>
      <joint type="slide" axis="0 1 0" damping="0"/>
      <geom type="cylinder" size="0.5 0.5" rgba="1 0.3 0 1" friction="0 0 0"/>
    </body>
  </worldbody>
</mujoco>"""

# ═══════ 障碍物 ═══════
def gen_random_obstacles(arr, w, h, seed, n=12):
    random.seed(seed); obs = []
    road_pix = set()
    for y in range(h):
        for x in range(w):
            v = arr[y,x] if arr.ndim==2 else int(np.mean(arr[y,x,:3]))
            if 50 <= v <= 200: road_pix.add((x,y))
    for _ in range(n*5):
        if len(obs) >= n or not road_pix: break
        px, py = random.sample(sorted(road_pix), 1)[0]
        wx, wy = px/PIX_PER_M, py/PIX_PER_M
        if 2 < wx < 98 and 2 < wy < 98:
            if not any(math.hypot(wx-ox, wy-oy) < 3.0 for ox,oy in obs):
                if math.hypot(wx-3, wy-3) > 5 and math.hypot(wx-FINISH[0], wy-FINISH[1]) > 5:
                    obs.append((wx, wy))
    return obs

# ═══════ 主循环 ═══════
FIXED_SEED = random.randint(0, 999999)
print(f"━━━ V5c 激光8m墙段+右手原则 ━━━ {LIDAR_RAYS}线 ━━━", flush=True)

track_arr, tw, th = Image.open(MAP), *[None]*2  # lazy
track_arr = np.array(track_arr)
obs_world = gen_random_obstacles(track_arr, tw if 'tw' in dir() else track_arr.shape[1],
                                  track_arr.shape[0], FIXED_SEED, 12)
print(f"[INIT] 起点(3,3) 障碍{len(obs_world)}个 seed={FIXED_SEED}", flush=True)

xml = build_xml(obs_world)
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mv = Mover(m, d)
d.qpos[0] = 3.0; d.qpos[1] = 3.0

with mujoco.viewer.launch_passive(m, d) as viewer:
    viewer.cam.azimuth = -90; viewer.cam.elevation = -45; viewer.cam.lookat = [50, 50, 0]
    step = 0; t0 = time.time(); target_gate = None

    while viewer.is_running():
        bx, by = d.qpos[0], d.qpos[1]

        # 终点
        if math.hypot(bx-FINISH[0], by-FINISH[1]) < 3.0:
            print(f"\n  ★ ARRIVED! @({bx:.1f},{by:.1f}) step={step}", flush=True); break

        # 规划
        if step % PLAN_INTERVAL == 0:
            walls = scan_walls(bx, by)
            gate = find_gates_from_walls(walls, bx, by, mv.yaw)
            if gate:
                target_gate = gate
                print(f"  [GATE] [{step}] →({gate[0]:.1f},{gate[1]:.1f}) walls={len(walls)}", flush=True)
            else:
                target_gate = FINISH

        # move
        if target_gate:
            mv.step(target_gate[0], target_gate[1], step)
            if math.hypot(target_gate[0]-bx, target_gate[1]-by) < ARRIVE_THRESH:
                target_gate = None
        else:
            mv.step(FINISH[0], FINISH[1], step)

        step += 1
        if step % 2000 == 0: print(f"  ... step={step}", flush=True)
        if step % RENDER_SKIP == 0: viewer.sync()

    print(f"done: step={step} t={time.time()-t0:.1f}s bounce={mv.bounce}", flush=True)
