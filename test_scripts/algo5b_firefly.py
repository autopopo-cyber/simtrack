#!/usr/bin/env python3
"""萤火算法 V5b — 右手原则: 激光→边界→门
无体素 无A* 无UNKNOWN — 只有WALL边和GATE边
右手原则: 从当前朝向顺时针扫第一条GATE=门, 冲中点
"""
import sys, os, math, time, random
import numpy as np
import mujoco, mujoco.viewer
from PIL import Image

# ═══════════ 参数 ═══════════
MAP = os.path.expanduser("~/workspace/simtrack/confirmed/track_clean.png")
SCAN_DIR = os.path.expanduser("~/workspace/simtrack/scans")
os.makedirs(SCAN_DIR, exist_ok=True)

VOXEL = 0.1; SPEED = 5.0; YAW_RATE = 6.0
LIDAR_RANGE = 15.0; LIDAR_STEPS = int(LIDAR_RANGE/VOXEL); LIDAR_RAYS = 120
GATE_GAP = 2.0  # 相邻命中点间距>2m=门
RENDER_SKIP = 100; PLAN_INTERVAL = 200  # 1Hz规划
BOUNCE_FORCE_DURATION = 0.3
STUCK_TIMEOUT = 300; STUCK_DIST_THRESH = 0.5
FINISH = (3.0, 95.0)
FIXED_SEED = random.randint(0, 999999)

UNKNOWN, FREE, WALL = 0, 1, 2
grid = {}; _cnt = {FREE:0, WALL:0}

def gget(vx,vy): return grid.get((vx,vy), UNKNOWN)
def gset(vx,vy,val):
    global _cnt
    old = gget(vx,vy)
    if old != UNKNOWN: _cnt[old] -= 1
    grid[(vx,vy)] = val; _cnt[val] = _cnt.get(val,0) + 1

def is_obstacle_world(wx, wy):
    vx, vy = int(wx/VOXEL), int(wy/VOXEL)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if gget(vx+dx, vy+dy) == WALL: return True
    return False

# ═══════════ 激光 → 边 ═══════════
def scan(bx, by):
    """120线激光 → 标记WALL, 返回命中点 [(角度, x, y, hit_wall), ...]"""
    hits = []
    for a in np.linspace(0, 2*math.pi, LIDAR_RAYS):
        cos_a, sin_a = math.cos(a), math.sin(a)
        prev_vx, prev_vy = int(bx/VOXEL), int(by/VOXEL)
        hit_wall = False; lx, ly = bx, by
        for step_i in range(1, LIDAR_STEPS+1):
            wx = bx + cos_a*step_i*VOXEL; wy = by + sin_a*step_i*VOXEL
            vx, vy = int(wx/VOXEL), int(wy/VOXEL)
            if is_obstacle_world(wx, wy):
                gset(vx,vy,WALL); gset(prev_vx,prev_vy,WALL)
                hit_wall = True; lx, ly = wx, wy; break
            if gget(vx,vy) == UNKNOWN: gset(vx,vy,FREE)
            prev_vx, prev_vy = vx, vy; lx, ly = wx, wy
        hits.append((a, lx, ly, hit_wall))
    hits.sort(key=lambda h: h[0])
    return hits

def build_edges(hits):
    """命中点→多边形边: WALL(间隙小) / GATE(间隙大)"""
    edges = []
    for i in range(len(hits)):
        j = (i+1) % len(hits)
        a1, x1, y1, hw1 = hits[i]
        a2, x2, y2, hw2 = hits[j]
        gap = math.hypot(x2-x1, y2-y1)
        if gap > GATE_GAP or (not hw1 and not hw2):
            edges.append((x1, y1, x2, y2, 'GATE', gap))
        else:
            edges.append((x1, y1, x2, y2, 'WALL', gap))
    return edges

def merge_edges(edges):
    """合并相邻同类型边"""
    if not edges: return []
    merged = []; i = 0; n = len(edges)
    while i < n:
        typ = edges[i][4]; j = i
        while j < n and edges[j][4] == typ: j += 1
        group = edges[i:j]
        if len(group) == 1:
            merged.append(group[0])
        else:
            merged.append((group[0][0], group[0][1], group[-1][2], group[-1][3], typ,
                          math.hypot(group[-1][2]-group[0][0], group[-1][3]-group[0][1])))
        i = j
    return merged

def find_gate_right_hand(merged, robot_yaw):
    """右手原则: 从当前yaw顺时针找第一条GATE边 → 返回门中点"""
    if not merged: return None
    # 将yaw归一化到[0, 2π)
    yaw_norm = robot_yaw % (2*math.pi)
    # 按角度排序所有边的中点 — 顺时针(角度递减)
    gate_candidates = []
    for e in merged:
        if e[4] != 'GATE': continue
        mx = (e[0] + e[2]) / 2; my = (e[1] + e[3]) / 2
        ang = math.atan2(my, mx)  # 以机器人为原点的角度
        # 顺时针: 从yaw往下找 (角度递减)
        rel = (yaw_norm - ang) % (2*math.pi)  # 顺时针偏差 [0, 2π)
        gate_candidates.append((rel, mx, my, e))
    if not gate_candidates: return None
    # 选顺时针偏差最小的（第一个右手边的门）
    gate_candidates.sort(key=lambda x: x[0])
    rel, mx, my, e = gate_candidates[0]
    return mx, my, e[5]

# ═══════════ Mover ═══════════
class Mover:
    def __init__(self, m, d):
        self.m, self.d = m, d
        self.yaw = math.pi/2  # 初始朝北(朝向y+)
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
            if math.hypot(bx-self.stuck_x, by-self.stuck_y) < STUCK_DIST_THRESH:
                self._bounce(90, 180)
            self.stuck_t = step_no; self.stuck_x = bx; self.stuck_y = by

        if self.force > 0:
            self.force -= 1; self.d.qvel[0] = vx; self.d.qvel[1] = vy
        elif self.escaping:
            self.escaping = False; self.d.qvel[0] = vx; self.d.qvel[1] = vy
        else:
            self.d.qvel[0] = vx; self.d.qvel[1] = vy
        mujoco.mj_step(self.m, self.d)

    def _bounce(self, lo, hi):
        if not self.escaping:
            self.bounce += 1; self.escaping = True
            if self.bounce % 5 == 0: print(f"  [BOUNCE] #{self.bounce}", flush=True)
        deg = random.uniform(lo, hi)*random.choice([-1,1])
        self.yaw += math.radians(deg); self.d.qvel[:] = 0
        self.force = int(BOUNCE_FORCE_DURATION/(SPEED*self.m.opt.timestep))

# ═══════════ 障碍物 ═══════════
def load_track():
    img = Image.open(MAP); arr = np.array(img)
    return arr, arr.shape[1], arr.shape[0]

def gen_random_obstacles(arr, w, h, seed, n=12):
    random.seed(seed); obs = []
    road_pix = set()
    for y in range(h):
        for x in range(w):
            if arr.ndim == 2:
                if 50 <= arr[y,x] <= 200: road_pix.add((x,y))
            elif len(arr.shape) == 3 and arr.shape[2] >= 3:
                r,g,b = arr[y,x,:3] if arr.shape[2]>=3 else (255,255,255)
                if 50 <= int(r) <= 200 and 50 <= int(g) <= 200: road_pix.add((x,y))
    for _ in range(n*5):
        if len(obs) >= n: break
        if not road_pix: break
        px, py = random.sample(sorted(road_pix), 1)[0]
        wx, wy = px/PIX_PER_M, py/PIX_PER_M
        if 2 < wx < 98 and 2 < wy < 98:
            if not any(math.hypot(wx-ox, wy-oy) < 3.0 for ox,oy in obs):
                if math.hypot(wx-3, wy-3) > 5 and math.hypot(wx-FINISH[0], wy-FINISH[1]) > 5:
                    obs.append((wx, wy))
    return obs

PIX_PER_M = 40

# ═══════════ MuJoCo ═══════════
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

# ═══════════ 主循环 ═══════════
print(f"━━━ 萤火 V5b 右手原则 ━━━ {LIDAR_RAYS}线 gap>{GATE_GAP}m=门 ━━━", flush=True)

track_arr, tw, th = load_track()
obs_world = gen_random_obstacles(track_arr, tw, th, FIXED_SEED, 12)
print(f"[INIT] 起点(3,3) 障碍{len(obs_world)}个 seed={FIXED_SEED}", flush=True)

xml = build_xml(obs_world)
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mv = Mover(m, d)
d.qpos[0] = 3.0; d.qpos[1] = 3.0

# 初始扫描
for _ in range(50): scan(3.0, 3.0)
print(f"[OK] FREE={_cnt[FREE]} WALL={_cnt[WALL]}", flush=True)

with mujoco.viewer.launch_passive(m, d) as viewer:
    viewer.cam.azimuth = -90; viewer.cam.elevation = -45; viewer.cam.lookat = [50, 50, 0]
    step = 0; t0 = time.time()
    target_gate = None; no_gate_count = 0; stuck_count = 0

    while viewer.is_running():
        bx, by = d.qpos[0], d.qpos[1]
        vx, vy = int(bx/VOXEL), int(by/VOXEL)
        if gget(vx, vy) == UNKNOWN: gset(vx, vy, FREE)

        # 终点检测
        if math.hypot(bx-FINISH[0], by-FINISH[1]) < 3.0:
            print(f"\n  ★ ARRIVED! @({bx:.1f},{by:.1f}) step={step}", flush=True)
            break

        # 激光
        if step % 20 == 0: scan(bx, by)

        # 1Hz: 右手原则找门
        if step % PLAN_INTERVAL == 0:
            hits = scan(bx, by)
            edges = build_edges(hits)
            merged = merge_edges(edges)
            gate = find_gate_right_hand(merged, mv.yaw)

            n_wall = sum(1 for e in merged if e[4]=='WALL')
            n_gate = sum(1 for e in merged if e[4]=='GATE')

            if gate:
                gx, gy, gap = gate
                target_gate = (gx, gy)
                no_gate_count = 0
                print(f"  [GATE] [{step}] →({gx:.1f},{gy:.1f}) gap={gap:.1f}m wall={n_wall} gate={n_gate}", flush=True)
            else:
                no_gate_count += 1
                target_gate = None
                print(f"  [NOGATE] [{step}] cnt={no_gate_count}", flush=True)
                # 无门时朝终点走
                target_gate = FINISH

        # 运动
        if target_gate:
            tx, ty = target_gate
            mv.step(tx, ty, step)
            if math.hypot(tx-bx, ty-by) < 1.0:
                target_gate = None  # 到达门, 下轮重新规划
        else:
            mv.step(FINISH[0], FINISH[1], step)

        step += 1
        if step % 2000 == 0:
            print(f"  ... step={step} F={_cnt[FREE]} W={_cnt[WALL]}", flush=True)
        if step % RENDER_SKIP == 0: viewer.sync()

    print(f"done: step={step} t={time.time()-t0:.1f}s bounce={mv.bounce}", flush=True)
