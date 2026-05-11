#!/usr/bin/env python3
"""bounce_obs — V8碰撞机器人 + 卡住检测 + 视角跟踪"""
import sys, os, math, random, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mujoco, mujoco.viewer, numpy as np, cv2
from simtrack import map as simmap

# ── 障碍物 ──
def gen_centerline():
    pts = []
    y0 = 2.5
    for seg in range(10):
        y = y0 + seg * 5.0
        x0, x1 = (5.0, 45.0) if seg % 2 == 0 else (45.0, 5.0)
        for j in range(10):
            pts.append((x0 + (j/9.0)*(x1-x0), y))
    right = [(46.5, 3.75), (47.5, 5.0), (46.5, 6.25)]
    for gy in range(5):
        for mx, my in right: pts.append((mx, my + gy*10.0))
    left = [(3.5, 8.75), (2.5, 10.0), (3.5, 11.25)]
    for gy in range(4):
        for mx, my in left: pts.append((mx, my + gy*10.0))
    return pts

rng = random.Random()
obstacles = []
cl = gen_centerline()
idx = 0
while idx < len(cl):
    cx, cy = cl[idx]
    wx, wy = simmap.maze_to_world(cx, cy)
    offset_y = rng.uniform(-2.0, 2.0)
    obstacles.append((wx, wy + offset_y))
    idx += rng.randint(3, 8)

obs_xml = ""
for i, (wx, wy) in enumerate(obstacles):
    obs_xml += (f'<body name="obs{i}" pos="{wx:.1f} {wy:.1f} 2.0">'
                f'<geom type="cylinder" size="1.0 2.0" rgba="0.9 0.2 0.2 0.9"/>'
                f'</body>\n')
print(f"障碍物: {len(obstacles)} 个", flush=True)

# ── hfield 碰撞检测 ──
hf = simmap.load()
def sample(wx, wy):
    mx, my = wx / simmap.SCALE, wy / simmap.SCALE
    px, py = int(mx * simmap.PIX_PER_M), simmap.HF_RES - 1 - int(my * simmap.PIX_PER_M)
    if 0 <= px < simmap.HF_RES and 0 <= py < simmap.HF_RES:
        return int(hf[py, px])
    return -1

def detect_wall(wx, wy, radius=0.6):
    for dy in np.arange(-radius, radius + 0.01, 0.15):
        max_dx = np.sqrt(max(0, radius**2 - dy**2))
        for dx in np.arange(-max_dx, max_dx + 0.01, 0.15):
            if sample(wx + dx, wy + dy) != simmap.ROAD_PIX:
                return True
    return False

def detect_obstacle(wx, wy, ox, oy, r=1.0):
    """检测是否与障碍物碰撞 (中心ox,oy, 半径r)"""
    return math.hypot(wx - ox, wy - oy) < (r + 0.55)

SPEED = 2.0
xml = f"""<mujoco>
<compiler angle="radian"/><option timestep="0.005"/>
<visual><global offwidth="1280" offheight="720"/></visual>
<asset><hfield name="track" size="50.0 50.0 4.0 2.0" file="{simmap.MAP_PATH}"/></asset>
<worldbody>
<light pos="50 50 80" dir="0 0 -1"/>
{obs_xml}
<geom type="hfield" hfield="track" pos="50 50 0.0" rgba="0.25 0.30 0.35 1.0"/>
<body name="bot" pos="6 6 1.5">
  <joint type="slide" axis="1 0 0" damping="0"/>
  <joint type="slide" axis="0 1 0" damping="0"/>
  <geom type="cylinder" size="0.5 0.5" rgba="1 0.3 0 1"/>
</body>
</worldbody>
</mujoco>"""

m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
d.qpos[0] = 6; d.qpos[1] = 6
mujoco.mj_forward(m, d)

yaw = 0.0
bounce = 0; step = 0
last_pos = (6.0, 6.0)
stuck_count = 0
t0 = time.time()
print(f"起点: (6,6) speed={SPEED}", flush=True)

with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
    v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.distance = 30; v.cam.elevation = -35; v.cam.azimuth = 90
    
    while v.is_running():
        bx, by = d.qpos[0], d.qpos[1]
        v.cam.lookat[:] = np.array([bx, by, 1.0], dtype=np.float64)
        
        # 卡住检测: 100步移动<0.1m → 转向
        dx = bx - last_pos[0]; dy = by - last_pos[1]
        if math.hypot(dx, dy) < 0.001:
            stuck_count += 1
        else:
            stuck_count = 0
        last_pos = (bx, by)
        
        if stuck_count > 100 or step % 500 == 0:
            if stuck_count > 100:
                yaw = random.uniform(0, 2 * math.pi)
                d.qvel[:] = 0
                bounce += 1
                print(f"STUCK#{bounce} step={step} ({bx:.1f},{by:.1f}) yaw={yaw:.2f}", flush=True)
                stuck_count = 0
        
        vx = np.cos(yaw) * SPEED
        vy = np.sin(yaw) * SPEED
        nx = bx + vx * m.opt.timestep
        ny = by + vy * m.opt.timestep
        
        # 墙碰撞 + 障碍物碰撞
        coll = detect_wall(nx, ny, 0.55)
        if not coll:
            for ox, oy in obstacles:
                if detect_obstacle(nx, ny, ox, oy, 1.0):
                    coll = True
                    break
        
        if coll:
            yaw = random.uniform(0, 2 * math.pi)
            d.qvel[:] = 0
            bounce += 1
            if bounce <= 10 or bounce % 50 == 0:
                print(f"BOUNCE#{bounce} step={step} ({bx:.1f},{by:.1f})", flush=True)
        else:
            d.qvel[0] = vx
            d.qvel[1] = vy
        
        mujoco.mj_step(m, d)
        step += 1
        v.sync()
        
        if step == 1 or step % 500 == 0:
            print(f"[{step}] ({bx:.1f},{by:.1f}) yaw={yaw:.2f} bounces={bounce}", flush=True)
    
    print(f"done: step={step} bounces={bounce} time={time.time()-t0:.1f}s", flush=True)
