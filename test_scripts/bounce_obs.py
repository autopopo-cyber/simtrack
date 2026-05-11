#!/usr/bin/env python3
"""bounce_obs — V8碰撞机器人 + 可碰撞障碍物"""
import sys, os, math, random, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mujoco, mujoco.viewer, numpy as np, cv2
from simtrack import map as simmap

# ── 障碍物 (同前) ──
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
        for mx, my in right:
            pts.append((mx, my + gy*10.0))
    left = [(3.5, 8.75), (2.5, 10.0), (3.5, 11.25)]
    for gy in range(4):
        for mx, my in left:
            pts.append((mx, my + gy*10.0))
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
    px = int(mx * simmap.PIX_PER_M)
    py = simmap.HF_RES - 1 - int(my * simmap.PIX_PER_M)
    if 0 <= px < simmap.HF_RES and 0 <= py < simmap.HF_RES:
        return int(hf[py, px])
    return -1

def detect_collision(wx, wy, radius=0.6):
    for dy in np.arange(-radius, radius + 0.01, 0.15):
        max_dx = np.sqrt(max(0, radius**2 - dy**2))
        for dx in np.arange(-max_dx, max_dx + 0.01, 0.15):
            if sample(wx + dx, wy + dy) != simmap.ROAD_PIX:
                return True
    return False

# ── 场景 ──
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
bounce = 0
step = 0
t0 = time.time()

with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
    v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.distance = 40; v.cam.elevation = -40; v.cam.azimuth = 90
    print("viewer ready", flush=True)

    while v.is_running():
        bx, by = d.qpos[0], d.qpos[1]
        v.cam.lookat[:] = np.array([bx, by, 1.0], dtype=np.float64)

        vx = np.cos(yaw) * SPEED
        vy = np.sin(yaw) * SPEED
        nx = bx + vx * m.opt.timestep
        ny = by + vy * m.opt.timestep

        if detect_collision(nx, ny, 0.55):
            yaw = random.uniform(0, 2 * np.pi)
            d.qvel[:] = 0
            bounce += 1
            if bounce <= 5 or bounce % 20 == 0:
                print(f"BOUNCE#{bounce} step={step} ({bx:.1f},{by:.1f})", flush=True)
        else:
            d.qvel[0] = vx
            d.qvel[1] = vy

        mujoco.mj_step(m, d)
        step += 1
        v.sync()

    print(f"done: step={step} bounces={bounce} time={time.time()-t0:.1f}s", flush=True)
