#!/usr/bin/env python3
"""simtrack runner — 10检查点导航 + 中心线障碍生成 + 扇区寻路"""
import sys, os, math, random, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mujoco, mujoco.viewer, numpy as np
from simtrack import map as simmap
from simtrack.lidar import LidarSensor
from simtrack.nav import SectorNav

# ── 中心线 (隐藏, 仅供障碍生成) ──
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

def gen_obstacles(centerline, rng):
    obs = []
    idx = 0
    while idx < len(centerline):
        cx, cy = centerline[idx]
        wx, wy = simmap.maze_to_world(cx, cy)
        offset_y = rng.uniform(-2.0, 2.0)
        obs.append((wx, wy + offset_y))
        idx += rng.randint(3, 8)
    return obs

# ── 导航点: 最开始的10个检查点 ──
nav_wps = simmap.get_checkpoints_world()  # 11个, 含起点

# ═════════════════════════════
SPEED = 2.0
LIDAR_RAYS = 360
LIDAR_RANGE = 15.0
LIDAR_HEIGHT = 1.2
SAFE_DIST = 2.0
CP_RADIUS = 3.0

rng = random.Random()
obstacles = gen_obstacles(gen_centerline(), rng)
print(f"导航点: {len(nav_wps)} 检查点", flush=True)
print(f"障碍物: {len(obstacles)} 个", flush=True)

# ── 场景 ──
cp_xml = ""
for i, (wx, wy) in enumerate(nav_wps[1:], 1):
    cp_xml += f'<body mocap="true" pos="{wx:.1f} {wy:.1f} 2"><geom type="sphere" size="1.5" rgba="0.2 0.5 1 0.8"/></body>\n'

obs_xml = ""
for i, (wx, wy) in enumerate(obstacles):
    obs_xml += (f'<body name="obs{i}" pos="{wx:.1f} {wy:.1f} 2.0">'
                f'<geom type="cylinder" size="1.0 2.0" rgba="0.9 0.2 0.2 0.9"/>'
                f'</body>\n')

start_wx, start_wy = nav_wps[0]

xml = f"""<mujoco>
<compiler angle="radian"/><option timestep="0.005"/>
<visual><global offwidth="1280" offheight="720"/></visual>
<asset><hfield name="track" size="50.0 50.0 4.0 2.0" file="{simmap.MAP_PATH}"/></asset>
<worldbody>
<light pos="50 50 80" dir="0 0 -1"/>
{cp_xml}
{obs_xml}
<geom type="hfield" hfield="track" pos="50 50 0.0" rgba="0.25 0.30 0.35 1.0"/>
<body name="bot" pos="{start_wx} {start_wy} {LIDAR_HEIGHT+0.3}">
  <joint type="slide" axis="1 0 0" damping="0"/>
  <joint type="slide" axis="0 1 0" damping="0"/>
  <geom type="cylinder" size="0.5 0.5" rgba="1 0.3 0 1"/>
  <site name="lidar_top" pos="0 0 {LIDAR_HEIGHT}" size="0.02"/>
</body>
</worldbody>
</mujoco>"""

m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
d.qpos[0] = start_wx
d.qpos[1] = start_wy
mujoco.mj_forward(m, d)

lidar = LidarSensor(m, d, site_name="lidar_top", rays=LIDAR_RAYS, lines=1, range_m=LIDAR_RANGE, hz=10)
nav = SectorNav(n_sectors=36, safe_dist=SAFE_DIST, speed=SPEED, cp_radius=CP_RADIUS)
hf = simmap.load()

wp_idx = 0
step = 0
lidar_step = 0
t0 = time.time()
print(f"起点: ({start_wx:.1f},{start_wy:.1f})", flush=True)
print("viewer starting...", flush=True)

with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
    v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.distance = 40
    v.cam.elevation = -40
    v.cam.azimuth = 90
    
    while v.is_running() and wp_idx < len(nav_wps):
        bx, by = d.qpos[0], d.qpos[1]
        v.cam.lookat[:] = np.array([bx, by, 2.0], dtype=np.float64)
        
        lidar_step += 1
        if lidar_step % lidar.step_interval == 0:
            lidar.update(bx, by, 0.0)
        
        tx, ty = nav_wps[wp_idx]
        vx, vy, reached = nav.steer(bx, by, tx, ty, lidar.points_2d, hf)
        
        if reached:
            wp_idx += 1
            print(f"  ✓ CP{wp_idx-1} step={step} ({bx:.1f},{by:.1f})", flush=True)
            if wp_idx >= len(nav_wps):
                print(f"🏁 FINISH step={step} time={time.time()-t0:.1f}s bounces={nav.bounces}", flush=True)
                break
            continue
        
        d.qvel[0] = vx
        d.qvel[1] = vy
        mujoco.mj_step(m, d)
        step += 1
        v.sync()
        
        if step % 200 == 0:
            d_cp = math.hypot(tx - bx, ty - by)
            print(f"  [{step}] ({bx:.1f},{by:.1f}) → CP{wp_idx}/{len(nav_wps)} d={d_cp:.1f}", flush=True)
    
    t_elapsed = time.time() - t0
    print(f"done: step={step} cp={wp_idx}/{len(nav_wps)} time={t_elapsed:.1f}s bounces={nav.bounces}", flush=True)
