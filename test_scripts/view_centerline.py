#!/usr/bin/env python3
"""test_scripts/view_centerline.py — 中心线3m黄球"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mujoco, mujoco.viewer, numpy as np, time
from simtrack import map as simmap

cps = simmap.CHECKPOINTS_MAZE  # 迷宫坐标

# 检查点之间直线插值
pts_maze = []
STEP = 0.1
for i in range(len(cps) - 1):
    x0, y0 = cps[i]
    x1, y1 = cps[i + 1]
    dist = math.hypot(x1 - x0, y1 - y0)
    n = max(2, int(dist / STEP))
    for j in range(n):
        t = j / (n - 1)
        pts_maze.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))

pts_world = [simmap.maze_to_world(x, y) for x, y in pts_maze]

# 3m 间隔黄球
balls = ""
cum = [0.0]
for i in range(1, len(pts_world)):
    cum.append(cum[-1] + math.hypot(
        pts_world[i][0] - pts_world[i-1][0],
        pts_world[i][1] - pts_world[i-1][1]))
total = cum[-1]

d = 0.0; idx = 0
while d < total:
    while idx < len(cum) - 1 and cum[idx+1] < d:
        idx += 1
    if idx >= len(pts_world) - 1: break
    seg = cum[idx+1] - cum[idx]
    t = (d - cum[idx]) / seg if seg > 0 else 0
    wx = pts_world[idx][0] + t * (pts_world[idx+1][0] - pts_world[idx][0])
    wy = pts_world[idx][1] + t * (pts_world[idx+1][1] - pts_world[idx][1])
    balls += f'<body mocap="true" pos="{wx:.1f} {wy:.1f} 1.5"><geom type="sphere" size="0.4" rgba="1 0.9 0 0.9"/></body>\n'
    d += 3.0

n_balls = balls.count("mocap")
print(f"中心线: {total:.0f}m, {n_balls} 个黄球(3m间隔)", flush=True)

xml = f"""<mujoco>
<compiler angle="radian"/><option timestep="0.005"/>
<visual><global offwidth="1280" offheight="720"/></visual>
<asset><hfield name="track" size="50.0 50.0 4.0 2.0" file="{simmap.MAP_PATH}"/></asset>
<worldbody>
<light pos="50 50 80" dir="0 0 -1"/>
{balls}
<geom type="hfield" hfield="track" pos="50 50 0.0" rgba="0.25 0.30 0.35 1.0"/>
</worldbody>
</mujoco>"""

m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
    v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.distance = 60; v.cam.elevation = -60; v.cam.azimuth = 90
    v.cam.lookat[:] = np.array([50, 25, 0])
    print("viewer ready", flush=True)
    while v.is_running(): mujoco.mj_step(m, d); v.sync(); time.sleep(0.01)
