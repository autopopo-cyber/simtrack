#!/usr/bin/env python3
"""test_scripts/view_centerline.py — 中心线，主人指定坐标"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mujoco, mujoco.viewer, numpy as np, time
from simtrack import map as simmap

balls = []  # (wx, wy) 世界坐标

# ── 10段直道，每段10球 ──
y0 = 2.5
for seg in range(10):
    y = y0 + seg * 5.0
    if seg % 2 == 0:  # 左→右
        x0, x1 = 5.0, 45.0
    else:              # 右→左
        x0, x1 = 45.0, 5.0
    for j in range(10):
        t = j / 9.0
        mx = x0 + t * (x1 - x0)
        my = y
        balls.append(simmap.maze_to_world(mx, my))

# ── 右侧U型弯 (5组, y+=10) ──
right_template = [(46.5, 3.75), (47.5, 5.0), (46.5, 6.25)]
for gy in range(5):
    for mx, my in right_template:
        balls.append(simmap.maze_to_world(mx, my + gy * 10.0))

# ── 左侧U型弯 (4组, y+=10) ──
left_template = [(3.5, 7.5), (2.5, 10.0), (3.5, 12.5)]
for gy in range(4):
    for mx, my in left_template:
        balls.append(simmap.maze_to_world(mx, my + gy * 10.0))

# ── 生成XML ──
balls_xml = ""
for wx, wy in balls:
    balls_xml += f'<body mocap="true" pos="{wx:.1f} {wy:.1f} 1.5"><geom type="sphere" size="0.3" rgba="1 0.9 0 0.9"/></body>\n'

print(f"中心线: {len(balls)} 个黄球 (10段×10 + 右弯5×3 + 左弯4×3)", flush=True)

xml = f"""<mujoco>
<compiler angle="radian"/><option timestep="0.005"/>
<visual><global offwidth="1280" offheight="720"/></visual>
<asset><hfield name="track" size="50.0 50.0 4.0 2.0" file="{simmap.MAP_PATH}"/></asset>
<worldbody>
<light pos="50 50 80" dir="0 0 -1"/>
{balls_xml}
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
