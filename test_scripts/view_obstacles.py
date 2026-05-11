#!/usr/bin/env python3
"""test_scripts/view_obstacles.py — 中心线115球 + 分段随机障碍物"""
import sys, os, math, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mujoco, mujoco.viewer, numpy as np, time
from simtrack import map as simmap

# ── 115个中心线坐标 (迷宫坐标) ──
centerline = []

# 10段直道
y0 = 2.5
for seg in range(10):
    y = y0 + seg * 5.0
    x0, x1 = (5.0, 45.0) if seg % 2 == 0 else (45.0, 5.0)
    for j in range(10):
        t = j / 9.0
        centerline.append((x0 + t * (x1 - x0), y))

# 右侧U型弯 (5组)
right_template = [(46.5, 3.75), (47.5, 5.0), (46.5, 6.25)]
for gy in range(5):
    for mx, my in right_template:
        centerline.append((mx, my + gy * 10.0))

# 左侧U型弯 (4组)
left_template = [(3.5, 8.75), (2.5, 10.0), (3.5, 11.25)]
for gy in range(4):
    for mx, my in left_template:
        centerline.append((mx, my + gy * 10.0))

assert len(centerline) == 115, f"期望115, 实际{len(centerline)}"
print(f"中心线: {len(centerline)} 点", flush=True)

# ── 黄球XML ──
balls_xml = ""
for mx, my in centerline:
    wx, wy = simmap.maze_to_world(mx, my)
    balls_xml += f'<body mocap="true" pos="{wx:.1f} {wy:.1f} 1.5"><geom type="sphere" size="0.3" rgba="1 0.9 0 0.9"/></body>\n'

# ── 分段随机障碍物 ──
rng = random.Random(42)
obstacles = []  # (mx, my) 迷宫坐标
idx = 0
while idx < len(centerline):
    cx, cy = centerline[idx]
    
    # Y轴随机偏移 ±2m
    offset_y = rng.uniform(-2.0, 2.0)
    ox = cx
    oy = cy + offset_y
    obstacles.append((ox, oy))
    
    # 下一个: index += random(3~8)
    idx += rng.randint(3, 8)

# ── 障碍物XML (红柱, 直径1m, 高2m) ──
obs_xml = ""
for i, (mx, my) in enumerate(obstacles):
    wx, wy = simmap.maze_to_world(mx, my)
    obs_xml += (f'<body name="obs{i}" pos="{wx:.1f} {wy:.1f} 1.0">'
                f'<geom type="cylinder" size="0.5 1.0" rgba="0.9 0.2 0.2 0.9"/></body>\n')

print(f"障碍物: {len(obstacles)} 个 (步长3~8, Y偏移±2m)", flush=True)

# ── 场景 ──
xml = f"""<mujoco>
<compiler angle="radian"/><option timestep="0.005"/>
<visual><global offwidth="1280" offheight="720"/></visual>
<asset><hfield name="track" size="50.0 50.0 4.0 2.0" file="{simmap.MAP_PATH}"/></asset>
<worldbody>
<light pos="50 50 80" dir="0 0 -1"/>
{balls_xml}
{obs_xml}
<geom type="hfield" hfield="track" pos="50 50 0.0" rgba="0.25 0.30 0.35 1.0"/>
</worldbody>
</mujoco>"""

m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
    v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.distance = 60; v.cam.elevation = -60; v.cam.azimuth = 90
    v.cam.lookat[:] = np.array([50, 25, 0])
    print("viewer ready — 黄球=中心线, 红柱=障碍物", flush=True)
    while v.is_running(): mujoco.mj_step(m, d); v.sync(); time.sleep(0.01)
