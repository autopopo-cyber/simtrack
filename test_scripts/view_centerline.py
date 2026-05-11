#!/usr/bin/env python3
"""test_scripts/view_centerline.py — 黄色小球显示中心线，3m间隔"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mujoco, mujoco.viewer, numpy as np, time
from simtrack import map as simmap

# ── 生成中心线 (迷宫坐标插值) ──
TURN_R = 4.0  # U型弯半径(m, 迷宫坐标)
STEP = 0.25   # 采样步长(m)

cps = simmap.CHECKPOINTS_MAZE  # 迷宫坐标
pts = []  # 中心线点 (迷宫坐标)

for i in range(len(cps) - 1):
    x0, y0 = cps[i]
    x1, y1 = cps[i + 1]
    
    # 判断方向: 从左(3)到右(47) 还是反过来
    going_right = x1 > x0
    
    # 直线段: 从弯道出口到下一个弯道入口
    turn_margin = TURN_R
    if going_right:
        straight_start = x0 + turn_margin
        straight_end = x1 - turn_margin
    else:
        straight_start = x0 - turn_margin
        straight_end = x1 + turn_margin
    
    # y 线性插值
    total_dx = abs(x1 - x0)
    dy_total = y1 - y0
    
    for x in np.arange(straight_start, straight_end + 0.01, STEP if going_right else -STEP):
        t = abs(x - x0) / total_dx if total_dx > 0 else 0
        y = y0 + t * dy_total
        pts.append((float(x), float(y)))
    
    # U型弯 (如果不是最后一段)
    if i < len(cps) - 2:
        cx = 47.0 if going_right else 3.0  # 弯心x
        cy = (y1 + cps[i+2][1]) / 2.0       # 弯心y (当前和下一个y的中点)
        
        sa = -math.pi/2 if going_right else math.pi/2
        ea = math.pi/2 if going_right else -math.pi/2
        n_arc = max(10, int(math.pi * TURN_R / STEP))
        
        for j in range(n_arc + 1):
            a = sa + (ea - sa) * j / n_arc
            pts.append((cx + TURN_R * math.cos(a), cy + TURN_R * math.sin(a)))

# 转为世界坐标
pts_world = [simmap.maze_to_world(x, y) for x, y in pts]

# 3m 间隔采样 → 黄色小球
balls_xml = ""
spacing = 3.0
cum = [0.0]
for i in range(1, len(pts_world)):
    dx = pts_world[i][0] - pts_world[i-1][0]
    dy = pts_world[i][1] - pts_world[i-1][1]
    cum.append(cum[-1] + math.hypot(dx, dy))
total = cum[-1]

d = 0.0
idx = 0
while d < total:
    while idx < len(cum) - 1 and cum[idx+1] < d:
        idx += 1
    if idx >= len(pts_world) - 1:
        break
    seg = cum[idx+1] - cum[idx]
    t = (d - cum[idx]) / seg if seg > 0 else 0
    wx = pts_world[idx][0] + t * (pts_world[idx+1][0] - pts_world[idx][0])
    wy = pts_world[idx][1] + t * (pts_world[idx+1][1] - pts_world[idx][1])
    balls_xml += f'<body mocap="true" pos="{wx:.1f} {wy:.1f} 1.5"><geom type="sphere" size="0.4" rgba="1 0.9 0 0.9"/></body>\n'
    d += spacing

print(f"中心线: {len(pts)} 点, {total:.0f}m, {int(total/spacing)} 个黄球", flush=True)

# ── MuJoCo 场景 ──
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
    print("viewer ready — 黄球=中心线(3m间隔)", flush=True)
    while v.is_running():
        mujoco.mj_step(m, d); v.sync(); time.sleep(0.01)
