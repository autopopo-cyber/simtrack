#!/usr/bin/env python3
"""2D圆柱体 + track_hd蛇形赛道 + 随机障碍 + Tangent Arc快速避障"""
import os, sys, time, random, math, numpy as np, mujoco, mujoco.viewer

sys.path.insert(0, os.path.expanduser("~/navigation"))
from tangent_arc_planner import TangentArcPlanner, TangentArcConfig

MAP_SZ = 50.0; ROAD_W = 5.0; TURN_R = 5.0; HF_RES = 2000
START_X, START_Y = 3.0, 47.0
SIM_DT = 0.008
ROBOT_R = 0.35; MAX_SPEED = 1.5; N_OBS = 30; OBS_R = 0.3

# ── 赛道中心线 (trackgen_v2: 10段40m直道 + R=5m U型弯) ──
cx, cy = [], []
x, y = START_X, START_Y
for i in range(10):
    if i % 2 == 0:
        for j in range(80): cx.append(x + j*0.5); cy.append(y)
        x = cx[-1]
        for j in range(1, int(math.pi*TURN_R/0.25)+1):
            a = math.pi/2*j/(math.pi*TURN_R/0.25)
            cx.append(x + TURN_R*(1-math.cos(a)))
            cy.append(y - TURN_R*math.sin(a))
        x = cx[-1]; y = cy[-1] - TURN_R*2
    else:
        for j in range(80): cx.append(x - j*0.5); cy.append(y)
        x = cx[-1]
        for j in range(1, int(math.pi*TURN_R/0.25)+1):
            a = math.pi/2*j/(math.pi*TURN_R/0.25)
            cx.append(x - TURN_R*(1-math.cos(a)))
            cy.append(y - TURN_R*math.sin(a))
        x = cx[-1]; y = cy[-1] - TURN_R*2
cx, cy = np.array(cx), np.array(cy)
cum_d = np.insert(np.cumsum(np.hypot(np.diff(cx), np.diff(cy))), 0, 0)
track_len = cum_d[-1]

# Waypoints: 每8m
waypoints, nd = [], 8.0
for i in range(len(cx)):
    if cum_d[i] >= nd: waypoints.append((cx[i], cy[i])); nd += 8.0
if not waypoints or waypoints[-1] != (cx[-1], cy[-1]): waypoints.append((cx[-1], cy[-1]))
print(f"Track: {track_len:.0f}m, {len(waypoints)} waypoints")

# ── 随机障碍物 (避开赛道中心线) ──
rng = random.Random(42)
obstacles = []
for _ in range(N_OBS):
    while True:
        ox = rng.uniform(2, MAP_SZ-2)
        oy = rng.uniform(2, MAP_SZ-2)
        # 离中心线至少2m
        min_d = min(((ox-cx[i])**2+(oy-cy[i])**2)**0.5 for i in range(0,len(cx),20))
        if min_d > 2.0:
            obstacles.append((ox, oy, OBS_R)); break

# ── 构建场景 ──
obs_xml = ""
for i, (ox, oy, r) in enumerate(obstacles):
    obs_xml += f'<body name="obs_{i}" pos="{ox} {oy} {r}">'
    obs_xml += f'<geom type="cylinder" size="{r} {r}" rgba="0.9 0.3 0.3 0.8"/></body>\n'

scene = f"""<mujoco>
  <compiler angle="radian"/>
  <option timestep="{SIM_DT}"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset>
    <hfield name="track" size="25.0 25.0 4.0 2.0" file="/tmp/track_hd.png"/>
    <material name="track_vis" rgba="0.25 0.30 0.35 1.0"/>
    <material name="invis" rgba="0.25 0.30 0.35 0.0"/>
  </asset>
  <worldbody>
    <light pos="25 25 80" dir="0 0 -1" diffuse="1.5 1.5 1.5" specular="0.5 0.5 0.5"/>
    <geom type="hfield" hfield="track" pos="25 25 0.0" material="track_vis" contype="0" conaffinity="0"/>
    <geom type="plane" size="0 0 0.05" material="invis"/>
    <body name="robot" pos="{START_X} {START_Y} 0.5">
      <joint name="x" type="slide" axis="1 0 0" damping="0"/>
      <joint name="y" type="slide" axis="0 1 0" damping="0"/>
      <joint name="yaw" type="hinge" axis="0 0 1" damping="0"/>
      <geom type="cylinder" size="{ROBOT_R} 0.3" rgba="0.2 0.6 0.2 0.9"/>
    </body>
    {obs_xml}
  </worldbody>
</mujoco>"""

with open("/home/qin/cyl_snake_scene.xml", "w") as f: f.write(scene)

# ── Planner ──
planner = TangentArcPlanner(TangentArcConfig(
    robot_radius=ROBOT_R, max_speed=MAX_SPEED, min_speed=0.1,
    max_yaw_rate=2.0, goal_tolerance=1.5,
    arc_samples=15, safety_margin=0.2, predict_time=4.0))
LIDAR_RAYS = 72; LIDAR_RANGE = 8.0

def lidar_scan(m, d, bx, by, yaw):
    """2D lidar from robot center"""
    pts = []; gid = np.array([-1], np.int32)
    start = np.array([bx, by, 0.3])
    for i in range(LIDAR_RAYS):
        a = yaw + 2*np.pi*i/LIDAR_RAYS
        dw = np.array([np.cos(a), np.sin(a), 0.0])
        dist = mujoco.mj_ray(m, d, start, dw, None, 1, -1, gid)
        if gid[0] >= 0 and 0 < dist < LIDAR_RANGE:
            h = start + dw*dist
            if h[2] > 0.1 and dist > ROBOT_R+0.05:  # 排除自身
                pts.append((h[0], h[1]))
    return pts

tmp_path = os.path.expanduser("~/unitree_rl_gym/resources/robots/g1_description/cyl_snake_scene.xml")
m = mujoco.MjModel.from_xml_path("/home/qin/cyl_snake_scene.xml")
d = mujoco.MjData(m)
d.qpos[0:2] = [START_X, START_Y]
print(f"Loaded: {N_OBS} obstacles, ready.")

cnt = 0; t0 = time.time()
current_wp = 0; goal = waypoints[0]
vx, vy = 0.0, 0.0

with mujoco.viewer.launch_passive(m, d) as v:
    v.cam.azimuth = 90; v.cam.elevation = -30; v.cam.distance = 30
    v.cam.lookat = (25, 25, 0)

    while v.is_running():
        bx, by, yaw = d.qpos[0], d.qpos[1], d.qpos[2]
        dist_to_goal = np.hypot(bx-goal[0], by-goal[1])
        
        # Lidar + plan
        lpts = lidar_scan(m, d, bx, by, yaw)
        vc, wc, _ = planner.plan(bx, by, yaw, vx, vy, goal, lpts)
        
        # WP transition
        if dist_to_goal < planner.cfg.goal_tolerance and current_wp < len(waypoints)-1:
            current_wp += 1; goal = waypoints[current_wp]
            elapsed = time.time() - t0
            print(f"[{cnt*SIM_DT:.0f}s] WP{current_wp} ({goal[0]:.0f},{goal[1]:.0f}) wall={elapsed:.0f}s")
        
        if current_wp >= len(waypoints)-1 and dist_to_goal < 2.0:
            avg = track_len/(cnt*SIM_DT)
            print(f"\n✓ ARRIVED sim={cnt*SIM_DT:.0f}s avg={avg:.2f}m/s")
            break
        
        # Velocity control
        vx, vy = vc * np.cos(yaw), vc * np.sin(yaw)
        d.qvel[0:2] = [vx, vy]
        d.qvel[2] = wc
        mujoco.mj_step(m, d)
        cnt += 1
        
        if cnt % 250 == 0:
            print(f"[{cnt*SIM_DT:.0f}s] ({bx:.1f},{by:.1f}) wp{current_wp} d={dist_to_goal:.1f}m "
                  f"v={vc:.1f} lidar={len(lpts)}")

print(f"\n=== sim={cnt*SIM_DT:.0f}s wall={time.time()-t0:.0f}s ===")
