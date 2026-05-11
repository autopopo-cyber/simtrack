#!/usr/bin/env python3
"""algo2_sector_nav — 扇区雷达导航 + 玩具车兜底

360° lidar → 36扇区 → 选目标方向最近的畅通扇区。
转速6rad/s，速度2→6渐进。
"""
import sys, os, math, time, random
import numpy as np
from PIL import Image
import mujoco, mujoco.viewer

MAP = os.path.expanduser("~/workspace/simtrack/confirmed/track_clean.png")
hf = np.array(Image.open(MAP))

SCALE = 2.0; HF_RES = 2000; PIX_PER_M = 40; ROAD_PIX = 128
DETECT_R = 0.5
SPEED_MIN = 2.0; SPEED_MAX = 6.0
YAW_RATE = 6.0          # 6 rad/s
CP_RADIUS = 3.0
N_SECTORS = 36           # 每10°一扇区
LIDAR_RAYS = 360         # 水平射线
LIDAR_RANGE = 12.0
SAFE_DIST = 2.0          # 扇区内点 > 此距离才算通畅

# ── 中心线 & 障碍物 ──
def gen_centerline():
    pts = []
    y0 = 2.5
    for seg in range(10):
        y = y0 + seg * 5.0
        x0, x1 = (5.0, 45.0) if seg % 2 == 0 else (45.0, 5.0)
        for j in range(10): pts.append((x0 + (j/9.0)*(x1-x0), y))
    for mx, my in [(46.5, 3.75), (47.5, 5.0), (46.5, 6.25)]:
        for gy in range(5): pts.append((mx, my + gy*10.0))
    for mx, my in [(3.5, 8.75), (2.5, 10.0), (3.5, 11.25)]:
        for gy in range(4): pts.append((mx, my + gy*10.0))
    return pts

rng = random.Random()
cl = gen_centerline()
obs_world = []
idx = 0
while idx < len(cl):
    cx, cy = cl[idx]; wx, wy = cx*SCALE, cy*SCALE
    obs_world.append((wx, wy + rng.uniform(-2.0, 2.0)))
    idx += rng.randint(3, 8)
obs_world = [(x,y) for x,y in obs_world if math.hypot(x-6, y-6) > 5.0]
OBS_R = 1.0; OBS_CLEAR = OBS_R + DETECT_R

# ── 航点 ──
cps_maze = [(3,3),(47,5),(3,10),(47,15),(3,20),(47,25),(3,30),(47,35),(3,40),(47,45),(3,48)]
nav_wps = [(x*SCALE, y*SCALE) for x,y in cps_maze]

# ── 检测 ──
def sample_hfield_at(wx, wy):
    mx, my = wx/SCALE, wy/SCALE
    px, py = int(mx*PIX_PER_M), HF_RES-1-int(my*PIX_PER_M)
    return int(hf[py,px]) if 0<=px<HF_RES and 0<=py<HF_RES else -1

def detect_wall(wx, wy, radius=DETECT_R):
    for dy in np.arange(-radius, radius+0.01, 0.15):
        md = np.sqrt(max(0,radius**2-dy**2))
        for dx in np.arange(-md, md+0.01, 0.15):
            if sample_hfield_at(wx+dx, wy+dy) != ROAD_PIX: return True
    return False

def detect_obs(wx, wy):
    for i,(ox,oy) in enumerate(obs_world):
        if math.hypot(wx-ox, wy-oy) < OBS_CLEAR: return i
    return -1

# ── 激光雷达 (软件) ──
def lidar_scan(bx, by, m, d, site_id, rays=360, range_m=12.0):
    """在(bx,by)位置向360°射 rays 条射线"""
    pts = []
    gg = np.ones(6, dtype=np.uint8) * 255
    gid = np.array([-1], np.int32)
    pos = d.site_xpos[site_id].copy()
    for i in range(rays):
        a = 2*math.pi*i/rays
        dw = np.array([math.cos(a), math.sin(a), 0.0])
        gid[0] = -1
        dist = mujoco.mj_ray(m, d, pos, dw, gg, 1, -1, gid)
        if gid[0] >= 0 and 0 < dist < range_m:
            hit = pos + dw*dist
            pts.append((float(hit[0]), float(hit[1])))
    return pts

# ── XML ──
cp_xml = "".join(f'<body mocap="true" pos="{x} {y} 2"><geom type="sphere" size="1.5" rgba="0.2 0.5 1 0.8"/></body>' for x,y in nav_wps[1:])
obs_xml = "".join(f'<body name="obs{i}" pos="{x:.1f} {y:.1f} 2.0"><geom type="cylinder" size="1.0 2.0" rgba="0.9 0.2 0.2 0.9"/></body>' for i,(x,y) in enumerate(obs_world))

xml = f"""<mujoco>
  <compiler angle="radian"/><option timestep="0.005"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset><hfield name="track" size="50.0 50.0 4.0 2.0" file="{MAP}"/></asset>
  <worldbody>
    <light pos="50 50 80" dir="0 0 -1"/>
    {cp_xml}{obs_xml}
    <geom type="hfield" hfield="track" pos="50 50 0.0" rgba="0.25 0.30 0.35 1.0" friction="0 0 0"/>
    <body name="bot" pos="0 0 0.5">
      <joint type="slide" axis="1 0 0" damping="0"/>
      <joint type="slide" axis="0 1 0" damping="0"/>
      <geom type="cylinder" size="0.5 0.5" rgba="1 0.3 0 1" friction="0 0 0"/>
      <site name="lidar_top" pos="0 0 0.8" size="0.02"/>
    </body>
  </worldbody>
</mujoco>"""

m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
d.qpos[0] = 6; d.qpos[1] = 6
mujoco.mj_forward(m, d)
lidar_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "lidar_top")

ESCAPE_STEPS = int(0.4 / (SPEED_MIN * m.opt.timestep))
yaw = 0.0; bounce = 0; force_steps = 0; escaping = False
wp_idx = 0; step = 0; speed = SPEED_MIN; t0 = time.time()
lidar_interval = int(1.0 / 10 / m.opt.timestep)  # 10Hz lidar
lidar_tick = 0

print(f"扇区{N_SECTORS} 射线{LIDAR_RAYS} safe={SAFE_DIST}m v={SPEED_MIN}→{SPEED_MAX}", flush=True)

with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
    v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.distance = 20; v.cam.elevation = -30; v.cam.azimuth = 180
    print("viewer ready", flush=True)

    while v.is_running() and wp_idx < len(nav_wps):
        bx, by = d.qpos[0], d.qpos[1]
        if bx<1 or bx>99 or by<1 or by>99:
            d.qpos[0]=max(1,min(99,bx)); d.qpos[1]=max(1,min(99,by))
            d.qvel[:]=0; yaw=random.uniform(0,2*math.pi)
            print(f"⚠ OOB step={step}", flush=True)
        v.cam.lookat[:] = np.array([bx, by, 0.5], dtype=np.float64)

        # 航点
        tx, ty = nav_wps[wp_idx]
        dist_to_cp = math.hypot(tx-bx, ty-by)
        if dist_to_cp < CP_RADIUS:
            wp_idx += 1
            print(f"✓ CP{wp_idx-1} step={step} v={speed:.1f}", flush=True)
            if wp_idx >= len(nav_wps):
                print(f"🏁 FINISH step={step} time={time.time()-t0:.1f}s bounces={bounce}", flush=True)
                break
            continue

        target_angle = math.atan2(ty-by, tx-bx)

        # ── 扇区导航 (非脱困时) ──
        if not escaping:
            lidar_tick += 1
            if lidar_tick % lidar_interval == 0:
                pts = lidar_scan(bx, by, m, d, lidar_site, LIDAR_RAYS, LIDAR_RANGE)
            else:
                pts = getattr(lidar_scan, '_cache', [])
            lidar_scan._cache = pts

            # 扇区最小距离
            sec_min = [float('inf')]*N_SECTORS
            for px, py in pts:
                dx, dy = px-bx, py-by
                dist = math.hypot(dx, dy)
                ang = math.atan2(dy, dx)
                rel = (ang - target_angle + math.pi) % (2*math.pi) - math.pi
                si = int((rel+math.pi)/(2*math.pi/N_SECTORS)) % N_SECTORS
                if dist < sec_min[si]: sec_min[si] = dist

            # 选最佳扇区: 从目标方向(0)向两侧找第一个通畅的
            best = 0
            for offset in range(N_SECTORS//2):
                for sign in [1, -1]:
                    si = (offset*sign) % N_SECTORS
                    if sec_min[si] >= SAFE_DIST:
                        best = si; break
                else: continue
                break

            # 目标方向 + 扇区偏移
            desired_yaw = target_angle + (best - 0) * 2*math.pi/N_SECTORS

            # 调速: 前方扇区越通畅越快
            front_clr = min(sec_min[0] / SAFE_DIST, 1.0) if sec_min[0] < float('inf') else 1.0
            if front_clr > 0.8: speed = min(speed+0.1, SPEED_MAX)
            elif front_clr < 0.3: speed = max(speed-0.5, SPEED_MIN)

            # 转向限速
            yaw_err = (desired_yaw - yaw + math.pi) % (2*math.pi) - math.pi
            dyaw = max(-YAW_RATE*m.opt.timestep, min(YAW_RATE*m.opt.timestep, yaw_err))
            yaw += dyaw

        # ── 碰撞检测 ──
        vx = math.cos(yaw)*speed; vy = math.sin(yaw)*speed
        nx = bx + vx*m.opt.timestep; ny = by + vy*m.opt.timestep
        wall = detect_wall(nx, ny, DETECT_R)
        obs_idx = detect_obs(nx, ny)
        colliding = wall or obs_idx >= 0

        if force_steps > 0:
            force_steps -= 1; d.qvel[0]=vx; d.qvel[1]=vy
        elif colliding:
            if not escaping:
                bounce += 1; escaping = True; speed = SPEED_MIN
                deg = random.uniform(30,90)*random.choice([-1,1])
                yaw += math.radians(deg)
                what = "墙" if wall else f"障碍#{obs_idx}"
                print(f"BOUNCE#{bounce} step={step} {what} Δ{deg:+.0f}°", flush=True)
            else:
                deg = random.uniform(30,90)*random.choice([-1,1])
                yaw += math.radians(deg)
            d.qvel[:]=0; force_steps=ESCAPE_STEPS
        else:
            escaping = False
            d.qvel[0]=vx; d.qvel[1]=vy

        mujoco.mj_step(m, d); step+=1; v.sync()

        if step%300==0:
            fsec = sec_min[0] if 'sec_min' in dir() else -1
            print(f"  [{step}] ({bx:.1f},{by:.1f}) CP{wp_idx} v={speed:.1f} d={dist_to_cp:.1f} f={fsec:.1f}", flush=True)

    print(f"done: {wp_idx}/{len(nav_wps)} step={step} time={time.time()-t0:.1f}s bounces={bounce}", flush=True)
