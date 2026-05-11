#!/usr/bin/env python3
"""bounce_nav — 路点弧线追踪 + 障碍规避 + 玩具车兜底

道路5m宽，机器人<1m宽，限宽1m通行。转速6rad/s，速度2→6m/s渐进。
"""
import sys, os, math, time, random
import numpy as np
from PIL import Image
import mujoco, mujoco.viewer

MAP = os.path.expanduser("~/workspace/simtrack/confirmed/track_clean.png")
hf = np.array(Image.open(MAP))

SCALE = 2.0; HF_RES = 2000; PIX_PER_M = 40; ROAD_PIX = 128
DETECT_R = 0.5          # 预判圈
SPEED_MIN = 2.0         # 起步速度
SPEED_MAX = 6.0         # 极速
YAW_RATE = 6.0          # 6 rad/s ≈ 344°/s
CP_RADIUS = 3.0         # 到达半径
LOOKAHEAD_CP = 1.5      # 提前1.5个CP距离开始弧线转弯

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
    cx, cy = cl[idx]
    wx, wy = cx * SCALE, cy * SCALE
    obs_world.append((wx, wy + rng.uniform(-2.0, 2.0)))
    idx += rng.randint(3, 8)
obs_world = [(x,y) for x,y in obs_world if math.hypot(x-6, y-6) > 5.0]
OBS_R = 1.0; OBS_CLEAR = OBS_R + DETECT_R

# ── 航点 ──
cps_maze = [(3,3),(47,5),(3,10),(47,15),(3,20),(47,25),(3,30),(47,35),(3,40),(47,45),(3,48)]
nav_wps = [(x*SCALE, y*SCALE) for x,y in cps_maze]

# ── 障碍物检测 ──
def sample_hfield_at(wx, wy):
    mx, my = wx / SCALE, wy / SCALE
    px = int(mx * PIX_PER_M)
    py = HF_RES - 1 - int(my * PIX_PER_M)
    if 0 <= px < HF_RES and 0 <= py < HF_RES:
        return int(hf[py, px])
    return -1

def detect_wall(wx, wy, radius=DETECT_R):
    for dy in np.arange(-radius, radius + 0.01, 0.15):
        max_dx = np.sqrt(max(0, radius**2 - dy**2))
        for dx in np.arange(-max_dx, max_dx + 0.01, 0.15):
            if sample_hfield_at(wx + dx, wy + dy) != ROAD_PIX:
                return True
    return False

def detect_obs(wx, wy):
    for i, (ox, oy) in enumerate(obs_world):
        if math.hypot(wx - ox, wy - oy) < OBS_CLEAR:
            return i
    return -1

def clearance_ahead(bx, by, yaw, dist=4.0):
    """探前方 clearance (0-1), 用于调速"""
    steps = int(dist / (SPEED_MAX * 0.005))
    hits = 0
    for s in range(steps):
        wx = bx + math.cos(yaw) * SPEED_MAX * 0.005 * (s+1)
        wy = by + math.sin(yaw) * SPEED_MAX * 0.005 * (s+1)
        if detect_wall(wx, wy, 0.4) or detect_obs(wx, wy) >= 0:
            hits += 1
    return 1.0 - hits / max(steps, 1)

def target_yaw(bx, by, wp_idx):
    """弧线转弯: 融合当前CP和下一个CP的方向"""
    tx, ty = nav_wps[wp_idx]
    angle_cur = math.atan2(ty - by, tx - bx)
    dist_cur = math.hypot(tx - bx, ty - by)

    # 快到当前CP时，提前向下一个CP弯
    if wp_idx + 1 < len(nav_wps) and dist_cur < CP_RADIUS * LOOKAHEAD_CP:
        nx, ny = nav_wps[wp_idx + 1]
        angle_next = math.atan2(ny - by, nx - bx)
        t = 1.0 - dist_cur / (CP_RADIUS * LOOKAHEAD_CP)  # 0→1 越近越偏向下一个
        # 角度融合（处理±π跳跃）
        diff = (angle_next - angle_cur + math.pi) % (2*math.pi) - math.pi
        return angle_cur + diff * t
    return angle_cur

# ── XML ──
cp_xml = "".join(
    f'<body mocap="true" pos="{x} {y} 2"><geom type="sphere" size="1.5" rgba="0.2 0.5 1 0.8"/></body>'
    for x, y in nav_wps[1:]
)
obs_xml = "".join(
    f'<body name="obs{i}" pos="{x:.1f} {y:.1f} 2.0">'
    f'<geom type="cylinder" size="1.0 2.0" rgba="0.9 0.2 0.2 0.9"/></body>'
    for i, (x, y) in enumerate(obs_world)
)

xml = f"""<mujoco>
  <compiler angle="radian"/><option timestep="0.005"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset><hfield name="track" size="50.0 50.0 4.0 2.0" file="{MAP}"/></asset>
  <worldbody>
    <light pos="50 50 80" dir="0 0 -1"/>
    {cp_xml}
    {obs_xml}
    <geom type="hfield" hfield="track" pos="50 50 0.0" rgba="0.25 0.30 0.35 1.0" friction="0 0 0"/>
    <body name="bot" pos="0 0 0.5">
      <joint type="slide" axis="1 0 0" damping="0"/>
      <joint type="slide" axis="0 1 0" damping="0"/>
      <geom type="cylinder" size="0.5 0.5" rgba="1 0.3 0 1" friction="0 0 0"/>
    </body>
  </worldbody>
</mujoco>"""

m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
d.qpos[0] = 6; d.qpos[1] = 6
mujoco.mj_forward(m, d)

ESCAPE_STEPS = int(0.4 / (SPEED_MIN * m.opt.timestep))
yaw = 0.0; bounce = 0; force_steps = 0; escaping = False
wp_idx = 0; step = 0; speed = SPEED_MIN; t0 = time.time()

print(f"航点:{len(nav_wps)} 障碍:{len(obs_world)} v={SPEED_MIN}→{SPEED_MAX}m/s ω={YAW_RATE}rad/s", flush=True)

with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
    v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.distance = 20; v.cam.elevation = -30; v.cam.azimuth = 180
    print("viewer ready", flush=True)

    while v.is_running() and wp_idx < len(nav_wps):
        bx, by = d.qpos[0], d.qpos[1]

        if bx < 1 or bx > 99 or by < 1 or by > 99:
            d.qpos[0] = max(1, min(99, bx)); d.qpos[1] = max(1, min(99, by))
            d.qvel[:] = 0; yaw = random.uniform(0, 2*math.pi)
            print(f"⚠ OOB step={step}", flush=True)
        v.cam.lookat[:] = np.array([bx, by, 0.5], dtype=np.float64)

        # 航点到达
        tx, ty = nav_wps[wp_idx]
        dist_to_cp = math.hypot(tx - bx, ty - by)
        if dist_to_cp < CP_RADIUS:
            wp_idx += 1
            print(f"✓ CP{wp_idx-1} step={step} v={speed:.1f} ({bx:.1f},{by:.1f})", flush=True)
            if wp_idx >= len(nav_wps):
                print(f"🏁 FINISH step={step} time={time.time()-t0:.1f}s bounces={bounce}", flush=True)
                break
            continue

        # ── 导航: 目标方向 (弧线)+ clearance 调速 ──
        desired_yaw = target_yaw(bx, by, wp_idx) if not escaping else yaw

        if not escaping:
            clr = clearance_ahead(bx, by, desired_yaw)
            # 调速: clearance高→加速, 低→减速
            if clr > 0.8:
                speed = min(speed + 0.1, SPEED_MAX)
            elif clr < 0.3:
                speed = max(speed - 0.5, SPEED_MIN)
            # 转向: 限角速度
            yaw_err = (desired_yaw - yaw + math.pi) % (2*math.pi) - math.pi
            dyaw = max(-YAW_RATE * m.opt.timestep, min(YAW_RATE * m.opt.timestep, yaw_err))
            yaw += dyaw

        vx = math.cos(yaw) * speed; vy = math.sin(yaw) * speed
        nx = bx + vx * m.opt.timestep; ny = by + vy * m.opt.timestep

        wall = detect_wall(nx, ny, DETECT_R)
        obs_idx = detect_obs(nx, ny)
        colliding = wall or obs_idx >= 0

        if force_steps > 0:
            force_steps -= 1; d.qvel[0] = vx; d.qvel[1] = vy
        elif colliding:
            if not escaping:
                bounce += 1; escaping = True; speed = SPEED_MIN
                deg = random.uniform(30, 90) * random.choice([-1, 1])
                yaw += math.radians(deg)
                what = "墙" if wall else f"障碍#{obs_idx}({obs_world[obs_idx][0]:.1f},{obs_world[obs_idx][1]:.1f})"
                print(f"BOUNCE#{bounce} step={step} {what} Δ{deg:+.0f}° CP{wp_idx} d={dist_to_cp:.1f}", flush=True)
            else:
                deg = random.uniform(30, 90) * random.choice([-1, 1])
                yaw += math.radians(deg)
            d.qvel[:] = 0; force_steps = ESCAPE_STEPS
        else:
            escaping = False
            d.qvel[0] = vx; d.qvel[1] = vy

        mujoco.mj_step(m, d)
        step += 1; v.sync()

        if step % 300 == 0:
            print(f"  [{step}] ({bx:.1f},{by:.1f}) CP{wp_idx}/{len(nav_wps)} v={speed:.1f} d={dist_to_cp:.1f} clr={clr:.2f}", flush=True)

    print(f"done: {wp_idx}/{len(nav_wps)} step={step} time={time.time()-t0:.1f}s bounces={bounce}", flush=True)
