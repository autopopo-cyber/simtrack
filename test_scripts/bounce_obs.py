#!/usr/bin/env python3
"""bounce_obs — V8 + 障碍物 + 1.5m预判碰撞 + 防卡死"""
import sys, os, math, time, random
import numpy as np, mujoco, mujoco.viewer
from PIL import Image

MAP = os.path.expanduser("~/workspace/simtrack/confirmed/track_clean.png")
hf = np.array(Image.open(MAP))

SCALE = 2.0; HF_RES = 2000; PIX_PER_M = 40; ROAD_PIX = 128
DETECT_R = 0.5  # 预判检测半径 (0.2s冲0.3m可脱身)

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

# ── 障碍物 ──
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
obs_world = []
idx = 0; cl = gen_centerline()
while idx < len(cl):
    cx, cy = cl[idx]
    wx, wy = cx * SCALE, cy * SCALE
    obs_world.append((wx, wy + rng.uniform(-2.0, 2.0)))
    idx += rng.randint(3, 8)
before = len(obs_world)
obs_world = [(x,y) for x,y in obs_world if math.hypot(x-6, y-6) > 5.0]
print(f"障碍物: {before}→{len(obs_world)}", flush=True)

obs_xml = "".join(
    f'<body name="obs{i}" pos="{x:.1f} {y:.1f} 2.0">'
    f'<geom type="cylinder" size="1.0 2.0" rgba="0.9 0.2 0.2 0.9"/></body>'
    for i, (x, y) in enumerate(obs_world)
)

OBS_R = 1.0; BOT_R = 0.55; OBS_CLEAR = OBS_R + DETECT_R

def detect_obs(wx, wy):
    for i, (ox, oy) in enumerate(obs_world):
        if math.hypot(wx - ox, wy - oy) < OBS_CLEAR:
            return i
    return -1

# ── 检查点 ──
cps_maze = [(3,3),(47,5),(3,10),(47,15),(3,20),(47,25),(3,30),(47,35),(3,40),(47,45),(3,48)]
cps_world = [(x*SCALE,y*SCALE) for x,y in cps_maze]
cp_xml = "".join(
    f'<body mocap="true" pos="{x} {y} 2"><geom type="sphere" size="1.5" rgba="0.2 0.5 1 0.8"/></body>'
    for x, y in cps_world[1:]
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

SPEED = 2.0; yaw = 0.0; bounce = 0
force_steps = 0; escaping = False
step = 0; t0 = time.time()

ESCAPE_DIST = 0.4  # 每次转向后冲0.4米
ESCAPE_STEPS = int(ESCAPE_DIST / (SPEED * m.opt.timestep))  # 0.4/(2*0.005)=40

print(f"speed={SPEED} 检测圈={DETECT_R}m 冲{ESCAPE_DIST}m/次 起点(6,6)", flush=True)

with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
    v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.distance = 20; v.cam.elevation = -30; v.cam.azimuth = 180
    print("viewer ready", flush=True)

    while v.is_running():
        bx, by = d.qpos[0], d.qpos[1]
        # 边界兜底: 出图拽回 (墙保护+障碍保护, 正常不应到)
        if bx < 1 or bx > 99 or by < 1 or by > 99:
            d.qpos[0] = max(1, min(99, bx))
            d.qpos[1] = max(1, min(99, by))
            d.qvel[:] = 0
            yaw = random.uniform(0, 2*math.pi)
            print(f"⚠ OOB step={step} 拽回({d.qpos[0]:.1f},{d.qpos[1]:.1f})", flush=True)
        v.cam.lookat[:] = np.array([d.qpos[0], d.qpos[1], 0.5], dtype=np.float64)

        vx = np.cos(yaw) * SPEED; vy = np.sin(yaw) * SPEED
        nx = bx + vx * m.opt.timestep; ny = by + vy * m.opt.timestep

        wall = detect_wall(nx, ny, DETECT_R)
        obs_idx = detect_obs(nx, ny)
        colliding = wall or obs_idx >= 0

        if force_steps > 0:
            # 冲0.4m中: 不检测碰撞, 闷头走
            force_steps -= 1
            d.qvel[0] = vx; d.qvel[1] = vy
        elif colliding:
            if not escaping:
                # 首次碰撞: 记一次bounce + 进入脱困循环
                bounce += 1
                escaping = True
                if wall:
                    deg = random.uniform(30, 90) * random.choice([-1, 1])
                    yaw += math.radians(deg)
                    print(f"BOUNCE#{bounce} step={step} ({bx:.1f},{by:.1f}) 墙 Δ{deg:+.0f}° yaw={math.degrees(yaw):.0f}°", flush=True)
                else:
                    ox, oy = obs_world[obs_idx]
                    deg = random.uniform(30, 90) * random.choice([-1, 1])
                    yaw += math.radians(deg)
                    print(f"BOUNCE#{bounce} step={step} ({bx:.1f},{by:.1f}) 障碍#{obs_idx}({ox:.1f},{oy:.1f}) Δ{deg:+.0f}° yaw={math.degrees(yaw):.0f}°", flush=True)
            else:
                # 脱困循环中: 0.4m后仍卡着→再转, 不记新bounce
                deg = random.uniform(30, 90) * random.choice([-1, 1])
                yaw += math.radians(deg)
            d.qvel[:] = 0
            force_steps = ESCAPE_STEPS
        else:
            # 畅通: 退出脱困状态
            escaping = False
            d.qvel[0] = vx; d.qvel[1] = vy

        mujoco.mj_step(m, d)
        step += 1; v.sync()

        if step % 500 == 0:
            print(f"  [{step}] ({bx:.1f},{by:.1f}) bounces={bounce}", flush=True)

    print(f"done: {step}s {bounce}b {time.time()-t0:.1f}s", flush=True)
