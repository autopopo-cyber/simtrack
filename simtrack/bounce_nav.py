#!/usr/bin/env python3
"""bounce_nav: 扇区寻路 + launch_passive，基于 track_clean.png"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simtrack.maze_coords import *
from simtrack.lidar import LidarSensor
import numpy as np, cv2, mujoco, mujoco.viewer, time, math, random

# ── 地图 ──
hf = cv2.imread(os.path.expanduser("~/workspace/simtrack/confirmed/track_clean.png"), cv2.IMREAD_GRAYSCALE)
TRACK_PNG = os.path.expanduser("~/workspace/simtrack/confirmed/track_clean.png")

def sample_hfield_at(wx, wy):
    mx, my = wx / SCALE, wy / SCALE
    px = int(mx * PIX_PER_M)
    py = HF_RES - 1 - int(my * PIX_PER_M)
    if 0 <= px < HF_RES and 0 <= py < HF_RES:
        return int(hf[py, px])
    return -1

def detect_collision(wx, wy, radius=0.6):
    for dy in np.arange(-radius, radius + 0.01, 0.15):
        max_dx = np.sqrt(max(0, radius**2 - dy**2))
        for dx in np.arange(-max_dx, max_dx + 0.01, 0.15):
            if sample_hfield_at(wx + dx, wy + dy) != 128:
                return True
    return False

# ── 检查点 ──
cps_world = get_checkpoints_world()
cp_spheres = "".join(
    f'<body mocap="true" pos="{x} {y} 2"><geom type="sphere" size="1.0" rgba="0.2 0.5 1 0.8"/></body>'
    for x, y in cps_world[1:]
)

# ── 场景 ──
xml = f"""<mujoco>
  <compiler angle="radian"/><option timestep="0.005"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset><hfield name="track" size="50.0 50.0 4.0 2.0" file="{TRACK_PNG}"/></asset>
  <worldbody>
    <light pos="50 50 80" dir="0 0 -1"/>
    {cp_spheres}
    <geom type="hfield" hfield="track" pos="50 50 0.0" rgba="0.25 0.30 0.35 1.0" friction="0 0 0"/>
    <body name="bot" pos="0 0 0.5">
      <joint type="slide" axis="1 0 0" damping="0"/>
      <joint type="slide" axis="0 1 0" damping="0"/>
      <geom type="cylinder" size="0.5 0.5" rgba="1 0.3 0 1" friction="0 0 0"/>
      <site name="lidar_top" pos="0 0 0.8" size="0.02"/>
    </body>
  </worldbody>
</mujoco>"""

xml_path = os.path.expanduser("~/workspace/simtrack/_bounce_nav.xml")
with open(xml_path, "w") as f: f.write(xml)

m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)
d.qpos[0] = 6; d.qpos[1] = 6
mujoco.mj_forward(m, d)

# ── 激光雷达 (10Hz) ──
lidar = LidarSensor(m, d, site_name="lidar_top", rays=240, lines=1, range_m=12.0, hz=10)

# ── 扇区寻路参数 ──
N_SECTORS = 36          # 每10°一个扇区
SECTOR_DEG = 360 / N_SECTORS
SAFE_DIST = 2.0         # 安全距离(m)，扇区内有点<此距离=阻塞
SPEED = 2.0
CP_RADIUS = 3.0         # 到达检查点半径

cp_idx = 0
step = 0
t0 = time.time()

print(f"bounce_nav | lidar={lidar.rays}rays | sectors={N_SECTORS} | cp={len(cps_world)} | speed={SPEED}", flush=True)

with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
    v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.distance = 20; v.cam.elevation = -30; v.cam.azimuth = 180
    print("viewer ready", flush=True)

    while v.is_running() and cp_idx < len(cps_world):
        bx, by = d.qpos[0], d.qpos[1]
        v.cam.lookat[:] = np.array([bx, by, 0.5], dtype=np.float64)

        # ── 当前目标检查点 ──
        tx, ty = cps_world[cp_idx]
        dist_to_cp = math.hypot(tx - bx, ty - by)
        if dist_to_cp < CP_RADIUS:
            cp_idx += 1
            print(f"✓ CP{cp_idx-1} reached step={step} ({bx:.1f},{by:.1f})", flush=True)
            if cp_idx >= len(cps_world):
                print(f"🏁 FINISH step={step} time={time.time()-t0:.1f}s", flush=True)
                break
            continue

        target_angle = math.atan2(ty - by, tx - bx)

        # ── 激光雷达扫描 ──
        lidar.update(bx, by, target_angle)
        pts = lidar.points_2d

        # 扇区距离统计
        sector_min = [float('inf')] * N_SECTORS
        for px, py in pts:
            dx, dy = px - bx, py - by
            dist = math.hypot(dx, dy)
            ang = math.atan2(dy, dx)
            # 角度差，归一到 [0, 2π)
            rel = (ang - target_angle) % (2 * math.pi)
            si = int(rel / (2 * math.pi / N_SECTORS)) % N_SECTORS
            if dist < sector_min[si]:
                sector_min[si] = dist

        # ── 选最优扇区 ──
        # 优先目标方向(扇区0)，然后向两侧扩展找第一个通畅扇区
        best_sector = 0
        for offset in range(N_SECTORS // 2):
            for sign in [1, -1]:
                si = (offset * sign) % N_SECTORS
                if sector_min[si] >= SAFE_DIST:
                    best_sector = si
                    break
            else:
                continue
            break

        # 扇区中心角度
        sector_mid = target_angle + best_sector * 2 * math.pi / N_SECTORS
        yaw = sector_mid

        # ── 碰撞安全兜底 ──
        vx = np.cos(yaw) * SPEED; vy = np.sin(yaw) * SPEED
        nx = bx + vx * m.opt.timestep; ny = by + vy * m.opt.timestep

        if detect_collision(nx, ny, 0.55):
            # 紧急避让: 随机转向
            yaw = random.uniform(0, 2 * math.pi)
            d.qvel[:] = 0
            print(f"⚠ BOUNCE step={step} pos=({bx:.1f},{by:.1f}) cp={cp_idx}", flush=True)
        else:
            d.qvel[0] = np.cos(yaw) * SPEED
            d.qvel[1] = np.sin(yaw) * SPEED

        mujoco.mj_step(m, d)
        step += 1; v.sync()

        if step <= 20 or step % 40 == 0:
            pts_str = ",".join(f"s{i}={sector_min[i]:.1f}" for i in range(0, N_SECTORS, 6))
            print(f"[{step}] ({bx:.1f},{by:.1f})→CP{cp_idx} d={dist_to_cp:.1f} best_s={best_sector} | {pts_str}", flush=True)

    t_elapsed = time.time() - t0
    print(f"done: step={step} cp={cp_idx}/{len(cps_world)} time={t_elapsed:.1f}s", flush=True)
