#!/usr/bin/env python3
"""bounce_view v8: cylinder采样碰撞检测 (0.6m半径) + launch_passive"""
import sys; sys.path.insert(0,"/tmp")
from maze_coords import *
import numpy as np, cv2, mujoco, mujoco.viewer, time, random

hf = cv2.imread("/tmp/track_clean.png", cv2.IMREAD_GRAYSCALE)

def sample_hfield_at(wx, wy):
    """世界坐标→hfield像素值，OOB返回-1"""
    mx, my = wx / SCALE, wy / SCALE
    px = int(mx * PIX_PER_M)
    py = HF_RES - 1 - int(my * PIX_PER_M)
    if 0 <= px < HF_RES and 0 <= py < HF_RES:
        return int(hf[py, px])
    return -1

def detect_collision(wx, wy, radius=0.6):
    """在(wx,wy)周围radius半径圆柱体内采样，非128=撞墙"""
    # 稀疏采样: 0.15m步长，覆盖圆盘
    for dy in np.arange(-radius, radius + 0.01, 0.15):
        max_dx = np.sqrt(max(0, radius**2 - dy**2))
        for dx in np.arange(-max_dx, max_dx + 0.01, 0.15):
            if sample_hfield_at(wx + dx, wy + dy) != 128:
                return True
    return False

cps_world = get_checkpoints_world()
cp_spheres = "".join(
    f'<body mocap="true" pos="{x} {y} 2"><geom type="sphere" size="1.5" rgba="0.2 0.5 1 0.8"/></body>'
    for x, y in cps_world[1:]
)

xml = f"""<mujoco>
  <compiler angle="radian"/><option timestep="0.005"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset><hfield name="track" size="50.0 50.0 4.0 2.0" file="/tmp/track_clean.png"/></asset>
  <worldbody>
    <light pos="50 50 80" dir="0 0 -1"/>
    {cp_spheres}
    <geom type="hfield" hfield="track" pos="50 50 0.0" rgba="0.25 0.30 0.35 1.0" friction="0 0 0"/>
    <body name="bot" pos="0 0 0.5">
      <joint type="slide" axis="1 0 0" damping="0"/>
      <joint type="slide" axis="0 1 0" damping="0"/>
      <geom type="cylinder" size="0.5 0.5" rgba="1 0.3 0 1" friction="0 0 0"/>
    </body>
  </worldbody>
</mujoco>"""

with open("/tmp/bounce_v8.xml", "w") as f: f.write(xml)

m = mujoco.MjModel.from_xml_path("/tmp/bounce_v8.xml")
d = mujoco.MjData(m)
d.qpos[0] = 6; d.qpos[1] = 6
mujoco.mj_forward(m, d)

SPEED = 1.5; yaw = 0.0; bounce = 0
print(f"v8 cylinder collision | radius=0.6m | {len(cps_world)} nav spheres", flush=True)

with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
    v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.distance = 20; v.cam.elevation = -30; v.cam.azimuth = 180
    print("viewer ready", flush=True)
    
    step = 0; t0 = time.time()
    while v.is_running():
        bx, by = d.qpos[0], d.qpos[1]
        v.cam.lookat[:] = np.array([bx, by, 0.5], dtype=np.float64)
        
        vx = np.cos(yaw) * SPEED; vy = np.sin(yaw) * SPEED
        nx = bx + vx * m.opt.timestep; ny = by + vy * m.opt.timestep
        
        # 碰撞预判: 机器人半径0.5m, 检测外扩0.05m → 总半径0.55m → 直径1.1m
        if detect_collision(nx, ny, 0.55):
            yaw = random.uniform(0, 2 * np.pi)
            d.qvel[:] = 0
            bounce += 1
            print(f"BOUNCE#{bounce} step={step} pos=({bx:.1f},{by:.1f})", flush=True)
        else:
            d.qvel[0] = vx; d.qvel[1] = vy
        
        mujoco.mj_step(m, d)
        step += 1; v.sync()
        
        if step <= 20 or step % 20 == 0:
            print(f"step={step} pos=({bx:.2f},{by:.2f}) bounce={bounce}", flush=True)
    
    print(f"done: {step}s {bounce}b {time.time()-t0:.1f}s", flush=True)
