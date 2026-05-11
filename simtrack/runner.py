#!/usr/bin/env python3
"""
simtrack runner — 扇区导航仿真入口

模块:
  1. map.py       — 地图 (track_clean.png, 碰撞检测, 中心线)
  2. waypoints.py — 导航点 + 分段障碍物
  3. lidar.py     — 激光雷达 (0.5m高度, 240射线, 12m)
  4. nav.py       — 扇区导航算法

用法: python -m simtrack.runner [speed]
"""
import sys, os, time, math
import numpy as np
import mujoco, mujoco.viewer
import cv2

# 确保项目根目录在 path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from simtrack import map as simmap
from simtrack import waypoints as simwp
from simtrack.lidar import LidarSensor
from simtrack.nav import SectorNav

# ── 配置 ──
SPEED = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
LIDAR_HEIGHT = 1.2       # 雷达高度(m), 2m墙内 (匹配G1)
LIDAR_RAYS = 360
LIDAR_RANGE = 15.0
LIDAR_HZ = 10
N_SECTORS = 36
SAFE_DIST = 2.0
CP_RADIUS = 3.0            # 到达检查点半径
WP_SPACING = 4.0           # 中心线路点间距(m)
NAV_SPACING = 8.0          # 导航点间距(m)
OBS_SPACING = 5.0          # 障碍物分段间距(m)
OBS_DENSITY = 0.3          # 障碍物密度

def main():
    print(f"simtrack runner — lidar={LIDAR_RAYS}rays/{LIDAR_RANGE}m, "
          f"sectors={N_SECTORS}, safe={SAFE_DIST}m, speed={SPEED}", flush=True)
    
    # ── 1. 地图 ──
    hf = simmap.load()
    print(f"  地图: {simmap.HF_RES}×{simmap.HF_RES}, 墙{simmap.WALL_PIX} 路{simmap.ROAD_PIX}", flush=True)
    
    # 中心线
    centerline = simmap.generate_centerline(hf, spacing_m=WP_SPACING)
    print(f"  中心线: {len(centerline)} 点", flush=True)
    
    # 导航点
    nav_wps = simwp.from_centerline(centerline, spacing_m=NAV_SPACING)
    print(f"  导航点: {len(nav_wps)} 个 ({NAV_SPACING}m间距)", flush=True)
    
    # 障碍物
    obstacles = simwp.generate_segment_obstacles(
        centerline, spacing_m=OBS_SPACING, density=OBS_DENSITY, seed=42)
    print(f"  障碍物: {len(obstacles)} 个 (密度{OBS_DENSITY})", flush=True)
    
    # ── 2. 构建场景 ──
    # 检查点球体 (蓝色)
    cp_world = simmap.get_checkpoints_world()
    cp_xml = simwp.checkpoints_to_xml(cp_world[1:], radius=1.0)
    
    # 障碍物圆柱体 (红色)
    obs_xml = simwp.obstacles_to_xml(obstacles, radius=0.3, height=1.0)
    
    # 起点
    start_wx, start_wy = cp_world[0]
    
    xml = f"""<mujoco>
<compiler angle="radian"/><option timestep="0.005"/>
<visual><global offwidth="1280" offheight="720"/></visual>
<asset><hfield name="track" size="50.0 50.0 4.0 2.0" file="{simmap.MAP_PATH}"/></asset>
<worldbody>
  <light pos="50 50 80" dir="0 0 -1"/>
  {cp_xml}
  {obs_xml}
  <geom type="hfield" hfield="track" pos="50 50 0.0" rgba="0.25 0.30 0.35 1.0" friction="0 0 0"/>
  <body name="bot" pos="{start_wx} {start_wy} {LIDAR_HEIGHT + 0.3}">
    <joint type="slide" axis="1 0 0" damping="0"/>
    <joint type="slide" axis="0 1 0" damping="0"/>
    <geom type="cylinder" size="0.5 0.5" rgba="1 0.3 0 1" friction="0 0 0"/>
    <site name="lidar_top" pos="0 0 {LIDAR_HEIGHT}" size="0.02"/>
  </body>
</worldbody>
</mujoco>"""
    
    xml_path = os.path.join(_project_root, "_scene.xml")
    with open(xml_path, "w") as f:
        f.write(xml)
    
    # ── 3. MuJoCo + Lidar + Nav ──
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    d.qpos[0] = start_wx
    d.qpos[1] = start_wy
    mujoco.mj_forward(m, d)
    
    lidar = LidarSensor(m, d, site_name="lidar_top", 
                        rays=LIDAR_RAYS, lines=1, range_m=LIDAR_RANGE, hz=LIDAR_HZ)
    nav = SectorNav(n_sectors=N_SECTORS, safe_dist=SAFE_DIST, 
                     speed=SPEED, cp_radius=CP_RADIUS)
    
    # ── 4. 仿真循环 ──
    wp_idx = 0
    step = 0
    t0 = time.time()
    lidar_step = 0
    
    print(f"  viewer starting...", flush=True)
    
    with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        v.cam.distance = 20
        v.cam.elevation = -30
        v.cam.azimuth = 180
        
        while v.is_running() and wp_idx < len(nav_wps):
            bx, by = d.qpos[0], d.qpos[1]
            v.cam.lookat[:] = np.array([bx, by, 1.0], dtype=np.float64)
            
            # 雷达更新 (10Hz)
            lidar_step += 1
            if lidar_step % lidar.step_interval == 0:
                pts = lidar.update(bx, by, 0.0)  # yaw is computed in steer
            pts_2d = lidar.points_2d
            
            # 导航
            tx, ty = nav_wps[wp_idx]
            vx, vy, reached = nav.steer(bx, by, tx, ty, pts_2d, hf)
            
            if reached:
                wp_idx += 1
                print(f"  ✓ WP{wp_idx-1} ({bx:.1f},{by:.1f}) step={step}", flush=True)
                if wp_idx >= len(nav_wps):
                    print(f"🏁 FINISH step={step} time={time.time()-t0:.1f}s "
                          f"bounces={nav.bounces}", flush=True)
                    break
                continue
            
            d.qvel[0] = vx
            d.qvel[1] = vy
            mujoco.mj_step(m, d)
            step += 1
            v.sync()
            
            if step % 100 == 0:
                d_cp = math.hypot(tx - bx, ty - by)
                print(f"  [{step}] ({bx:.1f},{by:.1f}) → WP{wp_idx}/{len(nav_wps)} "
                      f"d={d_cp:.1f} bounces={nav.bounces}", flush=True)
        
        t_elapsed = time.time() - t0
        print(f"done: step={step} wp={wp_idx}/{len(nav_wps)} "
              f"time={t_elapsed:.1f}s bounces={nav.bounces}", flush=True)

if __name__ == "__main__":
    main()
