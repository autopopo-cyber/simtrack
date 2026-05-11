#!/usr/bin/env python3
"""algo2_lane_switch — 世界体素 v5

数据: 1m³体素 0=未知 1=通畅 2=墙 3=已走
决策: 扫清四周→找未探索的体素→朝wp方向走
"""
import sys, os, math, time, random
import numpy as np
from PIL import Image
import mujoco, mujoco.viewer

MAP = os.path.expanduser("~/workspace/simtrack/confirmed/track_clean.png")
hf = np.array(Image.open(MAP))

SCALE = 2.0; HF_RES = 2000; PIX_PER_M = 40; ROAD_PIX = 128
SAFE_R = 1.0; SPEED = 4.0; SPEED_MAX = 6.0; YAW_RATE = 6.0
CP_RADIUS = 2.0; LIDAR_RANGE = 15.0; VOXEL = 1.0  # 1m体素

UNKNOWN, FREE, WALL, VISITED = 0, 1, 2, 3
W = 100; vox = np.zeros((W, W), dtype=np.int8)

# ── 中心线 & 障碍物 ──
def gen_centerline():
    pts = []; y0 = 2.5
    for seg in range(10):
        y = y0+seg*5.0; x0, x1 = (5.0,45.0) if seg%2==0 else (45.0,5.0)
        for j in range(10): pts.append((x0+(j/9.0)*(x1-x0), y))
    for mx, my in [(46.5,3.75),(47.5,5.0),(46.5,6.25)]:
        for gy in range(5): pts.append((mx, my+gy*10.0))
    for mx, my in [(3.5,8.75),(2.5,10.0),(3.5,11.25)]:
        for gy in range(4): pts.append((mx, my+gy*10.0))
    return pts

rng = random.Random(); cl = gen_centerline(); obs_world = []; idx = 0
while idx < len(cl):
    cx, cy = cl[idx]; wx, wy = cx*SCALE, cy*SCALE
    obs_world.append((wx, wy+rng.uniform(-2.0,2.0))); idx += rng.randint(3,8)
obs_world = [(x,y) for x,y in obs_world if math.hypot(x-6,y-6)>5.0]
OBS_R = 1.0; OBS_CLEAR = OBS_R+SAFE_R

cps_maze = [(3,3),(47,5),(3,10),(47,15),(3,20),(47,25),(3,30),(47,35),(3,40),(47,45),(3,48)]
nav_wps = [(x*SCALE, y*SCALE) for x,y in cps_maze]

# ── 检测 ──
def sample_hf(wx, wy):
    mx, my = wx/SCALE, wy/SCALE
    px, py = int(mx*PIX_PER_M), HF_RES-1-int(my*PIX_PER_M)
    return int(hf[py,px]) if 0<=px<HF_RES and 0<=py<HF_RES else -1

is_wall = lambda wx, wy: sample_hf(wx, wy) != ROAD_PIX
is_obs = lambda wx, wy: any(math.hypot(wx-ox, wy-oy) < OBS_CLEAR for ox, oy in obs_world)
blocked = lambda wx, wy: is_wall(wx, wy) or is_obs(wx, wy)

def road_dir(wp_idx):
    if wp_idx+1 >= len(nav_wps): return (1,0)
    cx,cy=nav_wps[wp_idx]; nx,ny=nav_wps[wp_idx+1]
    dx,dy=nx-cx,ny-cy; d=math.hypot(dx,dy)
    return (dx/d, dy/d) if d>0.01 else (1,0)

# ── 体素扫描 ──
def scan_voxels(bx, by):
    """射线扫描标记 FREE 和 WALL"""
    for a in np.linspace(0, 2*np.pi, 120):
        cos_a, sin_a = math.cos(a), math.sin(a)
        for d in np.arange(0.5, LIDAR_RANGE+0.1, 0.5):
            wx, wy = bx+cos_a*d, by+sin_a*d
            vx, vy = int(wx), int(wy)
            if not (0 <= vx < W and 0 <= vy < W): break
            if vox[vy, vx] == WALL: break
            if blocked(wx, wy):
                vox[vy, vx] = WALL; break
            vox[vy, vx] = max(vox[vy, vx], FREE)

def target_yaw(bx, by, wp_idx):
    tx, ty = nav_wps[wp_idx]; dist = math.hypot(tx-bx, ty-by)
    ang = math.atan2(ty-by, tx-bx)
    if wp_idx+1<len(nav_wps) and dist<CP_RADIUS*2.5:
        nx, ny = nav_wps[wp_idx+1]
        ang2 = math.atan2(ny-by, nx-bx)
        t = 1.0-dist/(CP_RADIUS*2.5)
        diff = (ang2-ang+math.pi)%(2*math.pi)-math.pi
        ang += diff*t
    return ang

# ── 体素决策 ──
def find_frontier(bx, by, wp_idx):
    """找最佳未探索体素: 邻接已探索区域的未知格子, 朝wp方向"""
    cx, cy = int(bx), int(by)
    wp_yaw = target_yaw(bx, by, wp_idx)
    best_score = -9999; best = None
    
    # 搜索范围: 当前位置周围20m
    for dy in range(-20, 21):
        for dx in range(-20, 21):
            vx, vy = cx+dx, cy+dy
            if not (0 <= vx < W and 0 <= vy < W): continue
            if vox[vy, vx] != FREE: continue  # 只要已扫描通畅但未走过的
            
            # 必须邻接VISITED或FREE(探索边缘)
            adjacent = False
            for ndy in (-1,0,1):
                for ndx in (-1,0,1):
                    nvx, nvy = vx+ndx, vy+ndy
                    if 0 <= nvx < W and 0 <= nvy < W:
                        if vox[nvy, nvx] in (VISITED, FREE):
                            adjacent = True; break
                if adjacent: break
            if not adjacent: continue
            
            # 评分
            score = 0
            # 朝wp加分
            ang = math.atan2(vy+0.5-by, vx+0.5-bx)
            diff = abs((ang-wp_yaw+math.pi)%(2*math.pi)-math.pi)
            score -= diff * 30  # 偏离wp扣分
            
            # 距离惩罚
            dist = math.hypot(dx, dy)
            score -= dist * 5
            
            # 未知邻接加分(前沿)
            unknown_neighbors = 0
            for ndy in (-1,0,1):
                for ndx in (-1,0,1):
                    nvx2, nvy2 = vx+ndx, vy+ndy
                    if 0 <= nvx2 < W and 0 <= nvy2 < W:
                        if vox[nvy2, nvx2] == UNKNOWN:
                            unknown_neighbors += 1
            score += unknown_neighbors * 20
            
            if score > best_score:
                best_score = score
                best = (vx+0.5, vy+0.5)  # 格子中心
    
    return best

# ── XML ──
CP_XML = "".join(f'<body mocap="true" pos="{x} {y} 2"><geom type="sphere" size="1.5" rgba="0.2 0.5 1 0.8"/></body>' for x,y in nav_wps[1:])
OBS_XML = "".join(f'<body name="obs{i}" pos="{x:.1f} {y:.1f} 2.0"><geom type="cylinder" size="1.0 2.0" rgba="0.9 0.2 0.2 0.9"/></body>' for i,(x,y) in enumerate(obs_world))

xml = f"""<mujoco>
  <compiler angle="radian"/><option timestep="0.005"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset><hfield name="track" size="50.0 50.0 4.0 2.0" file="{MAP}"/></asset>
  <worldbody>
    <light pos="50 50 80" dir="0 0 -1"/>{CP_XML}{OBS_XML}
    <geom type="hfield" hfield="track" pos="50 50 0.0" rgba="0.25 0.30 0.35 1.0" friction="0 0 0"/>
    <body name="bot" pos="0 0 0.5">
      <joint type="slide" axis="1 0 0" damping="0"/>
      <joint type="slide" axis="0 1 0" damping="0"/>
      <geom type="cylinder" size="0.5 0.5" rgba="1 0.3 0 1" friction="0 0 0"/>
    </body>
  </worldbody>
</mujoco>"""

m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
d.qpos[0]=6; d.qpos[1]=6; mujoco.mj_forward(m,d)

yaw=0.0; bounce=0; force_steps=0; escaping=False
wp_idx=0; step=0; speed=SPEED; t0=time.time()
stuck_t=0; stuck_x=0.0; stuck_y=0.0; target=None

print(f"=== algo2 体素 v5 === {VOXEL}m³ LIDAR={LIDAR_RANGE}m", flush=True)

with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
    v.cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.distance=25; v.cam.elevation=-35; v.cam.azimuth=180

    while v.is_running() and wp_idx<len(nav_wps):
        bx, by = d.qpos[0], d.qpos[1]
        if bx<1 or bx>99 or by<1 or by>99:
            d.qpos[0]=max(1,min(99,bx)); d.qpos[1]=max(1,min(99,by))
            d.qvel[:]=0; yaw=random.uniform(0,2*math.pi)
        v.cam.lookat[:]=np.array([bx, by, 0.5], dtype=np.float64)

        tx, ty = nav_wps[wp_idx]; dist_to_cp = math.hypot(tx-bx, ty-by)
        if dist_to_cp < CP_RADIUS:
            wp_idx+=1
            vis = int(np.sum(vox==VISITED)); f = int(np.sum(vox==FREE))
            print(f"✓ CP{wp_idx-1} step={step} visited={vis} free={f} walls={int(np.sum(vox==WALL))}", flush=True)
            if wp_idx>=len(nav_wps):
                print(f"🏁 FINISH step={step} time={time.time()-t0:.1f}s", flush=True)
                break
            continue

        # 标记已走
        vx, vy = int(bx), int(by)
        if 0 <= vx < W and 0 <= vy < W: vox[vy, vx] = VISITED

        # 扫描 (10Hz)
        if step % 20 == 0: scan_voxels(bx, by)

        # 决策
        target = find_frontier(bx, by, wp_idx)

        if target is None:
            if not escaping:
                bounce+=1; escaping=True
                deg=random.uniform(90,180)*random.choice([-1,1]); yaw+=math.radians(deg)
                d.qvel[:]=0; force_steps=int(0.3/(SPEED*m.opt.timestep))
            mujoco.mj_step(m,d); step+=1
            if step%RENDER_SKIP==0: v.sync()
            continue

        # 执行
        if not escaping:
            tgt_yaw = math.atan2(target[1]-by, target[0]-bx)
            yaw_err = (tgt_yaw-yaw+math.pi)%(2*math.pi)-math.pi
            dyaw = max(-YAW_RATE*m.opt.timestep, min(YAW_RATE*m.opt.timestep, yaw_err))
            yaw += dyaw
            clear = math.hypot(target[0]-bx, target[1]-by)
            speed = max(1.5, min(SPEED_MAX, clear*0.5))

        vx_m=math.cos(yaw)*speed; vy_m=math.sin(yaw)*speed
        nx, ny = bx+vx_m*m.opt.timestep, by+vy_m*m.opt.timestep

        # 卡死检测
        if step-stuck_t>300:
            if math.hypot(bx-stuck_x, by-stuck_y)<0.5:
                if not escaping:
                    bounce+=1; escaping=True
                    deg=random.uniform(90,180)*random.choice([-1,1]); yaw+=math.radians(deg)
                    d.qvel[:]=0; force_steps=int(0.3/(SPEED*m.opt.timestep))
            stuck_t=step; stuck_x=bx; stuck_y=by

        if force_steps>0:
            force_steps-=1; d.qvel[0]=vx_m; d.qvel[1]=vy_m
        elif blocked(nx, ny):
            if not escaping:
                bounce+=1; escaping=True
                deg=random.uniform(45,120)*random.choice([-1,1]); yaw+=math.radians(deg)
                d.qvel[:]=0; force_steps=int(0.4/(SPEED*m.opt.timestep))
        else:
            escaping=False; d.qvel[0]=vx_m; d.qvel[1]=vy_m

        mujoco.mj_step(m,d); step+=1
        if step%RENDER_SKIP==0: v.sync()
        if step%200==0:
            vis=int(np.sum(vox==VISITED)); f=int(np.sum(vox==FREE))
            print(f"  [{step}] ({bx:.0f},{by:.0f}) CP{wp_idx} v={speed:.1f} d={dist_to_cp:.0f} V{vis}/F{f}", flush=True)

    print(f"done: {wp_idx}/{len(nav_wps)} step={step} time={time.time()-t0:.1f}s", flush=True)
