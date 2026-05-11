#!/usr/bin/env python3
"""algo2_lane_lidar — 三车道 + 激光雷达象限法

15m lidar@10Hz → 车道投影 → 选最干净车道
目标: 零碰撞到终点
"""
import sys, os, math, time, random
import numpy as np
from PIL import Image
import mujoco, mujoco.viewer

MAP = os.path.expanduser("~/workspace/simtrack/confirmed/track_clean.png")
hf = np.array(Image.open(MAP))

SCALE = 2.0; HF_RES = 2000; PIX_PER_M = 40; ROAD_PIX = 128
SAFE_R = 1.0           # 安全半径 (机器人0.5m + 余量0.5m)
SPEED = 2.0            # 初速2m/s
SPEED_MAX = 6.0
YAW_RATE = 6.0
CP_RADIUS = 3.0
LIDAR_RANGE = 15.0     # 15m探测
LIDAR_RAYS = 240       # 射线数
LIDAR_HZ = 10
LANE_OFFSETS = {"左": -1.5, "中": 0.0, "右": 1.5}
LANE_WIDTH = 1.2       # 车道半宽
LOOKAHEAD = 8.0        # 前瞻距离

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

cps_maze = [(3,3),(47,5),(3,10),(47,15),(3,20),(47,25),(3,30),(47,35),(3,40),(47,45),(3,48)]
nav_wps = [(x*SCALE, y*SCALE) for x,y in cps_maze]

# ── 道路方向 ──
def road_direction(wp_idx):
    if wp_idx+1 >= len(nav_wps): return (1,0)
    cx,cy=nav_wps[wp_idx]; nx,ny=nav_wps[wp_idx+1]
    dx,dy=nx-cx,ny-cy; d=math.hypot(dx,dy)
    return (dx/d, dy/d) if d>0.01 else (1,0)

def road_normal(wp_idx):
    rdx, rdy = road_direction(wp_idx)
    return (-rdy, rdx)  # 左法向量

# ── 激光雷达 (MuJoCo rays) ──
def lidar_scan(bx, by, m, d, site_id, rays=LIDAR_RAYS, range_m=LIDAR_RANGE):
    pts = []
    gg = np.ones(6, dtype=np.uint8)*255; gid = np.array([-1], np.int32)
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

# ── 车道评分 ──
def lane_score_from_lidar(bx, by, wp_idx, lidar_pts):
    """用lidar点投影到道路坐标系, 评估每条车道的前方通畅度"""
    rdx, rdy = road_direction(wp_idx)
    nx_dir, ny_dir = road_normal(wp_idx)
    
    scores = {}
    for name, offset in LANE_OFFSETS.items():
        lane_cx = bx + nx_dir*offset  # 车道中心x (世界)
        lane_cy = by + ny_dir*offset
        
        # 统计该车道前方 LOOKAHEAD 米内的 lidar 点
        hits = 0; total_checks = 0
        for step in range(10):
            d = LOOKAHEAD*(step+1)/10
            cx = lane_cx + rdx*d; cy = lane_cy + rdy*d
            total_checks += 1
            # 检查该点周围 LANE_WIDTH 内是否有 lidar 命中点
            for px, py in lidar_pts:
                if math.hypot(px-cx, py-cy) < LANE_WIDTH:
                    hits += 1; break
        scores[name] = 1.0 - hits/max(total_checks, 1)
    return scores

def wall_proximity(bx, by, wp_idx):
    """检查到最近墙体的距离 (0=贴墙, 1=安全)"""
    rdx, rdy = road_direction(wp_idx)
    nx_dir, ny_dir = road_normal(wp_idx)
    # 检查左右两侧墙距
    min_dist = float('inf')
    for side, offset in [(-1, -5.0), (1, 5.0)]:  # 左右各5m
        wx = bx + nx_dir*offset; wy = by + ny_dir*offset
        val = sample_hfield_at(wx, wy)
        if val != ROAD_PIX:
            # 二分查找墙边界
            lo, hi = 0.0, abs(offset)
            for _ in range(8):
                mid = (lo+hi)/2
                mx = bx + nx_dir*side*mid; my = by + ny_dir*side*mid
                if sample_hfield_at(mx, my) == ROAD_PIX: lo = mid
                else: hi = mid
            min_dist = min(min_dist, hi)
    return min(1.0, min_dist/2.0)  # 2m内=危险

def sample_hfield_at(wx, wy):
    mx, my = wx/SCALE, wy/SCALE
    px, py = int(mx*PIX_PER_M), HF_RES-1-int(my*PIX_PER_M)
    return int(hf[py,px]) if 0<=px<HF_RES and 0<=py<HF_RES else -1

def is_blocked(wx, wy):
    if sample_hfield_at(wx, wy) != ROAD_PIX: return True
    for ox, oy in obs_world:
        if math.hypot(wx-ox, wy-oy) < (OBS_R+SAFE_R): return True
    return False

OBS_R = 1.0
def target_yaw(bx, by, wp_idx):
    tx, ty = nav_wps[wp_idx]; dist = math.hypot(tx-bx, ty-by)
    ang = math.atan2(ty-by, tx-bx)
    if wp_idx+1<len(nav_wps) and dist<CP_RADIUS*2.5:
        nx, ny = nav_wps[wp_idx+1]; ang2 = math.atan2(ny-by, nx-bx)
        t = 1.0-dist/(CP_RADIUS*2.5); diff=(ang2-ang+math.pi)%(2*math.pi)-math.pi
        ang+=diff*t
    return ang

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
      <site name="lidar_top" pos="0 0 1.0" size="0.02"/>
    </body>
  </worldbody>
</mujoco>"""

m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
d.qpos[0]=6; d.qpos[1]=6; mujoco.mj_forward(m,d)
lidar_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "lidar_top")

yaw=0.0; bounce=0; force_steps=0; escaping=False
wp_idx=0; step=0; speed=SPEED; t0=time.time()
lidar_interval = int(1.0/LIDAR_HZ/m.opt.timestep)
lidar_tick = 0; lidar_cache = []

print(f"=== algo2_lane_lidar === 15m lidar@10Hz 三车道 安全{r}={SAFE_R}m", flush=True)

with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
    v.cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.distance=25; v.cam.elevation=-35; v.cam.azimuth=180
    print("viewer ready", flush=True)

    while v.is_running() and wp_idx<len(nav_wps):
        bx, by = d.qpos[0], d.qpos[1]
        if bx<1 or bx>99 or by<1 or by>99:
            d.qpos[0]=max(1,min(99,bx)); d.qpos[1]=max(1,min(99,by))
            d.qvel[:]=0; yaw=random.uniform(0,2*math.pi)
            print(f"⚠ OOB step={step}", flush=True)
        v.cam.lookat[:]=np.array([bx, by, 0.5], dtype=np.float64)

        tx, ty = nav_wps[wp_idx]; dist_to_cp = math.hypot(tx-bx, ty-by)
        if dist_to_cp < CP_RADIUS:
            wp_idx+=1
            print(f"✓ CP{wp_idx-1} step={step} v={speed:.1f} ({bx:.1f},{by:.1f})", flush=True)
            if wp_idx>=len(nav_wps):
                print(f"🏁 FINISH step={step} time={time.time()-t0:.1f}s bounces={bounce}", flush=True)
                break
            continue

        # ── lidar ──
        lidar_tick += 1
        if lidar_tick % lidar_interval == 0:
            lidar_cache = lidar_scan(bx, by, m, d, lidar_site)

        # ── 导航 ──
        if not escaping:
            lane_scores = lane_score_from_lidar(bx, by, wp_idx, lidar_cache)
            best_lane = max(lane_scores, key=lane_scores.get)
            best_clr = lane_scores[best_lane]
            
            # 墙距惩罚: 贴墙时强切中路
            wprox = wall_proximity(bx, by, wp_idx)
            if wprox < 0.5:  # 离墙<1m
                lane_scores["中"] = min(1.0, lane_scores.get("中",0)+0.5)
                best_lane = max(lane_scores, key=lane_scores.get)
                best_clr = lane_scores[best_lane]

            # 速度: clearance高加速, 墙近减速
            if best_clr > 0.8 and wprox > 0.7: speed = min(speed+0.15, SPEED_MAX)
            elif best_clr < 0.3 or wprox < 0.3: speed = max(speed-0.5, SPEED)

            # 方向 = 目标方向 + 车道偏移
            tgt_yaw = target_yaw(bx, by, wp_idx)
            rdx, rdy = road_direction(wp_idx)
            nx_dir, ny_dir = road_normal(wp_idx)
            offset = LANE_OFFSETS[best_lane]
            # 车道目标点: 前方2m + 侧偏移
            lx = bx+rdx*3.0+nx_dir*offset; ly = by+rdy*3.0+ny_dir*offset
            lane_yaw = math.atan2(ly-by, lx-bx)
            # 融合: 70%车道 + 30%CP方向
            diff = (lane_yaw-tgt_yaw+math.pi)%(2*math.pi)-math.pi
            steer_yaw = tgt_yaw + diff*0.5

            yaw_err = (steer_yaw-yaw+math.pi)%(2*math.pi)-math.pi
            dyaw = max(-YAW_RATE*m.opt.timestep, min(YAW_RATE*m.opt.timestep, yaw_err))
            yaw += dyaw

            if step%200==0:
                scores_str = " ".join(f"{n}={lane_scores[n]:.2f}" for n in LANE_OFFSETS)
                print(f"  [{step}] ({bx:.1f},{by:.1f}) CP{wp_idx} v={speed:.1f} d={dist_to_cp:.1f} →{best_lane} wall={wprox:.1f} [{scores_str}]", flush=True)

        # ── 碰撞兜底 ──
        vx=math.cos(yaw)*speed; vy=math.sin(yaw)*speed
        nx=bx+vx*m.opt.timestep; ny=by+vy*m.opt.timestep
        blocked = is_blocked(nx, ny)
        wall_near = sample_hfield_at(nx-0.3, ny)!=ROAD_PIX or sample_hfield_at(nx+0.3, ny)!=ROAD_PIX or sample_hfield_at(nx, ny-0.3)!=ROAD_PIX or sample_hfield_at(nx, ny+0.3)!=ROAD_PIX

        if force_steps>0:
            force_steps-=1; d.qvel[0]=vx; d.qvel[1]=vy
        elif blocked or wall_near:
            if not escaping:
                bounce+=1; escaping=True; speed=SPEED
                deg=random.uniform(45,120)*random.choice([-1,1]); yaw+=math.radians(deg)
                print(f"💥 BOUNCE#{bounce} step={step} ({bx:.1f},{by:.1f}) Δ{deg:+.0f}° wall={wprox:.1f}", flush=True)
            else:
                deg=random.uniform(45,120)*random.choice([-1,1]); yaw+=math.radians(deg)
            d.qvel[:]=0; force_steps=int(0.4/(SPEED*m.opt.timestep))
        else:
            escaping=False; d.qvel[0]=vx; d.qvel[1]=vy

        mujoco.mj_step(m,d); step+=1; v.sync()

    print(f"done: {wp_idx}/{len(nav_wps)} step={step} time={time.time()-t0:.1f}s bounces={bounce}", flush=True)
