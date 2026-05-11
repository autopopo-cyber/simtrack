#!/usr/bin/env python3
"""algo2_astar_lidar — A*路径规划 + 三车道lidar导航

1. 用A*预计算经过相邻路点的最优路径（网格分辨率0.5m）
2. 沿A*路径走，lidar检测障碍→切换车道
3. 目标: 零碰撞到终点
"""
import sys, os, math, time, random, heapq
import numpy as np
from PIL import Image
import mujoco, mujoco.viewer

MAP = os.path.expanduser("~/workspace/simtrack/confirmed/track_clean.png")
hf = np.array(Image.open(MAP))

SCALE = 2.0; HF_RES = 2000; PIX_PER_M = 40; ROAD_PIX = 128
SAFE_R = 1.0
SPEED = 2.0; SPEED_MAX = 6.0; YAW_RATE = 6.0
CP_RADIUS = 3.0
GRID_RES = 0.5          # A*网格分辨率(m)
LIDAR_RANGE = 15.0; LIDAR_RAYS = 240; LIDAR_HZ = 10
LOOKAHEAD = 6.0

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

# ── hfield 采样 ──
def sample_hf(wx, wy):
    mx, my = wx/SCALE, wy/SCALE
    px, py = int(mx*PIX_PER_M), HF_RES-1-int(my*PIX_PER_M)
    return int(hf[py,px]) if 0<=px<HF_RES and 0<=py<HF_RES else -1

def is_wall(wx, wy): return sample_hf(wx, wy) != ROAD_PIX

def is_obs(wx, wy):
    for ox, oy in obs_world:
        if math.hypot(wx-ox, wy-oy) < OBS_CLEAR: return True
    return False

# ── A* 路径规划 ──
def astar_path(start_w, end_w):
    """A*在0.5m网格上找路径, 只走路面(ROAD_PIX)"""
    sx, sy = int(start_w[0]/GRID_RES), int(start_w[1]/GRID_RES)
    ex, ey = int(end_w[0]/GRID_RES), int(end_w[1]/GRID_RES)
    
    h = lambda x,y: abs(x-ex)+abs(y-ey)
    open_set = [(h(sx,sy), 0, sx, sy)]
    came_from = {}; g_score = {(sx,sy): 0}
    
    while open_set:
        _, cost, cx, cy = heapq.heappop(open_set)
        if (cx,cy) == (ex,ey):
            # 回溯
            path = [(ex*GRID_RES, ey*GRID_RES)]
            while (cx,cy) in came_from:
                cx, cy = came_from[(cx,cy)]
                path.append((cx*GRID_RES, cy*GRID_RES))
            path.reverse()
            return path
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
            nx, ny = cx+dx, cy+dy
            wwx, wwy = nx*GRID_RES, ny*GRID_RES
            if sample_hf(wwx, wwy) != ROAD_PIX: continue
            if is_obs(wwx, wwy): continue
            ng = cost + (1.414 if dx and dy else 1.0)
            if ng < g_score.get((nx,ny), float('inf')):
                g_score[(nx,ny)] = ng; came_from[(nx,ny)] = (cx,cy)
                heapq.heappush(open_set, (ng+h(nx,ny), ng, nx, ny))
    return None  # A*失败: 直连

def road_direction(wp_idx):
    if wp_idx+1 >= len(nav_wps): return (1,0)
    cx,cy=nav_wps[wp_idx]; nx,ny=nav_wps[wp_idx+1]
    dx,dy=nx-cx,ny-cy; d=math.hypot(dx,dy)
    return (dx/d, dy/d) if d>0.01 else (1,0)

def road_normal(wp_idx):
    rdx, rdy = road_direction(wp_idx); return (-rdy, rdx)

# ── 激光雷达 ──
def lidar_scan(bx, by, m, d, site_id):
    pts = []
    gg = np.ones(6, dtype=np.uint8)*255; gid = np.array([-1], np.int32)
    pos = d.site_xpos[site_id].copy()
    for i in range(LIDAR_RAYS):
        a = 2*math.pi*i/LIDAR_RAYS
        dw = np.array([math.cos(a), math.sin(a), 0.0])
        gid[0] = -1
        dist = mujoco.mj_ray(m, d, pos, dw, gg, 1, -1, gid)
        if gid[0] >= 0 and 0 < dist < LIDAR_RANGE:
            hit = pos + dw*dist; pts.append((float(hit[0]), float(hit[1])))
    return pts

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

# ── 预计算所有A*路径 ──
print("A*预计算...", flush=True)
all_astar_paths = []
for i in range(len(nav_wps)-1):
    path = astar_path(nav_wps[i], nav_wps[i+1])
    if path:
        all_astar_paths.append(path)
        print(f"  CP{i}→CP{i+1}: {len(path)}点", flush=True)
    else:
        print(f"  CP{i}→CP{i+1}: A*失败, 直连", flush=True)
        all_astar_paths.append([nav_wps[i], nav_wps[i+1]])

yaw=0.0; bounce=0; force_steps=0; escaping=False
wp_idx=0; path_idx=0; step=0; speed=SPEED; t0=time.time()
lidar_interval = int(1.0/LIDAR_HZ/m.opt.timestep); lidar_tick=0; lidar_cache=[]

def steer_to_goal(bx, by, gx, gy):
    """朝目标点转向, 限制角速度"""
    target = math.atan2(gy-by, gx-bx)
    err = (target-yaw+math.pi)%(2*math.pi)-math.pi
    return max(-YAW_RATE*m.opt.timestep, min(YAW_RATE*m.opt.timestep, err))

print(f"=== algo2_astar_lidar === A*就绪 ok", flush=True)

with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
    v.cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.distance=25; v.cam.elevation=-35; v.cam.azimuth=180
    print("viewer ready", flush=True)

    while v.is_running() and wp_idx<len(nav_wps):
        bx, by = d.qpos[0], d.qpos[1]
        if bx<1 or bx>99 or by<1 or by>99:
            d.qpos[0]=max(1,min(99,bx)); d.qpos[1]=max(1,min(99,by))
            d.qvel[:]=0; yaw=random.uniform(0,2*math.pi)
        v.cam.lookat[:]=np.array([bx, by, 0.5], dtype=np.float64)

        # 航点到达
        tx, ty = nav_wps[wp_idx]; dist_to_cp = math.hypot(tx-bx, ty-by)
        if dist_to_cp < CP_RADIUS:
            wp_idx+=1; path_idx=0
            print(f"✓ CP{wp_idx-1} step={step} v={speed:.1f}", flush=True)
            if wp_idx>=len(nav_wps):
                print(f"🏁 FINISH step={step} time={time.time()-t0:.1f}s bounces={bounce}", flush=True)
                break
            continue

        # ── A*路径跟随 ──
        astar = all_astar_paths[wp_idx]
        # 找最近路径点
        while path_idx < len(astar)-1:
            nx_pt, ny_pt = astar[path_idx+1]
            if math.hypot(nx_pt-bx, ny_pt-by) < math.hypot(astar[path_idx][0]-bx, astar[path_idx][1]-by):
                path_idx += 1
            else: break
        gx, gy = astar[min(path_idx+3, len(astar)-1)]  # 前瞻3步

        # ── lidar ──
        lidar_tick += 1
        if lidar_tick % lidar_interval == 0:
            lidar_cache = lidar_scan(bx, by, m, d, lidar_site)

        if not escaping:
            # 三车道检测: 沿A*路径方向分车道
            rdx, rdy = gx-bx, gy-by; dg = math.hypot(rdx, rdy)
            if dg > 0.01: rdx/=dg; rdy/=dg
            else: rdx, rdy = 1, 0
            nx_dir, ny_dir = -rdy, rdx
            
            lane_hits = {"左":0, "中":0, "右":0}
            offsets = {"左":-1.5, "中":0.0, "右":1.5}
            for name, off in offsets.items():
                lx = bx+nx_dir*off; ly = by+ny_dir*off
                for dd in np.arange(1.0, LOOKAHEAD+0.1, 1.0):
                    cx = lx+rdx*dd; cy = ly+rdy*dd
                    if is_wall(cx, cy) or is_obs(cx, cy):
                        lane_hits[name] += 1
                    elif any(math.hypot(px-cx, py-cy) < 0.8 for px, py in lidar_cache):
                        lane_hits[name] += 1
            
            best_lane = min(lane_hits, key=lane_hits.get) if lane_hits else "中"
            
            # 速度: 通畅加速
            if lane_hits[best_lane] == 0: speed = min(speed+0.2, SPEED_MAX)
            elif lane_hits[best_lane] >= 3: speed = max(speed-0.5, SPEED)
            else: speed = max(speed-0.1, SPEED)

            # 转向: 车道目标 + A*目标
            off = offsets[best_lane]
            lx = bx+nx_dir*off+rdx*2.0; ly = by+ny_dir*off+rdy*2.0
            lane_yaw = math.atan2(ly-by, lx-bx)
            astar_yaw = math.atan2(gy-by, gx-bx)
            diff = (lane_yaw-astar_yaw+math.pi)%(2*math.pi)-math.pi
            steer_yaw = astar_yaw + diff*0.6
            
            old_yaw = yaw
            err = (steer_yaw-old_yaw+math.pi)%(2*math.pi)-math.pi
            dyaw = max(-YAW_RATE*m.opt.timestep, min(YAW_RATE*m.opt.timestep, err))
            yaw += dyaw

            if step%200==0:
                print(f"  [{step}] ({bx:.1f},{by:.1f}) CP{wp_idx} v={speed:.1f} →{best_lane} hits={lane_hits} d={dist_to_cp:.1f}", flush=True)

        # ── 碰撞 ──
        vx=math.cos(yaw)*speed; vy=math.sin(yaw)*speed
        nx=bx+vx*m.opt.timestep; ny=by+vy*m.opt.timestep
        blocked = is_wall(nx,ny) or is_obs(nx,ny)
        if force_steps>0:
            force_steps-=1; d.qvel[0]=vx; d.qvel[1]=vy
        elif blocked:
            if not escaping:
                bounce+=1; escaping=True; speed=SPEED
                deg=random.uniform(45,120)*random.choice([-1,1]); yaw+=math.radians(deg)
                print(f"💥 BOUNCE#{bounce} step={step} ({bx:.1f},{by:.1f}) Δ{deg:+.0f}°", flush=True)
            else:
                deg=random.uniform(45,120)*random.choice([-1,1]); yaw+=math.radians(deg)
            d.qvel[:]=0; force_steps=int(0.4/(SPEED*m.opt.timestep))
        else:
            escaping=False; d.qvel[0]=vx; d.qvel[1]=vy

        mujoco.mj_step(m,d); step+=1; v.sync()

    print(f"done: {wp_idx}/{len(nav_wps)} step={step} time={time.time()-t0:.1f}s bounces={bounce}", flush=True)
