#!/usr/bin/env python3
"""algo2_lane_switch — 探索栅格 + 门穿越

策略: 1m=一道门。3车道=3扇并列门。LIDAR看15m=前面15道门。
      优先走未探索的门朝导航点方向。走过的门标记已探索，不再走。
"""
import sys, os, math, time, random
import numpy as np
from PIL import Image
import mujoco, mujoco.viewer

MAP = os.path.expanduser("~/workspace/simtrack/confirmed/track_clean.png")
hf = np.array(Image.open(MAP))

SCALE = 2.0; HF_RES = 2000; PIX_PER_M = 40; ROAD_PIX = 128
SAFE_R = 1.0; SPEED = 4.0; SPEED_MAX = 6.0; YAW_RATE = 6.0
CP_RADIUS = 2.0  # 2m内就算到达
LIDAR_RANGE = 15.0; DOOR_SIZE = 1.0; LIDAR_HZ = 10  # 每米一道门
LANES = {"左": -1.5, "中": 0.0, "右": 1.5}
LANE_ORDER = ["中", "右", "左"]  # 优先顺序

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

# ── 探索栅格 (0=未知, 1=通畅, 2=墙/障碍, 3=已走过) ──
GRID_W, GRID_H = 100, 100
UNKNOWN, FREE, WALL, VISITED = 0, 1, 2, 3
grid = np.zeros((GRID_H, GRID_W), dtype=np.int8)

def sample_hf(wx, wy):
    mx, my = wx/SCALE, wy/SCALE
    px, py = int(mx*PIX_PER_M), HF_RES-1-int(my*PIX_PER_M)
    return int(hf[py,px]) if 0<=px<HF_RES and 0<=py<HF_RES else -1

def is_wall_g(wx, wy): return sample_hf(wx, wy) != ROAD_PIX
def is_obs_g(wx, wy):
    for ox, oy in obs_world:
        if math.hypot(wx-ox, wy-oy) < OBS_CLEAR: return True
    return False
def blocked(wx, wy): return is_wall_g(wx, wy) or is_obs_g(wx, wy)

def mark_visible(bx, by, wp_idx):
    """用LIDAR扫描标记栅格: 射线碰到墙/障碍前=通畅, 碰到的点=墙"""
    rdx, rdy = road_direction(wp_idx)
    for a in np.linspace(0, 2*math.pi, 120):
        for d in np.arange(0.5, LIDAR_RANGE+0.1, 0.5):
            wx = bx + math.cos(a)*d; wy = by + math.sin(a)*d
            gx, gy = int(wx), int(wy)
            if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
                if blocked(wx, wy):
                    grid[gy, gx] = max(grid[gy, gx], WALL)
                    break
                else:
                    grid[gy, gx] = max(grid[gy, gx], FREE)

def mark_visited_row(bx, by, wp_idx):
    """标记当前位置+同行三道门为已探索"""
    rdx, rdy = road_direction(wp_idx)
    nx_dir, ny_dir = road_normal(wp_idx)
    for lane_name, off in LANES.items():
        wx = bx + nx_dir*off
        wy = by + ny_dir*off
        gx, gy = int(wx), int(wy)
        if 0 <= gx < GRID_W and 0 <= gy < GRID_H:
            grid[gy, gx] = VISITED

def road_direction(wp_idx):
    if wp_idx+1 >= len(nav_wps): return (1,0)
    cx,cy=nav_wps[wp_idx]; nx,ny=nav_wps[wp_idx+1]
    dx,dy=nx-cx,ny-cy; d=math.hypot(dx,dy)
    return (dx/d, dy/d) if d>0.01 else (1,0)

def road_normal(wp_idx):
    rdx, rdy = road_direction(wp_idx); return (-rdy, rdx)

def target_yaw(bx, by, wp_idx):
    tx, ty = nav_wps[wp_idx]; dist = math.hypot(tx-bx, ty-by)
    ang = math.atan2(ty-by, tx-bx)
    if wp_idx+1<len(nav_wps) and dist<CP_RADIUS*2.5:
        nx, ny = nav_wps[wp_idx+1]; ang2 = math.atan2(ny-by, nx-bx)
        t = 1.0-dist/(CP_RADIUS*2.5); diff=(ang2-ang+math.pi)%(2*math.pi)-math.pi
        ang+=diff*t
    return ang

# ── 门穿越决策 ──
def doors_ahead(bx, by, wp_idx):
    """扫描前方15道门×3车道, 返回最佳目标点 (tx, ty) 或 None"""
    rdx, rdy = road_direction(wp_idx)
    nx_dir, ny_dir = road_normal(wp_idx)
    
    # 对每个距离(门)和车道, 检查是否通畅+未探索
    best_dist = 999; best_lane = None; best_wx = best_wy = 0
    
    for dist_m in np.arange(DOOR_SIZE, LIDAR_RANGE+0.1, DOOR_SIZE):
        for lane_name in LANE_ORDER:
            off = LANES[lane_name]
            wx = bx + rdx*dist_m + nx_dir*off
            wy = by + rdy*dist_m + ny_dir*off
            gx, gy = int(wx), int(wy)
            
            if not (0 <= gx < GRID_W and 0 <= gy < GRID_H):
                continue
            if blocked(wx, wy):
                continue
            # 这扇门是通的, 检查它是否未探索
            cell = grid[gy, gx]
            if cell == VISITED:
                continue  # 走过的门不回头
            
            # 优先: 未探索 > 已看到通畅
            score = 0 if cell == UNKNOWN else 1  # UNKNOWN优先
            
            # 中道优先
            if lane_name == "中": score -= 0.5
            elif lane_name == "右": score -= 0.2
            
            # 综合: 距离越近越好(但要够远)
            if dist_m < best_dist or (dist_m == best_dist and score < 0):
                # 但至少走1m以上
                if dist_m >= 1.0:
                    best_dist = dist_m
                    best_lane = lane_name
                    best_wx, best_wy = wx, wy
    
    if best_lane is None:
        return None  # 前面无路
    
    return best_wx, best_wy

def find_any_frontier(bx, by):
    """无前路时: 找最近未探索边界"""
    gx0, gy0 = int(bx), int(by)
    best_dist = 999; best_x = best_y = 0
    
    for dy in range(-15, 16):
        for dx in range(-15, 16):
            gx, gy = gx0+dx, gy0+dy
            if not (0 <= gx < GRID_W and 0 <= gy < GRID_H):
                continue
            if grid[gy, gx] == UNKNOWN and not blocked(gx+0.5, gy+0.5):
                # 检查是否邻接已探索区域
                has_explored = False
                for ndy in range(-1, 2):
                    for ndx in range(-1, 2):
                        ngx, ngy = gx+ndx, gy+ndy
                        if 0 <= ngx < GRID_W and 0 <= ngy < GRID_H:
                            if grid[ngy, ngx] in (FREE, VISITED):
                                has_explored = True
                if has_explored:
                    d = math.hypot(dx, dy)
                    # 偏向wp方向
                    if d < best_dist:
                        best_dist = d
                        best_x, best_y = gx+0.5, gy+0.5
    
    if best_dist < 999:
        return best_x, best_y
    return None

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
wp_idx=0; step=0; speed=SPEED; current_lane="中"; t0=time.time()
lidar_interval = int(1.0/LIDAR_HZ/m.opt.timestep); lidar_tick=0
RENDER_SKIP = 3
stuck_step=0; stuck_x=0.0; stuck_y=0.0
explored_count = 0; last_explored = 0

print(f"=== algo2 探索栅格 v3 === {DOOR_SIZE}m/门 LIDAR={LIDAR_RANGE}m", flush=True)

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
            print(f"✓ CP{wp_idx-1} step={step} explored={explored_count}", flush=True)
            if wp_idx>=len(nav_wps):
                print(f"🏁 FINISH step={step} time={time.time()-t0:.1f}s bounces={bounce}", flush=True)
                break
            continue

        # ── 感知+探索 (10Hz) ──
        lidar_tick += 1
        if lidar_tick % lidar_interval == 0:
            mark_visible(bx, by, wp_idx)
            explored_count = int(np.sum(grid == VISITED))

        # 标记当前位置+同行三道门已走过
        mark_visited_row(bx, by, wp_idx)

        # ── 门穿越决策 ──
        target = doors_ahead(bx, by, wp_idx)
        if target is None:
            # 无前路: 找最近未探索边界
            target = find_any_frontier(bx, by)
            if target is None:
                # 彻底无路: 弹开
                if not escaping:
                    bounce+=1
                    deg=random.uniform(90,180)*random.choice([-1,1]); yaw+=math.radians(deg)
                    d.qvel[:]=0; force_steps=int(0.3/(SPEED*m.opt.timestep))
                    print(f"💥 无路 BOUNCE#{bounce} step={step} Δ{deg:+.0f}°", flush=True)
                    escaping=True
                else:
                    deg=random.uniform(45,120)*random.choice([-1,1]); yaw+=math.radians(deg)
                d.qvel[:]=0; force_steps=int(0.3/(SPEED*m.opt.timestep))
                mujoco.mj_step(m,d); step+=1
                if step%RENDER_SKIP==0: v.sync()
                continue
        
        # ── 朝目标转向 ──
        if not escaping:
            # 目标在车道系中的偏移
            rdx, rdy = road_direction(wp_idx)
            nx_dir, ny_dir = road_normal(wp_idx)
            # 计算目标相对于道路中线的偏移
            tx_rel = (target[0]-bx)*nx_dir + (target[1]-by)*ny_dir
            # 选最近的车道
            best = min(LANES, key=lambda n: abs(LANES[n]-tx_rel))
            if best != current_lane:
                current_lane = best
            
            # 转向
            tgt_yaw = math.atan2(target[1]-by, target[0]-bx)
            yaw_err = (tgt_yaw-yaw+math.pi)%(2*math.pi)-math.pi
            dyaw = max(-YAW_RATE*m.opt.timestep, min(YAW_RATE*m.opt.timestep, yaw_err))
            yaw += dyaw
            
            # 调速: 远处通畅加速, 近处或有墙减速
            clear_dist = math.hypot(target[0]-bx, target[1]-by)
            if clear_dist > 10: speed = min(speed+0.3, SPEED_MAX)
            elif clear_dist < 3: speed = max(speed-0.5, 1.5)
            else: speed = max(speed-0.1, SPEED)

        # ── 碰撞检测 ──
        vx=math.cos(yaw)*speed; vy=math.sin(yaw)*speed
        nx=bx+vx*m.opt.timestep; ny=by+vy*m.opt.timestep
        
        if step-stuck_step>300:
            if math.hypot(bx-stuck_x, by-stuck_y)<0.5:
                if not escaping:
                    bounce+=1; escaping=True
                    deg=random.uniform(90,180)*random.choice([-1,1]); yaw+=math.radians(deg)
                    d.qvel[:]=0; force_steps=int(0.3/(SPEED*m.opt.timestep))
                    print(f"💥 卡死 step={step}", flush=True)
            stuck_step=step; stuck_x=bx; stuck_y=by
        
        if force_steps>0:
            force_steps-=1; d.qvel[0]=vx; d.qvel[1]=vy
        elif blocked(nx, ny):
            if not escaping:
                bounce+=1; escaping=True
                deg=random.uniform(45,120)*random.choice([-1,1]); yaw+=math.radians(deg)
                d.qvel[:]=0; force_steps=int(0.4/(SPEED*m.opt.timestep))
            else:
                deg=random.uniform(45,120)*random.choice([-1,1]); yaw+=math.radians(deg)
        else:
            escaping=False; d.qvel[0]=vx; d.qvel[1]=vy

        mujoco.mj_step(m,d); step+=1
        if step%RENDER_SKIP==0: v.sync()
        if step%200==0:
            mode = "🚪" if target else "🔍"
            exp_pct = explored_count/(GRID_W*GRID_H)*100
            print(f"  [{step}] ({bx:.0f},{by:.0f}) CP{wp_idx} v={speed:.1f} {current_lane} doors={explored_count}({exp_pct:.0f}%) d={dist_to_cp:.0f} {mode}", flush=True)

    print(f"done: {wp_idx}/{len(nav_wps)} step={step} time={time.time()-t0:.1f}s bounces={bounce} explored={explored_count}", flush=True)
