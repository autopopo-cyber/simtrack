#!/usr/bin/env python3
"""algo3_lane_switch — 三车道切换导航

策略: 5m路分左(-1.5m)/中(0)/右(+1.5m)三条车道。
      默认走中路。15m lidar探测障碍→评三条车道→选通畅的。
      障碍通过后回归中路。每次变道打印log。
"""
import sys, os, math, time, random
import numpy as np
from PIL import Image
import mujoco, mujoco.viewer

MAP = os.path.expanduser("~/workspace/simtrack/confirmed/track_clean.png")
hf = np.array(Image.open(MAP))

SCALE = 2.0; HF_RES = 2000; PIX_PER_M = 40; ROAD_PIX = 128
SAFE_R = 1.0; SPEED = 4.0; SPEED_MAX = 6.0; YAW_RATE = 6.0
CP_RADIUS = 3.0
LIDAR_RANGE = 15.0; LIDAR_RAYS = 120; LIDAR_HZ = 10
LOOKAHEAD = 12.0  # 前瞻距离
LANES = {"左": -1.5, "中": 0.0, "右": 1.5}

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

def is_wall(wx, wy): return sample_hf(wx, wy) != ROAD_PIX

def is_obs(wx, wy):
    for ox, oy in obs_world:
        if math.hypot(wx-ox, wy-oy) < OBS_CLEAR: return True
    return False

def road_direction(wp_idx):
    if wp_idx+1 >= len(nav_wps): return (1,0)
    cx,cy=nav_wps[wp_idx]; nx,ny=nav_wps[wp_idx+1]
    dx,dy=nx-cx,ny-cy; d=math.hypot(dx,dy)
    return (dx/d, dy/d) if d>0.01 else (1,0)

def road_normal(wp_idx):
    rdx, rdy = road_direction(wp_idx); return (-rdy, rdx)

def turn_info(wp_idx):
    """检测是否接近弯道: (is_turn, outer_side) outer_side='左'/'右'"""
    if wp_idx >= len(nav_wps)-2: return (False, None)
    rdx1, rdy1 = road_direction(wp_idx)
    rdx2, rdy2 = road_direction(wp_idx+1)
    cross = rdx1*rdy2 - rdy1*rdx2
    dot = rdx1*rdx2 + rdy1*rdy2
    angle = math.degrees(math.acos(max(-1, min(1, dot))))
    if angle > 30:
        return (True, "左" if cross > 0 else "右")
    return (False, None)

def dist_to_turn(bx, by, wp_idx):
    """到达弯道起点的距离"""
    if wp_idx >= len(nav_wps): return 999
    tx, ty = nav_wps[wp_idx]
    return math.hypot(tx-bx, ty-by)

def target_yaw(bx, by, wp_idx):
    tx, ty = nav_wps[wp_idx]; dist = math.hypot(tx-bx, ty-by)
    ang = math.atan2(ty-by, tx-bx)
    if wp_idx+1<len(nav_wps) and dist<CP_RADIUS*2.5:
        nx, ny = nav_wps[wp_idx+1]; ang2 = math.atan2(ny-by, nx-bx)
        t = 1.0-dist/(CP_RADIUS*2.5); diff=(ang2-ang+math.pi)%(2*math.pi)-math.pi
        ang+=diff*t
    return ang

# ── 车道评估 ──
def lane_clearance(bx, by, wp_idx):
    """评估三条车道的前方通畅度: 沿道路方向采样LOOKAHEAD米, 返回{车道名: (通畅距离, 最窄宽度)}"""
    rdx, rdy = road_direction(wp_idx)
    nx_dir, ny_dir = road_normal(wp_idx)
    
    result = {}
    for name, offset in LANES.items():
        min_clear = LOOKAHEAD  # 通畅距离
        min_width = 5.0        # 最窄可用宽度
        for d in np.arange(0.5, LOOKAHEAD+0.1, 0.5):
            cx = bx + rdx*d + nx_dir*offset
            cy = by + rdy*d + ny_dir*offset
            if is_wall(cx, cy) or is_obs(cx, cy):
                min_clear = min(min_clear, d)
            # 测该点左右可用宽度
            w = 0
            for side in [-1, 1]:
                for ww in np.arange(0.1, 2.6, 0.2):
                    wx = cx + nx_dir*side*ww; wy = cy + ny_dir*side*ww
                    if is_wall(wx, wy) or is_obs(wx, wy): break
                    w += 0.2
            min_width = min(min_width, w)
        result[name] = (min_clear, min_width)
    return result

# ── lidar ──
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

yaw=0.0; bounce=0; force_steps=0; escaping=False
wp_idx=0; step=0; speed=SPEED; current_lane="中"; t0=time.time()
lidar_interval = int(1.0/LIDAR_HZ/m.opt.timestep); lidar_tick=0

print(f"=== algo3_lane_switch === 默认中路 探测{LIDAR_RANGE}m →{SPEED_MAX}m/s", flush=True)

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

        tx, ty = nav_wps[wp_idx]; dist_to_cp = math.hypot(tx-bx, ty-by)
        if dist_to_cp < CP_RADIUS:
            wp_idx+=1
            print(f"✓ CP{wp_idx-1} step={step} v={speed:.1f} lane={current_lane}", flush=True)
            if wp_idx>=len(nav_wps):
                print(f"🏁 FINISH step={step} time={time.time()-t0:.1f}s bounces={bounce}", flush=True)
                break
            continue

        # ── 三车道评估 (10Hz) ──
        lidar_tick += 1
        if lidar_tick % lidar_interval == 0:
            clr = lane_clearance(bx, by, wp_idx)
            is_turn, outer_side = turn_info(wp_idx)
            d_turn = dist_to_turn(bx, by, wp_idx)
            mid_clear, mid_width = clr["中"]
            
            # ── 弯道策略: 靠外墙扩视野 ──
            if is_turn and d_turn < 15.0:
                turn_mode = True
                # 外侧=视野更好的一侧
                best_lane = outer_side  # outer_side='左'或'右'
            else:
                turn_mode = False
                # 直道规则: 默认中路。如果中路前方<8m有障, 选最通畅的
                if mid_clear < 8.0:
                    best_lane = max(clr, key=lambda n: clr[n][0]*0.7 + clr[n][1]*0.3)
                elif current_lane != "中":
                    if mid_clear > 12.0:
                        best_lane = "中"
                    else:
                        best_lane = current_lane
                else:
                    best_lane = "中"
            
            if best_lane != current_lane:
                reason = f"弯道靠外={outer_side}" if turn_mode else f"中(clear={mid_clear:.1f}m)"
                print(f"  🔄 变道 step={step} {current_lane}→{best_lane} {reason}", flush=True)
                current_lane = best_lane
            
            # 调速: 弯道减速, 直道加速
            cur_clear = clr[current_lane][0]
            if turn_mode:
                speed = max(2.0, speed - 0.5)  # 弯道减速到2m/s
            elif cur_clear > 12:
                speed = min(speed+0.3, SPEED_MAX)
            elif cur_clear < 4:
                speed = max(speed-0.5, SPEED)
            else:
                speed = max(speed-0.1, SPEED)

        # ── 转向: 朝CP + 车道偏移 ──
        if not escaping:
            tgt_yaw = target_yaw(bx, by, wp_idx)
            rdx, rdy = road_direction(wp_idx)
            nx_dir, ny_dir = road_normal(wp_idx)
            off = LANES[current_lane]
            # 车道目标: 前方5m + 侧偏
            lx = bx + rdx*5.0 + nx_dir*off; ly = by + rdy*5.0 + ny_dir*off
            lane_yaw = math.atan2(ly-by, lx-bx)
            diff = (lane_yaw-tgt_yaw+math.pi)%(2*math.pi)-math.pi
            steer_yaw = tgt_yaw + diff*0.4
            
            yaw_err = (steer_yaw-yaw+math.pi)%(2*math.pi)-math.pi
            dyaw = max(-YAW_RATE*m.opt.timestep, min(YAW_RATE*m.opt.timestep, yaw_err))
            yaw += dyaw

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
                print(f"💥 BOUNCE#{bounce} step={step} ({bx:.1f},{by:.1f}) lane={current_lane} Δ{deg:+.0f}°", flush=True)
            else:
                deg=random.uniform(45,120)*random.choice([-1,1]); yaw+=math.radians(deg)
            d.qvel[:]=0; force_steps=int(0.4/(SPEED*m.opt.timestep))
        else:
            escaping=False; d.qvel[0]=vx; d.qvel[1]=vy

        mujoco.mj_step(m,d); step+=1; v.sync()

        if step%200==0:
            c = clr.get("中",(0,0)) if 'clr' in dir() else (0,0)
            print(f"  [{step}] ({bx:.1f},{by:.1f}) CP{wp_idx} v={speed:.1f} {current_lane} d={dist_to_cp:.1f}", flush=True)

    print(f"done: {wp_idx}/{len(nav_wps)} step={step} time={time.time()-t0:.1f}s bounces={bounce}", flush=True)
