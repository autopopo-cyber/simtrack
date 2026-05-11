#!/usr/bin/env python3
"""algo2_lane_nav — 三车道导航 + 可视化小球 + 玩具车兜底

可视化:
  🟢 绿球(左/中/右) = 车道通畅
  🟡 黄球 = 车道一般  
  🔴 红球 = 车道阻塞
  🔵 蓝球 = 选中车道方向指示

日志: 车道选择/clearance/速度全透明，我来盯。
"""
import sys, os, math, time, random
import numpy as np
from PIL import Image
import mujoco, mujoco.viewer

MAP = os.path.expanduser("~/workspace/simtrack/confirmed/track_clean.png")
hf = np.array(Image.open(MAP))

SCALE = 2.0; HF_RES = 2000; PIX_PER_M = 40; ROAD_PIX = 128
DETECT_R = 0.5; SPEED_MIN = 2.0; SPEED_MAX = 6.0; YAW_RATE = 6.0
CP_RADIUS = 3.0; LOOKAHEAD_M = 5.0; LANE_SAMPLES = 8
LANES = {"左": -1.5, "中": 0.0, "右": 1.5}
LANE_NAMES = list(LANES.keys())

# ── 中心线 & 障碍物 ──
def gen_centerline():
    pts = []; y0 = 2.5
    for seg in range(10):
        y = y0 + seg*5.0; x0, x1 = (5.0,45.0) if seg%2==0 else (45.0,5.0)
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
OBS_R=1.0; OBS_CLEAR=OBS_R+DETECT_R

cps_maze = [(3,3),(47,5),(3,10),(47,15),(3,20),(47,25),(3,30),(47,35),(3,40),(47,45),(3,48)]
nav_wps = [(x*SCALE, y*SCALE) for x,y in cps_maze]

# ── 检测 ──
def sample_hfield_at(wx, wy):
    mx, my = wx/SCALE, wy/SCALE
    px, py = int(mx*PIX_PER_M), HF_RES-1-int(my*PIX_PER_M)
    return int(hf[py,px]) if 0<=px<HF_RES and 0<=py<HF_RES else -1

def is_wall(wx, wy, r=0.45):
    for dy in np.arange(-r, r+0.01, 0.2):
        md = np.sqrt(max(0, r**2-dy**2))
        for dx in np.arange(-md, md+0.01, 0.2):
            if sample_hfield_at(wx+dx, wy+dy) != ROAD_PIX: return True
    return False

def is_obs(wx, wy):
    for ox, oy in obs_world:
        if math.hypot(wx-ox, wy-oy) < OBS_CLEAR: return True
    return False

def is_blocked(wx, wy): return is_wall(wx,wy) or is_obs(wx,wy)

def road_direction(wp_idx):
    if wp_idx+1 >= len(nav_wps): return (1,0)
    cx,cy=nav_wps[wp_idx]; nx,ny=nav_wps[wp_idx+1]
    dx,dy=nx-cx,ny-cy; d=math.hypot(dx,dy)
    return (dx/d, dy/d) if d>0.01 else (1,0)

def lane_clearance(bx, by, wp_idx, lane_offset):
    rdx, rdy = road_direction(wp_idx); nx_dir, ny_dir = -rdy, rdx
    hits = 0
    for i in range(LANE_SAMPLES):
        dist = LOOKAHEAD_M*(i+1)/LANE_SAMPLES
        if is_blocked(bx+rdx*dist+nx_dir*lane_offset, by+rdy*dist+ny_dir*lane_offset):
            hits += 1
    return 1.0 - hits/LANE_SAMPLES

def lane_preview_points(bx, by, wp_idx):
    """返回三车道前瞻采样点的世界坐标 [(lane_name, [(wx,wy,blocked),...]),...]"""
    rdx, rdy = road_direction(wp_idx); nx_dir, ny_dir = -rdy, rdx
    result = []
    for name in LANE_NAMES:
        offset = LANES[name]; pts = []
        for i in range(LANE_SAMPLES):
            dist = LOOKAHEAD_M*(i+1)/LANE_SAMPLES
            wx = bx+rdx*dist+nx_dir*offset; wy = by+rdy*dist+ny_dir*offset
            pts.append((wx, wy, is_blocked(wx,wy)))
        result.append((name, pts))
    return result

def target_yaw(bx, by, wp_idx):
    tx, ty = nav_wps[wp_idx]; dist = math.hypot(tx-bx, ty-by)
    ang = math.atan2(ty-by, tx-bx)
    if wp_idx+1<len(nav_wps) and dist<CP_RADIUS*2:
        nx, ny = nav_wps[wp_idx+1]; ang2 = math.atan2(ny-by, nx-bx)
        t = 1.0-dist/(CP_RADIUS*2); diff=(ang2-ang+math.pi)%(2*math.pi)-math.pi
        ang+=diff*t
    return ang

# ── 可视化球体色 ──
def clr_to_rgba(clr):
    """clearance→RGBA: >0.7绿, 0.3-0.7黄, <0.3红"""
    if clr > 0.7: return "0.1 0.9 0.1 0.9"
    elif clr > 0.3: return "1.0 0.85 0.1 0.9"
    return "0.9 0.1 0.1 0.9"

# ── XML ──
CP_XML = "".join(f'<body mocap="true" pos="{x} {y} 2"><geom type="sphere" size="1.5" rgba="0.2 0.5 1 0.8"/></body>' for x,y in nav_wps[1:])
OBS_XML = "".join(f'<body name="obs{i}" pos="{x:.1f} {y:.1f} 2.0"><geom type="cylinder" size="1.0 2.0" rgba="0.9 0.2 0.2 0.9"/></body>' for i,(x,y) in enumerate(obs_world))
# 可视化小球: 3车道 × 8采样 = 24球 + 1个方向指示
VIZ_XML = "".join(
    f'<body mocap="true" name="viz_{lane}_{i}" pos="0 0 -10"><geom type="sphere" size="0.15" rgba="0.5 0.5 0.5 0.5"/></body>'
    for lane in LANE_NAMES for i in range(LANE_SAMPLES)
)
VIZ_XML += '<body mocap="true" name="viz_dir" pos="0 0 -10"><geom type="sphere" size="0.3" rgba="0.1 0.5 1.0 0.9"/></body>'

xml = f"""<mujoco>
  <compiler angle="radian"/><option timestep="0.005"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset><hfield name="track" size="50.0 50.0 4.0 2.0" file="{MAP}"/></asset>
  <worldbody>
    <light pos="50 50 80" dir="0 0 -1"/>{CP_XML}{OBS_XML}{VIZ_XML}
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

# 预取可视化body ID
viz_ids = {}
for lane in LANE_NAMES:
    for i in range(LANE_SAMPLES):
        viz_ids[(lane,i)] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"viz_{lane}_{i}")
viz_dir_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "viz_dir")

ESCAPE_STEPS = int(0.4/(SPEED_MIN*m.opt.timestep))
yaw=0.0; bounce=0; force_steps=0; escaping=False
wp_idx=0; step=0; speed=SPEED_MIN; current_lane="中"; t0=time.time()
viz_update_tick=0

def update_viz(bx, by, wp_idx, best_lane):
    """更新可视化小球位置和颜色"""
    preview = lane_preview_points(bx, by, wp_idx)
    for lane_name, pts in preview:
        clr = lane_clearance(bx, by, wp_idx, LANES[lane_name])
        rgba_str = clr_to_rgba(clr)
        rgba = [float(x) for x in rgba_str.split()]
        for i, (wx, wy, blocked) in enumerate(pts):
            bid = viz_ids[(lane_name,i)]
            mid = m.body_mocapid[bid]
            d.mocap_pos[mid][0] = wx; d.mocap_pos[mid][1] = wy; d.mocap_pos[mid][2] = 0.5
            m.geom_rgba[m.body_geomadr[bid]] = rgba
    # 方向指示球
    offset = LANES.get(best_lane, 0.0)
    rdx, rdy = road_direction(wp_idx); nx_dir, ny_dir = -rdy, rdx
    mid = m.body_mocapid[viz_dir_id]
    d.mocap_pos[mid][0] = bx+rdx*1.5+nx_dir*offset
    d.mocap_pos[mid][1] = by+rdy*1.5+ny_dir*offset
    d.mocap_pos[mid][2] = 0.6
    m.geom_rgba[m.body_geomadr[viz_dir_id]] = [0.1, 0.5, 1.0, 0.9]

print(f"=== algo2_lane_nav === 车道:{LANE_NAMES} lookahead={LOOKAHEAD_M}m v={SPEED_MIN}→{SPEED_MAX}", flush=True)

with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
    v.cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.distance=25; v.cam.elevation=-35; v.cam.azimuth=180
    print("viewer ready (小球可视化已激活)", flush=True)

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

        # ── 三车道评估 ──
        if not escaping:
            lane_scores = {n: lane_clearance(bx,by,wp_idx,LANES[n]) for n in LANE_NAMES}
            best_lane = max(lane_scores, key=lane_scores.get)
            best_clr = lane_scores[best_lane]; current_lane = best_lane

            if best_clr > 0.8: speed = min(speed+0.2, SPEED_MAX)
            elif best_clr < 0.3: speed = max(speed-0.5, SPEED_MIN)

            desired_yaw = target_yaw(bx, by, wp_idx)
            rdx, rdy = road_direction(wp_idx); nx_dir, ny_dir = -rdy, rdx
            offset = LANES[best_lane]
            desired_x = bx+rdx+0.5*nx_dir*offset; desired_y = by+rdy+0.5*ny_dir*offset
            lane_yaw = math.atan2(desired_y-by, desired_x-bx)
            diff = (lane_yaw-desired_yaw+math.pi)%(2*math.pi)-math.pi
            steer_yaw = desired_yaw + diff*0.4

            yaw_err = (steer_yaw-yaw+math.pi)%(2*math.pi)-math.pi
            dyaw = max(-YAW_RATE*m.opt.timestep, min(YAW_RATE*m.opt.timestep, yaw_err))
            yaw += dyaw

            # 更新可视化 (每10步)
            if viz_update_tick%5==0:
                update_viz(bx, by, wp_idx, best_lane)
            viz_update_tick+=1

            if step%200==0:
                scores_str = " ".join(f"{n}={lane_scores[n]:.2f}" for n in LANE_NAMES)
                print(f"  [{step}] ({bx:.1f},{by:.1f}) CP{wp_idx} v={speed:.1f} d={dist_to_cp:.1f} →{best_lane} [{scores_str}]", flush=True)

        # ── 碰撞/玩具车 ──
        vx=math.cos(yaw)*speed; vy=math.sin(yaw)*speed
        nx=bx+vx*m.opt.timestep; ny=by+vy*m.opt.timestep
        wall = is_blocked(nx, ny)
        if force_steps>0:
            force_steps-=1; d.qvel[0]=vx; d.qvel[1]=vy
        elif wall:
            if not escaping:
                bounce+=1; escaping=True; speed=SPEED_MIN
                deg=random.uniform(30,90)*random.choice([-1,1]); yaw+=math.radians(deg)
                print(f"💥 BOUNCE#{bounce} step={step} ({bx:.1f},{by:.1f}) Δ{deg:+.0f}° lane={current_lane}", flush=True)
            else:
                deg=random.uniform(30,90)*random.choice([-1,1]); yaw+=math.radians(deg)
            d.qvel[:]=0; force_steps=ESCAPE_STEPS
        else:
            escaping=False; d.qvel[0]=vx; d.qvel[1]=vy

        mujoco.mj_step(m,d); step+=1; v.sync()

    print(f"done: {wp_idx}/{len(nav_wps)} step={step} time={time.time()-t0:.1f}s bounces={bounce}", flush=True)
