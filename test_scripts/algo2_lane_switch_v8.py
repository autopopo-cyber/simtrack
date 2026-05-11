#!/usr/bin/env python3
"""algo2_lane_switch — 世界体素 v8 (萤火算法)

v7→v8: wall_distance搜索半径固定8 + target hysteresis防晃荡 + 权重调优
"""
import sys, os, math, time, random
import numpy as np
from PIL import Image
import mujoco, mujoco.viewer

MAP = os.path.expanduser("~/workspace/simtrack/confirmed/track_clean.png")
hf = np.array(Image.open(MAP))

SCALE = 2.0; HF_RES = 2000; PIX_PER_M = 40; ROAD_PIX = 128
SAFE_R = 1.0; SPEED = 4.0; SPEED_MAX = 6.0; YAW_RATE = 6.0
CP_RADIUS = 2.0; LIDAR_RANGE = 15.0; VOXEL = 1.0

UNKNOWN, FREE, WALL, VISITED = 0, 1, 2, 3
W = 100; vox = np.zeros((W, W), dtype=np.int8)

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
obs_world = [(x,y) for x,y in obs_world if math.hypot(x-10,y-5)>5.0]
OBS_R = 1.0; OBS_CLEAR = OBS_R+SAFE_R

cps_maze = [(5,2.5),(47,5),(3,10),(47,15),(3,20),(47,25),(3,30),(47,35),(3,40),(47,45),(3,48)]
nav_wps = [(x*SCALE, y*SCALE) for x,y in cps_maze]

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

def scan_voxels(bx, by):
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

def line_clear(bx, by, wx, wy):
    dx, dy = wx-bx, wy-by
    dist = math.hypot(dx, dy)
    if dist < 0.1: return True
    steps = int(dist / 0.3)
    for i in range(1, steps):
        t = i / steps
        if blocked(bx+dx*t, by+dy*t): return False
    return True

def wall_distance(vx, vy):
    """到最近WALL体素的距离, 固定搜索半径8"""
    best = 999.0
    cx, cy = vx+0.5, vy+0.5
    r = 8
    for ndy in range(-r, r+1):
        for ndx in range(-r, r+1):
            nx, ny = vx+ndx, vy+ndy
            if not (0 <= nx < W and 0 <= ny < W): continue
            if vox[ny, nx] == WALL:
                d = math.hypot(cx-(nx+0.5), cy-(ny+0.5))
                if d < best: best = d
    return best

def find_frontier(bx, by, wp_idx, prev_target=None, verbose=True):
    """挑最佳体素: line_clear + 邻接已探索 + 离墙远 + 朝wp + hysteresis防晃"""
    cx, cy = int(bx), int(by)
    wp_yaw = target_yaw(bx, by, wp_idx)
    best_score = -9999; best = None
    
    # hysteresis: 如果上一轮选中的target仍然line_clear, 给它加分
    prev_bonus = None
    if prev_target:
        ptx, pty = prev_target
        pvx, pvy = int(ptx), int(pty)
        if (0 <= pvx < W and 0 <= pvy < W and vox[pvy, pvx] == FREE
            and line_clear(bx, by, ptx, pty)):
            # 算prev的分数, 加hysteresis bonus
            prev_bonus = True
    
    for dy in range(-20, 21):
        for dx in range(-20, 21):
            vx, vy = cx+dx, cy+dy
            if not (0 <= vx < W and 0 <= vy < W): continue
            if vox[vy, vx] != FREE: continue
            
            if not line_clear(bx, by, vx+0.5, vy+0.5): continue
            
            adjacent = any(vox[vy+ndy, vx+ndx] in (VISITED, FREE)
                          for ndy in (-1,0,1) for ndx in (-1,0,1)
                          if 0<=vx+ndx<W and 0<=vy+ndy<W)
            if not adjacent: continue
            
            # v9: 离墙太近的体素直接禁掉 (1.5m安全距离)
            if wall_distance(vx, vy) < 2.0: continue
            
            score = 0
            ang = math.atan2(vy+0.5-by, vx+0.5-bx)
            diff = abs((ang-wp_yaw+math.pi)%(2*math.pi)-math.pi)
            score -= diff * 20          # v8: 从30降到20, 给wall_distance更多话语权
            score -= math.hypot(dx, dy) * 3  # v8: 从5降到3
            
            wd = wall_distance(vx, vy)
            score += wd * 25            # v8: 从40降到25, 平衡方向
            
            unknown_nb = sum(1 for ndy in (-1,0,1) for ndx in (-1,0,1)
                           if 0<=vx+ndx<W and 0<=vy+ndy<W and vox[vy+ndy, vx+ndx]==UNKNOWN)
            score += unknown_nb * 15    # v8: 从20降到15
            

            
            if score > best_score:
                best_score = score
                best = (vx+0.5, vy+0.5)
    
    if best and wp_idx < 3:
        bx2, by2 = best
        print(f'  [DEBUG] best=({bx2:.1f},{by2:.1f}) wd={wall_distance(int(bx2),int(by2)):.1f} score={best_score:.0f}', flush=True)
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

class Mover:
    def __init__(self, m, d):
        self.m, self.d = m, d
        self.yaw = 0.0; self.speed = SPEED; self.bounce = 0
        self.force = 0; self.escaping = False
        self.stuck_t = 0; self.stuck_x = 0.0; self.stuck_y = 0.0
    
    def step(self, tx, ty, step):
        bx, by = self.d.qpos[0], self.d.qpos[1]
        dt = self.m.opt.timestep
        
        if not self.escaping:
            tgt_yaw = math.atan2(ty-by, tx-bx)
            err = (tgt_yaw-self.yaw+math.pi)%(2*math.pi)-math.pi
            dyaw = max(-YAW_RATE*dt, min(YAW_RATE*dt, err))
            self.yaw += dyaw
            clear = math.hypot(tx-bx, ty-by)
            self.speed = max(1.5, min(SPEED_MAX, clear*0.5))
        
        vx = math.cos(self.yaw)*self.speed
        vy = math.sin(self.yaw)*self.speed
        nx, ny = bx+vx*dt, by+vy*dt
        
        if step-self.stuck_t > 300:
            if math.hypot(bx-self.stuck_x, by-self.stuck_y) < 0.5:
                self._bounce(90, 180)
            self.stuck_t = step; self.stuck_x = bx; self.stuck_y = by
        
        if self.force > 0:
            self.force -= 1
            self.d.qvel[0] = vx; self.d.qvel[1] = vy
        elif blocked(nx, ny):
            self._bounce(45, 120)
        else:
            self.escaping = False
            self.d.qvel[0] = vx; self.d.qvel[1] = vy
        
        mujoco.mj_step(self.m, self.d); return True
    
    def _bounce(self, lo, hi):
        if not self.escaping:
            self.bounce += 1; self.escaping = True
        deg = random.uniform(lo, hi)*random.choice([-1,1])
        self.yaw += math.radians(deg)
        self.d.qvel[:] = 0
        self.force = int(0.3/(SPEED*self.m.opt.timestep))

m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
d.qpos[0]=10; d.qpos[1]=5; mujoco.mj_forward(m,d)

mv = Mover(m, d)
wp_idx=0; step=0; t0=time.time(); RENDER_SKIP=3

print(f"=== 萤火算法 v10 (safe radius 15) === {VOXEL}m³ Mover", flush=True)

with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
    v.cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.distance=25; v.cam.elevation=-35; v.cam.azimuth=180

    LIDAR_TICK = 20; DECIDE_TICK = 200
    target = None; stall_step = 0; stall_v = 0

    while v.is_running() and wp_idx<len(nav_wps):
        bx, by = d.qpos[0], d.qpos[1]
        if bx<1 or bx>99 or by<1 or by>99:
            d.qpos[0]=max(1,min(99,bx)); d.qpos[1]=max(1,min(99,by))
            d.qvel[:]=0; mv.yaw=random.uniform(0,2*math.pi)
        v.cam.lookat[:]=np.array([bx, by, 0.5], dtype=np.float64)

        tx, ty = nav_wps[wp_idx]; dist_to_cp = math.hypot(tx-bx, ty-by)
        if dist_to_cp < CP_RADIUS:
            wp_idx+=1
            vis = int(np.sum(vox==VISITED)); f = int(np.sum(vox==FREE))
            print(f"✓ CP{wp_idx-1} step={step} V{vis}/F{f}/W{int(np.sum(vox==WALL))}", flush=True)
            if wp_idx>=len(nav_wps):
                print(f"🏁 FINISH step={step} time={time.time()-t0:.1f}s bounce={mv.bounce}", flush=True)
                break
            continue

        vx, vy = int(bx), int(by)
        if 0 <= vx < W and 0 <= vy < W: vox[vy, vx] = VISITED
        if step % LIDAR_TICK == 0: scan_voxels(bx, by)

        if step % DECIDE_TICK == 0 or target is None:
            target = find_frontier(bx, by, wp_idx, target)
            if target and step % 400 == 0:
                print(f"  🎯 [{step}] target=({target[0]:.0f},{target[1]:.0f}) V{int(np.sum(vox==VISITED))}", flush=True)

        if target is None:
            mv._bounce(90, 180)
            mujoco.mj_step(m, d); step += 1
            if step % RENDER_SKIP == 0: v.sync()
            continue

        mv.step(target[0], target[1], step)
        step += 1
        if step % RENDER_SKIP == 0: v.sync()

    print(f"done: {wp_idx}/{len(nav_wps)} step={step} time={time.time()-t0:.1f}s", flush=True)
