#!/usr/bin/env python3
"""萤火算法 Firefly v5 — "门"方案 + A*墙距代价

体素: OLD=FREE+VISITED(WALL已分析) | NEW=UNKNOWN
A*: 在OLD里搜到最近NEW体素, 墙距罚分自动走宽路
循环: scan→A*→沿路径走→到门→scan→A*→...
"""
import sys, os, math, time, random, heapq
from collections import deque
import numpy as np
from PIL import Image
import mujoco, mujoco.viewer

MAP = os.path.expanduser("~/workspace/simtrack/confirmed/track_clean.png")
hf = np.array(Image.open(MAP))

SCALE = 2.0; HF_RES = 2000; PIX_PER_M = 40; ROAD_PIX = 128
SAFE_R = 1.0; SPEED = 5.0; SPEED_MAX = 8.0; YAW_RATE = 6.0
CP_RADIUS = 2.0; LIDAR_RANGE = 15.0
VOXEL = 0.5; W = 200

UNKNOWN, FREE, WALL, VISITED = 0, 1, 2, 3
vox = np.zeros((W, W), dtype=np.int8)

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
unexplored = lambda vx, vy: 0<=vx<W and 0<=vy<W and vox[vy,vx] in (FREE, UNKNOWN)
walkable = lambda vx, vy: 0<=vx<W and 0<=vy<W and vox[vy,vx] in (FREE, VISITED)

def target_yaw_dir(wp_idx):
    if wp_idx >= len(nav_wps): return (1,0)
    tx, ty = nav_wps[wp_idx]
    if wp_idx+1 < len(nav_wps):
        nx, ny = nav_wps[wp_idx+1]
        dx, dy = nx-tx, ny-ty
        d = math.hypot(dx, dy)
        if d > 0.01: return (dx/d, dy/d)
    return (1,0)

def scan_voxels(bx, by):
    for a in np.linspace(0, 2*np.pi, 120):
        cos_a, sin_a = math.cos(a), math.sin(a)
        for d in np.arange(0.5, LIDAR_RANGE+0.1, 0.5):
            wx, wy = bx+cos_a*d, by+sin_a*d
            vx, vy = int(wx/VOXEL), int(wy/VOXEL)
            if not (0 <= vx < W and 0 <= vy < W): break
            if vox[vy, vx] == WALL: break
            if blocked(wx, wy):
                vox[vy, vx] = WALL; break
            vox[vy, vx] = max(vox[vy, vx], FREE)

def line_clear(bx, by, wx, wy):
    dx, dy = wx-bx, wy-by
    dist = math.hypot(dx, dy)
    if dist < 0.01: return True
    steps = int(dist / 0.3)
    for i in range(1, steps):
        t = i / steps
        if blocked(bx+dx*t, by+dy*t): return False
    return True

def find_gate_path(bx, by, wp_idx):
    """A*找从当前位置到最佳"出口"的路径。
    出口 = 邻接UNKNOWN的UNEXPLORED体素。
    非出口的可达UNEXPLORED体素标为VISITED(消除回头路)。
    返回: [((vx,vy), (wx,wy)), ...] 体素路径列表, 或None"""
    sx, sy = int(bx/VOXEL), int(by/VOXEL)
    if sx<0 or sx>=W or sy<0 or sy>=W or vox[sy,sx]==WALL: return None
    
    wp_dx, wp_dy = target_yaw_dir(wp_idx)
    tx, ty = nav_wps[min(wp_idx, len(nav_wps)-1)]
    
    # A* 
    open_set = []
    heapq.heappush(open_set, (0, sx, sy))
    came_from = {}
    g_score = {(sx, sy): 0}
    best_gate = None; best_gate_score = -9999
    visited_a_star = set()
    
    while open_set and len(came_from) < 20000:
        _, cx, cy = heapq.heappop(open_set)
        if (cx, cy) in visited_a_star: continue
        visited_a_star.add((cx, cy))
        cg = g_score.get((cx, cy), 9999)
        
        # 出口检查: 这个体素邻接UNKNOWN?
        is_gate = any(vox[cy+ndy, cx+ndx] == UNKNOWN
                     for ndy in (-1,0,1) for ndx in (-1,0,1)
                     if 0<=cx+ndx<W and 0<=cy+ndy<W)
        if is_gate:
            wx, wy = (cx+0.5)*VOXEL, (cy+0.5)*VOXEL
            score = -cg  # 越近越好
            # 朝WP方向加分
            dot = ((cx-sx)*wp_dx + (cy-sy)*wp_dy) / max(1, math.hypot(cx-sx, cy-sy))
            score += (dot+1)*50
            if score > best_gate_score:
                best_gate_score = score
                best_gate = (cx, cy)
        
        # 扩4邻域
        for ndx, ndy in [(0,-1),(0,1),(-1,0),(1,0)]:
            nx, ny = cx+ndx, cy+ndy
            if not walkable(nx, ny): continue
            # wall距离惩罚: 搜周围找最近WALL, 越近代价越高
            wd = 999.0; wx2, wy2 = (nx+0.5)*VOXEL, (ny+0.5)*VOXEL
            for ndy2 in range(-6, 7):
                for ndx2 in range(-6, 7):
                    tnx, tny = nx+ndx2, ny+ndy2
                    if 0<=tnx<W and 0<=tny<W and vox[tny,tnx]==WALL:
                        d = math.hypot(wx2-(tnx+0.5)*VOXEL, wy2-(tny+0.5)*VOXEL)
                        if d < wd: wd = d
            penalty = max(0, 2.0 - wd) * 3  # 2m内开始罚, 越近越重
            ng = cg + 1 + penalty
            if (nx, ny) not in g_score or ng < g_score[(nx, ny)]:
                g_score[(nx, ny)] = ng
                came_from[(nx, ny)] = (cx, cy)
                h = math.hypot(tx-(nx+0.5)*VOXEL, ty-(ny+0.5)*VOXEL) * 0.1
                heapq.heappush(open_set, (ng+h, nx, ny))
    
    if best_gate is None:
        return None
    
    # 回溯路径
    path = []
    cur = best_gate
    while cur != (sx, sy):
        path.append(cur)
        if cur not in came_from: break
        cur = came_from[cur]
    path.reverse()  # 从sx,sy到gate
    
    # 转换: 体素坐标→世界坐标
    world_path = [((px+0.5)*VOXEL, (py+0.5)*VOXEL) for px, py in path]
    return world_path

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
            bx, by = self.d.qpos[0], self.d.qpos[1]
            print(f"  💥 bounce#{self.bounce} @({bx:.1f},{by:.1f})", flush=True)
        deg = random.uniform(lo, hi)*random.choice([-1,1])
        self.yaw += math.radians(deg)
        self.d.qvel[:] = 0
        self.force = int(0.3/(SPEED*self.m.opt.timestep))

m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
d.qpos[0]=10; d.qpos[1]=5; mujoco.mj_forward(m,d)

mv = Mover(m, d)
wp_idx=0; step=0; t0=time.time(); RENDER_SKIP=3
path = None; path_idx = 0

print(f"=== 萤火算法 Firefly v5 === GATE mode | {VOXEL}m voxel | A*→出口→扫描→循环", flush=True)

with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
    v.cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.distance=25; v.cam.elevation=-35; v.cam.azimuth=180

    LIDAR_TICK = 20
        # 初始扫描10轮 (2秒=400步)
    print("  🔍 initial scan 10 rounds...", flush=True)
    for _ in range(400):
        bx, by = d.qpos[0], d.qpos[1]
        vx, vy = int(bx/VOXEL), int(by/VOXEL)
        if 0 <= vx < W and 0 <= vy < W: vox[vy, vx] = VISITED
        if _ % LIDAR_TICK == 0: scan_voxels(bx, by)
        mujoco.mj_step(m, d)
    print(f"  ✅ scan done F{int(np.sum(vox==FREE))} W{int(np.sum(vox==WALL))}", flush=True)
    
    # 第一次A*
    path = find_gate_path(d.qpos[0], d.qpos[1], wp_idx)
    if path: gate = path[-1]; print(f"  🚪 init gate=({gate[0]:.0f},{gate[1]:.0f}) path_len={len(path)}", flush=True)

    while v.is_running() and wp_idx<len(nav_wps):
        bx, by = d.qpos[0], d.qpos[1]
        if bx<1 or bx>99 or by<1 or by>99:
            d.qpos[0]=max(1,min(99,bx)); d.qpos[1]=max(1,min(99,by))
            d.qvel[:]=0; mv.yaw=random.uniform(0,2*math.pi)
        v.cam.lookat[:]=np.array([bx, by, 0.5], dtype=np.float64)

        # CP检测
        new_wp = wp_idx
        for i in range(len(nav_wps)):
            if math.hypot(nav_wps[i][0]-bx, nav_wps[i][1]-by) < CP_RADIUS:
                new_wp = max(new_wp, i+1)
        if new_wp > wp_idx:
            wp_idx = new_wp
            vis = int(np.sum(vox==VISITED))
            print(f"  🏁 CP{wp_idx-1} @({bx:.1f},{by:.1f}) | step={step} V{vis}", flush=True)
            if wp_idx>=len(nav_wps):
                print(f"🏁 FINISH step={step} time={time.time()-t0:.1f}s bounce={mv.bounce}", flush=True)
                break
            continue

        vx, vy = int(bx/VOXEL), int(by/VOXEL)
        if 0 <= vx < W and 0 <= vy < W: vox[vy, vx] = VISITED
        if step % LIDAR_TICK == 0:
            scan_voxels(bx, by)

        # 路径规划/执行
        if path is None or path_idx >= len(path):
            print(f"  🚪 [{step}] A* gate search... V{int(np.sum(vox==VISITED))} FREE{int(np.sum(vox==FREE))}", flush=True)
            path = find_gate_path(bx, by, wp_idx)
            path_idx = 0
            if path:
                gate = path[-1]
                print(f"  🚪 [{step}] gate=({gate[0]:.0f},{gate[1]:.0f}) path_len={len(path)}", flush=True)
            else:
                print(f"  ⚡ [{step}] no gate found, bounce", flush=True)
                mv._bounce(90, 180)
                path = None
        
        if path is not None and path_idx < len(path):
            tx, ty = path[path_idx]
            if math.hypot(tx-bx, ty-by) < 1.0:
                path_idx += 1
                if path_idx >= len(path):
                    # 到出口: scan 10 rounds + re-A*
                    print(f"  🚪 [{step}] at gate, scanning...", flush=True)
                    for _ in range(200):
                        if _ % LIDAR_TICK == 0: scan_voxels(d.qpos[0], d.qpos[1])
                        mujoco.mj_step(m, d); step += 1
                        if step % RENDER_SKIP == 0: v.sync()
                    path = find_gate_path(d.qpos[0], d.qpos[1], wp_idx)
                    path_idx = 0
                    if path:
                        last_gate = path[-1]
                        print(f"  🚪 next gate=({last_gate[0]:.0f},{last_gate[1]:.0f}) len={len(path)}", flush=True)
                    else:
                        print(f"  ⚡ no next gate", flush=True)
            else:
                mv.step(tx, ty, step)
        else:
            mv._bounce(90, 180)
        
        step += 1
        if step % RENDER_SKIP == 0: v.sync()

    print(f"done: {wp_idx}/{len(nav_wps)} step={step} time={time.time()-t0:.1f}s", flush=True)
