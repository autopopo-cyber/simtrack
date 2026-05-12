#!/usr/bin/env python3
"""萤火算法 Firefly v18 — 自己修路: 有向路径表+扫图+延伸

体素:
  UNKNOWN=0 — 未扫描
  FREE=1    — 扫描过但没走过
  WALL=2    — 墙/障碍
  VISITED=3 — 实际到达过

修路:
  road = [体素A, 体素B, ... , 体素N=前线]
  1. 沿着road走
  2. 迷路→找road上最近直达点→回归
  3. 到前线→扫图→找新门→A*→append到road
  4. 终点只在扫到时才知道
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

# 导航点可视化用
nav_wps = [(x*SCALE, y*SCALE) for x,y in cl]
FINISH = nav_wps[-1]

def sample_hf(wx, wy):
    mx, my = wx/SCALE, wy/SCALE
    px, py = int(mx*PIX_PER_M), HF_RES-1-int(my*PIX_PER_M)
    return int(hf[py,px]) if 0<=px<HF_RES and 0<=py<HF_RES else -1

is_wall = lambda wx, wy: sample_hf(wx, wy) != ROAD_PIX
is_obs = lambda wx, wy: any(math.hypot(wx-ox, wy-oy) < OBS_CLEAR for ox, oy in obs_world)
blocked = lambda wx, wy: is_wall(wx, wy) or is_obs(wx, wy)
walkable = lambda vx, vy: 0<=vx<W and 0<=vy<W and vox[vy,vx] in (FREE, VISITED)

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

def find_nearest_gate(bx, by):
    """在FREE里找最近的、有UNKNOWN邻居的体素作为门"""
    sx, sy = int(bx/VOXEL), int(by/VOXEL)
    if not (0<=sx<W and 0<=sy<W): return None
    if vox[sy,sx] == WALL: return None
    
    open_set = []; heapq.heappush(open_set, (0, sx, sy))
    came_from = {}; g_score = {(sx,sy): 0}
    visited = set()
    best_gate = None; best_gate_score = -9999
    
    while open_set and len(came_from) < 20000:
        _, cx, cy = heapq.heappop(open_set)
        if (cx,cy) in visited: continue
        visited.add((cx,cy))
        cg = g_score.get((cx,cy), 9999)
        
        if vox[cy,cx] == FREE:
            unk_nb = sum(1 for dy in (-1,0,1) for dx in (-1,0,1)
                        if 0<=cx+dx<W and 0<=cy+dy<W and vox[cy+dy,cx+dx]==UNKNOWN)
            if unk_nb > 0:
                score = -cg + unk_nb*20
                # 离墙距离罚分
                gate_wd = 999
                for dy2 in range(-5,6):
                    for dx2 in range(-5,6):
                        tnx,tny = cx+dx2, cy+dy2
                        if 0<=tnx<W and 0<=tny<W and vox[tny,tnx]==WALL:
                            dd = math.hypot(dx2, dy2)
                            if dd < gate_wd: gate_wd = dd
                if gate_wd < 1.5:
                    score -= (1.5-gate_wd)*20
                if score > best_gate_score:
                    best_gate_score = score
                    best_gate = (cx, cy)
        
        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
            nx, ny = cx+dx, cy+dy
            if not walkable(nx, ny): continue
            wd = 999.0
            for dy2 in range(-6,7):
                for dx2 in range(-6,7):
                    tnx,tny = nx+dx2, ny+dy2
                    if 0<=tnx<W and 0<=tny<W and vox[tny,tnx]==WALL:
                        d = math.hypot(dx2, dy2)*VOXEL
                        if d < wd: wd = d
            penalty = max(0, 2.0-wd)*3
            ng = cg + 1 + penalty
            if (nx,ny) not in g_score or ng < g_score[(nx,ny)]:
                g_score[(nx,ny)] = ng
                came_from[(nx,ny)] = (cx,cy)
                heapq.heappush(open_set, (ng, nx, ny))
    
    if best_gate is None:
        return None
    
    path = []; cur = best_gate
    while cur != (sx,sy):
        path.append(cur)
        if cur not in came_from: break
        cur = came_from[cur]
    path.reverse()
    return [(px, py) for px, py in path]  # voxel coords

def find_nearest_road_voxel(bx, by, road):
    """在road上找离机器人最近的、4邻接可直达的体素"""
    sx, sy = int(bx/VOXEL), int(by/VOXEL)
    if not (0<=sx<W and 0<=sy<W): return None, -1
    
    best_i, best_d = -1, 99999
    for i, (rx, ry) in enumerate(road):
        # 用A*搜: (sx,sy)能否走VISITED/FREE到达(rx,ry)
        d = math.hypot(rx-sx, ry-sy)
        if d > 30: continue  # 太远跳过
        if d < best_d:
            # 简单4邻接检查: 能在VISITED空间里走到road点吗
            open_set = [(math.hypot(rx-sx,ry-sy), sx, sy)]
            came = {}; gs = {(sx,sy): 0}; vis = set()
            found = False
            while open_set and len(came) < 500:
                _, cx, cy = heapq.heappop(open_set)
                if (cx,cy) in vis: continue
                vis.add((cx,cy))
                if (cx,cy) == (rx,ry):
                    found = True; break
                for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
                    nx, ny = cx+dx, cy+dy
                    if not (0<=nx<W and 0<=ny<W and vox[ny,nx] in (VISITED, FREE)): continue
                    if (nx,ny) not in gs:
                        gs[(nx,ny)] = gs.get((cx,cy),999)+1
                        came[(nx,ny)] = (cx,cy)
                        heapq.heappush(open_set, (gs[(nx,ny)]+math.hypot(rx-nx,ry-ny), nx, ny))
            if found:
                best_i, best_d = i, d
    
    return best_i, best_d

def road_to_world(road_vox):
    """体素坐标→世界坐标"""
    return [((px+0.5)*VOXEL, (py+0.5)*VOXEL) for px, py in road_vox]

def voxel_path_to_world(bx, by, target_vox):
    """从当前世界位置A*到目标体素, 返回世界路径"""
    sx, sy = int(bx/VOXEL), int(by/VOXEL)
    tx, ty = target_vox
    if not (0<=sx<W and 0<=sy<W and 0<=tx<W and 0<=ty<W): return None
    if vox[sy,sx]==WALL or vox[ty,tx]==WALL: return None
    
    open_set = [(math.hypot(tx-sx, ty-sy), sx, sy)]
    came_from = {}; g_score = {(sx,sy): 0}
    visited = set()
    found = False
    while open_set and len(came_from) < 5000:
        _, cx, cy = heapq.heappop(open_set)
        if (cx,cy) in visited: continue
        visited.add((cx,cy))
        if (cx,cy) == (tx,ty):
            found = True; break
        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
            nx, ny = cx+dx, cy+dy
            if not walkable(nx, ny): continue
            ng = g_score.get((cx,cy), 999) + 1
            if (nx,ny) not in g_score or ng < g_score[(nx,ny)]:
                g_score[(nx,ny)] = ng
                came_from[(nx,ny)] = (cx,cy)
                heapq.heappush(open_set, (ng+math.hypot(tx-nx, ty-ny), nx, ny))
    if not found: return None
    
    path = []; cur = (tx,ty)
    while cur != (sx,sy):
        path.append(cur)
        if cur not in came_from: break
        cur = came_from[cur]
    path.reverse()
    return [((px+0.5)*VOXEL, (py+0.5)*VOXEL) for px, py in path]

# ── XML & Mover ──
NAV_DOTS = "".join(f'<body mocap="true" pos="{x:.1f} {y:.1f} 1.5"><geom type="sphere" size="0.25" rgba="0.3 0.6 1.0 0.6"/></body>' for x,y in nav_wps)
CP_XML = f'<body mocap="true" pos="{FINISH[0]:.1f} {FINISH[1]:.1f} 2"><geom type="sphere" size="1.5" rgba="0.2 1.0 0.2 0.8"/></body>'
OBS_XML = "".join(f'<body name="obs{i}" pos="{x:.1f} {y:.1f} 2.0"><geom type="cylinder" size="1.0 2.0" rgba="0.9 0.2 0.2 0.9"/></body>' for i,(x,y) in enumerate(obs_world))

xml = f"""<mujoco>
  <compiler angle="radian"/><option timestep="0.005"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset><hfield name="track" size="50.0 50.0 4.0 2.0" file="{MAP}"/></asset>
  <worldbody>
    <light pos="50 50 80" dir="0 0 -1"/>{NAV_DOTS}{CP_XML}{OBS_XML}
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
            if self.bounce % 5 == 0:
                print(f"  💥 bounce#{self.bounce} @({bx:.1f},{by:.1f})", flush=True)
        deg = random.uniform(lo, hi)*random.choice([-1,1])
        self.yaw += math.radians(deg)
        self.d.qvel[:] = 0
        self.force = int(0.3/(SPEED*self.m.opt.timestep))

# ── 主循环: 修路 ──
m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
d.qpos[0]=10; d.qpos[1]=5; mujoco.mj_forward(m,d)

mv = Mover(m, d)
step=0; t0=time.time(); RENDER_SKIP=20
road = []           # [(vx,vy), ...] 体素ID表, 从起点到前线
road_pos = []       # 对应世界坐标
road_idx = 0        # 当前位置在路上的索引
path = None; path_idx = 0  # 当前执行的A*子路径
finish_found = False; finish_vox = None
last_find_step = -999

print(f"=== 萤火算法 Firefly v18 === 自己修路: 找门+延伸路径", flush=True)

with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
    v.cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.distance=25; v.cam.elevation=-35; v.cam.azimuth=180

    LIDAR_TICK = 20

    # 初始扫描
    print("  🔍 initial scan (200 steps)...", flush=True)
    for _ in range(200):
        bx, by = d.qpos[0], d.qpos[1]
        vx, vy = int(bx/VOXEL), int(by/VOXEL)
        if 0 <= vx < W and 0 <= vy < W: vox[vy, vx] = VISITED
        if _ % LIDAR_TICK == 0: scan_voxels(bx, by)
        mujoco.mj_step(m, d)
    print(f"  ✅ F{int(np.sum(vox==FREE))} W{int(np.sum(vox==WALL))}", flush=True)

    # 第一条路: 从起点到最近的gate
    first_gate_vox = find_nearest_gate(d.qpos[0], d.qpos[1])
    if first_gate_vox:
        road = first_gate_vox
        road_pos = road_to_world(road)
        print(f"  🛤️  first road: {len(road)} voxels, gate=({road[-1][0]*VOXEL:.0f},{road[-1][1]*VOXEL:.0f})", flush=True)
        # 走向第一个gate
        path = road_pos; path_idx = 0

    while v.is_running() and not finish_found:
        bx, by = d.qpos[0], d.qpos[1]
        if bx<1 or bx>99 or by<1 or by>99:
            d.qpos[0]=max(1,min(99,bx)); d.qpos[1]=max(1,min(99,by))
            d.qvel[:]=0; mv.yaw=random.uniform(0,2*math.pi)
        v.cam.lookat[:]=np.array([bx, by, 0.5], dtype=np.float64)

        # 标记当前位置
        vx, vy = int(bx/VOXEL), int(by/VOXEL)
        if 0 <= vx < W and 0 <= vy < W: vox[vy, vx] = VISITED
        if step % LIDAR_TICK == 0:
            scan_voxels(bx, by)

        # 检查终点: LIDAR扫到终点了没
        if not finish_found and step % 50 == 0:
            # 扫描范围里有没有终点附近的FREE?
            fx, fy = int(FINISH[0]/VOXEL), int(FINISH[1]/VOXEL)
            # 简单检测: 终点体素或邻居是否被扫到FREE/VISITED
            if 0<=fx<W and 0<=fy<W and vox[fy,fx] in (FREE, VISITED):
                finish_found = True
                finish_vox = (fx, fy)
                print(f"  🏁 FINISH found! @({bx:.1f},{by:.1f}) step={step} time={time.time()-t0:.1f}s bounce={mv.bounce}", flush=True)
                break

        # 路径执行
        if path is None or path_idx >= len(path):
            # 需要新路径: 在road上往前走还是延伸road?
            if len(road) == 0:
                # 还没路, 或路被清空 → bounce
                mv._bounce(90, 180)
                step += 1; continue
            
            # 我在road上吗？找最近road点
            ri, _ = find_nearest_road_voxel(bx, by, road)
            if ri >= 0 and ri < len(road)-1:
                # 在road上, 继续往前走
                target_vox = road[min(ri+1, len(road)-1)]
                path = voxel_path_to_world(bx, by, target_vox)
                if path:
                    path_idx = 0
                    continue
            
            # 到前线了 → 扫图 → 找新门 → 延伸
            if step - last_find_step < 200:
                # 刚找过, 等一会儿
                mv._bounce(90, 180)
                step += 1; continue
            
            print(f"  🔍 [{step}] at frontier, scanning...", flush=True)
            scan_voxels(bx, by)  # 多扫几次
            scan_voxels(bx, by)
            new_gate_vox = find_nearest_gate(bx, by)
            last_find_step = step
            
            if new_gate_vox:
                # A*从前线到新门
                from_vox = road[-1]
                path = voxel_path_to_world(
                    (from_vox[0]+0.5)*VOXEL,
                    (from_vox[1]+0.5)*VOXEL,
                    new_gate_vox[-1]
                )
                if path:
                    road.extend(new_gate_vox)
                    road_pos = road_to_world(road)
                    print(f"  🛤️  [{step}] extended road: +{len(new_gate_vox)}vox → {len(road)}vox gate=({new_gate_vox[-1][0]*VOXEL:.0f},{new_gate_vox[-1][1]*VOXEL:.0f}) F{int(np.sum(vox==FREE))}", flush=True)
                    path = road_pos; path_idx = 0
                else:
                    mv._bounce(90, 180)
            else:
                # 找A*在VISITED空间里最远的点
                print(f"  🧭 [{step}] no gate, seeking farthest VISITED...", flush=True)
                sx, sy = int(bx/VOXEL), int(by/VOXEL)
                best_v = None; best_d = 0
                open_set = [(0, sx, sy)]; vis = set()
                while open_set and len(vis) < 5000:
                    _, cx, cy = heapq.heappop(open_set)
                    if (cx,cy) in vis: continue
                    vis.add((cx,cy))
                    d = math.hypot(cx-sx, cy-sy)
                    if vox[cy,cx] == VISITED and d > best_d:
                        best_d = d; best_v = (cx, cy)
                    for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
                        nx, ny = cx+dx, cy+dy
                        if 0<=nx<W and 0<=ny<W and vox[ny,nx]==VISITED and (nx,ny) not in vis:
                            heapq.heappush(open_set, (0, nx, ny))
                if best_v:
                    path = voxel_path_to_world(bx, by, best_v)
                    if path:
                        path_idx = 0
                    else:
                        mv._bounce(90, 180)
                else:
                    mv._bounce(90, 180)
        
        if path is not None and path_idx < len(path):
            tx, ty = path[path_idx]
            if math.hypot(tx-bx, ty-by) < 1.0:
                path_idx += 1
            else:
                mv.step(tx, ty, step)
        else:
            mv._bounce(90, 180)
        
        step += 1
        if step % 1000 == 0:
            print(f"  ... step={step} V{int(np.sum(vox==VISITED))} F{int(np.sum(vox==FREE))} road={len(road)}vox", flush=True)
        if step % RENDER_SKIP == 0: v.sync()
    
    if finish_found:
        print(f"done! road={len(road)}vox step={step} time={time.time()-t0:.1f}s bounce={mv.bounce}", flush=True)
    else:
        print(f"stopped: road={len(road)}vox step={step} time={time.time()-t0:.1f}s", flush=True)
