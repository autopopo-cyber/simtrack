#!/usr/bin/env python3
"""萤火算法 Firefly v19 — 迷宫探索: 自产路标(3m)+最近门优先+回溯

机器人只知道:
  自己的坐标
  LIDAR扫到的 FREE/WALL/UNKNOWN
  之前走过的路 (VISITED标记)
  自己留下的路标 (每3m一个)

不知道:
  地图全貌
  终点位置 (LIDAR扫到才知道)
  中心线/导航点

策略:
  1. 找离自己最近的"门"(FREE体素+邻接UNKNOWN)
  2. A*在VISITED空间规划到门的路径
  3. 走过去 → 扫描 → 新地面变FREE → 新门出现
  4. 每3m(6体素)留一个路标
  5. 没门了→回溯到最近的还有未探索门的路标
"""
import sys, os, math, time, random, heapq
import numpy as np
from PIL import Image
import mujoco, mujoco.viewer

MAP = os.path.expanduser("~/workspace/simtrack/confirmed/track_clean.png")
hf = np.array(Image.open(MAP))

SCALE = 2.0; HF_RES = 2000; PIX_PER_M = 40; ROAD_PIX = 128
SAFE_R = 1.0; SPEED = 5.0; SPEED_MAX = 8.0; YAW_RATE = 6.0
LIDAR_RANGE = 15.0
VOXEL = 0.5; W = 200
MILESTONE_INTERVAL = 6  # 3m = 6个0.5m体素

UNKNOWN, FREE, WALL, VISITED = 0, 1, 2, 3
vox = np.zeros((W, W), dtype=np.int8)

# 障碍物 (用于仿真)
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

# 终点 (机器人不知道, 只在LIDAR扫到时才检测)
FINISH = (7.0, 82.5)  # 世界坐标, 对应地图右下角出口
FINISH_VX, FINISH_VY = int(FINISH[0]/VOXEL), int(FINISH[1]/VOXEL)

def sample_hf(wx, wy):
    mx, my = wx/SCALE, wy/SCALE
    px, py = int(mx*PIX_PER_M), HF_RES-1-int(my*PIX_PER_M)
    return int(hf[py,px]) if 0<=px<HF_RES and 0<=py<HF_RES else -1

is_wall = lambda wx, wy: sample_hf(wx, wy) != ROAD_PIX
is_obs = lambda wx, wy: any(math.hypot(wx-ox, wy-oy) < OBS_CLEAR for ox, oy in obs_world)
blocked = lambda wx, wy: is_wall(wx, wy) or is_obs(wx, wy)
walkable = lambda vx, vy: 0<=vx<W and 0<=vy<W and vox[vy,vx] in (FREE, VISITED)

def scan_voxels(bx, by):
    """120条射线, 最远15m, FREE/WALL标记"""
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

def find_nearest_gate(sx, sy):
    """A*在VISITED+FREE空间, 找最近的FREE体素(邻接UNKNOWN)"""
    if not (0<=sx<W and 0<=sy<W and vox[sy,sx] in (FREE, VISITED)): 
        return None
    
    open_set = [(0, sx, sy)]
    came_from = {}; g_score = {(sx,sy): 0}
    visited = set()
    best_gate = None; best_dist = 99999
    
    while open_set and len(came_from) < 20000:
        _, cx, cy = heapq.heappop(open_set)
        if (cx,cy) in visited: continue
        visited.add((cx,cy))
        cg = g_score.get((cx,cy), 9999)
        
        if vox[cy,cx] == FREE:
            # 它有UNKNOWN邻居吗?
            has_unk = any(
                vox[cy+dy,cx+dx]==UNKNOWN
                for dy in (-1,0,1) for dx in (-1,0,1)
                if 0<=cx+dx<W and 0<=cy+dy<W
            )
            if has_unk and cg < best_dist:
                best_dist = cg
                best_gate = (cx, cy)
        
        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
            nx, ny = cx+dx, cy+dy
            if not walkable(nx, ny): continue
            # 墙距惩罚
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
    
    if best_gate is None: return None
    
    # 回溯路径
    path = []; cur = best_gate
    while cur != (sx,sy):
        path.append(cur)
        if cur not in came_from: break
        cur = came_from[cur]
    path.reverse()
    return path  # [(vx,vy), ...] voxel coords

def astar_voxel_path(sx, sy, tx, ty):
    """A*在VISITED+FREE空间, 从(sx,sy)到(tx,ty), 返回体素路径"""
    if not (0<=sx<W and 0<=sy<W and 0<=tx<W and 0<=ty<W): return None
    if vox[sy,sx]==WALL or vox[ty,tx]==WALL: return None
    
    open_set = [(math.hypot(tx-sx, ty-sy), sx, sy)]
    came_from = {}; g_score = {(sx,sy): 0}
    visited = set()
    
    while open_set and len(came_from) < 5000:
        _, cx, cy = heapq.heappop(open_set)
        if (cx,cy) in visited: continue
        visited.add((cx,cy))
        if (cx,cy) == (tx,ty): break
        
        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
            nx, ny = cx+dx, cy+dy
            if not walkable(nx, ny): continue
            ng = g_score.get((cx,cy), 999) + 1
            if (nx,ny) not in g_score or ng < g_score[(nx,ny)]:
                g_score[(nx,ny)] = ng
                came_from[(nx,ny)] = (cx,cy)
                heapq.heappush(open_set, (ng+math.hypot(tx-nx, ty-ny), nx, ny))
    
    if (tx,ty) not in came_from and (tx,ty) != (sx,sy): return None
    
    path = []; cur = (tx,ty)
    while cur != (sx,sy):
        path.append(cur)
        if cur not in came_from: break
        cur = came_from[cur]
    path.reverse()
    return path

def milestone_dist(mi, to_vx, to_vy):
    """路标到目标体素的曼哈顿距离"""
    mx, my = mi
    return abs(mx-to_vx) + abs(my-to_vy)

# ── XML & Mover ──
FINISH_WX, FINISH_WY = FINISH
CP_XML = f'<body mocap="true" pos="{FINISH_WX:.1f} {FINISH_WY:.1f} 2"><geom type="sphere" size="1.5" rgba="0.2 1.0 0.2 0.8"/></body>'
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
            if self.bounce % 5 == 0:
                print(f"  💥 bounce#{self.bounce} @({self.d.qpos[0]:.1f},{self.d.qpos[1]:.1f})", flush=True)
        deg = random.uniform(lo, hi)*random.choice([-1,1])
        self.yaw += math.radians(deg)
        self.d.qvel[:] = 0
        self.force = int(0.3/(SPEED*self.m.opt.timestep))

# ── 主循环: 迷宫探索 ──
m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
d.qpos[0]=6; d.qpos[1]=6; mujoco.mj_forward(m,d)

mv = Mover(m, d)
step=0; t0=time.time(); RENDER_SKIP=20

# 路标网络: [(vx,vy), ...] — 机器人自己生成的导航点
milestones = []
last_milestone_x, last_milestone_y = 0, 0  # 上次留路标时的体素坐标

path = None; path_idx = 0  # 当前A*路径
finish_found = False
no_gate_count = 0

print(f"=== 萤火算法 Firefly v19 === 迷宫探索: 自产路标(3m)+最近门+回溯", flush=True)

with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
    v.cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.distance=25; v.cam.elevation=-35; v.cam.azimuth=180
    LIDAR_TICK = 20

    # 初始扫描
    print("  🔍 initial scan...", flush=True)
    for _ in range(200):
        bx, by = d.qpos[0], d.qpos[1]
        vx, vy = int(bx/VOXEL), int(by/VOXEL)
        if 0 <= vx < W and 0 <= vy < W: vox[vy, vx] = VISITED
        if _ % LIDAR_TICK == 0: scan_voxels(bx, by)
        mujoco.mj_step(m, d)
    print(f"  ✅ F{int(np.sum(vox==FREE))} W{int(np.sum(vox==WALL))}", flush=True)

    # 第一个路标: 起点
    sx, sy = int(d.qpos[0]/VOXEL), int(d.qpos[1]/VOXEL)
    milestones.append((sx, sy))
    last_milestone_x, last_milestone_y = sx, sy

    while v.is_running() and not finish_found:
        bx, by = d.qpos[0], d.qpos[1]
        if bx<1 or bx>99 or by<1 or by>99:
            d.qpos[0]=max(1,min(99,bx)); d.qpos[1]=max(1,min(99,by))
            d.qvel[:]=0; mv.yaw=random.uniform(0,2*math.pi)
        v.cam.lookat[:]=np.array([bx, by, 0.5], dtype=np.float64)

        # 标记VISITED + 扫描
        vx, vy = int(bx/VOXEL), int(by/VOXEL)
        if 0 <= vx < W and 0 <= vy < W: 
            vox[vy, vx] = VISITED
            # 每3m留一个路标
            if abs(vx-last_milestone_x)+abs(vy-last_milestone_y) >= MILESTONE_INTERVAL:
                milestones.append((vx, vy))
                last_milestone_x, last_milestone_y = vx, vy
                if len(milestones) % 10 == 0:
                    print(f"  📍 milestones={len(milestones)} @({bx:.1f},{by:.1f})", flush=True)
        
        if step % LIDAR_TICK == 0:
            scan_voxels(bx, by)

        # 终点检测: LIDAR扫到终点附近的体素?
        if not finish_found:
            fx, fy = int(FINISH[0]/VOXEL), int(FINISH[1]/VOXEL)
            if 0<=fx<W and 0<=fy<W and vox[fy,fx] in (FREE, VISITED):
                d2 = math.hypot(FINISH[0]-bx, FINISH[1]-by)
                if d2 < 5.0:  # 5m内才算到了
                    finish_found = True
                    print(f"  🏁 FINISH found! @({bx:.1f},{by:.1f}) step={step} time={time.time()-t0:.1f}s bounce={mv.bounce}", flush=True)
                    break

        # 路径规划/执行
        if path is None or path_idx >= len(path):
            # 找最近的门
            gate_path = find_nearest_gate(vx, vy)
            
            if gate_path:
                no_gate_count = 0
                # 体素路径→世界路径
                path = [((px+0.5)*VOXEL, (py+0.5)*VOXEL) for px, py in gate_path]
                path_idx = 0
                gate = gate_path[-1]
                gate_wx, gate_wy = (gate[0]+0.5)*VOXEL, (gate[1]+0.5)*VOXEL
                print(f"  🚪 [{step}] gate=({gate_wx:.0f},{gate_wy:.0f}) len={len(path)} F{int(np.sum(vox==FREE))}", flush=True)
            else:
                no_gate_count += 1
                # 没门了: 回溯到最近的路标, 从那里再找
                if no_gate_count > 3 and len(milestones) > 1:
                    # 从路标里找最近的一个, A*回去
                    best_mi = -1; best_md = 99999
                    for i in range(len(milestones)-1):
                        mx, my = milestones[i]
                        if vox[my,mx] != VISITED: continue
                        d = abs(mx-vx) + abs(my-vy)
                        if d < best_md:
                            best_md, best_mi = d, i
                    
                    if best_mi >= 0:
                        mx, my = milestones[best_mi]
                        back_path = astar_voxel_path(vx, vy, mx, my)
                        if back_path:
                            path = [((px+0.5)*VOXEL, (py+0.5)*VOXEL) for px, py in back_path]
                            path_idx = 0
                            print(f"  🔙 [{step}] backtrack→milestone#{best_mi} len={len(path)}", flush=True)
                            no_gate_count = 0
                        else:
                            mv._bounce(90, 180)
                    else:
                        mv._bounce(90, 180)
                else:
                    mv._bounce(90, 180)
        
        if path is not None and path_idx < len(path):
            tx, ty = path[path_idx]
            d = math.hypot(tx-bx, ty-by)
            if d < 1.0:
                path_idx += 1
            else:
                mv.step(tx, ty, step)
        else:
            mv._bounce(90, 180)
        
        step += 1
        if step % 2000 == 0:
            print(f"  ... step={step} V{int(np.sum(vox==VISITED))} F{int(np.sum(vox==FREE))} milestones={len(milestones)}", flush=True)
        if step % RENDER_SKIP == 0: v.sync()
    
    result = "FINISH" if finish_found else "stopped"
    print(f"done({result}): milestones={len(milestones)} step={step} time={time.time()-t0:.1f}s bounce={mv.bounce}", flush=True)
