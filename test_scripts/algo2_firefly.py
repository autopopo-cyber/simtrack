#!/usr/bin/env python3
"""萤火算法 Firefly v20 — 文件持久化+动态路标球+固定种子

文件:
  scan_vox.npy      — 体素网格 (200x200 int8)
  scan_meta.json    — {seed, milestones, finished_flag}
  
可视化:
  [BLUE] 蓝色小球 = 路标 (每3m)
  [YELLOW] 黄色小球 = 当前门 (A*目标)

Log: 只有两个move动作: 导航到门(x,y) / 导航到路标(x,y)
"""
import sys, os, math, time, random, heapq, json
import numpy as np
from PIL import Image
import mujoco, mujoco.viewer

MAP = os.path.expanduser("~/workspace/simtrack/confirmed/track_clean.png")
SCAN_DIR = os.path.expanduser("~/workspace/simtrack/scans")
SCAN_VOX = os.path.join(SCAN_DIR, "scan_vox.npy")
SCAN_META = os.path.join(SCAN_DIR, "scan_meta.json")
os.makedirs(SCAN_DIR, exist_ok=True)

hf = np.array(Image.open(MAP))

SCALE = 2.0; HF_RES = 2000; PIX_PER_M = 40; ROAD_PIX = 128
SAFE_R = 1.0; SPEED = 5.0; SPEED_MAX = 8.0; YAW_RATE = 6.0
LIDAR_RANGE = 15.0
VOXEL = 0.5; W = 200
MILESTONE_INTERVAL = 6  # 3m = 6个0.5m体素

UNKNOWN, FREE, WALL, VISITED = 0, 1, 2, 3
vox = np.zeros((W, W), dtype=np.int8)

# 固定种子 (调试用)
FIXED_SEED = 42

# ── 障碍物生成 (固定种子) ──
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

def gen_obstacles(seed):
    rng = random.Random(seed)
    cl = gen_centerline()
    obs_world = []; idx = 0
    while idx < len(cl):
        cx, cy = cl[idx]; wx, wy = cx*SCALE, cy*SCALE
        obs_world.append((wx, wy+rng.uniform(-2.0,2.0)))
        idx += rng.randint(3,8)
    obs_world = [(x,y) for x,y in obs_world if math.hypot(x-6,y-6)>5.0]
    return obs_world

obs_world = gen_obstacles(FIXED_SEED)
OBS_R = 1.0; OBS_CLEAR = OBS_R+SAFE_R

# 终点
FINISH = (7.0, 82.5)

def sample_hf(wx, wy):
    mx, my = wx/SCALE, wy/SCALE
    px, py = int(mx*PIX_PER_M), HF_RES-1-int(my*PIX_PER_M)
    return int(hf[py,px]) if 0<=px<HF_RES and 0<=py<HF_RES else -1

is_wall = lambda wx, wy: sample_hf(wx, wy) != ROAD_PIX
is_obs = lambda wx, wy: any(math.hypot(wx-ox, wy-oy) < OBS_CLEAR for ox, oy in obs_world)
blocked = lambda wx, wy: is_wall(wx, wy) or is_obs(wx, wy)
walkable = lambda vx, vy: 0<=vx<W and 0<=vy<W and vox[vy,vx] in (FREE, VISITED)

# ── 文件读写 ──
def save_state(finished):
    np.save(SCAN_VOX, vox)
    meta = {"seed": FIXED_SEED, "finished": finished}
    with open(SCAN_META, 'w') as f:
        json.dump(meta, f)

def load_state():
    """返回 (vox_grid, finished) 或 None"""
    if not (os.path.exists(SCAN_VOX) and os.path.exists(SCAN_META)):
        return None
    loaded_vox = np.load(SCAN_VOX)
    if loaded_vox.shape != (W, W):
        return None
    with open(SCAN_META) as f:
        meta = json.load(f)
    if meta.get("seed") != FIXED_SEED:
        return None
    return loaded_vox, meta["finished"]

def state_exists_and_finished():
    if not (os.path.exists(SCAN_VOX) and os.path.exists(SCAN_META)):
        return False
    with open(SCAN_META) as f:
        meta = json.load(f)
    return meta.get("finished", False)

# ── 扫描 ──
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

# ── 最近门 ──
def find_nearest_gate(sx, sy):
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
            has_unk = any(
                vox[cy+dy,cx+dx]==UNKNOWN
                for dy in (-1,0,1) for dx in (-1,0,1)
                if 0<=cx+dx<W and 0<=cy+dy<W
            )
            if not has_unk: pass  # 没UNKNOWN邻居不算门
            else:
                # 过滤: 门本身离墙<3体素(1.5m)? 跳过
                too_near_wall = False
                for dy2 in range(-3, 4):
                    for dx2 in range(-3, 4):
                        tnx, tny = cx+dx2, cy+dy2
                        if 0<=tnx<W and 0<=tny<W and vox[tny,tnx]==WALL:
                            if abs(dx2)<=1 or abs(dy2)<=1:  # 邻接或紧贴
                                too_near_wall = True
                                break
                    if too_near_wall: break
                if not too_near_wall and cg < best_dist:
                    best_dist = cg; best_gate = (cx, cy)
        
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
    
    if best_gate is None: return None
    path = []; cur = best_gate
    while cur != (sx,sy):
        path.append(cur)
        if cur not in came_from: break
        cur = came_from[cur]
    path.reverse()
    return path

# ── A*到路标 ──
def astar_to(sx, sy, tx, ty):
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

# ── 可视化球管理 ──
MAX_MILESTONE_BALLS = 300
MAX_GATE_BALLS = 50

class BallManager:
    """管理mocap小球: [BLUE]路标球 [YELLOW]门球"""
    def __init__(self, m, d):
        self.m = m; self.d = d
        self.mstone_bodies = []  # body names
        self.gate_bodies = []
        self.mstone_count = 0
        self.gate_count = 0
    
    def add_milestone(self, wx, wy):
        """在(wx,wy)放一个蓝球"""
        i = self.mstone_count
        if i < MAX_MILESTONE_BALLS:
            body_name = f"mstone_{i}"
            if body_name in self.mstone_bodies:
                self.d.mocap_pos[self.m.body(body_name).mocapid] = [wx, wy, 1.5]
            self.mstone_count += 1
    
    def add_gate(self, wx, wy):
        """在(wx,wy)放一个黄球"""
        i = self.gate_count
        if i < MAX_GATE_BALLS:
            body_name = f"gate_{i}"
            if body_name in self.gate_bodies:
                self.d.mocap_pos[self.m.body(body_name).mocapid] = [wx, wy, 2.0]
            self.gate_count += 1
    
    def clear_gates(self):
        """隐藏所有门球"""
        for name in self.gate_bodies:
            mcap = self.m.body(name).mocapid
            self.d.mocap_pos[mcap] = [0, 0, -10]  # 地下=不可见
        self.gate_count = 0

# ── XML ──
def build_xml(milestone_count=0, gate_count=0):
    """构建含预分配可视化球的XML"""
    ms_xml = ""
    for i in range(MAX_MILESTONE_BALLS):
        z = 1.5 if i < milestone_count else -10
        ms_xml += f'<body name="mstone_{i}" mocap="true" pos="0 0 {z}"><geom type="sphere" size="0.2" rgba="0.3 0.6 1.0 0.8"/></body>\n'
    gt_xml = ""
    for i in range(MAX_GATE_BALLS):
        z = 2.0 if i < gate_count else -10
        gt_xml += f'<body name="gate_{i}" mocap="true" pos="0 0 {z}"><geom type="sphere" size="0.25" rgba="1.0 0.8 0.2 0.9"/></body>\n'
    
    FINISH_XML = f'<body mocap="true" pos="{FINISH[0]:.1f} {FINISH[1]:.1f} 2"><geom type="sphere" size="1.5" rgba="0.2 1.0 0.2 0.8"/></body>'
    OBS_XML = "".join(f'<body name="obs{i}" pos="{x:.1f} {y:.1f} 2.0"><geom type="cylinder" size="1.0 2.0" rgba="0.9 0.2 0.2 0.9"/></body>' for i,(x,y) in enumerate(obs_world))
    
    return f"""<mujoco>
  <compiler angle="radian"/><option timestep="0.005"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset><hfield name="track" size="50.0 50.0 4.0 2.0" file="{MAP}"/></asset>
  <worldbody>
    <light pos="50 50 80" dir="0 0 -1"/>
    {FINISH_XML}{OBS_XML}
    {ms_xml}{gt_xml}
    <geom type="hfield" hfield="track" pos="50 50 0.0" rgba="0.25 0.30 0.35 1.0" friction="0 0 0"/>
    <body name="bot" pos="0 0 0.5">
      <joint type="slide" axis="1 0 0" damping="0"/>
      <joint type="slide" axis="0 1 0" damping="0"/>
      <geom type="cylinder" size="0.5 0.5" rgba="1 0.3 0 1" friction="0 0 0"/>
    </body>
  </worldbody>
</mujoco>"""

# ── Mover (不变) ──
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
                print(f"  [BOUNCE] bounce#{self.bounce} @({self.d.qpos[0]:.1f},{self.d.qpos[1]:.1f})", flush=True)
        deg = random.uniform(lo, hi)*random.choice([-1,1])
        self.yaw += math.radians(deg)
        self.d.qvel[:] = 0
        self.force = int(0.3/(SPEED*self.m.opt.timestep))

# ═══════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════

# 1. 加载或初始化扫图
existing = load_state()
if existing is not None:
    loaded_vox, loaded_finished = existing
    print(f"[LOAD] 加载扫图: seed={FIXED_SEED} finished={loaded_finished}")
    np.copyto(vox, loaded_vox)
    milestones = []  # 路标不持久化, 运行时重建
    if loaded_finished:
        print("  [OK] 扫图已完成, 任意点A*直达终点")
else:
    print(f"[NEW] 新扫图: seed={FIXED_SEED}")
    milestones = []

# 2. 构建场景
xml = build_xml(len(milestones), 0)
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
d.qpos[0]=6; d.qpos[1]=6; mujoco.mj_forward(m,d)

mv = Mover(m, d)
balls = BallManager(m, d)
for name in [f"mstone_{i}" for i in range(MAX_MILESTONE_BALLS)]:
    balls.mstone_bodies.append(name)
for name in [f"gate_{i}" for i in range(MAX_GATE_BALLS)]:
    balls.gate_bodies.append(name)

# 加载已有路标到可视化
for wx, wy in milestones:
    balls.add_milestone(wx, wy)

step=0; t0=time.time(); RENDER_SKIP=20
last_milestone_x = last_milestone_y = 0
path = None; path_idx = 0
finish_found = existing is not None and existing[1]
no_gate_count = 0
current_target_type = ""  # "gate" or "milestone"

if milestones:
    mx, my = milestones[-1]
    last_milestone_x, last_milestone_y = mx, my

print(f"=== 萤火算法 Firefly v20 === 种子{FIXED_SEED} 路标{len(milestones)} 完成{finish_found}", flush=True)

with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
    v.cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.distance=25; v.cam.elevation=-35; v.cam.azimuth=180
    LIDAR_TICK = 20

    # 初始扫描
    print("  [SCAN] initial scan...", flush=True)
    for _ in range(200):
        bx, by = d.qpos[0], d.qpos[1]
        vx, vy = int(bx/VOXEL), int(by/VOXEL)
        if 0 <= vx < W and 0 <= vy < W: vox[vy, vx] = VISITED
        if _ % LIDAR_TICK == 0: scan_voxels(bx, by)
        mujoco.mj_step(m, d)
    print(f"  [OK] F{int(np.sum(vox==FREE))} W{int(np.sum(vox==WALL))}", flush=True)

    # 第一个路标
    if not milestones:
        sx, sy = int(d.qpos[0]/VOXEL), int(d.qpos[1]/VOXEL)
        milestones.append((sx, sy))
        last_milestone_x, last_milestone_y = sx, sy
        balls.add_milestone(d.qpos[0], d.qpos[1])

    # ── 主循环 ──
    while v.is_running():
        bx, by = d.qpos[0], d.qpos[1]
        if bx<1 or bx>99 or by<1 or by>99:
            d.qpos[0]=max(1,min(99,bx)); d.qpos[1]=max(1,min(99,by))
            d.qvel[:]=0; mv.yaw=random.uniform(0,2*math.pi)
        v.cam.lookat[:]=np.array([bx, by, 0.5], dtype=np.float64)

        vx, vy = int(bx/VOXEL), int(by/VOXEL)
        if 0 <= vx < W and 0 <= vy < W:
            vox[vy, vx] = VISITED
            if abs(vx-last_milestone_x)+abs(vy-last_milestone_y) >= MILESTONE_INTERVAL:
                milestones.append((vx, vy))
                last_milestone_x, last_milestone_y = vx, vy
                balls.add_milestone(bx, by)
                save_state(False)
                if len(milestones) % 20 == 0:
                    print(f"  [WAYPOINT] milestones={len(milestones)} @({bx:.1f},{by:.1f})", flush=True)
        
        if step % LIDAR_TICK == 0:
            scan_voxels(bx, by)

        # 终点检测
        if not finish_found:
            fx, fy = int(FINISH[0]/VOXEL), int(FINISH[1]/VOXEL)
            if 0<=fx<W and 0<=fy<W and vox[fy,fx] in (FREE, VISITED):
                if math.hypot(FINISH[0]-bx, FINISH[1]-by) < 5.0:
                    finish_found = True
                    save_state(True)
                    print(f"  [FINISH] FINISH! step={step} time={time.time()-t0:.1f}s bounce={mv.bounce}", flush=True)
                    break

        # 路径规划/执行
        if path is None or path_idx >= len(path):
            if finish_found:
                # 已完成: 沿路标走
                if path_idx >= len(path) and path is not None:
                    # 找下一个路标
                    pass
                break
            
            # 找最近的门
            gate_path = find_nearest_gate(vx, vy)
            
            if gate_path:
                no_gate_count = 0
                path = [((px+0.5)*VOXEL, (py+0.5)*VOXEL) for px, py in gate_path]
                path_idx = 0
                gate = gate_path[-1]
                gate_wx, gate_wy = (gate[0]+0.5)*VOXEL, (gate[1]+0.5)*VOXEL
                current_target_type = "gate"
                balls.clear_gates()
                balls.add_gate(gate_wx, gate_wy)
                print(f"  [GATE] [{step}] 导航到门({gate_wx:.0f},{gate_wy:.0f}) len={len(path)} F{int(np.sum(vox==FREE))}", flush=True)
            else:
                no_gate_count += 1
                if no_gate_count > 3 and len(milestones) > 1:
                    # 回退一个路标 (路标链保证可直达)
                    mx, my = milestones[-2]  # 上一个路标
                    back_path = astar_to(vx, vy, mx, my)
                    if back_path:
                        path = [((px+0.5)*VOXEL, (py+0.5)*VOXEL) for px, py in back_path]
                        path_idx = 0
                        current_target_type = "milestone"
                        balls.clear_gates()
                        mwx, mwy = (mx+0.5)*VOXEL, (my+0.5)*VOXEL
                        print(f"  [BACK] [{step}] 导航到路标({mwx:.0f},{mwy:.0f}) len={len(path)}", flush=True)
                        no_gate_count = 0
                    else:
                        # 上一个路标也到不了? 退起点
                        mx, my = milestones[0]
                        back_path = astar_to(vx, vy, mx, my)
                        if back_path:
                            path = [((px+0.5)*VOXEL, (py+0.5)*VOXEL) for px, py in back_path]
                            path_idx = 0
                            print(f"  [BACK] [{step}] 导航到起点({(mx+0.5)*VOXEL:.0f},{(my+0.5)*VOXEL:.0f})", flush=True)
                            no_gate_count = 0
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
        if step % 2000 == 0:
            print(f"  ... step={step} V{int(np.sum(vox==VISITED))} F{int(np.sum(vox==FREE))} milestones={len(milestones)}", flush=True)
        if step % RENDER_SKIP == 0: v.sync()
    
    save_state(finish_found)
    result = "FINISH" if finish_found else "stopped"
    print(f"done({result}): milestones={len(milestones)} step={step} time={time.time()-t0:.1f}s bounce={mv.bounce}", flush=True)
