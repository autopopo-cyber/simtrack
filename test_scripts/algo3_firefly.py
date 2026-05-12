#!/usr/bin/env python3
"""萤火算法 Firefly v3 SLAM — 字典地图 + 可变体素A* + far探索

可变体素:
  细格 0.1m — LIDAR精度, 边缘/门检测
  粗格 0.5m — A*寻路, 5×5细格聚合
  粗格状态: 全FREE→FREE, 有WALL→WALL, 否则UNKNOWN
  A*节点减少~25×, 配合缓存→O(1)查粗格
"""

import sys, os, math, time, random, heapq, json
import numpy as np
from PIL import Image
import mujoco, mujoco.viewer

# ═══════════════════════════════════════════
# 全部可配置参数
# ═══════════════════════════════════════════

MAP = os.path.expanduser("~/workspace/simtrack/confirmed/track_clean.png")
SCAN_DIR = os.path.expanduser("~/workspace/simtrack/scans")
SCAN_STATE = os.path.join(SCAN_DIR, "scan_dict.npz")
os.makedirs(SCAN_DIR, exist_ok=True)

SCALE = 2.0; HF_RES = 2000; PIX_PER_M = 40; ROAD_PIX = 128
SAFE_R = 0.5; SPEED = 5.0; SPEED_MAX = 8.0; YAW_RATE = 6.0
LIDAR_RANGE = 15.0

VOXEL = 0.1                                   # 细格精度
COARSE_FACTOR = 5                              # 粗格=5细格=0.5m
ROBOT_R = max(1, int(SAFE_R / VOXEL))         # 5
CLEARANCE = ROBOT_R
MILESTONE_STEP = int(3.0 / VOXEL)              # 30
LIDAR_STEPS = int(LIDAR_RANGE / VOXEL)         # 150
LIDAR_RAYS = 120

MAX_GATES = 200                                # 细格门候选多, 200个
WALL_SCAN_RADIUS = 10
WALL_BUFFER_M = 2.0; WALL_BUFFER_CELLS = int(WALL_BUFFER_M / VOXEL)  # 20
WALL_PENALTY = 3
MAX_GATE_DIST_COARSE = 60                      # 粗格门搜索上限(格=30m)
ASTAR_MAX_EXPAND = 20000

MIN_SPEED = 1.5; SPEED_FACTOR = 0.5
BOUNCE_FORCE_DURATION = 0.3
STUCK_TIMEOUT = 300; STUCK_DIST_THRESH = 0.5

EXPLORE_MODE = "far"
MIX_THRESHOLD = 50
INIT_SCAN_STEPS = 200
LIDAR_TICK = 20; RENDER_SKIP = 20
ARRIVE_THRESH = 1.0
WANDER_TIMEOUT = 600; WANDER_DRIFT_RATIO = 1.05
MAX_NO_GATE = 5
RESCUE_MS_COUNT = 5

FIXED_SEED = 42
MAX_MILESTONE_BALLS = 300; MAX_GATE_BALLS = 50
FINISH = (7.0, 82.5)

# ═══════════════════════════════════════════
# SLAM字典地图
# ═══════════════════════════════════════════
UNKNOWN, FREE, WALL = 0, 1, 2
grid = {}          # {(vx,vy): state}  细格
_wd = {}           # wall_dist缓存
_coarse = {}       # 粗格状态缓存: (cvx,cvy)→state

# 增量计数器 (避免gcount()扫全字典)
_cnt = {FREE: 0, WALL: 0}

def gget(vx, vy):
    return grid.get((vx, vy), UNKNOWN)

def gset(vx, vy, val):
    old = gget(vx, vy)
    if old == val: return
    if old != UNKNOWN: _cnt[old] -= 1
    grid[(vx, vy)] = val
    _cnt[val] += 1
    if val == WALL:
        _wd.clear()
        _coarse.pop((vx//COARSE_FACTOR, vy//COARSE_FACTOR), None)

def cget(cvx, cvy):
    """粗格状态 (缓存)"""
    key = (cvx, cvy)
    if key in _coarse: return _coarse[key]
    for dy in range(COARSE_FACTOR):
        for dx in range(COARSE_FACTOR):
            s = gget(cvx*COARSE_FACTOR+dx, cvy*COARSE_FACTOR+dy)
            if s == WALL:
                _coarse[key] = WALL; return WALL
    # 无WALL → 检查是否全FREE
    for dy in range(COARSE_FACTOR):
        for dx in range(COARSE_FACTOR):
            if gget(cvx*COARSE_FACTOR+dx, cvy*COARSE_FACTOR+dy) != FREE:
                _coarse[key] = UNKNOWN; return UNKNOWN
    _coarse[key] = FREE; return FREE

# ═══════════════════════════════════════════
# 地图加载 + 障碍物
# ═══════════════════════════════════════════

hf = np.array(Image.open(MAP))

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
    return [(x,y) for x,y in obs_world if math.hypot(x-6,y-6)>5.0]

obs_world = gen_obstacles(FIXED_SEED)
OBS_R = 1.0; OBS_CLEAR = OBS_R + SAFE_R

def sample_hf(wx, wy):
    mx, my = wx/SCALE, wy/SCALE
    px, py = int(mx*PIX_PER_M), HF_RES-1-int(my*PIX_PER_M)
    return int(hf[py,px]) if 0<=px<HF_RES and 0<=py<HF_RES else -1

def is_obstacle_world(wx, wy):
    if sample_hf(wx, wy) != ROAD_PIX: return True
    for ox, oy in obs_world:
        if math.hypot(wx-ox, wy-oy) < OBS_CLEAR: return True
    return False

# ═══════════════════════════════════════════
# 扫描
# ═══════════════════════════════════════════

def scan(bx, by):
    for a in np.linspace(0, 2*math.pi, LIDAR_RAYS):
        cos_a, sin_a = math.cos(a), math.sin(a)
        prev_vx, prev_vy = int(bx/VOXEL), int(by/VOXEL)
        for step_i in range(1, LIDAR_STEPS+1):
            wx, wy = bx + cos_a*step_i*VOXEL, by + sin_a*step_i*VOXEL
            vx, vy = int(wx/VOXEL), int(wy/VOXEL)
            if is_obstacle_world(wx, wy):
                gset(vx, vy, WALL)
                gset(prev_vx, prev_vy, WALL)
                break
            if gget(vx, vy) == UNKNOWN:
                gset(vx, vy, FREE)
            prev_vx, prev_vy = vx, vy

def blocked(wx, wy):
    vx, vy = int(wx/VOXEL), int(wy/VOXEL)
    for dy in range(-ROBOT_R, ROBOT_R+1):
        for dx in range(-ROBOT_R, ROBOT_R+1):
            if dx*dx+dy*dy <= ROBOT_R*ROBOT_R:
                nx, ny = vx+dx, vy+dy
                if is_obstacle_world((nx+0.5)*VOXEL, (ny+0.5)*VOXEL):
                    return True
    return False

# ── A* 辅助 ──
def wall_dist(vx, vy):
    key = (vx, vy)
    if key in _wd: return _wd[key]
    best = 999
    for dy in range(-WALL_SCAN_RADIUS, WALL_SCAN_RADIUS+1):
        for dx in range(-WALL_SCAN_RADIUS, WALL_SCAN_RADIUS+1):
            if gget(vx+dx, vy+dy) == WALL:
                d = abs(dx)+abs(dy)
                if d < best: best = d
    _wd[key] = best
    return best

def walkable(vx, vy):
    return gget(vx, vy) == FREE and wall_dist(vx, vy) > ROBOT_R

def line_clear(vx1, vy1, vx2, vy2):
    steps = max(abs(vx2-vx1), abs(vy2-vy1))
    if steps == 0: return True
    for i in range(steps+1):
        if gget(int(vx1+(vx2-vx1)*i/steps), int(vy1+(vy2-vy1)*i/steps)) == WALL:
            return False
    return True

# ── 粗格A*辅助 ──
ROBOT_R_COARSE = max(1, ROBOT_R // COARSE_FACTOR)  # 1
WALL_SCAN_COARSE = max(3, WALL_SCAN_RADIUS // COARSE_FACTOR)  # 2
CLEARANCE_COARSE = max(1, CLEARANCE // COARSE_FACTOR)  # 1

def coarse_wall_dist(cvx, cvy):
    best = 999
    for dy in range(-WALL_SCAN_COARSE, WALL_SCAN_COARSE+1):
        for dx in range(-WALL_SCAN_COARSE, WALL_SCAN_COARSE+1):
            if cget(cvx+dx, cvy+dy) == WALL:
                d = abs(dx)+abs(dy)
                if d < best: best = d
    return best

def coarse_walkable(cvx, cvy):
    return cget(cvx, cvy) == FREE and coarse_wall_dist(cvx, cvy) > ROBOT_R_COARSE

# ═══════════════════════════════════════════
# 门查找 (粗格A* + 细格门)
# ═══════════════════════════════════════════

def find_gates(fvx, fvy):
    """粗格A*找门, 返回细格门列表"""
    cvx, cvy = fvx//COARSE_FACTOR, fvy//COARSE_FACTOR
    if not coarse_walkable(cvx, cvy):
        return [], {}

    open_set = [(0, cvx, cvy)]
    came_from = {}; g_score = {(cvx, cvy): 0}
    visited = set()
    gates = []

    while open_set and len(came_from) < ASTAR_MAX_EXPAND and len(gates) < MAX_GATES:
        _, cx, cy = heapq.heappop(open_set)
        if (cx,cy) in visited: continue
        visited.add((cx,cy))
        cg = g_score.get((cx,cy), 9999)

        if gates and cg > MAX_GATE_DIST_COARSE:
            break

        # 检查粗格内的细格门
        if cget(cx, cy) == FREE:
            for dy in range(COARSE_FACTOR):
                for dx in range(COARSE_FACTOR):
                    gx, gy = cx*COARSE_FACTOR+dx, cy*COARSE_FACTOR+dy
                    if gget(gx, gy) != FREE: continue
                    if wall_dist(gx, gy) <= CLEARANCE: continue
                    has_unk = any(
                        gget(gx+ddx, gy+ddy) == UNKNOWN
                        for ddy in (-1,0,1) for ddx in (-1,0,1)
                    )
                    if has_unk:
                        gates.append((cg*COARSE_FACTOR, gx, gy))

        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
            nx, ny = cx+dx, cy+dy
            if not coarse_walkable(nx, ny): continue
            wd = coarse_wall_dist(nx, ny)
            penalty = max(0, WALL_BUFFER_CELLS//COARSE_FACTOR - wd) * WALL_PENALTY
            ng = cg + 1 + penalty
            if (nx,ny) not in g_score or ng < g_score[(nx,ny)]:
                g_score[(nx,ny)] = ng
                came_from[(nx,ny)] = (cx,cy)
                heapq.heappush(open_set, (ng, nx, ny))

    return gates, came_from

def pick_gate(gates, mode="far", stuck=False):
    if not gates: return None
    if stuck: return gates[0]
    if mode == "far": return gates[-1]
    if mode == "near": return gates[0]
    if mode == "mix":
        return gates[-1] if len(gates) >= MIX_THRESHOLD else gates[0]
    return gates[0]

def coarse_path_to_fine(cvx1, cvy1, gx, gy, came_from):
    """粗格路径→细格世界坐标路径"""
    # 从目标粗格回溯
    coarse_path = []
    cur = (gx//COARSE_FACTOR, gy//COARSE_FACTOR)
    start = (cvx1, cvy1)
    while cur != start:
        coarse_path.append(cur)
        if cur not in came_from: break
        cur = came_from[cur]
    coarse_path.reverse()

    # 粗格→世界坐标
    path = []
    for cvx, cvy in coarse_path:
        wx = (cvx*COARSE_FACTOR + COARSE_FACTOR//2 + 0.5) * VOXEL
        wy = (cvy*COARSE_FACTOR + COARSE_FACTOR//2 + 0.5) * VOXEL
        path.append((wx, wy))
    return path

def astar_to(fvx, fvy, tfx, tfy):
    """粗格A*到点, 返回细格世界坐标路径"""
    scx, scy = fvx//COARSE_FACTOR, fvy//COARSE_FACTOR
    tcx, tcy = tfx//COARSE_FACTOR, tfy//COARSE_FACTOR
    if not (coarse_walkable(scx, scy) and coarse_walkable(tcx, tcy)):
        return None

    open_set = [(math.hypot(tcx-scx, tcy-scy), scx, scy)]
    came_from = {}; g_score = {(scx, scy): 0}
    visited_set = set()
    while open_set and len(came_from) < ASTAR_MAX_EXPAND:
        _, cx, cy = heapq.heappop(open_set)
        if (cx,cy) in visited_set: continue
        visited_set.add((cx,cy))
        if (cx,cy) == (tcx,tcy): break
        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
            nx, ny = cx+dx, cy+dy
            if not coarse_walkable(nx, ny): continue
            ng = g_score.get((cx,cy), 999) + 1
            if (nx,ny) not in g_score or ng < g_score[(nx,ny)]:
                g_score[(nx,ny)] = ng
                came_from[(nx,ny)] = (cx,cy)
                heapq.heappush(open_set, (ng+math.hypot(tcx-nx, tcy-ny), nx, ny))

    if (tcx,tcy) not in came_from and (tcx,tcy) != (scx,scy): return None

    coarse_path = []; cur = (tcx,tcy)
    while cur != (scx,scy):
        coarse_path.append(cur)
        if cur not in came_from: break
        cur = came_from[cur]
    coarse_path.reverse()

    path = []
    for cvx, cvy in coarse_path:
        wx = (cvx*COARSE_FACTOR + COARSE_FACTOR//2 + 0.5) * VOXEL
        wy = (cvy*COARSE_FACTOR + COARSE_FACTOR//2 + 0.5) * VOXEL
        path.append((wx, wy))
    return path

# ═══════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════

class BallManager:
    def __init__(self, m, d):
        self.m = m; self.d = d
        self.mstone_bodies = []; self.gate_bodies = []
        self.mstone_count = 0; self.gate_count = 0
    def add_milestone(self, wx, wy):
        i = self.mstone_count
        if i < MAX_MILESTONE_BALLS:
            body_name = f"mstone_{i}"
            if body_name in self.mstone_bodies:
                self.d.mocap_pos[self.m.body(body_name).mocapid] = [wx, wy, 1.5]
            self.mstone_count += 1
    def add_gate(self, wx, wy):
        i = self.gate_count
        if i < MAX_GATE_BALLS:
            body_name = f"gate_{i}"
            if body_name in self.gate_bodies:
                self.d.mocap_pos[self.m.body(body_name).mocapid] = [wx, wy, 2.0]
            self.gate_count += 1
    def clear_gates(self):
        for name in self.gate_bodies:
            self.d.mocap_pos[self.m.body(name).mocapid] = [0, 0, -10]
        self.gate_count = 0

def build_xml():
    ms_xml = "".join(
        f'<body name="mstone_{i}" mocap="true" pos="0 0 -10">'
        f'<geom type="sphere" size="0.2" rgba="0.3 0.6 1.0 0.8"/></body>\n'
        for i in range(MAX_MILESTONE_BALLS))
    gt_xml = "".join(
        f'<body name="gate_{i}" mocap="true" pos="0 0 -10">'
        f'<geom type="sphere" size="0.25" rgba="1.0 0.8 0.2 0.9"/></body>\n'
        for i in range(MAX_GATE_BALLS))
    FINISH_XML = f'<body mocap="true" pos="{FINISH[0]:.1f} {FINISH[1]:.1f} 2"><geom type="sphere" size="1.5" rgba="0.2 1.0 0.2 0.8"/></body>'
    OBS_XML = "".join(
        f'<body name="obs{i}" pos="{x:.1f} {y:.1f} 2.0">'
        f'<geom type="cylinder" size="1.0 2.0" rgba="0.9 0.2 0.2 0.9"/></body>'
        for i,(x,y) in enumerate(obs_world))
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
            self.speed = max(MIN_SPEED, min(SPEED_MAX, math.hypot(tx-bx, ty-by)*SPEED_FACTOR))
        vx = math.cos(self.yaw)*self.speed; vy = math.sin(self.yaw)*self.speed
        nx, ny = bx+vx*dt, by+vy*dt
        if step-self.stuck_t > STUCK_TIMEOUT:
            if math.hypot(bx-self.stuck_x, by-self.stuck_y) < STUCK_DIST_THRESH:
                self._bounce(90, 180)
            self.stuck_t = step; self.stuck_x = bx; self.stuck_y = by
        if self.force > 0:
            self.force -= 1; self.d.qvel[0] = vx; self.d.qvel[1] = vy
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
        self.force = int(BOUNCE_FORCE_DURATION/(SPEED*self.m.opt.timestep))

# ═══════════════════════════════════════════
# 文件读写
# ═══════════════════════════════════════════

def save_state():
    if not grid: return
    xs = sorted(set(k[0] for k in grid)); ys = sorted(set(k[1] for k in grid))
    minx, maxx = xs[0], xs[-1]; miny, maxy = ys[0], ys[-1]
    w, h = maxx-minx+1, maxy-miny+1
    arr = np.full((h, w), UNKNOWN, dtype=np.int8)
    for (vx, vy), val in grid.items():
        arr[vy-miny, vx-minx] = val
    np.savez(SCAN_STATE, grid=arr, offset=(minx, miny), seed=FIXED_SEED, mode=EXPLORE_MODE)

def load_state():
    if not os.path.exists(SCAN_STATE): return None
    data = np.load(SCAN_STATE, allow_pickle=True)
    if data["seed"] != FIXED_SEED: return None
    arr = data["grid"]; ox, oy = data["offset"]
    loaded = {}
    for vy in range(arr.shape[0]):
        for vx in range(arr.shape[1]):
            if arr[vy, vx] != UNKNOWN:
                loaded[(vx+ox, vy+oy)] = int(arr[vy, vx])
    return loaded, str(data["mode"])

# ═══════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════

print(f"━━━ 萤火 Firefly v3 SLAM ━━━ {VOXEL}m细格/{VOXEL*COARSE_FACTOR}m粗格 模式={EXPLORE_MODE} ━━━", flush=True)

existing = load_state()
if existing is not None:
    loaded_grid, loaded_mode = existing
    print(f"[LOAD] 加载扫图: {len(loaded_grid)} cells mode={loaded_mode}")
    for k, v in loaded_grid.items():
        gset(*k, v)
    milestones = []
else:
    print(f"[NEW] 新扫图: mode={EXPLORE_MODE}")
    milestones = []

xml = build_xml()
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
d.qpos[0]=6; d.qpos[1]=6; mujoco.mj_forward(m,d)

mv = Mover(m, d)
balls = BallManager(m, d)
for name in [f"mstone_{i}" for i in range(MAX_MILESTONE_BALLS)]:
    balls.mstone_bodies.append(name)
for name in [f"gate_{i}" for i in range(MAX_GATE_BALLS)]:
    balls.gate_bodies.append(name)
for wx, wy in milestones:
    balls.add_milestone(wx, wy)

step = 0; t0 = time.time()
last_mx = last_my = 0
path = None; path_idx = 0
no_gate_count = 0
wander = 0; last_dist = 999

if milestones:
    last_mx, last_my = milestones[-1]

print(f"=== Firefly v3 start: fine={VOXEL}m coarse={VOXEL*COARSE_FACTOR}m gates={MAX_GATES} ===", flush=True)

with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
    v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.distance = 25; v.cam.elevation = -35; v.cam.azimuth = 180

    print("  [SCAN] initial scan...", flush=True)
    for _ in range(INIT_SCAN_STEPS):
        bx, by = d.qpos[0], d.qpos[1]
        if _ % LIDAR_TICK == 0: scan(bx, by)
        mujoco.mj_step(m, d)
    print(f"  [OK] FREE={_cnt[FREE]} WALL={_cnt[WALL]}", flush=True)

    if not milestones:
        sx, sy = int(d.qpos[0]/VOXEL), int(d.qpos[1]/VOXEL)
        milestones.append((sx, sy))
        last_mx, last_my = sx, sy
        balls.add_milestone(d.qpos[0], d.qpos[1])

    while v.is_running():
        try:
            bx, by = d.qpos[0], d.qpos[1]
        except AttributeError:
            print("  [!] viewer线程崩溃", flush=True)
            d = mujoco.MjData(m); mv.d = d
            bx, by = d.qpos[0], d.qpos[1]

        v.cam.lookat[:] = np.array([bx, by, 0.5], dtype=np.float64)

        vx, vy = int(bx/VOXEL), int(by/VOXEL)
        if gget(vx, vy) == UNKNOWN:
            gset(vx, vy, FREE)
        if abs(vx-last_mx)+abs(vy-last_my) >= MILESTONE_STEP:
            if wall_dist(vx, vy) > CLEARANCE:
                milestones.append((vx, vy))
                last_mx, last_my = vx, vy
                balls.add_milestone(bx, by)
                save_state()
                if len(milestones) % 20 == 0:
                    print(f"  [WAYPOINT] ms={len(milestones)} @({bx:.1f},{by:.1f})", flush=True)

        if step % LIDAR_TICK == 0:
            scan(bx, by)

        if path is None or path_idx >= len(path):
            gates, came_from = find_gates(vx, vy)
            gate = pick_gate(gates, EXPLORE_MODE, stuck=(no_gate_count > 0))

            if gate is not None:
                cg, gx, gy = gate
                cvx, cvy = vx//COARSE_FACTOR, vy//COARSE_FACTOR
                path = coarse_path_to_fine(cvx, cvy, gx, gy, came_from)
                path_idx = 0; wander = 0; last_dist = 999
                no_gate_count = 0
                gate_wx, gate_wy = (gx+0.5)*VOXEL, (gy+0.5)*VOXEL
                balls.clear_gates()
                balls.add_gate(gate_wx, gate_wy)
                print(f"  [GATE] [{step}] →({gate_wx:.1f},{gate_wy:.1f}) path={len(path)} gates={len(gates)} F={_cnt[FREE]} W={_cnt[WALL]}", flush=True)
            else:
                no_gate_count += 1
                if no_gate_count > MAX_NO_GATE and len(milestones) > 1:
                    mx, my = milestones[-2]
                    bp = astar_to(vx, vy, mx, my)
                    if bp:
                        path = bp; path_idx = 0; wander = 0; last_dist = 999
                        balls.clear_gates()
                        print(f"  [BACK] [{step}] →路标({(mx+0.5)*VOXEL:.1f},{(my+0.5)*VOXEL:.1f})", flush=True)
                        no_gate_count = 0
                    else:
                        mx, my = milestones[0]
                        bp = astar_to(vx, vy, mx, my)
                        if bp:
                            path = bp; path_idx = 0; wander = 0; last_dist = 999
                            print(f"  [BACK] [{step}] →起点({(mx+0.5)*VOXEL:.1f},{(my+0.5)*VOXEL:.1f})", flush=True)
                            no_gate_count = 0
                        else:
                            mv._bounce(90, 180)
                else:
                    mv._bounce(90, 180)

        if path is not None and path_idx < len(path):
            tx, ty = path[path_idx]
            ddist = math.hypot(tx-bx, ty-by)
            if ddist < ARRIVE_THRESH:
                path_idx += 1
                last_dist = 999; wander = 0
            elif ddist > last_dist * WANDER_DRIFT_RATIO:
                wander += 1
                if wander > WANDER_TIMEOUT:
                    for mx, my in reversed(milestones[-RESCUE_MS_COUNT:]):
                        if line_clear(vx, vy, mx, my):
                            path = [((mx+0.5)*VOXEL, (my+0.5)*VOXEL)]
                            path_idx = 0; wander = 0; last_dist = 999
                            print(f"  [LOST] [{step}] →路标({(mx+0.5)*VOXEL:.1f},{(my+0.5)*VOXEL:.1f})", flush=True)
                            break
                    else:
                        path = None; path_idx = 0; wander = 0; last_dist = 999
                        print(f"  [LOST] [{step}] 重新规划", flush=True)
                else:
                    last_dist = ddist; mv.step(tx, ty, step)
            else:
                wander = max(0, wander-1); last_dist = ddist
                mv.step(tx, ty, step)
        else:
            mv._bounce(90, 180)

        step += 1
        if step % 2000 == 0:
            print(f"  ... step={step} F={_cnt[FREE]} W={_cnt[WALL]} ms={len(milestones)}", flush=True)
        if step % RENDER_SKIP == 0:
            v.sync()

    save_state()
    print(f"done: ms={len(milestones)} step={step} t={time.time()-t0:.1f}s bounce={mv.bounce} mode={EXPLORE_MODE}", flush=True)
