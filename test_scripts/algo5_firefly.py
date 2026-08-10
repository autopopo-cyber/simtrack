#!/usr/bin/env python3
"""萤火算法 V5 — 边界簇门 + 黄球路点
概念: 门=FREE/UNKNOWN边界的连通域簇中心
      不是单个体素, 不是多边形边 — 是"已知探索到哪了"的集群
"""
import sys, os, math, time, random, heapq
import numpy as np
import mujoco, mujoco.viewer
from PIL import Image

# ═══════════════ 参数 ═══════════════
MAP = os.path.expanduser("~/workspace/simtrack/confirmed/track_clean.png")
SCAN_DIR = os.path.expanduser("~/workspace/simtrack/scans")
SCAN_STATE = os.path.join(SCAN_DIR, "scan_dict.npz")
os.makedirs(SCAN_DIR, exist_ok=True)

SCALE = 2.0; PIX_PER_M = 40
SAFE_R = 0.5; SPEED = 5.0; YAW_RATE = 6.0
LIDAR_RANGE = 15.0; VOXEL = 0.1
ROBOT_R = max(1, int(SAFE_R / VOXEL))
CLEARANCE = ROBOT_R
LIDAR_STEPS = int(LIDAR_RANGE / VOXEL)
LIDAR_RAYS = 120; RENDER_SKIP = 100; LIDAR_TICK = 20

# 门参数
MAX_GATES = 200; MAX_GATE_DIST = 3000
WALL_SCAN_RADIUS = 10; WALL_BUFFER_CELLS = 20; WALL_PENALTY = 3
ASTAR_MAX_EXPAND = 30000

# 跳步
JUMP_1M = 10; JUMP_03 = 3; JUMP_NEAR = 1

# 运动
MIN_SPEED = 3.0; SPEED_FACTOR = 2.0
BOUNCE_FORCE_DURATION = 0.3
STUCK_TIMEOUT = 300; STUCK_DIST_THRESH = 0.5
ARRIVE_THRESH = 1.0; PLAN_INTERVAL = 200

# 可视化
MAX_MILESTONE_BALLS = 1000; MAX_GATE_BALLS = 50; MAX_WAYPOINT_BALLS = 200
FINISH = (3.0, 95.0)
FIXED_SEED = random.randint(0, 999999)

# ═══════════════ SLAM字典 ═══════════════
UNKNOWN, FREE, WALL = 0, 1, 2
grid = {}; _wd = {}; _cnt = {FREE: 0, WALL: 0}

def gget(vx, vy): return grid.get((vx, vy), UNKNOWN)
def gset(vx, vy, val):
    global _cnt
    old = gget(vx, vy)
    if old != UNKNOWN: _cnt[old] -= 1
    grid[(vx, vy)] = val
    _cnt[val] = _cnt.get(val, 0) + 1
    if val == WALL: _wd.clear()

# ═══════════════ 激光 ═══════════════
def is_obstacle_world(wx, wy):
    vx, vy = int(wx/VOXEL), int(wy/VOXEL)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if gget(vx+dx, vy+dy) == WALL: return True
    return False

def scan(bx, by):
    """120线激光扫描, 标记FREE/WALL"""
    for a in np.linspace(0, 2*math.pi, LIDAR_RAYS):
        cos_a, sin_a = math.cos(a), math.sin(a)
        prev_vx, prev_vy = int(bx/VOXEL), int(by/VOXEL)
        for step_i in range(1, LIDAR_STEPS+1):
            wx = bx + cos_a * step_i * VOXEL
            wy = by + sin_a * step_i * VOXEL
            vx, vy = int(wx/VOXEL), int(wy/VOXEL)
            if is_obstacle_world(wx, wy):
                gset(vx, vy, WALL); gset(prev_vx, prev_vy, WALL)
                break
            if gget(vx, vy) == UNKNOWN: gset(vx, vy, FREE)
            prev_vx, prev_vy = vx, vy
    _wd.clear()

# ═══════════════ 边界门系统 ═══════════════
def wall_dist(vx, vy):
    key = (vx, vy)
    if key in _wd: return _wd[key]
    best = 999
    for dy in range(-WALL_SCAN_RADIUS, WALL_SCAN_RADIUS+1):
        for dx in range(-WALL_SCAN_RADIUS, WALL_SCAN_RADIUS+1):
            if gget(vx+dx, vy+dy) == WALL:
                d = abs(dx)+abs(dy)
                if d < best: best = d
    _wd[key] = best; return best

def walkable(vx, vy):
    return gget(vx, vy) == FREE and wall_dist(vx, vy) > ROBOT_R

def jump_steps(vx, vy, dx, dy):
    wd = wall_dist(vx, vy)
    max_jump = JUMP_1M if wd >= JUMP_1M else (JUMP_03 if wd >= JUMP_03 else JUMP_NEAR)
    for step in range(1, max_jump+1):
        nx, ny = vx+dx*step, vy+dy*step
        if not walkable(nx, ny): return step-1
    return max_jump

def find_gates(fvx, fvy):
    """A*探索FREE空间 → 找FREE邻接UNKNOWN的边界体素 → 连通域聚类"""
    if not walkable(fvx, fvy): return [], {}
    open_set = [(0, fvx, fvy)]
    came_from = {}; g_score = {(fvx, fvy): 0}
    visited = set(); raw_gates = []

    while open_set and len(came_from) < ASTAR_MAX_EXPAND and len(raw_gates) < MAX_GATES:
        _, cx, cy = heapq.heappop(open_set)
        if (cx,cy) in visited: continue
        visited.add((cx,cy))
        cg = g_score.get((cx,cy), 9999)
        if raw_gates and cg > MAX_GATE_DIST: break

        # 边界体素: FREE + 邻接UNKNOWN + 安全
        if gget(cx, cy) == FREE:
            has_unk = any(gget(cx+dx, cy+dy) == UNKNOWN for dy in (-1,0,1) for dx in (-1,0,1))
            if has_unk and wall_dist(cx, cy) > CLEARANCE:
                raw_gates.append((cg, wall_dist(cx, cy), cx, cy))

        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
            js = jump_steps(cx, cy, dx, dy)
            if js < 1: continue
            nx, ny = cx+dx*js, cy+dy*js
            penalty = max(0, WALL_BUFFER_CELLS - wall_dist(nx, ny)) * WALL_PENALTY
            ng = cg + js + penalty
            if (nx,ny) not in g_score or ng < g_score[(nx,ny)]:
                g_score[(nx,ny)] = ng; came_from[(nx,ny)] = (cx,cy)
                heapq.heappush(open_set, (ng, nx, ny))

    return merge_gates(raw_gates, came_from), came_from

def merge_gates(raw_gates, came_from):
    """BFS连通域聚类 — 每簇选最接近几何中心的门体素"""
    if not raw_gates: return []
    gate_set = set(); gate_info = {}
    for cg, wd, cx, cy in raw_gates:
        gate_set.add((cx,cy)); gate_info[(cx,cy)] = (cg, wd)
    visited = set(); clusters = []
    for sx, sy in gate_set:
        if (sx,sy) in visited: continue
        cluster = []; stack = [(sx,sy)]
        while stack:
            cx, cy = stack.pop()
            if (cx,cy) in visited or (cx,cy) not in gate_set: continue
            visited.add((cx,cy)); cluster.append((cx,cy))
            for dx,dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx,ny = cx+dx, cy+dy
                if (nx,ny) in gate_set and (nx,ny) not in visited: stack.append((nx,ny))
        if not cluster: continue
        n = len(cluster)
        ctr_x = sum(x for x,y in cluster)/n; ctr_y = sum(y for x,y in cluster)/n
        best, best_dist = None, 999999
        for cx, cy in cluster:
            if (cx,cy) not in came_from: continue
            d = (cx-ctr_x)**2 + (cy-ctr_y)**2
            if d < best_dist: best_dist = d; cg, wd = gate_info[(cx,cy)]; best = (cg, wd, cx, cy)
        if best: clusters.append(best)
    clusters.sort(key=lambda g: g[0])
    return clusters

def pick_gate(gates, mode="far", stuck=False):
    """门: (cg, wd, cx, cy). 选朝向终点的最近门"""
    if not gates: return None
    if stuck: return min(gates, key=lambda g: g[0])
    # V5: 朝向终点 — A*距离+到终点距离加权
    best, best_score = None, float('inf')
    for g in gates:
        _, _, cx, cy = g
        wx, wy = (cx+0.5)*VOXEL, (cy+0.5)*VOXEL
        fd = math.hypot(wx-FINISH[0], wy-FINISH[1])
        score = g[0] * 0.3 + fd * 0.7  # 30% A*距离 + 70% 离终点近
        if score < best_score:
            best_score, best = score, g
    return best

# ═══════════════ A* 路径 ═══════════════
def fine_path(sx, sy, gx, gy, came_from):
    path = []; cur = (gx, gy)
    while cur != (sx, sy):
        path.append(cur)
        if cur not in came_from: break
        cur = came_from[cur]
    path.reverse()
    return [((px+0.5)*VOXEL, (py+0.5)*VOXEL) for px, py in path]

def astar_to(fvx, fvy, tfx, tfy):
    if not walkable(fvx, fvy): return None
    if gget(tfx, tfy) == WALL: return None
    open_set = [(math.hypot(tfx-fvx, tfy-fvy), fvx, fvy)]
    came_from = {}; g_score = {(fvx,fvy): 0}; visited_set = set()
    while open_set and len(came_from) < ASTAR_MAX_EXPAND:
        _, cx, cy = heapq.heappop(open_set)
        if (cx,cy) in visited_set: continue
        visited_set.add((cx,cy))
        if (cx,cy) == (tfx,tfy): break
        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
            js = jump_steps(cx, cy, dx, dy)
            if js < 1: continue
            nx, ny = cx+dx*js, cy+dy*js
            ng = g_score.get((cx,cy),999) + js
            if (nx,ny) not in g_score or ng < g_score[(nx,ny)]:
                g_score[(nx,ny)] = ng; came_from[(nx,ny)] = (cx,cy)
                heapq.heappush(open_set, (ng+math.hypot(tfx-nx, tfy-ny), nx, ny))
    if (tfx,tfy) not in came_from and (tfx,tfy) != (fvx,fvy): return None
    return fine_path(fvx, fvy, tfx, tfy, came_from)

def gen_yellow_waypoints(raw_path):
    """沿A*路径每1米一个黄球"""
    if not raw_path: return []
    wp = [raw_path[0]]; acc = 0.0; px, py = raw_path[0]
    for wx, wy in raw_path[1:]:
        acc += math.hypot(wx-px, wy-py)
        if acc >= 1.0: wp.append((wx,wy)); acc = 0.0
        px, py = wx, wy
    last = raw_path[-1]
    if not wp or (abs(wp[-1][0]-last[0])>0.01 or abs(wp[-1][1]-last[1])>0.01): wp.append(last)
    return wp

# ═══════════════ 渲染 ═══════════════
class BallManager:
    def __init__(self, m, d):
        self.m, self.d = m, d
        self.mstone_bodies = []; self.gate_bodies = []; self.waypoint_bodies = []
        self.mstone_count = 0; self.gate_count = 0; self.waypoint_count = 0

    def add_milestone(self, wx, wy):
        i = self.mstone_count
        if i < MAX_MILESTONE_BALLS:
            name = f"mstone_{i}"
            if name in self.mstone_bodies: self.d.mocap_pos[self.m.body(name).mocapid] = [wx, wy, 1.5]
            self.mstone_count += 1

    def add_gate(self, wx, wy):
        i = self.gate_count
        if i < MAX_GATE_BALLS:
            name = f"gate_{i}"
            if name in self.gate_bodies: self.d.mocap_pos[self.m.body(name).mocapid] = [wx, wy, 2.0]
            self.gate_count += 1

    def add_waypoint(self, wx, wy):
        i = self.waypoint_count
        if i < MAX_WAYPOINT_BALLS:
            name = f"wp_{i}"
            if name in self.waypoint_bodies: self.d.mocap_pos[self.m.body(name).mocapid] = [wx, wy, 0.8]
            self.waypoint_count += 1

    def clear_gates(self):
        for name in self.gate_bodies: self.d.mocap_pos[self.m.body(name).mocapid] = [0, 0, -10]
        self.gate_count = 0

    def clear_waypoints(self):
        for name in self.waypoint_bodies: self.d.mocap_pos[self.m.body(name).mocapid] = [0, 0, -10]
        self.waypoint_count = 0

    def clear_milestones(self):
        for name in self.mstone_bodies: self.d.mocap_pos[self.m.body(name).mocapid] = [0, 0, -10]
        self.mstone_count = 0

def build_xml(obs_world):
    ms_xml = "".join(f'<body name="mstone_{i}" mocap="true" pos="0 0 -10"><geom type="sphere" size="0.2" rgba="0.3 0.6 1.0 0.8"/></body>\n' for i in range(MAX_MILESTONE_BALLS))
    gt_xml = "".join(f'<body name="gate_{i}" mocap="true" pos="0 0 -10"><geom type="sphere" size="0.25" rgba="1.0 0.8 0.2 0.9"/></body>\n' for i in range(MAX_GATE_BALLS))
    wp_xml = "".join(f'<body name="wp_{i}" mocap="true" pos="0 0 -10"><geom type="sphere" size="0.15" rgba="1.0 1.0 0.0 0.9"/></body>\n' for i in range(MAX_WAYPOINT_BALLS))
    fin_xml = f'<body mocap="true" pos="{FINISH[0]:.1f} {FINISH[1]:.1f} 2"><geom type="sphere" size="1.5" rgba="0.2 1.0 0.2 0.8"/></body>'
    obs_xml = "".join(f'<body name="obs{i}" pos="{x:.1f} {y:.1f} 2.0"><geom type="cylinder" size="1.0 2.0" rgba="0.9 0.2 0.2 0.9"/></body>' for i,(x,y) in enumerate(obs_world))
    return f"""<mujoco>
  <compiler angle="radian"/><option timestep="0.005"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset><hfield name="track" size="50.0 50.0 4.0 2.0" file="{MAP}"/></asset>
  <worldbody>
    <light pos="50 50 80" dir="0 0 -1"/>
    {fin_xml}{obs_xml}{ms_xml}{gt_xml}{wp_xml}
    <geom type="hfield" hfield="track" pos="50 50 0.0" rgba="0.25 0.30 0.35 1.0" friction="0 0 0"/>
    <body name="bot" pos="3.0 3.0 0.5">
      <joint type="slide" axis="1 0 0" damping="0"/>
      <joint type="slide" axis="0 1 0" damping="0"/>
      <geom type="cylinder" size="0.5 0.5" rgba="1 0.3 0 1" friction="0 0 0"/>
    </body>
  </worldbody>
</mujoco>"""

# ═══════════════ Mover ═══════════════
class Mover:
    def __init__(self, m, d):
        self.m, self.d = m, d
        self.yaw = 0.0; self.speed = SPEED; self.bounce = 0
        self.force = 0; self.escaping = False
        self.stuck_t = 0; self.stuck_x = 0.0; self.stuck_y = 0.0

    def step(self, tx, ty, step_no):
        bx, by = self.d.qpos[0], self.d.qpos[1]
        dt = self.m.opt.timestep
        if not self.escaping:
            tgt_yaw = math.atan2(ty-by, tx-bx)
            err = (tgt_yaw - self.yaw + math.pi) % (2*math.pi) - math.pi
            dyaw = max(-YAW_RATE*dt, min(YAW_RATE*dt, err))
            self.yaw += dyaw; self.speed = SPEED
        vx = math.cos(self.yaw)*self.speed; vy = math.sin(self.yaw)*self.speed
        if step_no - self.stuck_t > STUCK_TIMEOUT:
            if math.hypot(bx-self.stuck_x, by-self.stuck_y) < STUCK_DIST_THRESH:
                self._bounce(90, 180)
            self.stuck_t = step_no; self.stuck_x = bx; self.stuck_y = by
        if self.force > 0:
            self.force -= 1; self.d.qvel[0] = vx; self.d.qvel[1] = vy
        elif self.escaping:
            self.escaping = False; self.d.qvel[0] = vx; self.d.qvel[1] = vy
        else:
            self.d.qvel[0] = vx; self.d.qvel[1] = vy
        mujoco.mj_step(self.m, self.d)

    def _bounce(self, lo, hi):
        if not self.escaping:
            self.bounce += 1; self.escaping = True
            if self.bounce % 5 == 0: print(f"  [BOUNCE] #{self.bounce}", flush=True)
        deg = random.uniform(lo, hi)*random.choice([-1,1])
        self.yaw += math.radians(deg); self.d.qvel[:] = 0
        self.force = int(BOUNCE_FORCE_DURATION/(SPEED*self.m.opt.timestep))

# ═══════════════ 障碍物生成 ═══════════════
def load_track():
    img = Image.open(MAP); arr = np.array(img)
    h, w = arr.shape if arr.ndim == 2 else arr.shape[:2]
    return arr, w, h

def gen_random_obstacles(arr, w, h, seed, n=12):
    random.seed(seed); obs = []
    road_pix = set()
    for y in range(h):
        for x in range(w):
            if arr.ndim == 2:
                if 50 <= arr[y,x] <= 200: road_pix.add((x,y))
            elif len(arr.shape) == 3 and arr.shape[2] >= 3:
                r,g,b = arr[y,x,:3] if arr.shape[2]>=3 else (255,255,255)
                if 50 <= int(r) <= 200 and 50 <= int(g) <= 200: road_pix.add((x,y))
    attempts = 0
    for _ in range(n*5):
        if len(obs) >= n: break
        attempts += 1
        px, py = random.sample(sorted(road_pix), 1)[0] if road_pix else (random.randint(50,w-50), random.randint(50,h-50))
        wx = (px) / PIX_PER_M; wy = (py) / PIX_PER_M
        if 2 < wx < 98 and 2 < wy < 98:
            too_close = any(math.hypot(wx-ox, wy-oy) < 3.0 for ox,oy in obs)
            too_close_to_start = math.hypot(wx-3, wy-3) < 5.0
            too_close_to_finish = math.hypot(wx-FINISH[0], wy-FINISH[1]) < 5.0
            if not too_close and not too_close_to_start and not too_close_to_finish:
                obs.append((wx, wy))
    return obs

# ═══════════════ 存档 ═══════════════
def save_state():
    if not grid: return
    xs = sorted(set(k[0] for k in grid)); ys = sorted(set(k[1] for k in grid))
    if not xs: return
    minx, maxx = xs[0], xs[-1]; miny, maxy = ys[0], ys[-1]
    w, h = maxx-minx+1, maxy-miny+1
    arr = np.full((h, w), UNKNOWN, dtype=np.int8)
    for (vx, vy), val in grid.items(): arr[vy-miny, vx-minx] = val
    np.savez(SCAN_STATE, grid=arr, offset=(minx, miny), seed=FIXED_SEED)

def load_state():
    if not os.path.exists(SCAN_STATE): return None
    data = np.load(SCAN_STATE, allow_pickle=True)
    if data["seed"] != FIXED_SEED: return None
    arr = data["grid"]; ox, oy = data["offset"]
    loaded = {}
    for vy in range(arr.shape[0]):
        for vx in range(arr.shape[1]):
            if arr[vy, vx] != UNKNOWN: loaded[(vx+ox, vy+oy)] = int(arr[vy, vx])
    return loaded

# ═══════════════ 主入口 ═══════════════
print(f"━━━ 萤火 V5 边界簇门 + 黄球 ━━━ {VOXEL}m voxel {LIDAR_RAYS}线 ━━━", flush=True)

# 加载或初始化地图
existing = load_state()
if existing:
    for k, v in existing.items(): grid[k] = v
    _cnt[FREE] = sum(1 for v in existing.values() if v==FREE)
    _cnt[WALL] = sum(1 for v in existing.values() if v==WALL)
    milestones = []; print(f"[LOAD] {len(grid)} cells", flush=True)
else:
    milestones = []; print(f"[NEW] 新扫图", flush=True)

# 障碍物
track_arr, tw, th = load_track()
obs_world = gen_random_obstacles(track_arr, tw, th, FIXED_SEED, 12)
print(f"[OBS] {len(obs_world)}个障碍物", flush=True)

# MuJoCo
xml = build_xml(obs_world)
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mv = Mover(m, d)

# 路标球管理
balls = BallManager(m, d)
for name in [f"mstone_{i}" for i in range(MAX_MILESTONE_BALLS)]: balls.mstone_bodies.append(name)
for name in [f"gate_{i}" for i in range(MAX_GATE_BALLS)]: balls.gate_bodies.append(name)
for name in [f"wp_{i}" for i in range(MAX_WAYPOINT_BALLS)]: balls.waypoint_bodies.append(name)

# ── 初始扫描 ──
d.qpos[0] = 3.0; d.qpos[1] = 3.0
print(f"[INIT] 起点(3,3) seed={FIXED_SEED}", flush=True)
for _ in range(50): scan(3.0, 3.0)  # 初始扫描标记地面
print(f"[OK] FREE={_cnt[FREE]} WALL={_cnt[WALL]}", flush=True)

# ── Viewer ──
with mujoco.viewer.launch_passive(m, d) as viewer:
    viewer.cam.azimuth = -90; viewer.cam.elevation = -45; viewer.cam.lookat = [50, 50, 0]
    step = 0; t0 = time.time(); no_gate_count = 0
    yellow_wps = []; target_wp = None
    bp = []; path_idx = 0

    while viewer.is_running():
        bx, by = d.qpos[0], d.qpos[1]
        vx, vy = int(bx/VOXEL), int(by/VOXEL)
        if gget(vx, vy) == UNKNOWN: gset(vx, vy, FREE)

        # ── 终点检测 ──
        if math.hypot(bx-FINISH[0], by-FINISH[1]) < 3.0:
            print(f"\n  ★ ARRIVED! @({bx:.1f},{by:.1f}) step={step}", flush=True)
            break

        # ── 激光 ──
        if step % LIDAR_TICK == 0: scan(bx, by)

        # ── 1Hz边界门规划 ──
        if step % PLAN_INTERVAL == 0:
            gates, _ = find_gates(vx, vy)
            gate = pick_gate(gates, stuck=(no_gate_count > 0))
            if gate is not None:
                _, _, gx, gy = gate
                raw = astar_to(vx, vy, gx, gy)
                if raw:
                    yellow_wps = gen_yellow_waypoints(raw)
                    target_wp = yellow_wps[0] if yellow_wps else None
                    no_gate_count = 0
                    gate_wx, gate_wy = (gx+0.5)*VOXEL, (gy+0.5)*VOXEL
                    balls.clear_gates(); balls.add_gate(gate_wx, gate_wy)
                    balls.clear_waypoints()
                    for wwx, wwy in yellow_wps: balls.add_waypoint(wwx, wwy)
                    print(f"  [GATE] [{step}] →({gate_wx:.1f},{gate_wy:.1f}) yellow={len(yellow_wps)} gates={len(gates)}", flush=True)
            else:
                no_gate_count += 1
                print(f"  [NOGATE] [{step}] cnt={no_gate_count}", flush=True)

        # ── 运动: 冲黄球路点 ──
        if target_wp:
            tx, ty = target_wp
            mv.step(tx, ty, step)
            if math.hypot(tx-bx, ty-by) < ARRIVE_THRESH:
                yellow_wps.pop(0)
                target_wp = yellow_wps[0] if yellow_wps else None
                if target_wp: mv.stuck_t = step; mv.stuck_x = bx; mv.stuck_y = by
        else:
            # 无黄球: 直接冲路标
            if milestones:
                mx, my = milestones[-1]
                tx, ty = (mx+0.5)*VOXEL, (my+0.5)*VOXEL
                mv.step(tx, ty, step)
            else:
                mv.step(FINISH[0], FINISH[1], step)

        # ── 加路标 ──
        if step % 1000 == 0 and len(milestones) < MAX_MILESTONE_BALLS:
            mx, my = vx, vy
            if not milestones or milestones[-1] != (mx, my):
                milestones.append((mx, my))
                balls.add_milestone((mx+0.5)*VOXEL, (my+0.5)*VOXEL)

        step += 1
        if step % 2000 == 0: print(f"  ... step={step} F={_cnt[FREE]} W={_cnt[WALL]} ms={len(milestones)}", flush=True)
        if step % RENDER_SKIP == 0: viewer.sync()

    save_state()
    print(f"done: ms={len(milestones)} step={step} t={time.time()-t0:.1f}s bounce={mv.bounce}", flush=True)
