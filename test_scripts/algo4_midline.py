#!/usr/bin/env python3
"""萤火算法 Firefly v4 最大体素优先 — 选道路最宽处(wall_dist最大)推进前线

核心思路:
  道路中间的格 → wall_dist大 (体素大)
  贴墙的格     → wall_dist小 (体素小)
  机器人优先走最大的体素 → 自然在道路中央
  
  每步: 前方扇形采样 → 选wall_dist最大的FREE格 → A*过去
  障碍物出现 → wall_dist地图更新 → 宽处偏移 → 轨迹自动扭过去
  渐进放置: 目标不放最远处, 放中间距离 → 下一步再算 → 自然前推
"""

import sys, os, math, time, random, heapq
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

VOXEL = 0.1
ROBOT_R = max(1, int(SAFE_R / VOXEL))
CLEARANCE = ROBOT_R
MILESTONE_STEP = int(3.0 / VOXEL)
LIDAR_STEPS = int(LIDAR_RANGE / VOXEL)
LIDAR_RAYS = 120

# ── v4 最大体素优先参数 ──
FAN_FAR      = 25.0    # 扇形最远距离(m)
FAN_NEAR     = 3.0     # 扇形最近距离(m) — 别选脚下的格
FAN_ANGLE    = 60      # 扇形半角(度)
FAN_DSTEP    = 2.0     # 距离采样步长(m)
FAN_ASTEP    = 15      # 角度采样步长(度)

MAX_TARGET_DIST = 10.0    # 目标渐进距离(m) — 不放最远, 放中间
MIN_TARGET_WD   = 5       # wall_dist最小阈值(格) — 低于此的不选
WALL_SCAN_RADIUS = 10
ASTAR_MAX_EXPAND = 30000

MIN_SPEED = 1.5; SPEED_FACTOR = 0.5
BOUNCE_FORCE_DURATION = 0.3
STUCK_TIMEOUT = 300; STUCK_DIST_THRESH = 0.5
ARRIVE_THRESH = 1.0
WANDER_TIMEOUT = 600; WANDER_DRIFT_RATIO = 1.05
MAX_NO_PATH = 5

INIT_SCAN_STEPS = 200
LIDAR_TICK = 20; RENDER_SKIP = 20
FIXED_SEED = 42
MAX_MILESTONE_BALLS = 300; MAX_TARGET_BALLS = 10
FINISH = (7.0, 82.5)

# ═══════════════════════════════════════════
# SLAM字典地图 (与v3相同)
# ═══════════════════════════════════════════
UNKNOWN, FREE, WALL = 0, 1, 2
grid = {}
_wd = {}
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

# ═══════════════════════════════════════════
# 地图 + 障碍物
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
# 扫描 + 碰撞
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

# ═══════════════════════════════════════════
# wall_dist + A* (三级跳)
# ═══════════════════════════════════════════

JUMP_1M = 10; JUMP_03 = 3; JUMP_NEAR = 1

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

def jump_steps(vx, vy, dx, dy):
    wd = wall_dist(vx, vy)
    if wd >= JUMP_1M:   max_jump = JUMP_1M
    elif wd >= JUMP_03: max_jump = JUMP_03
    else:               max_jump = JUMP_NEAR
    for step in range(1, max_jump+1):
        nx, ny = vx + dx*step, vy + dy*step
        if not walkable(nx, ny):
            return step - 1
    return max_jump

def line_clear(vx1, vy1, vx2, vy2):
    steps = max(abs(vx2-vx1), abs(vy2-vy1))
    if steps == 0: return True
    for i in range(steps+1):
        if gget(int(vx1+(vx2-vx1)*i/steps), int(vy1+(vy2-vy1)*i/steps)) == WALL:
            return False
    return True

def astar_to(fvx, fvy, tfx, tfy):
    """跳步A*到点, 返回世界坐标路径"""
    if not (walkable(fvx, fvy) and walkable(tfx, tfy)):
        return None
    open_set = [(math.hypot(tfx-fvx, tfy-fvy), fvx, fvy)]
    came_from = {}; g_score = {(fvx, fvy): 0}
    visited_set = set()
    while open_set and len(came_from) < ASTAR_MAX_EXPAND:
        _, cx, cy = heapq.heappop(open_set)
        if (cx,cy) in visited_set: continue
        visited_set.add((cx,cy))
        if (cx,cy) == (tfx,tfy): break
        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
            js = jump_steps(cx, cy, dx, dy)
            if js < 1: continue
            nx, ny = cx + dx*js, cy + dy*js
            ng = g_score.get((cx,cy), 999) + js
            if (nx,ny) not in g_score or ng < g_score[(nx,ny)]:
                g_score[(nx,ny)] = ng
                came_from[(nx,ny)] = (cx,cy)
                heapq.heappush(open_set, (ng+math.hypot(tfx-nx, tfy-ny), nx, ny))
    if (tfx,tfy) not in came_from and (tfx,tfy) != (fvx,fvy): return None
    path = []; cur = (tfx, tfy)
    while cur != (fvx, fvy):
        path.append(cur)
        if cur not in came_from: break
        cur = came_from[cur]
    path.reverse()
    return [((px+0.5)*VOXEL, (py+0.5)*VOXEL) for px, py in path]

# ═══════════════════════════════════════════
# ★ v4 核心: 最大体素优先规划器 ★
# ═══════════════════════════════════════════

def pick_max_voxel_target(bx, by, heading):
    """在机器人前方扇形区域采样，选wall_dist最大的FREE格作为目标。
    
    原理:
      wall_dist = 该格到最近墙的格数
      道路中间 → wall_dist大 (宽阔, "大体素")
      贴墙/窄路 → wall_dist小
      机器人优先走wall_dist最大的地方 = 自然在道路中间
    
    渐进放置:
      不选扇形最远端的格, 选中间距离(MAX_TARGET_DIST以内)
      下一步走到那里后再算 → 轨迹自然前推
      障碍物出现 → wall_dist地图更新 → 宽处偏移 → 轨迹自动扭过去
    """
    best_vx = best_vy = None
    best_wd = MIN_TARGET_WD - 1
    best_dist = 999

    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    for dist_m in np.arange(FAN_NEAR, FAN_FAR + VOXEL, FAN_DSTEP):
        if dist_m > MAX_TARGET_DIST and best_vx is not None:
            break  # 已经找到候选, 不再往更远看 (渐进放置)

        for ang_deg in np.arange(-FAN_ANGLE, FAN_ANGLE + 1, FAN_ASTEP):
            ang = heading + math.radians(ang_deg)
            wx = bx + math.cos(ang) * dist_m
            wy = by + math.sin(ang) * dist_m
            vx, vy = int(wx/VOXEL), int(wy/VOXEL)

            if not walkable(vx, vy):
                continue

            wd = wall_dist(vx, vy)
            if wd < MIN_TARGET_WD:
                continue

            # 选wall_dist最大的; 平局时选更远的
            if wd > best_wd or (wd == best_wd and dist_m > best_dist):
                best_wd = wd
                best_dist = dist_m
                best_vx, best_vy = vx, vy

    if best_vx is None:
        return None

    return best_vx, best_vy, best_wd

# ═══════════════════════════════════════════
# 可视化
# ═══════════════════════════════════════════

class BallManager:
    def __init__(self, m, d):
        self.m = m; self.d = d
        self.mstone_bodies = []; self.target_bodies = []
        self.mstone_count = 0; self.target_count = 0
    def add_milestone(self, wx, wy):
        i = self.mstone_count
        if i < MAX_MILESTONE_BALLS:
            body_name = f"mstone_{i}"
            if body_name in self.mstone_bodies:
                self.d.mocap_pos[self.m.body(body_name).mocapid] = [wx, wy, 1.5]
            self.mstone_count += 1
    def add_target(self, wx, wy):
        i = self.target_count
        if i < MAX_TARGET_BALLS:
            body_name = f"target_{i}"
            if body_name in self.target_bodies:
                self.d.mocap_pos[self.m.body(body_name).mocapid] = [wx, wy, 2.0]
            self.target_count += 1
    def clear_targets(self):
        for name in self.target_bodies:
            self.d.mocap_pos[self.m.body(name).mocapid] = [0, 0, -10]
        self.target_count = 0

def build_xml():
    ms_xml = "".join(
        f'<body name="mstone_{i}" mocap="true" pos="0 0 -10">'
        f'<geom type="sphere" size="0.2" rgba="0.3 0.6 1.0 0.8"/></body>\n'
        for i in range(MAX_MILESTONE_BALLS))
    tg_xml = "".join(
        f'<body name="target_{i}" mocap="true" pos="0 0 -10">'
        f'<geom type="sphere" size="0.25" rgba="1.0 0.8 0.2 0.9"/></body>\n'
        for i in range(MAX_TARGET_BALLS))
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
    {ms_xml}{tg_xml}
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
                print(f"  [BOUNCE] #{self.bounce} @({self.d.qpos[0]:.1f},{self.d.qpos[1]:.1f})", flush=True)
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
    np.savez(SCAN_STATE, grid=arr, offset=(minx, miny), seed=FIXED_SEED)

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
    return loaded

# ═══════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════

print(f"━━━ 萤火 Firefly v4 最大体素优先 ━━━ {VOXEL}m lookahead={FAN_FAR}m max_gate={MAX_TARGET_DIST}m ━━━", flush=True)

existing = load_state()
if existing is not None:
    print(f"[LOAD] 加载扫图: {len(existing)} cells")
    # 直接写字典, 不用gset — 避免每个WALL都触发_wd.clear()
    for k, v in existing.items():
        if v != UNKNOWN:
            grid[k] = v
            _cnt[v] = _cnt.get(v, 0) + 1
    milestones = []
else:
    print(f"[NEW] 新扫图")
    milestones = []

xml = build_xml()
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
d.qpos[0]=6; d.qpos[1]=6; mujoco.mj_forward(m,d)

mv = Mover(m, d)
balls = BallManager(m, d)
for name in [f"mstone_{i}" for i in range(MAX_MILESTONE_BALLS)]:
    balls.mstone_bodies.append(name)
for name in [f"target_{i}" for i in range(MAX_TARGET_BALLS)]:
    balls.target_bodies.append(name)
for wx, wy in milestones:
    balls.add_milestone(wx, wy)

step = 0; t0 = time.time()
last_mx = last_my = 0
path = None; path_idx = 0
no_path_count = 0
wander = 0; last_dist = 999

if milestones:
    last_mx, last_my = milestones[-1]

print(f"=== Firefly v4 start: max voxel first, fan={FAN_FAR}m max_target={MAX_TARGET_DIST}m ===", flush=True)

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

        # ── ★ v4 主逻辑: 最大体素优先 ★ ──
        if path is None or path_idx >= len(path):
            result = pick_max_voxel_target(bx, by, mv.yaw)

            if result is not None:
                gvx, gvy, gwd = result
                new_path = astar_to(vx, vy, gvx, gvy)

                if new_path:
                    path = new_path; path_idx = 0
                    wander = 0; last_dist = 999; no_path_count = 0
                    gate_wx, gate_wy = (gvx+0.5)*VOXEL, (gvy+0.5)*VOXEL
                    balls.clear_targets()
                    balls.add_target(gate_wx, gate_wy)
                    if step % 100 == 0:
                        print(f"  [MAXWD] [{step}] →({gate_wx:.1f},{gate_wy:.1f}) wd={gwd} path={len(path)}", flush=True)
                else:
                    no_path_count += 1
            else:
                no_path_count += 1

            # A*失败或无候选 → 恢复
            if no_path_count > MAX_NO_PATH:
                if len(milestones) > 1:
                    for mx, my in reversed(milestones[-5:]):
                        bp = astar_to(vx, vy, mx, my)
                        if bp:
                            path = bp; path_idx = 0
                            wander = 0; last_dist = 999
                            print(f"  [BACK] [{step}] →路标({(mx+0.5)*VOXEL:.1f},{(my+0.5)*VOXEL:.1f})", flush=True)
                            no_path_count = 0
                            break
                    else:
                        mv._bounce(90, 180)
                else:
                    mv._bounce(90, 180)

        # 沿路径移动
        if path is not None and path_idx < len(path):
            tx, ty = path[path_idx]
            ddist = math.hypot(tx-bx, ty-by)
            if ddist < ARRIVE_THRESH:
                path_idx += 1
                last_dist = 999; wander = 0
            elif ddist > last_dist * WANDER_DRIFT_RATIO:
                wander += 1
                if wander > WANDER_TIMEOUT:
                    rescued = False
                    for mx, my in reversed(milestones[-5:]):
                        if line_clear(vx, vy, mx, my):
                            path = [((mx+0.5)*VOXEL, (my+0.5)*VOXEL)]
                            path_idx = 0; wander = 0; last_dist = 999
                            print(f"  [LOST] [{step}] →路标({(mx+0.5)*VOXEL:.1f},{(my+0.5)*VOXEL:.1f})", flush=True)
                            rescued = True
                            break
                    if not rescued:
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
        # ── 终点检测 ──
        if math.hypot(bx-FINISH[0], by-FINISH[1]) < 3.0:
            print(f"\n  ★ ARRIVED! @({bx:.1f},{by:.1f}) 终点({FINISH[0]:.1f},{FINISH[1]:.1f}) step={step} ms={len(milestones)}", flush=True)
            break

        if step % 2000 == 0:
            print(f"  ... step={step} F={_cnt[FREE]} W={_cnt[WALL]} ms={len(milestones)}", flush=True)
        if step % RENDER_SKIP == 0:
            v.sync()

    save_state()
    print(f"done: ms={len(milestones)} step={step} t={time.time()-t0:.1f}s bounce={mv.bounce}", flush=True)
