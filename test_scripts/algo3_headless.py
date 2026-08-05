#!/usr/bin/env python3
"""萤火 Firefly v3 SLAM — headless 闭环测试版

基于 algo3_firefly.py (v3-locked)，仅替换渲染层：
- viewer.launch_passive → EGL 离屏渲染（无 DISPLAY 可跑）
- 算法逻辑零改动：find_gates / astar_to / Mover / milestones 原样
- 输出成绩单 JSON + 渲染帧 PNG

用法：
  python3 algo3_headless.py --seed 42 --max-steps 200000 --render-every 200
"""

import sys, os, math, time, random, heapq, json, argparse
import numpy as np
from PIL import Image
import mujoco

# ═══════════════════════════════════════════
# 全部可配置参数（与 algo3_firefly.py 一致）
# ═══════════════════════════════════════════

PROJ = os.path.expanduser("~/workspace/simtrack")
MAP = os.path.join(PROJ, "confirmed/track_clean.png")
SCAN_DIR = os.path.join(PROJ, "scans")
SCAN_STATE = os.path.join(SCAN_DIR, "scan_dict.npz")
os.makedirs(SCAN_DIR, exist_ok=True)

SCALE = 1.0; HF_RES = 2000; PIX_PER_M = 40; ROAD_PIX = 128
SAFE_R = 0.2; SPEED = 4.0; SPEED_MAX = 4.0; YAW_RATE = 1.0
LIDAR_RANGE = 15.0

VOXEL = 0.1
ROBOT_R = max(1, int(SAFE_R / VOXEL))
CLEARANCE = ROBOT_R
MILESTONE_STEP = int(3.0 / VOXEL)
LIDAR_STEPS = int(LIDAR_RANGE / VOXEL)
LIDAR_RAYS = 120

MAX_GATES = 200
WALL_SCAN_RADIUS = 10
WALL_BUFFER_M = 2.0; WALL_BUFFER_CELLS = int(WALL_BUFFER_M / VOXEL)
WALL_PENALTY = 3
MAX_GATE_DIST = 3000
ASTAR_MAX_EXPAND = 30000

MIN_SPEED = 3.0; SPEED_FACTOR = 2.0
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

FIXED_SEED = random.randint(0, 999999)
MAX_MILESTONE_BALLS = 300; MAX_GATE_BALLS = 50
FINISH = (2.5, 47.5)
HIT_BACKOFF = 0.2
GAP_YELLOW_M = 1.0
DECIDE_RADIUS = 15.0
DECIDE_TICK = 200

# ═══════════════════════════════════════════
# 命令行参数
# ═══════════════════════════════════════════
ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=None, help="随机种子（默认随机）")
ap.add_argument("--max-steps", type=int, default=300000, help="最大步数上限")
ap.add_argument("--render-every", type=int, default=200, help="离屏渲染间隔（步）")
ap.add_argument("--out-dir", type=str, default="/tmp/firefly_frames", help="渲染帧输出目录")
ap.add_argument("--timeout", type=float, default=900, help="墙钟超时（秒）")
ap.add_argument("--save-name", type=str, default="", help="成绩单文件名（默认 auto）")
args = ap.parse_args()

if args.seed is not None:
    FIXED_SEED = args.seed

# ═══════════════════════════════════════════
# SLAM字典地图
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

def sample_hf(wx, wy):
    mx, my = wx/SCALE, wy/SCALE
    px, py = int(mx*PIX_PER_M), HF_RES-1-int(my*PIX_PER_M)
    return int(hf[py,px]) if 0<=px<HF_RES and 0<=py<HF_RES else -1

def gen_obstacles(seed):
    rng = random.Random(seed)
    cl = gen_centerline()
    obs_world = []; idx = 0
    while idx < len(cl):
        cx, cy = cl[idx]; wx, wy = cx*SCALE, cy*SCALE
        ox, oy = wx, wy + rng.uniform(-1.5, 1.5)
        # 障碍必须完全在道路内（不嵌墙）：中心在道路上 + 半径范围内无墙
        if sample_hf(ox, oy) == ROAD_PIX and not _obs_hits_wall(ox, oy, 0.5):
            obs_world.append((ox, oy))
        # 密度降为原来的 1/4（间距 4 倍）
        idx += rng.randint(12, 32)
    return [(x,y) for x,y in obs_world if math.hypot(x-6,y-6)>5.0]

def _obs_hits_wall(ox, oy, r):
    """检查以 (ox,oy) 为中心 r 半径的圆是否碰到墙（确保障碍不嵌墙）"""
    steps = 12
    for i in range(steps):
        a = 2 * math.pi * i / steps
        wx, wy = ox + r * math.cos(a), oy + r * math.sin(a)
        if sample_hf(wx, wy) != ROAD_PIX:
            return True
    return False

obs_world = gen_obstacles(FIXED_SEED)
OBS_R = 0.5; OBS_CLEAR = OBS_R + SAFE_R

def is_obstacle_world(wx, wy):
    if sample_hf(wx, wy) != ROAD_PIX: return True
    for ox, oy in obs_world:
        if math.hypot(wx-ox, wy-oy) < OBS_CLEAR: return True
    return False

# ═══════════════════════════════════════════
# 扫描 + 碰撞检测
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
    # 越界保护：赛道外直接视为 blocked（防穿墙跑出地图）
    if not (0.0 <= wx <= 50.0 and 0.0 <= wy <= 50.0):
        return True
    vx, vy = int(wx/VOXEL), int(wy/VOXEL)
    for dy in range(-ROBOT_R, ROBOT_R+1):
        for dx in range(-ROBOT_R, ROBOT_R+1):
            if dx*dx+dy*dy <= ROBOT_R*ROBOT_R:
                nx, ny = vx+dx, vy+dy
                if is_obstacle_world((nx+0.5)*VOXEL, (ny+0.5)*VOXEL):
                    return True
    return False

# ── 三级跳A* ──

JUMP_1M = 10
JUMP_03 = 3
JUMP_NEAR = 1

def jump_steps(vx, vy, dx, dy):
    wd = wall_dist(vx, vy)
    if wd >= JUMP_1M:   max_jump = JUMP_1M
    elif wd >= JUMP_03: max_jump = JUMP_03
    else:               max_jump = JUMP_NEAR
    for step in range(1, max_jump + 1):
        nx, ny = vx + dx*step, vy + dy*step
        if not walkable(nx, ny):
            return step - 1
    return max_jump

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

# ═══════════════════════════════════════════
# 跳步门查找
# ═══════════════════════════════════════════

def find_gates(fvx, fvy):
    if not walkable(fvx, fvy):
        return [], {}
    open_set = [(0, fvx, fvy)]
    came_from = {}; g_score = {(fvx, fvy): 0}
    visited = set()
    gates = []
    while open_set and len(came_from) < ASTAR_MAX_EXPAND and len(gates) < MAX_GATES:
        _, cx, cy = heapq.heappop(open_set)
        if (cx,cy) in visited: continue
        visited.add((cx,cy))
        cg = g_score.get((cx,cy), 9999)
        if gates and cg > MAX_GATE_DIST:
            break
        if gget(cx, cy) == FREE:
            has_unk = any(gget(cx+dx, cy+dy) == UNKNOWN
                          for dy in (-1,0,1) for dx in (-1,0,1))
            if has_unk and wall_dist(cx, cy) > CLEARANCE:
                gates.append((cg, cx, cy))
        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
            js = jump_steps(cx, cy, dx, dy)
            if js < 1: continue
            nx, ny = cx + dx*js, cy + dy*js
            wd = wall_dist(nx, ny)
            penalty = max(0, WALL_BUFFER_CELLS - wd) * WALL_PENALTY
            ng = cg + js + penalty
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

def fine_path(sx, sy, gx, gy, came_from, to_world=True):
    path = []; cur = (gx, gy)
    while cur != (sx, sy):
        path.append(cur)
        if cur not in came_from: break
        cur = came_from[cur]
    path.reverse()
    if to_world:
        return [((px+0.5)*VOXEL, (py+0.5)*VOXEL) for px, py in path]
    return path

def astar_to(fvx, fvy, tfx, tfy):
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
    return fine_path(fvx, fvy, tfx, tfy, came_from)

# ═══════════════════════════════════════════
# MuJoCo 模型（headless 版：EGL 离屏）
# ═══════════════════════════════════════════

def build_xml():
    OBS_XML = "".join(
        f'<body name="obs{i}" pos="{x:.1f} {y:.1f} 2.0">'
        f'<geom type="cylinder" size="0.5 1.0" rgba="0.9 0.2 0.2 0.9"/></body>'
        for i,(x,y) in enumerate(obs_world))
    FINISH_XML = f'<body mocap="true" pos="{FINISH[0]:.1f} {FINISH[1]:.1f} 2"><geom type="sphere" size="1.5" rgba="0.2 1.0 0.2 0.8"/></body>'
    return f"""<mujoco>
  <compiler angle="radian"/><option timestep="0.005"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset><hfield name="track" size="25.0 25.0 4.0 2.0" file="{MAP}"/></asset>
  <worldbody>
    <light pos="25 25 80" dir="0 0 -1"/>
    {FINISH_XML}{OBS_XML}
    <geom type="hfield" hfield="track" pos="25 25 0.0" rgba="0.25 0.30 0.35 1.0" friction="0 0 0"/>
    <body name="bot" pos="0 0 0.5">
      <joint type="slide" axis="1 0 0" damping="0"/>
      <joint type="slide" axis="0 1 0" damping="0"/>
      <joint name="yaw" type="hinge" axis="0 0 1" damping="0"/>
      <!-- 机器狗：水平圆柱（长轴沿 yaw 方向），0.8m 长 × 0.4m 径，保留物理碰撞防穿墙 -->
      <geom type="capsule" fromto="0 -0.4 0 0 0.4 0" size="0.2" rgba="1 0.3 0 1" friction="0 0 0"/>
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
        # 同步 yaw joint（qpos[2]），让狗身体实际转向（沿轴向前进，不横着走）
        self.d.qpos[2] = self.yaw
        vx = math.cos(self.yaw)*self.speed; vy = math.sin(self.yaw)*self.speed
        nx, ny = bx+vx*dt, by+vy*dt
        if step-self.stuck_t > STUCK_TIMEOUT:
            if math.hypot(bx-self.stuck_x, by-self.stuck_y) < STUCK_DIST_THRESH:
                self._bounce(90, 180)
            self.stuck_t = step; self.stuck_x = bx; self.stuck_y = by
        if self.force > 0:
            self.force -= 1; self.d.qvel[0] = vx; self.d.qvel[1] = vy
            if self.force <= 0:
                self.escaping = False  # force 耗尽复位
        elif blocked(nx, ny):
            # 诊断：区分被墙挡还是被障碍物挡
            _bx, _by = self.d.qpos[0], self.d.qpos[1]
            _hit_wall = sample_hf(nx, ny) != 128
            _near_obs = any(math.hypot(nx-ox, ny-oy) < OBS_CLEAR for ox, oy in obs_world)
            if self.bounce % 5 == 0:
                print(f"  [BOUNCE] bounce#{self.bounce} @({_bx:.1f},{_by:.1f})→({nx:.1f},{ny:.1f}) wall={_hit_wall} obs={_near_obs}", flush=True)
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
        # 随机转向，但检查新方向是否可通行（避免 bounce 后直冲进墙/越界）
        bx, by = self.d.qpos[0], self.d.qpos[1]
        dt = self.m.opt.timestep
        lookahead = max(1.0, self.speed * dt * 8)
        for _attempt in range(8):
            deg = random.uniform(lo, hi) * random.choice([-1, 1])
            cand = self.yaw + math.radians(deg)
            cx = bx + math.cos(cand) * lookahead
            cy = by + math.sin(cand) * lookahead
            if not blocked(cx, cy):
                self.yaw = cand
                self.d.qvel[:] = 0
                self.force = int(BOUNCE_FORCE_DURATION/(SPEED*self.m.opt.timestep))
                return
        # 全方向堵 → 180° 掉头（防死角死循环）
        self.yaw += math.pi
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
# 指标统计
# ═══════════════════════════════════════════
stats = {
    "gates_selected": 0,
    "backtracks": 0,
    "lost_rescues": 0,
    "bounces": 0,
    "milestones": 0,
    "steps": 0,
    "arrived": False,
    "time_sec": 0.0,
    "final_coverage": 0.0,
    "final_pos": None,
}

def coverage_pct():
    """探索覆盖率：FREE+WALL 占总可通行格（用地图真值，世界 0-100m）"""
    road_total = 0
    for wy in range(0, 1000):  # 100m / 0.1m（世界坐标 SCALE=2.0）
        for wx in range(0, 1000):
            if not is_obstacle_world((wx+0.5)*VOXEL, (wy+0.5)*VOXEL):
                road_total += 1
    explored = _cnt[FREE] + _cnt[WALL]
    return explored / road_total * 100 if road_total else 0

# ═══════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════

print(f"━━━ 萤火 Firefly v3 SLAM headless ━━━ {VOXEL}m 三级跳A* 模式={EXPLORE_MODE} seed={FIXED_SEED} ━━━", flush=True)

xml = build_xml()
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
d.qpos[0]=2.5; d.qpos[1]=2.5; mujoco.mj_forward(m,d)

# EGL 离屏渲染
os.makedirs(args.out_dir, exist_ok=True)
try:
    from mujoco import egl
    _ctx = egl.GLContext(1280, 720)
    _ctx.make_current()
    renderer = mujoco.Renderer(m, 720, 1280)
    RENDER_OK = True
    print("  [RENDER] EGL 离屏渲染 OK", flush=True)
except Exception as e:
    renderer = None
    RENDER_OK = False
    print(f"  [RENDER] 离屏渲染不可用: {e}", flush=True)

def render_frame(step):
    """离屏渲染当前帧保存 PNG"""
    if renderer is None:
        return
    try:
        renderer.update_scene(d, camera=-1)
        img = renderer.render()
        Image.fromarray(img).save(os.path.join(args.out_dir, f"frame_{step:06d}.png"))
    except Exception as e:
        pass  # 渲染失败不阻塞主循环

mv = Mover(m, d)

step = 0; t0 = time.time()
last_mx = last_my = 0
path = None; path_idx = 0
no_gate_count = 0
wander = 0; last_dist = 999
milestones = []
start_pos = (d.qpos[0], d.qpos[1])
milestones.append((int(start_pos[0]/VOXEL), int(start_pos[1]/VOXEL)))
last_mx, last_my = milestones[0]

print(f"=== Firefly v3 headless start: seed={FIXED_SEED} max_steps={args.max_steps} ===", flush=True)

# 初始扫描
for _ in range(INIT_SCAN_STEPS):
    bx, by = d.qpos[0], d.qpos[1]
    if _ % LIDAR_TICK == 0: scan(bx, by)
    mujoco.mj_step(m, d)
print(f"  [OK] FREE={_cnt[FREE]} WALL={_cnt[WALL]}", flush=True)

frame_idx = 0
while step < args.max_steps and time.time() - t0 < args.timeout:
    bx, by = d.qpos[0], d.qpos[1]
    vx, vy = int(bx/VOXEL), int(by/VOXEL)
    if gget(vx, vy) == UNKNOWN:
        gset(vx, vy, FREE)

    # 路标放置
    if abs(vx-last_mx)+abs(vy-last_my) >= MILESTONE_STEP:
        if wall_dist(vx, vy) > CLEARANCE:
            milestones.append((vx, vy))
            last_mx, last_my = vx, vy
            save_state()

    if step % LIDAR_TICK == 0:
        scan(bx, by)

    # 决策
    if path is None or path_idx >= len(path):
        gates, came_from = find_gates(vx, vy)
        gate = pick_gate(gates, EXPLORE_MODE, stuck=(no_gate_count > 0))
        if gate is not None:
            cg, gx, gy = gate
            path = fine_path(vx, vy, gx, gy, came_from)
            path_idx = 0; wander = 0; last_dist = 999
            no_gate_count = 0
            stats["gates_selected"] += 1
            print(f"  [GATE] [{step}] →({(gx+0.5)*VOXEL:.1f},{(gy+0.5)*VOXEL:.1f}) path={len(path)} gates={len(gates)}", flush=True)
        else:
            no_gate_count += 1
            if no_gate_count > MAX_NO_GATE and len(milestones) > 1:
                saved = False
                for i in range(len(milestones)-2, -1, -1):
                    mx, my = milestones[i]
                    bp = astar_to(vx, vy, mx, my)
                    if bp:
                        path = bp; path_idx = 0; wander = 0; last_dist = 999
                        no_gate_count = 0
                        stats["backtracks"] += 1
                        print(f"  [BACK] [{step}] →路标#{i}", flush=True)
                        saved = True
                        break
                if not saved:
                    if no_gate_count < 10: mv._bounce(90, 180)
                    else: mv._bounce(150, 210)

    # 执行
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
                for mx, my in reversed(milestones[-RESCUE_MS_COUNT:]):
                    if line_clear(vx, vy, mx, my):
                        path = [((mx+0.5)*VOXEL, (my+0.5)*VOXEL)]
                        path_idx = 0; wander = 0; last_dist = 999
                        stats["lost_rescues"] += 1
                        print(f"  [LOST] [{step}] →路标", flush=True)
                        rescued = True
                        break
                if not rescued:
                    path = None; path_idx = 0; wander = 0; last_dist = 999
                    stats["lost_rescues"] += 1
                    print(f"  [LOST] [{step}] 重新规划", flush=True)
            else:
                last_dist = ddist; mv.step(tx, ty, step)
        else:
            wander = max(0, wander-1); last_dist = ddist
            mv.step(tx, ty, step)
    else:
        mv._bounce(90, 180)

    step += 1

    # 终点检测
    if math.hypot(bx-FINISH[0], by-FINISH[1]) < 3.0:
        print(f"\n  ★ ARRIVED! @({bx:.1f},{by:.1f}) step={step} ms={len(milestones)}", flush=True)
        stats["arrived"] = True
        break

    # 离屏渲染
    if RENDER_OK and args.render_every > 0 and step % args.render_every == 0:
        render_frame(step)

    # 进度日志
    if step % 20000 == 0:
        cov = coverage_pct()
        print(f"  ... step={step} F={_cnt[FREE]} W={_cnt[WALL]} ms={len(milestones)} cov={cov:.1f}% t={time.time()-t0:.0f}s", flush=True)

# ── 收尾统计 ──
stats["steps"] = step
stats["time_sec"] = round(time.time() - t0, 2)
stats["bounces"] = mv.bounce
stats["milestones"] = len(milestones)
stats["final_pos"] = [round(d.qpos[0], 2), round(d.qpos[1], 2)]
stats["final_coverage"] = round(coverage_pct(), 2)
stats["seed"] = FIXED_SEED
stats["free_cells"] = _cnt[FREE]
stats["wall_cells"] = _cnt[WALL]
stats["mode"] = EXPLORE_MODE

if RENDER_OK:
    render_frame(step)  # 最后帧

save_state()

# 成绩单
if args.save_name:
    out_json = os.path.join(SCAN_DIR, args.save_name)
else:
    out_json = os.path.join(SCAN_DIR, f"baseline_seed{FIXED_SEED}.json")
with open(out_json, "w") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print("\n=== 成绩单 ===")
for k, v in stats.items():
    print(f"  {k}: {v}")
print(f"\n[SAVE] {out_json}")
print(f"done: ms={len(milestones)} step={step} t={time.time()-t0:.1f}s bounce={mv.bounce} mode={EXPLORE_MODE}", flush=True)
