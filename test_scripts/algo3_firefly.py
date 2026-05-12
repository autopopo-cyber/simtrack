#!/usr/bin/env python3
"""萤火算法 Firefly v3 — 0.1m精度 障碍物boolean表

坐标系统: 0.1m为单位, 50m×50m → 500×500
数据结构:
  obstacles[500,500] bool  — 激光扫到的障碍物 (True=障碍)
  explored[500,500] bool   — 激光扫过的区域
  路标链 []                — 运行时3m间隔路标
  门/路径/前线             — 概念不变

规则:
  1. 激光命中障碍物 → 标记该格+朝机器人方向1格=障碍 (永远0.1m间隙)
  2. A*展开邻居→跳过wall_distance≤5格(0.5m=半径)
  3. 路标→必须离最近障碍>5格
  4. 探索模式: near(扫全图每个角落) / far(先铺更深更广的前线)

v2 → v3 核心变化:
  1m体素 → 0.1m布尔表 (250K cells, 31KB压缩)
  UNKNOWN/FREE/WALL/VISITED → obstacles+explored双表
  地面真值预计算500×500, 扫描和碰撞O(1)查表
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

# ── 物理参数 ──
SCALE = 2.0; HF_RES = 2000; PIX_PER_M = 40; ROAD_PIX = 128
SAFE_R = 0.5; SPEED = 5.0; SPEED_MAX = 8.0; YAW_RATE = 6.0
LIDAR_RANGE = 15.0

# ── 0.1m精度 ──
VOXEL = 0.1; W3 = 500                     # 50m/0.1m
ROBOT_R = 5                                # 0.5m = 5格
CLEARANCE = 5                              # 路标离墙最少5格(0.5m)
MILESTONE_STEP = 30                        # 3m = 30格
LIDAR_STEPS = int(LIDAR_RANGE / 0.1)      # 150步
LIDAR_RAYS = 120

# ── 探索模式: near(就近) / far(就远) ──
EXPLORE_MODE = "near"  # 改这里切换行为

FIXED_SEED = 42

# ── 数据结构 ──
obstacles = np.zeros((W3, W3), dtype=bool)    # 激光扫到的障碍
explored = np.zeros((W3, W3), dtype=bool)     # 激光扫过的区域
gt = np.zeros((W3, W3), dtype=bool)           # 地面真值(预计算)

FINISH = (7.0, 82.5)  # 终点(世界坐标) — 写完才知在哪

# ═══════════════════════════════════════════
# 障碍物生成
# ═══════════════════════════════════════════

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
OBS_R = 1.0; OBS_CLEAR = OBS_R + SAFE_R

# ── 地图采样 ──
def sample_hf(wx, wy):
    mx, my = wx/SCALE, wy/SCALE
    px, py = int(mx*PIX_PER_M), HF_RES-1-int(my*PIX_PER_M)
    return int(hf[py,px]) if 0<=px<HF_RES and 0<=py<HF_RES else -1

# ── 预计算地面真值 ──
print("[GT] 预计算 500x500 地面真值...", flush=True)
for vy in range(W3):
    for vx in range(W3):
        wx, wy = (vx+0.5)*VOXEL, (vy+0.5)*VOXEL
        if sample_hf(wx, wy) != ROAD_PIX:
            gt[vy, vx] = True; continue
        for ox, oy in obs_world:
            if math.hypot(wx-ox, wy-oy) < OBS_CLEAR:
                gt[vy, vx] = True; break
print(f"  [OK] 障碍格 {int(np.sum(gt))} / {W3*W3}", flush=True)

# ═══════════════════════════════════════════
# 扫描 (LIDAR 10Hz)
# ═══════════════════════════════════════════

def scan(bx, by):
    """120射线, 步长0.1m, 最大15m。命中障碍→标记该格+朝机器人方向1格"""
    for a in np.linspace(0, 2*math.pi, LIDAR_RAYS):
        cos_a, sin_a = math.cos(a), math.sin(a)
        prev_vx, prev_vy = int(bx/VOXEL), int(by/VOXEL)
        for step_i in range(1, LIDAR_STEPS+1):
            d = step_i * 0.1
            wx, wy = bx + cos_a*d, by + sin_a*d
            vx, vy = int(wx/VOXEL), int(wy/VOXEL)
            if not (0 <= vx < W3 and 0 <= vy < W3):
                break
            # 标记已探索
            explored[vy, vx] = True
            # 命中障碍物?
            if gt[vy, vx]:
                obstacles[vy, vx] = True          # 命中点
                # 膨胀: 朝机器人方向回退1格也标为障碍
                if 0 <= prev_vx < W3 and 0 <= prev_vy < W3:
                    obstacles[prev_vy, prev_vx] = True
                break
            prev_vx, prev_vy = vx, vy

# ── 碰撞检测 (200Hz) ──
def blocked(wx, wy):
    """机器人半径0.5m=5格, 检测圈内是否有障碍物(用地面真值)"""
    vx, vy = int(wx/VOXEL), int(wy/VOXEL)
    for dy in range(-ROBOT_R, ROBOT_R+1):
        for dx in range(-ROBOT_R, ROBOT_R+1):
            if dx*dx + dy*dy <= ROBOT_R*ROBOT_R:
                nx, ny = vx+dx, vy+dy
                if 0 <= nx < W3 and 0 <= ny < W3 and gt[ny, nx]:
                    return True
    return False

# ── A* 辅助 ──
def wall_dist(vx, vy):
    """在obstacles表中找最近障碍的距离(格数)"""
    best = 999
    for dy in range(-6, 7):
        for dx in range(-6, 7):
            nx, ny = vx+dx, vy+dy
            if 0 <= nx < W3 and 0 <= ny < W3 and obstacles[ny, nx]:
                d = abs(dx) + abs(dy)  # Manhattan, 简单够用
                if d < best: best = d
    return best

def walkable(vx, vy):
    """A*可达: 在界内+非障碍+离墙>机器人半径"""
    if not (0 <= vx < W3 and 0 <= vy < W3):
        return False
    if obstacles[vy, vx]:
        return False
    # 离墙至少ROBOT_R格
    return wall_dist(vx, vy) > ROBOT_R

def line_clear(vx1, vy1, vx2, vy2):
    """两点间直线是否无障碍。Bresenham采样, 每步查 obstacles"""
    steps = max(abs(vx2-vx1), abs(vy2-vy1))
    if steps == 0: return True
    for i in range(steps+1):
        x = int(vx1 + (vx2-vx1)*i/steps)
        y = int(vy1 + (vy2-vy1)*i/steps)
        if 0<=x<W3 and 0<=y<W3 and obstacles[y, x]:
            return False
    return True

# ═══════════════════════════════════════════
# 门查找 (最近UNKNOWN相邻的FREE格)
# ═══════════════════════════════════════════

def find_gates(sx, sy, max_gates=20):
    """A*找前N个门。返回 [(距离, x, y), ...] 按距离排序"""
    if not (0<=sx<W3 and 0<=sy<W3 and not obstacles[sy,sx]):
        return [], {}
    open_set = [(0, sx, sy)]
    came_from = {}; g_score = {(sx,sy): 0}
    visited = set()
    gates = []

    while open_set and len(came_from) < 50000 and len(gates) < max_gates:
        _, cx, cy = heapq.heappop(open_set)
        if (cx,cy) in visited: continue
        visited.add((cx,cy))
        cg = g_score.get((cx,cy), 9999)

        if explored[cy, cx] and not obstacles[cy, cx]:
            has_unk = any(
                not explored[cy+dy, cx+dx]
                for dy in (-1,0,1) for dx in (-1,0,1)
                if 0<=cx+dx<W3 and 0<=cy+dy<W3
            )
            if has_unk and wall_dist(cx, cy) > CLEARANCE:
                gates.append((cg, cx, cy))

        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
            nx, ny = cx+dx, cy+dy
            if not walkable(nx, ny): continue
            wd = wall_dist(nx, ny)
            penalty = max(0, 20-wd)*3
            ng = cg + 1 + penalty
            if (nx,ny) not in g_score or ng < g_score[(nx,ny)]:
                g_score[(nx,ny)] = ng
                came_from[(nx,ny)] = (cx,cy)
                heapq.heappush(open_set, (ng, nx, ny))

    return gates, came_from

def pick_gate(gates, mode="near", stuck=False):
    """从门列表选一个。卡住→最近 / far→最远 / near→中位"""
    if not gates: return None
    if stuck: return gates[0]           # 卡死→最近, 逃命
    if mode == "far": return gates[-1]  # 就远→最远, 铺前线
    return gates[len(gates)//2]         # 就近→中位, 不远不近均匀覆盖

def gate_path(sx, sy, gx, gy, came_from):
    """从came_from回溯出到指定门的路径"""
    path = []; cur = (gx, gy)
    while cur != (sx, sy):
        path.append(cur)
        if cur not in came_from: break
        cur = came_from[cur]
    path.reverse()
    return path

# ── A* 到点 ──
def astar_to(sx, sy, tx, ty):
    if not (0<=sx<W3 and 0<=sy<W3 and 0<=tx<W3 and 0<=ty<W3): return None
    if obstacles[sy,sx] or obstacles[ty,tx]: return None
    open_set = [(math.hypot(tx-sx, ty-sy), sx, sy)]
    came_from = {}; g_score = {(sx,sy): 0}
    visited_set = set()
    while open_set and len(came_from) < 20000:
        _, cx, cy = heapq.heappop(open_set)
        if (cx,cy) in visited_set: continue
        visited_set.add((cx,cy))
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

# ═══════════════════════════════════════════
# 可视化球管理
# ═══════════════════════════════════════════

MAX_MILESTONE_BALLS = 300
MAX_GATE_BALLS = 50

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
            mcap = self.m.body(name).mocapid
            self.d.mocap_pos[mcap] = [0, 0, -10]
        self.gate_count = 0

# ═══════════════════════════════════════════
# XML 场景构建
# ═══════════════════════════════════════════

def build_xml():
    ms_xml = ""
    for i in range(MAX_MILESTONE_BALLS):
        ms_xml += f'<body name="mstone_{i}" mocap="true" pos="0 0 -10"><geom type="sphere" size="0.2" rgba="0.3 0.6 1.0 0.8"/></body>\n'
    gt_xml = ""
    for i in range(MAX_GATE_BALLS):
        gt_xml += f'<body name="gate_{i}" mocap="true" pos="0 0 -10"><geom type="sphere" size="0.25" rgba="1.0 0.8 0.2 0.9"/></body>\n'

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

# ═══════════════════════════════════════════
# Mover (200Hz 运动控制)
# ═══════════════════════════════════════════

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

        # 卡住检测
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
# 文件读写
# ═══════════════════════════════════════════

def save_state():
    """保存 obstacles + explored 双表"""
    state = np.zeros((2, W3, W3), dtype=bool)
    state[0] = obstacles; state[1] = explored
    np.save(SCAN_VOX, state)
    meta = {"seed": FIXED_SEED, "mode": EXPLORE_MODE}
    with open(SCAN_META, 'w') as f:
        json.dump(meta, f)

def load_state():
    if not (os.path.exists(SCAN_VOX) and os.path.exists(SCAN_META)):
        return None
    with open(SCAN_META) as f:
        meta = json.load(f)
    if meta.get("seed") != FIXED_SEED:
        return None
    loaded = np.load(SCAN_VOX)
    if loaded.shape != (2, W3, W3):
        return None
    return loaded[0], loaded[1], meta.get("mode", "near")

# ═══════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════

print(f"━━━ 萤火算法 Firefly v3 ━━━ 0.1m精度 {W3}x{W3} 模式={EXPLORE_MODE} ━━━", flush=True)

# 1. 加载扫图
existing = load_state()
if existing is not None:
    loaded_obs, loaded_exp, loaded_mode = existing
    print(f"[LOAD] 加载扫图: seed={FIXED_SEED} mode={loaded_mode}")
    np.copyto(obstacles, loaded_obs)
    np.copyto(explored, loaded_exp)
    if loaded_mode != EXPLORE_MODE:
        print(f"  [!] 模式不匹配: 文件{loaded_mode} vs 当前{EXPLORE_MODE}")
    milestones = []
else:
    print(f"[NEW] 新扫图: seed={FIXED_SEED} mode={EXPLORE_MODE}")
    milestones = []

# 2. 构建场景
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

step=0; t0=time.time()
last_mx = last_my = 0
path = None; path_idx = 0
no_gate_count = 0
wander = 0; last_dist = 999
current_target_type = ""

if milestones:
    last_mx, last_my = milestones[-1]

print(f"=== Firefly v3 start: VOXEL={VOXEL}m W={W3} mode={EXPLORE_MODE} 路标{len(milestones)} ===", flush=True)

with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
    v.cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.distance=25; v.cam.elevation=-35; v.cam.azimuth=180
    LIDAR_TICK = 20

    # 初始扫描
    print("  [SCAN] initial scan...", flush=True)
    for _ in range(200):
        bx, by = d.qpos[0], d.qpos[1]
        if _ % LIDAR_TICK == 0:
            scan(bx, by)
        mujoco.mj_step(m, d)
    print(f"  [OK] obstacles={int(np.sum(obstacles))} explored={int(np.sum(explored))}", flush=True)

    # 第一个路标
    if not milestones:
        sx, sy = int(d.qpos[0]/VOXEL), int(d.qpos[1]/VOXEL)
        milestones.append((sx, sy))
        last_mx, last_my = sx, sy
        balls.add_milestone(d.qpos[0], d.qpos[1])

    # ── 主循环 ──
    while v.is_running():
        try:
            bx, by = d.qpos[0], d.qpos[1]
        except AttributeError:
            print("  [!] viewer线程崩溃, 尝试恢复...", flush=True)
            d = mujoco.MjData(m); mv.d = d
            bx, by = d.qpos[0], d.qpos[1]
        # 边界保护
        if bx<1 or bx>99 or by<1 or by>99:
            d.qpos[0]=max(1,min(99,bx)); d.qpos[1]=max(1,min(99,by))
            d.qvel[:]=0; mv.yaw=random.uniform(0,2*math.pi)
        v.cam.lookat[:]=np.array([bx, by, 0.5], dtype=np.float64)

        vx, vy = int(bx/VOXEL), int(by/VOXEL)
        if 0 <= vx < W3 and 0 <= vy < W3:
            explored[vy, vx] = True
            # 路标放置
            if abs(vx-last_mx)+abs(vy-last_my) >= MILESTONE_STEP:
                # 路标本身必须离墙足够远
                if wall_dist(vx, vy) > CLEARANCE:
                    milestones.append((vx, vy))
                    last_mx, last_my = vx, vy
                    balls.add_milestone(bx, by)
                    save_state()
                    if len(milestones) % 20 == 0:
                        print(f"  [WAYPOINT] milestones={len(milestones)} @({bx:.1f},{by:.1f})", flush=True)

        # LIDAR扫描
        if step % LIDAR_TICK == 0:
            scan(bx, by)

        # 路径规划/执行
        if path is None or path_idx >= len(path):
            gates, came_from = find_gates(vx, vy)
            gate = pick_gate(gates, EXPLORE_MODE, stuck=(no_gate_count > 0))

            if gate is not None:
                cg, gx, gy = gate
                gp = gate_path(vx, vy, gx, gy, came_from)
                no_gate_count = 0
                path = [((px+0.5)*VOXEL, (py+0.5)*VOXEL) for px, py in gp]
                path_idx = 0; wander = 0; last_dist = 999
                gate_wx, gate_wy = (gx+0.5)*VOXEL, (gy+0.5)*VOXEL
                current_target_type = "gate"
                balls.clear_gates()
                balls.add_gate(gate_wx, gate_wy)
                print(f"  [GATE] [{step}] →({gate_wx:.1f},{gate_wy:.1f}) len={len(path)} gates={len(gates)} stuck={no_gate_count>0} obs={int(np.sum(obstacles))} exp={int(np.sum(explored))}", flush=True)
            else:
                no_gate_count += 1
                if no_gate_count > 3 and len(milestones) > 1:
                    mx, my = milestones[-2]
                    back_path = astar_to(vx, vy, mx, my)
                    if back_path:
                        path = [((px+0.5)*VOXEL, (py+0.5)*VOXEL) for px, py in back_path]
                        path_idx = 0; wander = 0; last_dist = 999
                        current_target_type = "milestone"
                        balls.clear_gates()
                        mwx, mwy = (mx+0.5)*VOXEL, (my+0.5)*VOXEL
                        print(f"  [BACK] [{step}] →路标({mwx:.1f},{mwy:.1f}) len={len(path)}", flush=True)
                        no_gate_count = 0
                    else:
                        mx, my = milestones[0]
                        back_path = astar_to(vx, vy, mx, my)
                        if back_path:
                            path = [((px+0.5)*VOXEL, (py+0.5)*VOXEL) for px, py in back_path]
                            path_idx = 0; wander = 0; last_dist = 999
                            print(f"  [BACK] [{step}] →起点({(mx+0.5)*VOXEL:.1f},{(my+0.5)*VOXEL:.1f})", flush=True)
                            no_gate_count = 0
                        else:
                            mv._bounce(90, 180)
                else:
                    mv._bounce(90, 180)

        if path is not None and path_idx < len(path):
            tx, ty = path[path_idx]
            d = math.hypot(tx-bx, ty-by)
            if d < 1.0:
                path_idx += 1
                last_dist = 999; wander = 0
            elif d > last_dist * 1.05:
                wander += 1
                if wander > 600:
                    vx, vy = int(bx/VOXEL), int(by/VOXEL)
                    for mx, my in reversed(milestones[-5:]):
                        if line_clear(vx, vy, mx, my):
                            path = [((mx+0.5)*VOXEL, (my+0.5)*VOXEL)]
                            path_idx = 0; wander = 0; last_dist = 999
                            print(f"  [LOST] [{step}] →路标({(mx+0.5)*VOXEL:.1f},{(my+0.5)*VOXEL:.1f})", flush=True)
                            break
                    else:
                        path = None; path_idx = 0; wander = 0; last_dist = 999
                        print(f"  [LOST] [{step}] 重新规划", flush=True)
                else:
                    last_dist = d
                    mv.step(tx, ty, step)
            else:
                wander = max(0, wander - 1)
                last_dist = d
                mv.step(tx, ty, step)
        else:
            mv._bounce(90, 180)

        step += 1
        if step % 2000 == 0:
            print(f"  ... step={step} obs={int(np.sum(obstacles))} exp={int(np.sum(explored))} ms={len(milestones)}", flush=True)

    save_state()
    print(f"done: milestones={len(milestones)} step={step} time={time.time()-t0:.1f}s bounce={mv.bounce} mode={EXPLORE_MODE}", flush=True)
