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

# 地标标牌系统（30 个 ArUco+数字标牌）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from test_scripts.landmarks import landmark_xml, landmark_positions, BOT_Z, wall_xml

# ═══════════════════════════════════════════
# 全部可配置参数（与 algo3_firefly.py 一致）
# ═══════════════════════════════════════════

PROJ = os.path.expanduser("~/workspace/simtrack")
MAP = os.path.join(PROJ, "confirmed/track_clean.png")
# 渲染用降分辨率 hfield（2000→500）：EGL/OSMesa 软渲染下 2000x2000 高度图 2.7s/帧，
# 500x500 只要 223ms（快12倍）。sample_hf 碰撞检测仍用原图 MAP，精度不降。
RENDER_MAP = os.path.join(PROJ, "confirmed/track_500.png")
SCAN_DIR = os.path.join(PROJ, "scans")
SCAN_STATE = os.path.join(SCAN_DIR, "scan_dict.npz")
os.makedirs(SCAN_DIR, exist_ok=True)

SCALE = 1.0; HF_RES = 2000; PIX_PER_M = 40; ROAD_PIX = 128
SAFE_R = 0.2; SPEED = 4.0; SPEED_MAX = 4.0; YAW_RATE = 1.5
LIDAR_RANGE = 30.0  # 15→30m（主人指令 2026-08-06）：真实狗雷达10万点/s覆盖前半球，30m 探测更远

VOXEL = 0.1
ROBOT_R = max(1, int(SAFE_R / VOXEL))
CLEARANCE = ROBOT_R
MILESTONE_STEP = int(3.0 / VOXEL)
LIDAR_STEPS = int(LIDAR_RANGE / VOXEL)
LIDAR_RAYS = 360  # 120→360（1°间隔）：15m处射线间距0.26m，覆盖0.1m薄斜墙（120时0.78m漏扫）

MAX_GATES = 200
WALL_SCAN_RADIUS = 20  # 10→20：配合 JUMP_1M=20，wall_dist 能分辨 2m 空旷
WALL_BUFFER_M = 2.0; WALL_BUFFER_CELLS = int(WALL_BUFFER_M / VOXEL)
WALL_PENALTY = 3
UNKNOWN_PENALTY = 8  # 未知格可通行但代价高（探索规划，优先已知路）
VORONOI_C = 2.0      # 走中间代价系数：penalty = C/d² (d=离墙格数)，KNOWN_MAP_MODE 下启用
MAX_GATE_DIST = 300  # 门搜索距离上限(格=30m)——蛇形通道门常在20-30m外，30m覆盖整通道；避免全图(500格)搜索
ASTAR_MAX_EXPAND = 250000  # 30000→250000：蛇形迷宫(476m) A* 从起点到终点需扩展远超3万格（BFS全图23.5万格）

MIN_SPEED = 1.0; SPEED_FACTOR = 1.5
# 运动学约束（主人：现实中不允许碰撞）
# 限速/限加速度/限减速度 + 前瞻测距 + 制动约束 v≤sqrt(2·A_DECEL·d)，物理上保证碰撞=0
A_ACCEL = 5.0      # 加速度 (m/s²)：速度爬升上限
A_DECEL = 8.0      # 减速度 (m/s²)：制动能力，任何速度都能在障碍前停住
STOP_MARGIN = 0.4  # 停车时距障碍的安全余量 (m)
LOOKAHEAD = 4.0    # 前瞻测距上限 (m)
STUCK_TIMEOUT = 300; STUCK_DIST_THRESH = 0.5

EXPLORE_MODE = "score"
MIX_THRESHOLD = 50
INIT_SCAN_STEPS = 200
LIDAR_TICK = 10; RENDER_SKIP = 20  # LIDAR_TICK 20→10：scan更频繁，减少两次scan间盲区（斜墙漏检）
ARRIVE_THRESH = 1.0
PATH_LOOKAHEAD = 3.0  # pure pursuit 前瞻：目标=路径上直线距离≥此值的点（直道满速，接近自动减速）
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
ap.add_argument("--save-map", type=str, default="", help="跑完保存地图到文件 (npz)")
ap.add_argument("--load-map", type=str, default="", help="加载旧地图为 static_grid (npz)")
ap.add_argument("--known-raw", type=int, default=1, help="KNOWN_MAP 用原始 track_clean 地图（sample_hf 判定，与物理一致）")
ap.add_argument("--vision", type=int, default=1, help="视觉识别（hfield降分辨率后223ms/帧，默认开）")
ap.add_argument("--landmarks", type=int, default=1, help="标牌几何（contype=0零物理开销，默认开）")
ap.add_argument("--obs-reseed", type=int, default=0, help="运行中障碍变化步数：到该步换新障碍 seed（B阶段）")
ap.add_argument("--no-obs", type=int, default=0, help="1=纯墙版（去掉所有障碍，只留 hfield 墙）")
ap.add_argument("--lidar-rays", type=int, default=0, help="雷达射线数（0=用代码默认360）。真实狗~1万点/帧前半球")
ap.add_argument("--lidar-tick", type=int, default=0, help="雷达扫描间隔步数（0=用代码默认10）。真实狗10Hz")
ap.add_argument("--trail-every", type=int, default=0, help="轨迹记录间隔（步，0=关）。存 scans/trail_*.npz")
ap.add_argument("--random-start", type=int, default=0, help="1=从随机位置(避开墙/障碍)出发")
ap.add_argument("--target", type=str, default="finish", help="目标: start|finish")
args = ap.parse_args()

if args.seed is not None:
    FIXED_SEED = args.seed

# 雷达参数可调（实验：点云密度/频率是不是瓶颈）
if args.lidar_rays > 0:
    LIDAR_RAYS = args.lidar_rays
if args.lidar_tick > 0:
    LIDAR_TICK = args.lidar_tick

# 纯墙版：去掉所有障碍（只留 hfield 墙）
if args.no_obs:
    obs_world = []
    print("  [CFG] 纯墙版：障碍已清空（--no-obs）", flush=True)

# 目标动态设置（--target start|finish|ch<N>|任意坐标）
if args.target == "start":
    FINISH = (2.5, 2.5)
elif args.target == "finish":
    FINISH = (2.5, 47.5)
elif args.target.startswith("ch"):
    # 任意通道二维码：每通道 1 个标牌（landmark_positions() 已含全部通道位置）
    ch = int(args.target[2:])
    _lps = landmark_positions()
    if 0 <= ch < len(_lps):
        idx, cnum, side, lx, ly, wz, quat = _lps[ch]
        FINISH = (float(lx), float(ly))
        print(f"  [ARG] 目标=通道{ch} 二维码 ({lx:.1f},{ly:.1f})", flush=True)
    else:
        print(f"  [ARG] 通道{ch} 超出范围(0~{len(_lps)-1})，用 finish", flush=True)
        FINISH = (2.5, 47.5)
elif args.target.startswith("("):
    import ast
    FINISH = ast.literal_eval(args.target)
    print(f"  [ARG] 目标=自定义坐标 {FINISH}", flush=True)
else:
    print(f"  [ARG] 未知目标: {args.target}，用 finish", flush=True)
    FINISH = (2.5, 47.5)

# ═══════════════════════════════════════════
# SLAM字典地图
# ═══════════════════════════════════════════
UNKNOWN, FREE, WALL = 0, 1, 2
grid = {}
static_grid = {}   # 旧地图背景（只读）：WALL=永久的墙；加载地图时填充
KNOWN_MAP_MODE = False  # True=阶段2（加载旧地图），规划叠加 static_grid
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

def gget_plan(vx, vy):
    """规划用叠加视图：static 的墙永远 WALL；live 障碍/自由优先；其余回退 static。
    探索模式（KNOWN_MAP_MODE=False）下 static 空 → 等价于 gget。"""
    s = static_grid.get((vx, vy), UNKNOWN)
    if s == WALL:
        return WALL
    l = grid.get((vx, vy), UNKNOWN)
    if l != UNKNOWN:
        return l
    return s

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

def random_road_pos(seed, min_dist_from=5.0, from_pos=(2.5, 2.5)):
    """随机采样道路位置（不在墙里、不在墙边、不在障碍里、离给定点够远）。绑架测试用。"""
    rng = random.Random(seed)
    for _ in range(8000):
        wx, wy = rng.uniform(1.0, 49.0), rng.uniform(1.0, 49.0)
        if sample_hf(wx, wy) != ROAD_PIX:
            continue                       # 不在墙里
        if _obs_hits_wall(wx, wy, 0.7):
            continue                       # 不在墙边
        if any(math.hypot(wx-ox, wy-oy) < OBS_CLEAR + 0.3 for ox, oy in obs_world):
            continue                       # 不在障碍里
        if math.hypot(wx-from_pos[0], wy-from_pos[1]) < min_dist_from:
            continue                       # 离目标别太近（绑架测试才有意义）
        return wx, wy
    return 25.0, 25.0  # fallback 中心（大概率是路）

obs_world = [] if args.no_obs else gen_obstacles(FIXED_SEED)
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
                # prev 格是墙前可通行格（机器人可能站上面），标 FREE 不标 WALL
                if gget(prev_vx, prev_vy) == UNKNOWN:
                    gset(prev_vx, prev_vy, FREE)
                break
            # 射线清除：穿过格强制 FREE（旧障碍被"照"掉）。
            # gset 只写 live 层，static 的墙不受影响。
            if gget(vx, vy) != FREE:
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

# 跨步尺寸（主人指令 2026-08-06：雷达 30m 后跨步可扩展减少计算）
JUMP_1M = 20   # 10→20：空旷区 1m→2m 一跳（离墙≥2m 时）
JUMP_03 = 6    # 3→6：离墙≥0.6m 时
JUMP_NEAR = 1  # 贴墙保持 0.1m

def jump_steps(vx, vy, dx, dy):
    wd = wall_dist(vx, vy)
    if wd >= JUMP_1M:   max_jump = JUMP_1M
    elif wd >= JUMP_03: max_jump = JUMP_03
    else:               max_jump = JUMP_NEAR
    for step in range(1, max_jump + 1):
        nx, ny = vx + dx*step, vy + dy*step
        if not traversable(nx, ny):
            return step - 1
    return max_jump

def wall_dist(vx, vy):
    key = (vx, vy)
    if key in _wd: return _wd[key]
    best = 999
    for dy in range(-WALL_SCAN_RADIUS, WALL_SCAN_RADIUS+1):
        for dx in range(-WALL_SCAN_RADIUS, WALL_SCAN_RADIUS+1):
            if gget_plan(vx+dx, vy+dy) == WALL:
                d = abs(dx)+abs(dy)
                if d < best: best = d
    _wd[key] = best
    return best

def walkable(vx, vy):
    return gget_plan(vx, vy) == FREE and wall_dist(vx, vy) > ROBOT_R

def traversable(vx, vy):
    """探索可通行：FREE 或 UNKNOWN 都行（WALL 不行）。
    核心：frontier 探索允许走向未知——A* 规划激进，执行层(Mover)实时避障兜底。
    真实机器人也是这么干的：未知区域可通行，撞到才知道有墙。
    """
    return gget_plan(vx, vy) != WALL

def line_clear(vx1, vy1, vx2, vy2):
    steps = max(abs(vx2-vx1), abs(vy2-vy1))
    if steps == 0: return True
    for i in range(steps+1):
        if gget_plan(int(vx1+(vx2-vx1)*i/steps), int(vy1+(vy2-vy1)*i/steps)) == WALL:
            return False
    return True

# ═══════════════════════════════════════════
# 跳步门查找
# ═══════════════════════════════════════════

def _nearest_walkable(vx, vy, max_r=8):
    """墙边脱困：从 (vx,vy) BFS 找最近的 walkable 格（机器人贴墙时跳不出 jump_steps）
    放宽到 traversable：机器人位置贴墙边但前方可能是未知区域，也能起步"""
    if traversable(vx, vy):
        return vx, vy
    seen = {(vx, vy)}
    q = [(vx, vy, 0)]
    while q:
        cx, cy, dist = q.pop(0)
        if dist >= max_r:
            continue
        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
            nx, ny = cx+dx, cy+dy
            if (nx,ny) in seen:
                continue
            seen.add((nx,ny))
            if traversable(nx, ny):
                return nx, ny
            q.append((nx, ny, dist+1))
    return None

def find_gates(fvx, fvy):
    # 起点放宽：机器人物理位置可能是 UNKNOWN（刚起步未扫描）或贴墙边，
    # 但已实际站在那，必须允许寻路，否则 find_gates 返回空 → 主循环死循环
    if gget_plan(fvx, fvy) == WALL:
        return [], {}
    start = _nearest_walkable(fvx, fvy)
    if start is None:
        return [], {}
    fvx, fvy = start
    open_set = [(0, fvx, fvy)]
    came_from = {}; g_score = {(fvx, fvy): 0}
    visited = set()
    gates = []
    search_radius = MAX_GATE_DIST  # 门搜索半径（格）：只搜附近，避免全图 BFS 拖慢
    while open_set and len(came_from) < ASTAR_MAX_EXPAND and len(gates) < MAX_GATES:
        _, cx, cy = heapq.heappop(open_set)
        if (cx,cy) in visited: continue
        visited.add((cx,cy))
        cg = g_score.get((cx,cy), 9999)
        if gates and cg > MAX_GATE_DIST:
            break
        # 距离剪枝：路径代价超过搜索半径的格不扩展（远处门走过去再找）
        # 用 g_score(路径代价) 而非曼哈顿——蛇形通道路径近但直线远的门不会被误剪
        if cg > search_radius:
            continue
        if gget_plan(cx, cy) == FREE:
            has_unk = any(gget_plan(cx+dx, cy+dy) == UNKNOWN
                          for dy in (-1,0,1) for dx in (-1,0,1))
            # 门=未知前沿。黑名单门（bounce 撞墙的）跳过
            # wall_dist 过滤：>ROBOT_R-1 保证机器人能站（缺口处放宽 1 格）
            if has_unk and wall_dist(cx, cy) > ROBOT_R - 1 and (cx, cy) not in bad_gates:
                gates.append((cg, cx, cy))
        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
            js = jump_steps(cx, cy, dx, dy)
            if js < 1: continue
            nx, ny = cx + dx*js, cy + dy*js
            wd = wall_dist(nx, ny)
            penalty = max(0, WALL_BUFFER_CELLS - wd) * WALL_PENALTY
            # 走中间（KNOWN_MAP_MODE）：代价 + C/d²，贴墙爆炸、中线最低（Voronoi 骨架）
            if KNOWN_MAP_MODE:
                penalty += VORONOI_C / (max(1, wd) * max(1, wd))
            # UNKNOWN 格可通行但代价高（优先已知路，必要时才穿未知）
            # KNOWN_MAP_MODE：穿越未知无惩罚（已知地图模式要快速穿行，执行层实时避障）
            if gget_plan(nx, ny) == UNKNOWN and not KNOWN_MAP_MODE:
                penalty += UNKNOWN_PENALTY
            ng = cg + js + penalty
            if (nx,ny) not in g_score or ng < g_score[(nx,ny)]:
                g_score[(nx,ny)] = ng
                came_from[(nx,ny)] = (cx,cy)
                heapq.heappush(open_set, (ng, nx, ny))
    return cluster_gates(gates), came_from

def cluster_gates(gates, min_size=8):
    """门格聚类：相邻门格 BFS 聚成 region，取质心+size。借鉴 frontier_exploration。
    gates: [(cg, vx, vy)] → [(cg, cx, cy, size)]，按质心格 cg 升序（近→远）
    修复（2026-08-06 根因）：探索早期前沿窄，门格 <min_size 被全过滤 → 0 门卡死。
    min_size 动态化：门少时降低阈值；全过滤时保底返回最近门。
    """
    if not gates:
        return []
    # 门少时（探索早期窄前沿）降低聚类阈值，避免全过滤
    dyn_min = min(min_size, max(2, len(gates) // 2))
    cells = [(vx, vy) for _, vx, vy in gates]
    cell_set = set(cells)
    visited = set()
    regions = []
    for cg, vx, vy in gates:
        if (vx, vy) in visited:
            continue
        # BFS 聚簇（4连通）
        cluster = []
        q = [(vx, vy)]
        visited.add((vx, vy))
        while q:
            cx, cy = q.pop()
            cluster.append((cx, cy))
            for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
                nx, ny = cx+dx, cy+dy
                if (nx, ny) in cell_set and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    q.append((nx, ny))
        if len(cluster) < dyn_min:
            continue  # 过滤小区域（噪声/缝隙），阈值随门数动态
        sx = sum(c[0] for c in cluster) // len(cluster)
        sy = sum(c[1] for c in cluster) // len(cluster)
        best_cg = min(c for c, x, y in gates if (x, y) in cluster)
        regions.append((best_cg, sx, sy, len(cluster)))
    if not regions and gates:
        # 保底：全被过滤（极窄前沿）→ 返回最近门，避免 0 门主循环死循环
        cg, vx, vy = min(gates)  # cg 最小 = 最近
        regions.append((cg, vx, vy, 1))
    regions.sort(key=lambda r: r[0])
    return regions

def pick_gate(gates, mode="score", stuck=False, robot=(0, 0), fin=(0, 0)):
    if not gates: return None
    if stuck: return gates[0]
    if mode == "far": return gates[-1]
    if mode == "near": return gates[0]
    if mode == "mix":
        return gates[-1] if len(gates) >= MIX_THRESHOLD else gates[0]
    if mode == "score":
        # 终点导向：任务=到达终点，优先向终点方向推进的门
        # score = 0.55·advance + 0.25·(1/dist) + 0.20·(size/50)
        # advance 主导：蛇形赛道 y 从 2.5→47.5，推进门 = y 增大方向
        # 探索模式加信息增益：门附近 UNKNOWN 格数（可新扫面积）
        bx, by = robot
        fx, fy = fin
        best = None; best_score = -1
        for g in gates:
            cg, gx, gy, size = g
            wx, wy = (gx+0.5)*VOXEL, (gy+0.5)*VOXEL
            d = math.hypot(wx - bx, wy - by)
            d = max(d, 1.0)
            # 向终点推进度：目标-机器人 在 终点方向上的投影（归一化到 0~1）
            advance = 0.0
            denom = math.hypot(fx-bx, fy-by)
            if denom > 1e-6:
                adv = ((wx-bx)*(fx-bx) + (wy-by)*(fy-by)) / (denom * max(d, 0.01))
                advance = max(0.0, min(1.0, adv))
            if not KNOWN_MAP_MODE:
                # 探索：终点导向 score（信息增益实验失败——导致反复横跳，回退纯 advance 评分）
                score = 0.55 * advance + 0.25 * (1.0/d) + 0.20 * (size / 50.0)
            else:
                score = 0.55 * advance + 0.25 * (1.0/d) + 0.20 * (size / 50.0)
            if score > best_score:
                best_score = score; best = g
        return best
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

ASTAR_CALLS = [0]
def astar_to(fvx, fvy, tfx, tfy):
    ASTAR_CALLS[0] += 1
    if ASTAR_CALLS[0] % 100 == 1:
        print(f"  [A*] call#{ASTAR_CALLS[0]} from=({fvx},{fvy}) to=({tfx},{tfy})", flush=True)
    # 起点放宽：未知/已知都可起步（WALL 才拒绝），机器人物理位置贴墙边也必须能回溯寻路
    if gget_plan(fvx, fvy) == WALL:
        return None
    # 终点要求：探索模式要 walkable（已确认自由区）；KNOWN_MAP_MODE 放宽到 traversable（未知也可达，执行层实时避障）
    if KNOWN_MAP_MODE:
        if not traversable(tfx, tfy):
            return None
    elif not walkable(tfx, tfy):
        return None
    start = _nearest_walkable(fvx, fvy)
    if start is None:
        return None
    fvx, fvy = start
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
            # 走中间（KNOWN_MAP_MODE）：代价 + C/d²
            if KNOWN_MAP_MODE:
                wd = wall_dist(nx, ny)
                ng += VORONOI_C / (max(1, wd) * max(1, wd))
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
        f'<geom type="cylinder" size="0.5 1.0" rgba="0.9 0.2 0.2 0.9" contype="0" conaffinity="0"/></body>'
        for i,(x,y) in enumerate(obs_world))
    FINISH_XML = f'<body mocap="true" pos="{FINISH[0]:.1f} {FINISH[1]:.1f} 2"><geom type="sphere" size="1.5" rgba="0.2 1.0 0.2 0.8"/></body>'
    # 地标标牌（贴墙方案，--landmarks 1 启用；导航测试默认关避免干扰）
    if args.landmarks:
        LM_ASSETS, LM_WORLD = landmark_xml()
        WALL_XML = wall_xml()
    else:
        LM_ASSETS, LM_WORLD, WALL_XML = "", "", ""
    # 机器人前置相机：桅杆 0.5m（世界 HF_SURF+1.0=5.008），平视看墙上标牌
    # euler="0 -1.5708 -1.5708"（extrinsic xyz）：fwd=+x, up=+z(天空上), right=-y
    # 注意：只有 euler="0 -1.5708 0" 时 right=+z 天空在水平方向（画面 roll 90°），必须加 z=-90°
    CAM_XML = '<camera name="bot_cam" pos="0.4 0 0.5" mode="fixed" euler="0 -1.5708 -1.5708"/>'
    return f"""<mujoco>
  <compiler angle="radian"/><option timestep="0.005"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset><hfield name="track" size="25.0 25.0 4.0 2.0" file="{RENDER_MAP}"/>
    {LM_ASSETS}
  </asset>
  <worldbody>
    <light pos="25 25 80" dir="0 0 -1"/>
    {FINISH_XML}{OBS_XML}
    <geom type="hfield" hfield="track" pos="25 25 0.0" rgba="0.25 0.30 0.35 1.0" friction="0 0 0" contype="0" conaffinity="0"/>
    {WALL_XML}
    {LM_WORLD}
    <!-- 方案A：边界几何纯可视化（contype=0），防穿墙靠算法走廊检查 -->
    <geom type="box" size="1.0 25.0 2.0" pos="-1.0 25 1.0" rgba="0.2 0.2 0.2 0" contype="0" conaffinity="0"/>
    <geom type="box" size="1.0 25.0 2.0" pos="51.0 25 1.0" rgba="0.2 0.2 0.2 0" contype="0" conaffinity="0"/>
    <geom type="box" size="25.0 1.0 2.0" pos="25 -1.0 1.0" rgba="0.2 0.2 0.2 0" contype="0" conaffinity="0"/>
    <geom type="box" size="25.0 1.0 2.0" pos="25 51.0 1.0" rgba="0.2 0.2 0.2 0" contype="0" conaffinity="0"/>
    <body name="bot" pos="0 0 {BOT_Z}">
      <joint type="slide" axis="1 0 0" damping="0"/>
      <joint type="slide" axis="0 1 0" damping="0"/>
      <joint name="yaw" type="hinge" axis="0 0 1" damping="0"/>
      <!-- 机器狗：水平圆柱（长轴沿 yaw 方向），0.8m 长 × 0.4m 径，contype=0 纯算法控制 -->
      <geom type="capsule" fromto="0 -0.4 0 0 0.4 0" size="0.2" rgba="1 0.3 0 1" friction="0 0 0" contype="0" conaffinity="0"/>
      {CAM_XML}
    </body>
  </worldbody>
</mujoco>"""

class Mover:
    def __init__(self, m, d):
        self.m, self.d = m, d
        self.yaw = 0.0; self.speed = 0.0; self.bounce = 0
        self.stuck_t = 0; self.stuck_x = 0.0; self.stuck_y = 0.0
        self.target = (FINISH[0], FINISH[1])  # 当前目标（GATE 方向），bounce 时优先朝向
        self.escape_steps = 0   # bounce 逃生冷却：沿 escape_yaw 强制走 N 步（绕出墙角再回归路径）
        self.escape_yaw = 0.0

    def _forward_clear(self, bx, by, yaw_ang):
        """沿 yaw 方向前瞻测距：返回前方最近障碍距离 (m)。blocked() 已含机器人半径膨胀。"""
        for k in range(1, int(LOOKAHEAD / 0.05) + 1):
            px = bx + math.cos(yaw_ang) * 0.05 * k
            py = by + math.sin(yaw_ang) * 0.05 * k
            if blocked(px, py):
                return 0.05 * k
        return LOOKAHEAD

    def step(self, tx, ty, step):
        bx, by = self.d.qpos[0], self.d.qpos[1]
        dt = self.m.opt.timestep
        # bounce 逃生冷却：沿 escape_yaw 方向走（目标=前方 2m 点），绕出墙角再回归
        if self.escape_steps > 0:
            self.escape_steps -= 1
            tx = bx + math.cos(self.escape_yaw) * 2.0
            ty = by + math.sin(self.escape_yaw) * 2.0
        self.target = (tx, ty)  # 记录当前目标，bounce 用
        if step % 10000 == 0:
            print(f"    [MOVER] step={step} pos=({bx:.1f},{by:.1f}) target=({tx:.1f},{ty:.1f}) yaw={math.degrees(self.yaw):.0f}° speed={self.speed:.2f}", flush=True)
        # 转向目标（yaw 转向 + 沿轴向速度）
        tgt_yaw = math.atan2(ty-by, tx-bx)
        err = (tgt_yaw-self.yaw+math.pi)%(2*math.pi)-math.pi
        dyaw = max(-YAW_RATE*dt, min(YAW_RATE*dt, err))
        self.yaw += dyaw
        # ── 大转向限速（DWA 思想：曲率越大速度越低）──
        # 目标方向偏离当前朝向 >57° 时限速 1.0——先转身再加速，防止直冲偏离（绑架随机起点 yaw=0 但目标在背后 170°）
        turn_limited = False
        if abs(err) > 1.0:
            turn_limited = True
        # ── 前瞻测距：沿当前 yaw 方向量到障碍的距离 ──
        d_clear = self._forward_clear(bx, by, self.yaw)
        # ── 期望速度（限速）：目标距离速度 vs 制动约束，取小 ──
        v_des = min(SPEED_MAX, math.hypot(tx-bx, ty-by)*SPEED_FACTOR)
        if turn_limited:
            v_des = min(v_des, 1.0)
        # 近墙限速：前方空间小（<2m）时降速，窄连接段/迷宫缺口慢速通过防过冲
        if d_clear < 2.0:
            v_des = min(v_des, 1.5)
        # 制动约束：当前速度 v 需满足 v² ≤ 2·A_DECEL·(d_clear-STOP_MARGIN)
        # → 任何速度下急刹都能在障碍前停住，物理上碰撞=0
        v_brake = math.sqrt(max(0.0, 2.0*A_DECEL*max(0.0, d_clear-STOP_MARGIN)))
        v_des = min(v_des, v_brake)
        # ── 加速度/减速度限制：速度不突变，按 A_ACCEL 爬升、A_DECEL 下降 ──
        if self.speed < v_des:
            self.speed = min(v_des, self.speed + A_ACCEL*dt)
        else:
            self.speed = max(v_des, self.speed - A_DECEL*dt)
        # ── 前方被堵且已停住（speed≈0）→ 预判转向，不碰撞 ──
        if self.speed <= 0.05 and d_clear < STOP_MARGIN + 0.15:
            _hit_wall = sample_hf(bx, by) != ROAD_PIX
            _near_obs = any(math.hypot(bx-ox, by-oy) < OBS_CLEAR for ox, oy in obs_world)
            if self.bounce % 5 == 0:
                print(f"  [STOP] bounce#{self.bounce} @({bx:.1f},{by:.1f}) d_clear={d_clear:.2f} wall={_hit_wall} obs={_near_obs}", flush=True)
            self._bounce(45, 120)
            # 转向后本帧不再移动（下帧从新 yaw 重新测距加速）
        # 卡死检测
        if step-self.stuck_t > STUCK_TIMEOUT:
            if math.hypot(bx-self.stuck_x, by-self.stuck_y) < STUCK_DIST_THRESH:
                self._bounce(90, 180)
            self.stuck_t = step; self.stuck_x = bx; self.stuck_y = by
        # 执行移动（沿轴向速度，物理 yaw 由 hinge 积分）
        vx = math.cos(self.yaw)*self.speed; vy = math.sin(self.yaw)*self.speed
        nx, ny = bx+vx*dt, by+vy*dt
        # 硬防穿墙/障碍：下一位置 blocked（中心0.2m圆触障碍）就不动——物理上不可能穿。
        # bounce 决策已由上方 STOP 分支负责，这里只保证不移动（防低速漂移滑入）
        if blocked(nx, ny):
            self.speed = 0.0
            self.d.qvel[0] = 0; self.d.qvel[1] = 0; self.d.qvel[2] = 0
            # 从经验学习：撞到的未知格写回地图为 WALL（A* 下次就不会规划穿墙路径）
            gvx, gvy = int(nx/VOXEL), int(ny/VOXEL)
            if gget(gvx, gvy) == UNKNOWN:
                gset(gvx, gvy, WALL)
        else:
            self.d.qvel[0] = vx; self.d.qvel[1] = vy; self.d.qvel[2] = 0
            self.d.qpos[2] = self.yaw  # 控制 yaw 直接写回物理（滑动模型 friction=0，mj_step 保持）
        mujoco.mj_step(self.m, self.d)
        # 不再读回 qpos[2] 覆盖 self.yaw——yaw 是控制变量（旧代码读回导致每步转向被重置为初始 0）
        return True

    def _bounce(self, lo, hi):
        # 无条件计数+转向（防 escaping 卡死）；转向后 speed 已≈0，由 step() 重新加速
        self.bounce += 1
        if args.trail_every > 0:
            bounce_pts.append([round(self.d.qpos[0], 3), round(self.d.qpos[1], 3), self.bounce])
        if self.bounce % 5 == 0:
            print(f"  [BOUNCE] bounce#{self.bounce} @({self.d.qpos[0]:.1f},{self.d.qpos[1]:.1f})", flush=True)
        # 转向：先试目标方向（当前 GATE）的小角度偏转，再试随机（防墙边死循环）
        bx, by = self.d.qpos[0], self.d.qpos[1]
        tx, ty = self.target
        tgt_yaw = math.atan2(ty - by, tx - bx)
        # 全向每 5° 扫（防稀疏角度漏掉窄缺口——迷宫段间墙缺口只有 ~10° 宽）
        # 评分：推进优先，但推进度差距小(<0.15)时距离只做轻 tie-break（防 Z 字斜穿震荡）
        candidates = []
        for deg in range(0, 360, 5):
            candidates.append(tgt_yaw + math.radians(deg))
        best_yaw, best_d, best_score = None, -1.0, -1e9
        for cand in candidates:
            d = self._forward_clear(bx, by, cand)
            if d < STOP_MARGIN:
                continue
            dot = math.cos(cand - tgt_yaw)
            # 混合评分：推进优先但保留逃逸能力——dot*4 + 距离(截断2)
            # 平衡点：正前方被挡(d 小)时，侧向 dot=0.7 且 d=3m → 2.8+2=4.8 > 正前方 4+0.5=4.5
            # 但斜穿对面墙 dot=0.3,d=5 → 1.2+2=3.2 < 侧向贴墙走 dot=0.9,d=2 → 3.6+2=5.6
            score = dot * 4.0 + min(d, 2.0)
            if score > best_score:
                best_score, best_d, best_yaw = score, d, cand
        if best_yaw is not None and best_d >= STOP_MARGIN:
            self.yaw = best_yaw
            self.d.qpos[2] = best_yaw  # 同步物理 yaw，狗身体立即转到新方向
            self.d.qvel[:] = 0
            self.speed = 0.0
            # 逃生冷却：沿刚选的方向强制走 120 步（防 step() 又转向被挡目标 → bounce 死循环）
            self.escape_steps = 120
            self.escape_yaw = best_yaw
            return
        # 全部方向都堵（理论死角）→ 随机试几个方向，选能走最远的（防蹭墙死循环）
        best_yaw, best_d = None, -1.0
        for _ in range(12):
            cand = random.uniform(0, 2*math.pi)
            d = self._forward_clear(bx, by, cand)
            if d > best_d:
                best_d, best_yaw = d, cand
        if best_yaw is not None and best_d >= STOP_MARGIN:
            self.yaw = best_yaw
        else:
            # 连随机都全堵（不可能，兜底）：原地掉头
            self.yaw += math.pi
        self.d.qpos[2] = self.yaw
        self.d.qvel[:] = 0
        self.speed = 0.0

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

def save_map(path):
    """保存地图到文件（阶段1→阶段2 传递）。
    只保存**结构墙**（hfield 墙）——障碍是变化的，不存（live 层射线清除会重新发现）。
    用 sample_hf 判断结构墙：hfield 非路 = 永久墙。
    """
    if not grid: return
    xs = sorted(set(k[0] for k in grid)); ys = sorted(set(k[1] for k in grid))
    minx, maxx = xs[0], xs[-1]; miny, maxy = ys[0], ys[-1]
    arr = np.full((maxy-miny+1, maxx-minx+1), UNKNOWN, dtype=np.int8)
    for (vx, vy), val in grid.items():
        wx, wy = (vx+0.5)*VOXEL, (vy+0.5)*VOXEL
        if sample_hf(wx, wy) != ROAD_PIX:
            arr[vy-miny, vx-minx] = WALL   # 结构墙
        elif val == FREE:
            arr[vy-miny, vx-minx] = FREE    # 已知路
        # 障碍 (WALL 但 hfield 是路) 不存 —— 变化的东西
    np.savez(path, grid=arr, offset=(minx, miny), seed=FIXED_SEED)
    n_wall = int((arr == WALL).sum()); n_free = int((arr == FREE).sum())
    print(f"  [MAP] saved {n_free} FREE + {n_wall} WALL(结构墙) → {path}", flush=True)

def load_map(path):
    """加载旧地图为 static_grid，开启 KNOWN_MAP_MODE（阶段2）"""
    global static_grid, KNOWN_MAP_MODE
    if not os.path.exists(path):
        print(f"  [MAP] 警告: {path} 不存在", flush=True)
        return False
    data = np.load(path, allow_pickle=True)
    arr = data["grid"]; ox, oy = data["offset"]
    static_grid.clear()
    for vy in range(arr.shape[0]):
        for vx in range(arr.shape[1]):
            if arr[vy, vx] != UNKNOWN:
                static_grid[(vx+ox, vy+oy)] = int(arr[vy, vx])
    KNOWN_MAP_MODE = True
    print(f"  [MAP] loaded {len(static_grid)} cells from {path} (KNOWN_MAP_MODE)", flush=True)
    return True

# ═══════════════════════════════════════════
# 指标统计
# ═══════════════════════════════════════════
stats = {
    "gates_selected": 0,
    "backtracks": 0,
    "lost_rescues": 0,
    "bounces": 0,
    "collisions": 0,
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
if args.random_start:
    rs_x, rs_y = random_road_pos(FIXED_SEED + 1000, min_dist_from=5.0, from_pos=FINISH)
    d.qpos[0] = rs_x; d.qpos[1] = rs_y
    print(f"  [KIDNAP] 随机起点: ({rs_x:.1f}, {rs_y:.1f}) → 目标 {FINISH}", flush=True)
else:
    d.qpos[0]=2.5; d.qpos[1]=2.5
mujoco.mj_forward(m,d)

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

# 阶段2：加载旧地图为 static_grid（已知地图快速寻路）
if args.load_map:
    load_map(args.load_map)
# --known-raw：KNOWN_MAP 直接用原始 track_clean 地图（不依赖 npz 生成，与物理 sample_hf 判定完全一致）
# 区域判定：格覆盖 4×4 像素任一为墙 → WALL（单点采样会漏薄分界墙，如 y=30 row798-801 但中心采样 py=797）
if args.known_raw and not args.load_map:
    KNOWN_MAP_MODE = True
    bin_wall = (hf != ROAD_PIX).astype(np.int8)
    arr_wall = bin_wall.reshape(500, 4, 500, 4).max(axis=(1, 3))
    # MAX-pool 行块 = 图像行（row0=y=50m 顶部），格坐标 gy=0 是 y=0m → 必须 flip 对齐世界坐标！
    arr_wall = arr_wall[::-1, :]
    for vy in range(500):
        for vx in range(500):
            if arr_wall[vy, vx]:
                static_grid[(vx, vy)] = WALL
    print(f"  [MAP] 原始 track_clean 地图就绪 (KNOWN_MAP_MODE, {len(static_grid)} 墙格, MAX-pool 区域判定)", flush=True)

# HPA* 分层规划器（KNOWN_MAP 用；成熟算法替代全程 A*，构建~1.5s）
hpa = None
if KNOWN_MAP_MODE:
    try:
        sys.path.insert(0, os.path.join(PROJ, "scripts"))
        from hpa_star import HPAStar, CELL_M as HPA_CELL_M
        # wall_fn 用 live 视图（static 墙 + 实时障碍），距离场用 static 墙
        def _hpa_wall(vx, vy):
            return gget_plan(vx, vy) == WALL
        hpa = HPAStar(_hpa_wall, verbose=False)
        print(f"  [HPA] 分层规划器就绪 (cell={HPA_CELL_M}m)", flush=True)
    except Exception as e:
        hpa = None
        print(f"  [HPA] 不可用: {e}（回退全程 A*）", flush=True)

# 视觉地标识别（前置相机 + 颜色识别；GPU渲染 32ms/帧，--vision 1 开启）
if args.vision:
    try:
        from test_scripts.vision_landmark import VisionLandmark
        # detect_every=40：GPU 渲染 32ms/帧，每 40 步(0.2s物理)一帧开销可接受，标牌识别灵敏
        vis = VisionLandmark(m, d, renderer, cam_name="bot_cam", detect_every=40)
        print("  [VISION] 视觉地标识别已启用 (GPU 32ms/帧, detect_every=40)", flush=True)
    except Exception as e:
        vis = None
        print(f"  [VISION] 视觉不可用: {e}", flush=True)
else:
    vis = None

step = 0; t0 = time.time()
last_mx = last_my = 0
path = None; path_idx = 0; _plan_bounce_base = 0
gate = None; gates = []
no_gate_count = 0
wander = 0; last_dist = 999
milestones = []
start_pos = (d.qpos[0], d.qpos[1])
milestones.append((int(start_pos[0]/VOXEL), int(start_pos[1]/VOXEL)))
last_mx, last_my = milestones[0]
back_blacklist = set()  # BACK 过的路标索引（防死循环）
bad_gates = set()       # bounce 撞墙的门格黑名单（防反复选同一死门）

print(f"=== Firefly v3 headless start: seed={FIXED_SEED} max_steps={args.max_steps} ===", flush=True)

# 初始扫描
for _ in range(INIT_SCAN_STEPS):
    bx, by = d.qpos[0], d.qpos[1]
    if _ % LIDAR_TICK == 0: scan(bx, by)
    mujoco.mj_step(m, d)
print(f"  [OK] FREE={_cnt[FREE]} WALL={_cnt[WALL]}", flush=True)

frame_idx = 0
trail = []          # 轨迹记录：每 trail-every 步存 (step, x, y, yaw, bounce)
bounce_pts = []     # bounce 位置（分析卡点用）
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

    # B阶段：运行中障碍变化（--obs-reseed 指定步数换新障碍 seed）
    if args.obs_reseed > 0 and step == args.obs_reseed:
        old_obs = list(obs_world)
        obs_world = gen_obstacles(FIXED_SEED + 777)  # 新障碍位置（顶层代码直接赋值，全局生效）
        print(f"  [OBS] step={step} 障碍变化: {len(old_obs)}→{len(obs_world)} 个", flush=True)
        # 强制重规划（当前路径可能穿过新障碍）
        path = None; path_idx = 0; wander = 0; last_dist = 999
        _plan_bounce_base = mv.bounce
        stats["obs_changes"] = stats.get("obs_changes", 0) + 1

    # 视觉地标识别（相机帧 + ArUco；看到标牌记录唯一ID）
    if vis is not None:
        vis.scan_once(step)

    if path is None or path_idx >= len(path):
        if KNOWN_MAP_MODE:
            # KNOWN_MAP_MODE：地图已知，HPA* 分层规划到终点（成熟算法，长距离规划消失）
            if hpa is not None:
                t_plan0 = time.time()
                gp = hpa.plan(vx, vy, int(FINISH[0]/VOXEL), int(FINISH[1]/VOXEL))
                dt_plan = time.time() - t_plan0
                stats["hpa_plan_ms"] = stats.get("hpa_plan_ms", 0) + dt_plan
                stats["hpa_plans"] = stats.get("hpa_plans", 0) + 1
                path = [((px+0.5)*VOXEL, (py+0.5)*VOXEL) for px, py in gp] if gp else None
            else:
                path = astar_to(vx, vy, int(FINISH[0]/VOXEL), int(FINISH[1]/VOXEL))
            if path:
                path_idx = 0; wander = 0; last_dist = 999; no_gate_count = 0
                stats["gates_selected"] += 1
                if step % 10000 == 0:
                    print(f"    [DECIDE] KNOWN_MAP path={len(path)} →{FINISH} pos=({bx:.1f},{by:.1f})", flush=True)
            else:
                # 规划失败（新障碍挡住旧地图路径）→ 先避障，下轮重规划
                path = None; path_idx = 0; no_gate_count += 1
                mv._bounce(60, 120)
        else:
            # 强制重规划：bounce 过多说明当前路径失效（偏离/被挡），换目标
            if path is not None and mv.bounce - _plan_bounce_base > 8:
                if gate is not None and len(gates) > 1:
                    bad_gates.add((gate[1], gate[2]))
                path = None; path_idx = 0; wander = 0; last_dist = 999
                _plan_bounce_base = mv.bounce
            gates, came_from = find_gates(vx, vy)
            gate = pick_gate(gates, EXPLORE_MODE, stuck=(no_gate_count > 0),
                             robot=(bx, by), fin=FINISH)
            if step % 10000 == 0:
                print(f"    [DECIDE] step={step} gates={len(gates)} gate={'None' if gate is None else f'({gate[1]*VOXEL:.1f},{gate[2]*VOXEL:.1f})'} pos=({bx:.1f},{by:.1f})", flush=True)
            if gate is not None:
                cg, gx, gy, _gsize = gate
                path = fine_path(vx, vy, gx, gy, came_from)
                # 失败换门：A* 找不到当前门就试下一个（不 bounce，借鉴 frontier rank 机制）
                try:
                    gidx = gates.index(gate)
                except ValueError:
                    gidx = -1
                tries = 0
                while not path and tries < 3:
                    tries += 1
                    gidx += 1
                    if gidx >= len(gates):
                        break
                    gate = gates[gidx]
                    cg, gx, gy, _gsize = gate
                    path = fine_path(vx, vy, gx, gy, came_from)
                if not path:
                    # 所有候选门都不可达：不卡死，bounce 找路（下轮重试）
                    path = None; path_idx = 0; wander = 0; last_dist = 999
                    no_gate_count += 1
                    mv._bounce(60, 120)
                else:
                    path_idx = 0; wander = 0; last_dist = 999
                    no_gate_count = 0
                    stats["gates_selected"] += 1
                    print(f"  [GATE] [{step}] →({(gx+0.5)*VOXEL:.1f},{(gy+0.5)*VOXEL:.1f}) path={len(path)} gates={len(gates)}", flush=True)
            else:
                no_gate_count += 1
                if no_gate_count > MAX_NO_GATE and len(milestones) > 1:
                    saved = False
                    for i in range(len(milestones)-2, -1, -1):
                        if i in back_blacklist:
                            continue  # 已 BACK 过但没新门，跳过（防死循环）
                        mx, my = milestones[i]
                        bp = astar_to(vx, vy, mx, my)
                        if bp:
                            path = bp; path_idx = 0; wander = 0; last_dist = 999
                            no_gate_count = 0
                            stats["backtracks"] += 1
                            back_blacklist.add(i)
                            print(f"  [BACK] [{step}] →路标#{i}", flush=True)
                            saved = True
                            break
                    if not saved:
                        if no_gate_count < 10: mv._bounce(90, 180)
                        else: mv._bounce(150, 210)

    # 执行
    if step % 10000 == 0:
        print(f"    [EXEC] step={step} path={'None' if path is None else len(path)} idx={path_idx} no_gate={no_gate_count} pos=({d.qpos[0]:.1f},{d.qpos[1]:.1f})", flush=True)
    if path is not None and path_idx < len(path):
        # pure pursuit：找路径上最近点 → 目标 = 其前方 PATH_LOOKAHEAD 处的点
        # path_idx 持续推进（上次最近点索引），避免 bounce 后目标跳回起点
        best_i, best_d = path_idx, 1e18
        for _i in range(path_idx, min(path_idx + 400, len(path))):
            lx, ly = path[_i]
            dd = (lx-bx)*(lx-bx) + (ly-by)*(ly-by)
            if dd < best_d:
                best_d = dd; best_i = _i
        if best_i > path_idx:
            path_idx = best_i  # 推进：已走过的路径段不再回头
        # 目标 = best_i 之后第一个 ≥ PATH_LOOKAHEAD 的点
        tx, ty = path[path_idx]
        look_target = None
        for _i in range(path_idx, len(path)):
            lx, ly = path[_i]
            if math.hypot(lx-bx, ly-by) >= PATH_LOOKAHEAD:
                look_target = (lx, ly)
                break
        if look_target is not None:
            tx, ty = look_target
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
        # path 不可用（find_gates 空/回溯失败）：向终点走，运动学约束自己避障
        # （不能只 _bounce 不 step——那会原地转向永不移动，stuck 检测也不触发）
        mv.step(FINISH[0], FINISH[1], step)

    step += 1
    if step <= 50 or step % 200 == 0:
        import time as _t
        _now = _t.time()
        if step <= 50:
            _perf = getattr(__import__('__main__', fromlist=['x']), '_perf', None)
        else:
            _perf = None
        print(f"  [PERF] step={step} t={_now - t0:.1f}s rate={step/(_now - t0):.0f}步/s", flush=True)

    # 轨迹记录（分析行为用）
    if args.trail_every > 0 and step % args.trail_every == 0:
        trail.append([step, round(d.qpos[0], 3), round(d.qpos[1], 3),
                      round(math.degrees(d.qpos[2]), 1), mv.bounce])

    # 终点检测（3.5m：二维码在通道尽头转弯口前，够近即可；3.0 在斜墙边会差 7mm 判不到）
    if math.hypot(bx-FINISH[0], by-FINISH[1]) < 3.5:
        print(f"\n  ★ ARRIVED! @({bx:.1f},{by:.1f}) step={step} ms={len(milestones)}", flush=True)
        stats["arrived"] = True
        break

    # 碰撞检测（主人要求碰撞=0）：真实几何——机器人中心进入障碍安全圈才算碰撞
    # is_obstacle_world 已含 OBS_CLEAR=0.7（障碍半径0.5+机器人半径0.2），中心<0.7m即碰撞
    if is_obstacle_world(d.qpos[0], d.qpos[1]):
        stats["collisions"] += 1
        if stats["collisions"] == 1 or stats["collisions"] % 100 == 0:
            print(f"  ⚠ COLLISION #{stats['collisions']} @({d.qpos[0]:.2f},{d.qpos[1]:.2f}) step={step}", flush=True)

    # 离屏渲染
    if RENDER_OK and args.render_every > 0 and step % args.render_every == 0:
        render_frame(step)

    # 进度日志
    if step % 20000 == 0:
        cov = coverage_pct()
        print(f"  ... step={step} F={_cnt[FREE]} W={_cnt[WALL]} ms={len(milestones)} cov={cov:.1f}% t={time.time()-t0:.0f}s pos=({d.qpos[0]:.1f},{d.qpos[1]:.1f}) yaw={math.degrees(d.qpos[2]):.0f}°", flush=True)

# ── 收尾统计 ──
stats["steps"] = step
stats["time_sec"] = round(time.time() - t0, 2)
stats["bounces"] = mv.bounce
stats["collisions"] = stats.get("collisions", 0)
if vis is not None:
    stats["landmarks_seen"] = vis.total_detected
    stats["landmarks_unique"] = len(vis.seen_ids)
    stats["landmark_channels"] = sorted(vis.seen_ids)
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

# 阶段1：保存地图供后续阶段2加载
if args.save_map:
    save_map(args.save_map)

# 轨迹保存（分析行为用）
if args.trail_every > 0 and trail:
    import numpy as _np
    trail_path = os.path.join(SCAN_DIR, f"trail_seed{FIXED_SEED}_steps{step}_b{len(bounce_pts)}.npz")
    _np.savez(trail_path,
              trail=_np.array(trail, dtype=float),
              bounce_pts=_np.array(bounce_pts, dtype=float) if bounce_pts else _np.zeros((0, 3)),
              seed=FIXED_SEED, mode=EXPLORE_MODE,
              lidar_rays=LIDAR_RAYS, lidar_tick=LIDAR_TICK,
              no_obs=args.no_obs, obs_reseed=args.obs_reseed)
    print(f"\n[TRAIL] {trail_path} trail={len(trail)} bounce_pts={len(bounce_pts)}", flush=True)

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
