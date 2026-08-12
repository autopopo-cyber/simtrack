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
import cv2
from PIL import Image
import mujoco

# 地标标牌系统（30 个 ArUco+数字标牌）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from test_scripts.landmarks import landmark_xml, landmark_positions, BOT_Z, wall_xml, HF_SURF
from simtrack.obstacles_random import RandomObstacleField, mix_bend_positions
from simtrack.algorithms.dwa import DWAAlgorithm
from simtrack.odometry import Odometry
from simtrack.scan_matching import ScanMatcher
from simtrack.obstacle_tracker import ObstacleTracker

# ═══════════════════════════════════════════
# 全部可配置参数（与 algo3_firefly.py 一致）
# ═══════════════════════════════════════════

PROJ = os.path.expanduser("~/workspace/simtrack")
MAP = os.path.join(PROJ, "confirmed/track_clean.png")
# 渲染用降分辨率 hfield（2000→500）：EGL/OSMesa 软渲染下 2000x2000 高度图 2.7s/帧，
# 500x500 只要 223ms（快12倍）。sample_hf 碰撞检测仍用原图 MAP，精度不降。
# 2026-08-08 修正：track_500_bin.png 是从 clean 块判定生成的无抗锯齿二值图——
# 抗锯齿版 track_500.png 的墙边界有 0.3m 斜坡（狗贴墙走渲染时踩斜坡上 = 视觉上墙）
RENDER_MAP = os.path.join(PROJ, "confirmed/track_500_bin.png")
SCAN_DIR = os.path.join(PROJ, "scans")
SCAN_STATE = os.path.join(SCAN_DIR, "scan_dict.npz")
os.makedirs(SCAN_DIR, exist_ok=True)

SCALE = 1.0; HF_RES = 2000; PIX_PER_M = 40; ROAD_PIX = 128
SAFE_R = 0.2; SPEED = 4.0; SPEED_MAX = 4.0; YAW_RATE = 1.5
LIDAR_RANGE = 30.0  # 15→30m（主人指令 2026-08-06）：真实狗雷达10万点/s覆盖前半球，30m 探测更远

VOXEL = 0.1
ROBOT_R = max(1, int(SAFE_R / VOXEL))
CLEARANCE = ROBOT_R
# ── 门宽度判断（主人 2026-08-10 指令）：窄缝/栅栏陷阱识别 ──
# 现实有大量"能透过激光但狗过不去"的窄缝。识别手段=规划层净空场 PASS（ROS costmap
# inflation 思想）：地图本身保持精确边缘（KEEP_M=0.05 不动），通行性判断放规划层。
# PASS_CLEAR=0.6m = 狗半径 0.2 + 执行余量 0.4，对齐 HPA 细层 ROBOT_DIA=6 既有标准
# （已知地图模式 bounce 0 实测验证）——通行宽 <2×0.6=1.2m 的缝不存在可规划中线 →
# 对规划自动封闭（不是门也不是路径），狗不会被引进 STOP/bounce 陷阱。
PASS_CLEAR_M = 0.6
PASS_CLEAR = PASS_CLEAR_M / VOXEL   # 格（EDT 精确欧氏距离，浮点比较）
MILESTONE_STEP = int(3.0 / VOXEL)
LIDAR_STEPS = int(LIDAR_RANGE / VOXEL)
LIDAR_RAYS = 360  # 120→360（1°间隔）：15m处射线间距0.26m，覆盖0.1m薄斜墙（120时0.78m漏扫）

MAX_GATES = 200
WALL_BUFFER_M = 2.0; WALL_BUFFER_CELLS = int(WALL_BUFFER_M / VOXEL)
WALL_PENALTY = 3
UNKNOWN_PENALTY = 8  # 未知格可通行但代价高（探索规划，优先已知路）
VORONOI_C = 2.0      # 走中间代价系数：penalty = C/d² (d=离墙格数)，KNOWN_MAP_MODE 下启用
MAX_GATE_DIST = 500  # 全图搜索：左端缺口被斜边窄缝堵（感知格化后 <0.4m 狗过不去）时必须能找到右端宽缺口（距起点 44.5m）
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
ap.add_argument("--known-raw", type=int, default=0, help="⚠️测试作弊: KNOWN_MAP 直接读 track_clean 真值(非感知, 仅调试用, 真实流程用 save_map/load_map)")
ap.add_argument("--vision", type=int, default=1, help="视觉识别（hfield降分辨率后223ms/帧，默认开）")
ap.add_argument("--landmarks", type=int, default=1, help="标牌几何（contype=0零物理开销，默认开）")
ap.add_argument("--obs-reseed", type=int, default=0, help="运行中障碍变化步数：到该步换新障碍 seed（B阶段）")
ap.add_argument("--no-obs", type=int, default=0, help="1=纯墙版（去掉所有障碍，只留 hfield 墙）")
ap.add_argument("--lidar-rays", type=int, default=0, help="雷达射线数（0=用代码默认360）。真实狗~1万点/帧前半球")
ap.add_argument("--lidar-tick", type=int, default=0, help="雷达扫描间隔步数（0=用代码默认10）。真实狗10Hz")
ap.add_argument("--lidar-fov", type=float, default=180.0, help="雷达水平视场角(度)：180=前方半圆(相对狗yaw, 无特权), 360=全向(作弊)")
ap.add_argument("--lidar-lines", type=int, default=1, help="雷达线数：1=单线水平扫描, 3/10=多俯仰角(2D取最近,冗余确认)")
ap.add_argument("--trail-every", type=int, default=0, help="轨迹记录间隔（步，0=关）。存 scans/trail_*.npz")
ap.add_argument("--random-start", type=int, default=0, help="1=从随机位置(避开墙/障碍)出发")
ap.add_argument("--target", type=str, default="finish", help="目标: start|finish")
ap.add_argument("--obs-straight", type=int, default=0, help="每段直道障碍数（渐进：1→N。0=用默认密度生成）")
ap.add_argument("--obs-turn", type=int, default=0, help="每段弯道障碍数（渐进：1→N。0=不加）")
ap.add_argument("--obs-min-dist", type=float, default=6.0, help="直道障碍距通道两端转弯口的最小距离（m），避免堵死转弯")
ap.add_argument("--obs-patrol", type=int, default=0, help="巡逻障碍数：沿两直道+一弯道蛇形路径往返走（1m/s，~50m）")
ap.add_argument("--obs-random", type=int, default=0, help="随机反弹障碍数(2-4)：每段20m×5m, 1m/s, 20%/s变向, 撞墙/虚拟墙反弹")
ap.add_argument("--obs-random-ch", type=str, default="1,4,6,8", help="反弹区段通道列表(逗号分隔)")
ap.add_argument("--obs-mix", type=int, default=0, help="混合场：每拐弯处1个固定障碍(9个U型弯)+每直道1个随机反弹障碍(1m/s, x∈[4.5,45]不出本通道)")
ap.add_argument("--odom", type=int, default=1, help="1=里程计模式(默认,无特权)：决策/建图只用带噪里程计位姿（不用 d.qpos 真值），scan-matching 连续修正 + 二维码标牌绝对修正；0=真值位姿调试模式")
ap.add_argument("--odom-noise", type=float, default=0.05, help="里程计噪声量级（0.05=5%/s，四足 IMU+步态推算水平；含慢变偏差与陀螺漂移，见 odometry.py）")
ap.add_argument("--match", type=int, default=1, help="1=scan-to-map 匹配修正（默认开，激光里程计）；0=纯推算（仅二维码修正，A/B 调试用）")
ap.add_argument("--obs-feature", type=int, default=0, help="1=长直道特征障碍场：每段直道按间隔 15±5m 放固定障碍（每次建图随机），作为长直道激光特征防迷路")
ap.add_argument("--dwa-truth-vel", type=int, default=0, help="⚠️A/B 调试：DWA 障碍速度用真值 velocities（作弊，仅用于隔离跟踪器 bug——审核整改后默认 0=感知跟踪估计）")
ap.add_argument("--pass-clear", type=float, default=-1, help="规划净空(m)：通行宽<2×此值的窄缝封闭（默认用 PASS_CLEAR_M=0.6；0=关闭门宽度判断，回退旧行为做 A/B）")
args = ap.parse_args()

if args.pass_clear >= 0:
    PASS_CLEAR_M = args.pass_clear
    PASS_CLEAR = PASS_CLEAR_M / VOXEL

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
GRID_N = int(50.0 / VOXEL)   # 500×500 感知格
# 2026-08-09 性能：dict → numpy 数组。gget/gget_plan/blocked 是全局最热路径
# （profile：gget 430 万次/call ~6s/4000 步）；数组化后 scan 批量标记用 fancy indexing，
# 省掉全部逐格 Python 循环。语义不变：越界读=UNKNOWN，越界写=丢弃（旧 dict 会存垃圾格）。
G = np.zeros((GRID_N, GRID_N), dtype=np.int8)      # live 感知栅格 [vx, vy]
SG = np.zeros((GRID_N, GRID_N), dtype=np.int8)     # static 背景（旧地图/已知墙，只读）
KNOWN_MAP_MODE = False  # True=阶段2（加载旧地图），规划叠加 SG
_wd = {}
OBS_SEEN = {}      # 雷达扫到的障碍格 key → 命中时扫描序号（纯感知障碍记忆，替代 obs_world 真值查询）
OBS_PTS_LAST = []  # 最近一次扫描的障碍命中点（估计系世界坐标）——ObstacleTracker 输入
HIT_CONFIRMED = np.zeros((GRID_N, GRID_N), dtype=bool)  # 激光**直接命中**过的格（经验墙感知确认凭据）
# 滚动局部障碍层（ROS costmap obstacle layer 思想，2026-08-12 审核整改新增）：
# 全局地图随累计位姿漂移错位（幻影墙），但**最近几秒的激光观测在局部是准的**
# （3s 窗口内 odom 增量误差 ~1-2%=0.1m 级）。执行层避障（前瞻/制动/DWA）以新鲜
# 观测为准，全局地图只做规划——碰撞不再依赖"地图多准"。
LOCAL_STAMP = np.zeros((GRID_N, GRID_N), dtype=np.int32)  # 格 → 最近直接命中（+0.2m 膨胀）的扫描序号
LOCAL_WIN = 60   # 60 次扫描 ≈ 3s 观测有效期
LOCAL_STAMP[:] = -10**6   # 零初始化会被当"新鲜"（0 > scan_step-60 在开局成立）→ 全图堵死起点瘫痪（实测）
_dog_est = [0.0, 0.0]   # 狗当前估计位（DWA 局部层 0.3m 自清用）
_scan_step = [0]   # scan 调用计数

def gget(vx, vy):
    if 0 <= vx < GRID_N and 0 <= vy < GRID_N:
        return G[vx, vy]
    return UNKNOWN

def gset(vx, vy, val):
    if not (0 <= vx < GRID_N and 0 <= vy < GRID_N):
        return
    if G[vx, vy] == val:
        return
    G[vx, vy] = val
    _pg_touch()
    if val == WALL:
        _wd.clear()

def gget_plan(vx, vy):
    """规划用叠加视图：static 的墙永远 WALL；live 障碍/自由优先；其余回退 static。
    2026-08-09 修复：探索模式也叠加 static——撞障碍写的安全圈在 static 层，
    旧版探索模式不读 static → 圈对规划不可见 → 路径反复穿同一障碍。"""
    if 0 <= vx < GRID_N and 0 <= vy < GRID_N:
        if _pg_dirty[0]:
            _pg_ensure()
        return PG[vx, vy]
    return UNKNOWN

# 物化规划视图（2026-08-09 性能）：gget_plan 230 万次/call 是 find_gates 主成本。
# scan/gset/SG 写入置脏，读取方（find_gates/wall_dist/_open_frontier）批量前一次性重建。
PG = np.zeros((GRID_N, GRID_N), dtype=np.int8)
_pg_dirty = [True]

def _pg_ensure():
    if _pg_dirty[0]:
        np.copyto(PG, SG)
        _m = G != UNKNOWN
        PG[_m] = G[_m]
        PG[SG == WALL] = WALL
        _pg_dirty[0] = False

def _pg_touch():
    _pg_dirty[0] = True
    _dist_dirty[0] = True

# ── 净空场（门宽度判断的数据核心）──
# DIST：各格到最近感知墙的精确欧氏距离（格，cv2 EDT，500×500 ~2ms）；
# PASS：规划可通行布尔场 = 非墙 且 净空≥PASS_CLEAR。WALL 格 DIST=0 天然排除。
# PG 任何变更经 _pg_touch 置脏；规划器首次使用时重建（不是每次 scan 都算——规划才付账）。
DIST = np.zeros((GRID_N, GRID_N), dtype=np.float32)
PASS = np.zeros((GRID_N, GRID_N), dtype=bool)
_dist_dirty = [True]
_NARROW_REJ = [0]   # 窄缝门拒绝计数（能透过激光但 <2×PASS_CLEAR 过不去的前沿门——诊断用）

def _dist_ensure():
    if not _dist_dirty[0]: return
    _pg_ensure()
    DIST[:] = cv2.distanceTransform((PG != WALL).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    PASS[:] = DIST >= PASS_CLEAR
    _dist_dirty[0] = False

def _count(val):
    """live 层某值格数（原 _cnt 计数器，数组化后按需 count_nonzero，500×500 ~0.2ms）"""
    return int(np.count_nonzero(G == val))

# ═══════════════════════════════════════════
# 地图加载 + 障碍物
# ═══════════════════════════════════════════

hf = np.array(Image.open(MAP))
_hf_bin = hf != ROAD_PIX   # 真值墙像素（向量化 scan 用）
SCAN_STEP = 0.025          # 沿射线采样步长 = hfield 1px（40px/m）：地图最小特征 1px，采样跨不过任何缝
_scan_k = np.arange(1, int(LIDAR_RANGE / SCAN_STEP) + 1, dtype=np.float64) * SCAN_STEP
_scan_k32 = _scan_k.astype(np.float32)   # 2026-08-09 性能：xs/ys float32（内存带宽减半，精度 1e-4px 级无感）

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

def gen_obstacles_progressive(seed):
    """渐进式障碍（主人方案）：每段直道 N 个 + 每段弯道 N 个。
    - 直道障碍：通道中心线 ±1.5m 随机，距两端转弯口 >= --obs-min-dist（默认 6m，不堵转弯）
    - 弯道障碍：放转弯口道路内（U 型转弯段），偏离转弯中心线但留足够通道
    返回世界坐标列表。种子固定可复现。
    """
    rng = random.Random(seed)
    out = []
    obs_r = 0.5          # OBS_R（定义在下方，这里用字面量避免初始化顺序问题）
    obs_clear = obs_r + SAFE_R
    # 每通道中心线 y（世界坐标）
    for ch in range(10):
        yc = 2.5 + ch * 5.0
        # ── 直道障碍：x 在 [min_dist, 50-min_dist] 内均匀随机 ──
        # 2026-08-08 收窄横向偏移 ±1.5→±1.0：墙边禁入区(0.1) + 碰撞(0.7) 后缝隙 0.2m 卡死；
        # 距墙 ≥1.5m → 缝隙 ≥0.7m 狗轻松穿（主人预期"各种障碍都能穿过去"）
        for i in range(args.obs_straight):
            for _try in range(200):
                ox = rng.uniform(args.obs_min_dist, 50.0 - args.obs_min_dist)
                oy = yc + rng.uniform(-1.0, 1.0)
                if sample_hf(ox, oy) == ROAD_PIX and not _obs_hits_wall(ox, oy, obs_r):
                    # 不与其他障碍重叠
                    if all(math.hypot(ox-a, oy-b) > obs_clear + 0.3 for a, b in out):
                        out.append((ox, oy)); break
        # ── 弯道障碍：放 U 型转弯段内部（偶通道右端 x>46，奇通道左端 x<4）──
        # 关键：转弯段 y 在分界墙两侧（通道中心 ±2.0m 靠近墙），避开直道末端转弯入口——
        # 放直道末端(x=44-46)会堵死转弯路径（狗正撞上）；放转弯段内部狗绕行空间大
        # 2026-08-08 偏移 ±2.2→±1.2（同上：禁入区+碰撞缝隙保障）
        for i in range(args.obs_turn):
            for _try in range(200):
                if ch % 2 == 0:
                    ox = rng.uniform(46.5, 49.0)     # 右端转弯段内部（x 大端，通道4→5 U 型弯中）
                    oy = yc + rng.uniform(-1.2, 1.2)
                else:
                    ox = rng.uniform(1.0, 3.5)       # 左端转弯段内部
                    oy = yc + rng.uniform(-1.2, 1.2)
                if sample_hf(ox, oy) == ROAD_PIX and not _obs_hits_wall(ox, oy, obs_r):
                    if all(math.hypot(ox-a, oy-b) > obs_clear + 0.3 for a, b in out):
                        out.append((ox, oy)); break
    return out

def gen_feature_obstacles(seed):
    """长直道特征障碍（主人 2026-08-12 指令）：每段直道按间隔 15m±5m 放固定障碍，
    每次新建图重新随机（种子=建图种子）。
    用途：长直道两侧墙 45m 无特征——激光 scan-matching 沿走廊方向退化不可观测，
    二维码每通道只有 1 个；这些障碍作为直道上的激光/视觉特征锚点，防长直道迷路。
    约束：x∈[7,43]（距两端转弯口 ≥6m，不堵转弯）、y=通道中心±1.2m、
    不嵌墙、互不重叠、离起点 (2.5,2.5) ≥4m。"""
    rng = random.Random(seed)
    out = []
    obs_r = 0.5
    for ch in range(10):
        yc = 2.5 + ch * 5.0
        x = 7.0 + rng.uniform(0.0, 2.0)   # 首锚点 7~9m（转弯口 6m 外）
        while x <= 43.0:
            for _try in range(100):
                ox = x + rng.uniform(-1.5, 1.5)
                oy = yc + rng.uniform(-1.2, 1.2)
                if not (6.0 <= ox <= 44.0):
                    continue
                if math.hypot(ox - 2.5, oy - 2.5) < 4.0:
                    continue
                if sample_hf(ox, oy) == ROAD_PIX and not _obs_hits_wall(ox, oy, obs_r):
                    if all(math.hypot(ox - a, oy - b) > obs_r * 2 + 0.3 for a, b in out):
                        out.append((ox, oy)); break
            x += 15.0 + rng.uniform(-5.0, 5.0)   # 间隔 15±5m
    return out

# ═══════════════════════════════════════════
# 巡逻障碍（主人指令 08-07：两直道+一弯道，1m/s，~50m 往返）
# ═══════════════════════════════════════════
PATROL_SPEED = 1.0        # m/s（主人指定）
patrol_paths = []         # 每条路径 = [(x,y),...] 折线点（世界坐标）
patrol_phase = []         # 每个障碍沿路径的累计弧长 (m)
patrol_dir = []           # 1=正向 -1=反向

def gen_patrol_path(ch_a, ch_b):
    """巡逻路径：通道A中点 → 右端U型弯 → 通道B中点（蛇形，~50m）。
    ch_a < ch_b，两通道相邻（如 2→3）。折线点含转弯口内弧，全程在路内。"""
    yc_a = 2.5 + ch_a * 5.0
    yc_b = 2.5 + ch_b * 5.0
    # U 型弯：右端 x=47 区域（y=15 分界墙开口在 x>45）
    return [(25.0, yc_a), (47.0, yc_a), (47.0, yc_b), (25.0, yc_b)]

def patrol_total_len(path):
    return sum(math.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
               for i in range(len(path)-1))

def patrol_pos(path, s):
    """沿折线路径走弧长 s（m），返回 (x, y)。s 超长则循环往返。"""
    n = len(path)
    total = patrol_total_len(path)
    # 往返：正向走 0~total，反向走 total~2total
    s = s % (2 * total)
    if s > total:
        s = 2 * total - s   # 反向
        pts = list(reversed(path))
    else:
        pts = path
    acc = 0.0
    for i in range(len(pts)-1):
        x1, y1 = pts[i]; x2, y2 = pts[i+1]
        seg = math.hypot(x2-x1, y2-y1)
        if acc + seg >= s:
            t = (s - acc) / seg if seg > 0 else 0
            return (x1 + (x2-x1)*t, y1 + (y2-y1)*t)
        acc += seg
    return pts[-1]

def init_patrol_obstacles():
    """生成巡逻障碍：--obs-patrol N 个，分布在不同通道对（2-3, 6-7, ...）。"""
    global patrol_paths, patrol_phase, patrol_dir, obs_world
    patrol_paths = []
    patrol_phase = []
    patrol_dir = []
    pairs = [(2,3), (6,7)]          # 已排好的通道对（相邻，U型弯在右端）
    for i in range(args.obs_patrol):
        ch_a, ch_b = pairs[i % len(pairs)]
        path = gen_patrol_path(ch_a, ch_b)
        patrol_paths.append(path)
        patrol_phase.append(0.0 + i * 15.0)   # 错开相位（间隔 15m），两障碍不同步
        patrol_dir.append(1)
        obs_world.append(path[0])    # 初始位置 = 通道A中点
    print(f"  [CFG] 巡逻障碍 {args.obs_patrol} 个 (1m/s, ~50m 蛇形往返)", flush=True)

def update_patrol(dt):
    """每 tick 更新巡逻障碍位置（沿路径匀速走）。dt = 物理步长。"""
    global obs_world
    for i in range(len(patrol_paths)):
        patrol_phase[i] += PATROL_SPEED * dt * patrol_dir[i]
        obs_world[i] = patrol_pos(patrol_paths[i], patrol_phase[i])

# ── 随机反弹障碍（主人指令 08-07：20m×5m 段, 1m/s, 20%/s 变向, 弹性反弹）──
random_field = None   # RandomObstacleField 实例

def init_random_obstacles():
    global random_field, obs_world
    chs = [int(c) for c in args.obs_random_ch.split(",")][:args.obs_random]
    random_field = RandomObstacleField(channels=chs, seed=FIXED_SEED)
    obs_world = random_field.positions
    print(f"  [CFG] 随机反弹障碍 {len(obs_world)} 个 @通道{chs} (1m/s, 20%/s变向)", flush=True)

def update_random(dt):
    global obs_world
    random_field.update(dt)
    obs_world = list(_obs_fixed_prefix) + random_field.positions   # 保留固定障碍前缀（混合场）

def _moving_obs_positions():
    """当前**移动**障碍位置列表（接触区豁免只豁免移动障碍）。
    2026-08-10 根修：旧版豁免对 obs_world 全部生效——混合场里狗贴近**固定**弯道障碍时，
    障碍命中格也被标 FREE → 地图把固定障碍从狗身边擦掉 → 狗开进障碍体内持续碰撞
    （实测 5 号弯 (46.9,24.8) 卡 7700 步 / collision 8000+）。固定障碍有安全圈+刹车
    机械兜底，不需要豁免；移动障碍会主动逼近狗，豁免保留。"""
    if random_field is not None:
        return random_field.positions
    if patrol_paths:
        return obs_world[:len(patrol_paths)]
    return []

_obs_fixed_prefix = []   # 混合场（--obs-mix）的固定弯道障碍；update_random 重写 obs_world 时保留

def init_mix_obstacles():
    """混合障碍场（主人指令 2026-08-09）：每拐弯处 1 个固定障碍（9 个 U 型弯）
    + 每直道 1 个随机反弹障碍（1m/s，20%/s 变向，撞墙反弹，
    活动范围 x∈[4.5,45]——不超出本通道直道段，不进转弯开口）。
    2026-08-10：弯道障碍改种子随机分布（mix_bend_positions，贴外侧墙一带但不再
    刚性一列；范围核算见 obstacles_random.py——不嵌墙、不堵转弯走线）。"""
    global random_field, obs_world, _obs_fixed_prefix
    bends = [(x, y) for x, y in mix_bend_positions(FIXED_SEED)
             if sample_hf(x, y) == ROAD_PIX and not _obs_hits_wall(x, y, 0.5)]
    _obs_fixed_prefix = bends
    chs = list(range(10))
    random_field = RandomObstacleField(channels=chs, seed=FIXED_SEED, x0=4.5, x1=45.0)
    obs_world = list(_obs_fixed_prefix) + random_field.positions
    print(f"  [CFG] 混合障碍场：弯道固定 {len(bends)} 个(种子随机贴外侧一带) + 直道移动 {len(chs)} 个 (1m/s, x∈[4.5,45])", flush=True)
    print(f"  [CFG] 弯道障碍: {[(round(x,1), round(y,1)) for x, y in bends]}", flush=True)

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

if args.no_obs:
    obs_world = []
elif args.obs_mix > 0:
    obs_world = []
    init_mix_obstacles()             # 混合场：弯道固定 + 直道移动（random_field 驱动）
elif args.obs_feature > 0:
    obs_world = gen_feature_obstacles(FIXED_SEED)   # 长直道特征障碍（纯固定，15±5m 间隔）
    print(f"  [CFG] 直道特征障碍 {len(obs_world)} 个（间隔15±5m，种子随机）: "
          f"{[(round(x,1), round(y,1)) for x, y in obs_world]}", flush=True)
elif args.obs_random > 0:
    obs_world = []
    init_random_obstacles()          # 随机反弹障碍（obs_world[0:N] 是随机障碍）
elif args.obs_patrol > 0:
    obs_world = []
    init_patrol_obstacles()          # 巡逻障碍独立于固定障碍（obs_world[0:N] 是巡逻）
elif args.obs_straight > 0 or args.obs_turn > 0:
    obs_world = gen_obstacles_progressive(FIXED_SEED)
    print(f"  [CFG] 渐进障碍：直道{args.obs_straight}/段 + 弯道{args.obs_turn}/段 → 共 {len(obs_world)} 个", flush=True)
else:
    obs_world = gen_obstacles(FIXED_SEED)
OBS_R = 0.5; OBS_CLEAR = OBS_R + SAFE_R

# 障碍格预计算（性能：is_obstacle_world 从 20 次 hypot 循环 → set 查 O(1)）
# profile 显示 scan 占 99.5% 时间，其中 is_obstacle_world 663 万次 × 20 障碍 hypot = 1.15 亿次
OBS_CELLS = set()
OBS_CELLS_VALID = False

def rebuild_obs_cells():
    """预计算固定障碍 0.7m 范围内的所有格（set 查表）"""
    global OBS_CELLS, OBS_CELLS_VALID
    OBS_CELLS = set()
    r_grid = int(math.ceil(OBS_CLEAR / VOXEL)) + 1
    for ox, oy in obs_world:
        cx, cy = int(round(ox/VOXEL)), int(round(oy/VOXEL))
        for dy in range(-r_grid, r_grid+1):
            for dx in range(-r_grid, r_grid+1):
                if dx*dx + dy*dy <= r_grid*r_grid:
                    OBS_CELLS.add((cx+dx, cy+dy))
    OBS_CELLS_VALID = True

def is_obstacle_world(wx, wy, inflation=0.0):
    if sample_hf(wx, wy) != ROAD_PIX: return True
    # 移动障碍（巡逻/随机）：数量少（≤4）直接循环
    if random_field is not None or patrol_paths:
        for ox, oy in obs_world:
            if math.hypot(wx-ox, wy-oy) < OBS_CLEAR + inflation: return True
        return False
    # 固定障碍：精确圆判定（0.7m 物理接触）。格查表 r_grid=8+格量化把 0.9m 安全通过
    # 误判成碰撞（实测 (8.9,8.06) 距障碍中心 0.90m 被记碰撞）——调用方都是每步一次，开销可忽略
    for ox, oy in obs_world:
        if math.hypot(wx-ox, wy-oy) < OBS_CLEAR + inflation: return True
    return False

# ═══════════════════════════════════════════
# 扫描 + 碰撞检测
# ═══════════════════════════════════════════

MATCH_RANGE = 8.0   # scan-matching 只用 ≤8m 近距命中点（远处位姿误差杠杆大）
_in_init_scan = [False]   # 初始自旋期间禁匹配（地图空 + 每帧转 18° 超出搜索窗）
_STORM = [False]   # 保留位：风暴隔离机制已实测否决并移除（冻结会切断射线清除自救通道，
                   # 见踩坑文档 16.6）——恒 False，代码里留的隔离门全部惰性

def scan(cast_x, cast_y, cast_yaw, est_x=None, est_y=None, est_yaw=None):
    """前方 FOV 扇形扫描（相对狗 yaw）——无特权：只感知狗能看到的物理真实。
    多线（--lidar-lines N）：不同俯仰角的线，2D 导航取最近（任一线命中=检测到，冗余确认）。

    2026-08-12 物理/估计坐标系分离（代码审核整改）：
    - 投射（物理层）用**真值位姿** (cast_*)：激光从狗的真实位置发出，打的是真实世界
      ——真实激光雷达不知道自己位姿错了，点云相对机身是精确的；
    - 写图（认知层）用**估计位姿** (est_*)：狗把点云按自己以为的位姿放进地图，
      位姿误差 → 地图错位——这正是真实 SLAM 要面对的问题；
    - 写图前做 scan-to-map 匹配（ScanMatcher，ROS Cartographer 思想），
      用当前帧近距墙命中点对已建地图求位姿修正量，连续压住里程计漂移。
    --odom 0 调试模式：est=cast，行为与旧版一致。

    2026-08-09 重写（修"幽灵门"泄漏）：
    - 旧版沿射线 0.1m 步进逐点 is_obstacle_world——会跨过 <0.1m 的真缝
      （U 型弯 45° 斜边端点与墙端之间的亚格子缝隙），把墙另一侧标 FREE
      → 产生不可达的幽灵门，狗反复去穿缝卡死（实测 bounce 412 次卡 202s）。
    - 现改像素级步进 SCAN_STEP=0.025m（=hfield 40px/m 的 1px）：地图最小
      特征 1px，采样跨不过任何缝。numpy 向量化（每射线 1200 步一次性算），
      命中后批量去重写 grid，成本与旧版相当。
    """
    if est_x is None:
        est_x, est_y, est_yaw = cast_x, cast_y, cast_yaw
    fov_rad = math.radians(args.lidar_fov)
    n_lines = max(1, args.lidar_lines)
    _scan_step[0] += 1
    # OBS_SEEN 过期清理（每 10 次扫描清一次即可，200 次 ≈33s 消退；射线清除也会照掉）
    if OBS_SEEN and _scan_step[0] % 10 == 0:
        for _k in [_k for _k, _v in OBS_SEEN.items() if _scan_step[0] - _v > 200]:
            del OBS_SEEN[_k]
    OBS_PTS_LAST.clear()
    # 2D 简化：多线 = 同 FOV 的多条水平线（不同俯仰角扫不同高度，2D 判定等价；
    # 真实多线雷达主要价值是 3D 感知/鲁棒性，导航 2D 下取最近点）
    for _line in range(n_lines):
        rel = np.linspace(-fov_rad / 2, fov_rad / 2, LIDAR_RAYS)
        # ── 物理投射：真值位姿（激光打的是真实世界）──
        angles = cast_yaw + rel
        cos_a = np.cos(angles).astype(np.float32)
        sin_a = np.sin(angles).astype(np.float32)
        xs = cos_a[:, None] * _scan_k32[None, :]     # (R,S) float32
        ys = sin_a[:, None] * _scan_k32[None, :]
        xs += np.float32(cast_x); ys += np.float32(cast_y)   # 世界坐标（真值系）
        px = (xs * PIX_PER_M).astype(np.int32)
        py = HF_RES - 1 - (ys * PIX_PER_M).astype(np.int32)
        inb = (px >= 0) & (px < HF_RES) & (py >= 0) & (py < HF_RES)
        pxc = np.clip(px, 0, HF_RES - 1)
        pyc = np.clip(py, 0, HF_RES - 1)
        wall = (_hf_bin[pyc, pxc]) & inb               # 真值墙像素
        obs_hit = None
        obs_mov_hit = None
        if len(obs_world):
            # 障碍命中用 1/4 分辨率（0.1m 栅格级）计算再扩回：障碍盘 0.7m，
            # 0.1m 采样必命中——全分辨率是 4 倍浪费（实测 20 障碍广播是 scan 最大热点）
            xs4 = xs[:, ::4]; ys4 = ys[:, ::4]
            obs4 = np.zeros(xs4.shape, dtype=bool)
            for ox, oy in obs_world:                    # 物理障碍体（雷达可见）
                obs4 |= (xs4 - ox) ** 2 + (ys4 - oy) ** 2 <= OBS_CLEAR ** 2
            obs_hit = np.repeat(obs4, 4, axis=1)[:, :wall.shape[1]]
            wall |= obs_hit
            # 移动障碍单独一张掩码（接触区豁免只豁免移动障碍，固定障碍贴近也必须保持 WALL）。
            # 注：动/静区分是**物理层标注**（与 hfield 真值模拟激光回波同层），
            # 决策层从不读它——决策用的障碍运动信息全部来自 ObstacleTracker 感知估计。
            _mov = _moving_obs_positions()
            if _mov:
                obs4m = np.zeros(xs4.shape, dtype=bool)
                for ox, oy in _mov:
                    obs4m |= (xs4 - ox) ** 2 + (ys4 - oy) ** 2 <= OBS_CLEAR ** 2
                obs_mov_hit = np.repeat(obs4m, 4, axis=1)[:, :wall.shape[1]]
        hit_any = wall | (~inb)                         # 出界=射线终止（不标 WALL）
        R, S = hit_any.shape
        _hr = np.arange(R)
        first = np.argmax(hit_any, axis=1)              # 每条射线首个终止下标
        has = hit_any[_hr, first]
        stop = np.where(has, first, S)                  # 终止步；无命中=S（全程 FREE）
        hi = np.minimum(first, S - 1)
        _hit_inb = has & inb[_hr, hi]   # 真命中（排除出界终止）
        _rd = first.astype(np.float32) * np.float32(SCAN_STEP)    # 命中距 (R,)
        if obs_hit is not None:
            _wall_hit0 = _hit_inb & ~obs_hit[_hr, hi]   # 纯墙命中（匹配专用：障碍会动，不能当地图锚）
        else:
            _wall_hit0 = _hit_inb
        if odom is not None:
            # 里程计模式：只写近距墙标记（远处命中随位姿误差错位涂抹地图，实测 1m 漂移
            # 时墙根陷阱卡死）——射线仍被远墙终止（FREE 标记不穿墙），只是远墙格留 UNKNOWN，
            # 狗走近再标（那时位姿已被 scan-matching/二维码修正）
            _real_hit = _hit_inb & (_rd <= WALL_MAP_RANGE)
        else:
            _real_hit = _hit_inb

        # ── scan-to-map 匹配（激光里程计修正，写图前）──
        # 参照 = 全局感知墙掩码（PG）。风暴期/初始自旋期不匹配（位姿不可信时修正=加注正反馈）。
        # 匹配点只用纯墙命中（障碍会动/被写过位姿误差，不当锚）。
        _quarantine = _STORM[0] and odom is not None   # 风暴隔离：全局写图/匹配/绝对修正冻结
        if (odom is not None and matcher is not None and not _in_init_scan[0] and not _quarantine):
            _mp = _wall_hit0 & (_rd <= MATCH_RANGE)
            if _mp.any():
                _ma = est_yaw + rel[_mp]
                _mpts = np.stack([est_x + _rd[_mp] * np.cos(_ma),
                                  est_y + _rd[_mp] * np.sin(_ma)], axis=1)
                _pg_ensure()
                _wdil = cv2.dilate((PG == WALL).astype(np.uint8),
                                   np.ones((3, 3), np.uint8))
                _corr = matcher.match(_mpts, est_x, est_y, _wdil)
                if _corr is not None:
                    _dxm, _dym, _dam, _msc = _corr
                    odom.x += _dxm; odom.y += _dym
                    odom.yaw = (odom.yaw + _dam + math.pi) % (2 * math.pi) - math.pi
                    est_x += _dxm; est_y += _dym; est_yaw = odom.yaw

        # ── 写图坐标系：估计系（狗以为自己在哪里，点云就放在哪里）──
        if odom is not None:
            angles_e = est_yaw + rel
            cos_e = np.cos(angles_e).astype(np.float32)
            sin_e = np.sin(angles_e).astype(np.float32)
            xe = cos_e[:, None] * _scan_k32[None, :]
            ye = sin_e[:, None] * _scan_k32[None, :]
            xe += np.float32(est_x); ye += np.float32(est_y)
            cx = np.floor(xe / VOXEL).astype(np.int32)   # 世界→感知格（floor 语义，负坐标也正确越界）
            cy = np.floor(ye / VOXEL).astype(np.int32)
            _hx = xe[_hr, hi]                     # 命中点世界坐标（估计系，掠射填充用）
            _hy = ye[_hr, hi]
        else:
            cx = px >> 2    # 0.1m 感知格 = 4 个 hfield 像素（非负区等价 int(x/VOXEL)，越界格下面 mask 掉）
            cy = (HF_RES - 1 - py) >> 2   # py 是图像行（row0=y=50m 顶部，y 翻转）——必须翻回世界格！
            _hx = px[_hr, hi].astype(np.float32) / PIX_PER_M          # 命中点世界坐标 (R,)
            _hy = (HF_RES - 1 - py[_hr, hi]).astype(np.float32) / PIX_PER_M
        free_mask = np.arange(S)[None, :] < stop[:, None]
        # 2026-08-09 性能：numpy 数组批量标记（原 dict 逐格 gset 循环 + np.unique 是 scan 一半耗时）。
        # 先写 FREE（射线清除：穿过格强制 FREE，旧障碍被"照"掉）再写 WALL → 同格命中 WALL 优先。
        # 风暴隔离：全局地图写入全部暂停（防不可信位姿涂抹），只留局部层/障碍记忆保执行安全
        fm = free_mask & (cx >= 0) & (cx < GRID_N) & (cy >= 0) & (cy < GRID_N)
        if not _quarantine:
            G[cx[fm], cy[fm]] = FREE
            _pg_touch()
        # 墙厚先验（本地图墙 ≥0.2m=2格）：命中点再往里 0.1m/0.2m 深处也是墙。
        # 旧版只标命中格 → 掠射射线把墙面格又清成 FREE、墙背行留 UNKNOWN →
        # 墙脸上长出"伪前沿门"（门后是墙体，狗去确认=贴墙 bounce，实测之字打转每条墙 8 段×6-8s）
        # 接触区豁免（仅移动障碍场）：障碍命中格距狗 <0.5m 时**强制标 FREE**（不写 WALL）——
        # 标 WALL：脚边格进 keepout → 狗自己的格 blocked → 永冻 → 障碍继续逼近持续接触
        # （实测 collision 800+）；留 UNKNOWN：贴脸障碍周围冒未知泡 → 幽灵门把狗吸回去（实测游荡）。
        # 标 FREE 让狗始终能拉开距离；移动障碍有 DWA 运动预测兜底（r+0.3 排斥）不会真撞。
        # 固定障碍场不豁免（无 DWA，靠标记+安全圈刹停）
        # 2026-08-10：豁免掩码改为**仅移动障碍**（obs_mov_hit）——混合场里固定弯道障碍
        # 贴狗时也被豁免擦掉 → 狗开进障碍体内持续碰撞 8000+（5 号弯卡 7700 步实测）
        _close_obs = None
        _obs_hit_r0 = None
        if obs_mov_hit is not None:
            _obs_hit_r = obs_hit[_hr, hi] & _real_hit
            _mov_hit_r = obs_mov_hit[_hr, hi] & _real_hit
            _close_obs = _mov_hit_r & (_rd < 0.5)
            _mark_hit = (_real_hit & ~_obs_hit_r) | (_obs_hit_r & ~_close_obs)
            _obs_hit_r0 = _obs_hit_r & ~_close_obs   # 局部层用：剔除贴脸豁免格
        else:
            _mark_hit = _real_hit
            if obs_hit is not None:
                _obs_hit_r0 = obs_hit[_hr, hi] & _real_hit
        # 直接命中格登记（HIT_CONFIRMED）：经验墙写入的感知确认凭据——
        # 只认激光**直接打中**的格，墙厚先验/掠射填充推断的格不算（防假墙自指扩散）
        _dhx = cx[_hr, hi][_mark_hit]; _dhy = cy[_hr, hi][_mark_hit]
        _dok = (_dhx >= 0) & (_dhx < GRID_N) & (_dhy >= 0) & (_dhy < GRID_N)
        if _dok.any():
            if not _quarantine:
                HIT_CONFIRMED[_dhx[_dok], _dhy[_dok]] = True
        # 滚动局部障碍层（风暴期照常——执行层逃生靠它；**只在有障碍体的场启用**——
        # 纯墙场地图管线已被充分验证，加戳只增刹车噪声）：
        # ① 全部直接命中格（墙+障碍）**无膨胀**打戳——距离沿视线方向是精确的（位姿误差
        #    只带来横向偏移），正前方的墙/障碍永远被新鲜直读兜底（实测无墙戳时地图漂移
        #    会把狗嵌进真墙：地图自洽地错，blocked 永不触发，contype=0 物理不设防）。
        # ② 障碍命中额外 +0.4m 膨胀（命中面 0.5m 外还有狗半径+余量）。
        # 接触区豁免格（贴脸移动障碍）不打（打狗脚上=永冻）。墙戳无膨胀故走廊不变窄。
        # 注：所有场启用（纯墙场也要——漂移嵌墙没有障碍也一样发生）。
        # ③ 狗身 0.35m 自清区：膨胀戳可能盖住狗当前格（障碍逼近到 0.5-0.9m 时），
        #    把移动护卫在狗脚下锁死（实测混合场冻结：d_fwd=4.0 明明畅通但一步不动
        #    bounce 刷到 1100+）——打戳时跳过狗身边格（该处由接触区豁免+逃逸逻辑接管）。
        _evx2, _evy2 = int(est_x / VOXEL), int(est_y / VOXEL)
        if _dok.any():
            _sx = _dhx[_dok]; _sy = _dhy[_dok]
            _sf = (_sx - _evx2) ** 2 + (_sy - _evy2) ** 2 > 12   # >0.35m（格平方）
            if _sf.any():
                LOCAL_STAMP[_sx[_sf], _sy[_sf]] = _scan_step[0]
        if _obs_hit_r0 is not None and _obs_hit_r0.any():
            _ohx = cx[_hr, hi][_obs_hit_r0]; _ohy = cy[_hr, hi][_obs_hit_r0]
            _oko = (_ohx >= 0) & (_ohx < GRID_N) & (_ohy >= 0) & (_ohy < GRID_N)
            if _oko.any():
                _lh = np.zeros((GRID_N, GRID_N), dtype=np.uint8)
                _lh[_ohx[_oko], _ohy[_oko]] = 1
                _lh = cv2.dilate(_lh, np.ones((9, 9), np.uint8))
                # 自清区清零（0.35m ≈ ±3 格；数组 [vx, vy] 索引 x 在前）
                _lh[max(0, _evx2-3):_evx2+4, max(0, _evy2-3):_evy2+4] = 0
                LOCAL_STAMP[_lh.astype(bool)] = _scan_step[0]
        _wc = [(cx[_hr, hi], cy[_hr, hi])]
        for _extra in (4, 8):   # +0.1m、+0.2m 深处（4 采样=1 格）
            _di = np.minimum(first + _extra, S - 1)
            _wc.append((cx[_hr, _di], cy[_hr, _di]))
        wcx = np.concatenate([_w[0][_mark_hit] for _w in _wc])
        wcy = np.concatenate([_w[1][_mark_hit] for _w in _wc])
        _ok = (wcx >= 0) & (wcx < GRID_N) & (wcy >= 0) & (wcy < GRID_N)
        wcx = wcx[_ok]; wcy = wcy[_ok]
        if wcx.size and not _quarantine:
            if (G[wcx, wcy] != WALL).any():   # 有新墙格才清 wall_dist 缓存（原 gset 语义）
                _wd.clear()
            G[wcx, wcy] = WALL
        if _close_obs is not None and _close_obs.any() and not _quarantine:
            # 接触区格主动标 FREE（在 WALL 写之后，防止同格被墙厚先验覆盖）
            _ccx = cx[_hr, hi][_close_obs]
            _ccy = cy[_hr, hi][_close_obs]
            _cok = (_ccx >= 0) & (_ccx < GRID_N) & (_ccy >= 0) & (_ccy < GRID_N)
            if _cok.any():
                G[_ccx[_cok], _ccy[_cok]] = FREE
                _pg_touch()
        # 掠射填充（2026-08-09 v2）：相邻射线对命中同一平面墙 → 命中点连线补 WALL。
        # 0.5° 射线间隔在掠射墙面上留数米未知缝（尤其 30m 量程边界处，命中点间距 4-5m），
        # 墙脸前长出"幽灵前沿门"（如底墙脸 (32.5,0.7)），狗被吸到墙脸反复 STOP/bounce
        # （实测 bounce 270 次的主头，每次~2s）。
        # 判定 = 折中条件初筛 + **自由空间反证**（关键）：弦内部采样格若已被射线穿过标 FREE
        # （>25%），说明弦横穿开阔空间（开口/拐角/不同墙）→ 拒填；同墙掠射弦贴着墙皮，
        # 无任何射线能穿过 → 填充。障碍命中对不填（防假墙封死障碍↔墙的缝）。
        if obs_hit is not None:
            _wall_hit = _real_hit & ~obs_hit[_hr, hi]
        else:
            _wall_hit = _real_hit
        _vp = _wall_hit[:-1] & _wall_hit[1:]
        if _vp.any() and not _quarantine:
            _x0, _x1 = _hx[:-1], _hx[1:]
            _y0, _y1 = _hy[:-1], _hy[1:]
            _ch = np.hypot(_x1-_x0, _y1-_y0)                     # 相邻命中点弦长
            _dr = np.abs(_rd[1:] - _rd[:-1])                     # 命中距差
            _sel = _vp & (_ch > 0.3) & (_ch < 6.0) & (_dr < 5.5)
            for _i in np.nonzero(_sel)[0]:
                _n = max(3, int(_ch[_i] / 0.05) + 1)
                _t = np.linspace(0.08, 0.92, _n)                 # 内部采样（端点贴命中格，擦边容忍）
                _scx = ((_x0[_i] + (_x1[_i]-_x0[_i])*_t) / VOXEL).astype(np.int32)
                _scy = ((_y0[_i] + (_y1[_i]-_y0[_i])*_t) / VOXEL).astype(np.int32)
                if not ((_scx >= 0).all() and (_scx < GRID_N).all()
                        and (_scy >= 0).all() and (_scy < GRID_N).all()):
                    continue
                if (G[_scx, _scy] == FREE).mean() > 0.25:        # 自由空间反证：横穿开阔 → 拒填
                    continue
                if (G[_scx, _scy] != WALL).any():
                    _wd.clear()
                G[_scx, _scy] = WALL
                _pg_touch()
        # 障碍命中格登记（纯感知）：执行层"附近有无障碍"判定用——替代 obs_world 真值查询
        if obs_hit is not None:
            _oh = obs_hit[_hr, hi] & _real_hit
            for _r in np.nonzero(_oh)[0]:
                OBS_SEEN[int(cx[_r, hi[_r]]) * 4096 + int(cy[_r, hi[_r]])] = _scan_step[0]
                OBS_PTS_LAST.append((float(_hx[_r]), float(_hy[_r])))   # 跟踪器输入（估计系）
    # 匹配修正过的估计位姿回传（主循环刷新决策位姿）
    if odom is not None:
        return (est_x, est_y, est_yaw)
    return None

# ── 墙边禁入区（主人指令 2026-08-08：墙边 10cm 禁止进入）──
# ⚠️ 不许作弊：禁入区基于**雷达感知**的墙（grid WALL），不是真值地图 track_clean！
# 现实狗看不到真实坐标，只能靠激光雷达扫到的墙。没扫到的墙不禁入（敢走，撞了才知道=现实）。
# 2026-08-09 调优（主人指令）：两侧只留 10cm 防撞宽度，其余额外不可通行距离全取消——
# 门过滤/开阔前沿/DWA 附加余量同步收到 10cm 级。
# 注：keepout 本体维持 0.05m/4邻域（实测 0.10m+对角禁入让狗贴墙 STOP 翻倍——
# 宽路通过性的瓶颈在规划层附加余量，不在执行层 keepout）
KEEP_M = 0.05   # 距感知墙禁入距离 (m，表面距离)
WALL_MAP_RANGE = 10.0   # 里程计模式只标记 10m 内的墙（远处命中随位姿误差涂抹地图；射线仍被远墙终止）

# 距感知墙表面 < KEEP_M 的格偏移集合（按连续距离算）：
# KEEP_M=0.05 → 自身+4邻域（对角 d=1.41→表面 0.091m 不算贴墙）
_KO_OFF = tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                if (math.hypot(dx, dy) - 0.5) * VOXEL < KEEP_M + 1e-6)

def in_keepout(vx, vy):
    """距感知墙表面 < KEEP_M → True（禁入）。
    2026-08-09 性能：数组直读（plan==WALL ⟺ G==WALL 或 SG==WALL），
    原 3×3 gget_plan 循环是 blocked 的主成本（32.8 万次/call → 5.6s/4000 步）。"""
    for dx, dy in _KO_OFF:
        nx = vx + dx; ny = vy + dy
        if 0 <= nx < GRID_N and 0 <= ny < GRID_N and (G[nx, ny] == WALL or SG[nx, ny] == WALL):
            return True
    return False

def blocked(wx, wy, inflation=0.0):
    # 越界保护：赛道外直接视为 blocked（防穿墙跑出地图）
    if not (0.0 <= wx <= 50.0 and 0.0 <= wy <= 50.0):
        return True
    # 墙边禁入区（感知版）+ 中心格墙判定合并：_KO_OFF 偏移集（KEEP_M=0.10 → 自身+8邻域）任一感知墙即 blocked
    # ⚠️ 不用 is_obstacle_world（真值墙+真值障碍）——不许作弊：
    #   狗只知道雷达前方 180° 扫到的物理真实，墙后/死角/未扫描区域 = 不知道
    vx = int(wx / VOXEL); vy = int(wy / VOXEL)
    for dx, dy in _KO_OFF:
        nx = vx + dx; ny = vy + dy
        if 0 <= nx < GRID_N and 0 <= ny < GRID_N and (G[nx, ny] == WALL or SG[nx, ny] == WALL):
            return True
    return False

def blocked_batch(pts):
    """批量 blocked（DWA 快路径用）：pts (N,2) 世界坐标 → bool (N,)。
    与 blocked() 同语义：越界 True；自身+4邻域（keepout）任一感知墙 True。
    边界处邻格 clip 到自身——与标量版"越界邻格=非墙"效果等价（自身格已被 (0,0) 覆盖）。
    2026-08-12：叠加滚动局部层（新鲜直读命中）——DWA 轨迹模拟看到**当前真实**障碍，
    不被全局地图漂移误导；狗位 0.3m 内局部层不算（轨迹起点在狗身上，防全碰撞假死锁）。"""
    x = pts[:, 0]; y = pts[:, 1]
    oob = (x < 0.0) | (x > 50.0) | (y > 50.0) | (y < 0.0)
    vx = np.clip((x / VOXEL).astype(np.int32), 0, GRID_N - 1)
    vy = np.clip((y / VOXEL).astype(np.int32), 0, GRID_N - 1)
    hit = np.zeros(len(pts), dtype=bool)
    for dx, dy in _KO_OFF:
        nx = np.clip(vx + dx, 0, GRID_N - 1)
        ny = np.clip(vy + dy, 0, GRID_N - 1)
        hit |= (G[nx, ny] == WALL) | (SG[nx, ny] == WALL)
    _fresh = LOCAL_STAMP[vx, vy] > _scan_step[0] - LOCAL_WIN
    _dd2 = (x - _dog_est[0]) ** 2 + (y - _dog_est[1]) ** 2
    hit |= _fresh & (_dd2 > 0.09)
    return hit | oob

# ── 三级跳A* ──



# 跨步尺寸（主人指令 2026-08-06：雷达 30m 后跨步可扩展减少计算）
JUMP_1M = 20   # 10→20：空旷区 1m→2m 一跳（离墙≥2m 时）
JUMP_03 = 6    # 3→6：离墙≥0.6m 时
JUMP_NEAR = 1  # 贴墙保持 0.1m

def jump_steps(vx, vy, dx, dy, tfx=None, tfy=None):
    wd = wall_dist(vx, vy)
    if wd >= JUMP_1M:   max_jump = JUMP_1M
    elif wd >= JUMP_03: max_jump = JUMP_03
    else:               max_jump = JUMP_NEAR
    for step in range(1, max_jump + 1):
        nx, ny = vx + dx*step, vy + dy*step
        # 地图边界外视为不可通行（与 blocked() 一致）——否则 BFS 会探索地图外的虚空
        if not (0 <= nx < 500 and 0 <= ny < 500):
            return step - 1
        if not plan_clear(nx, ny):   # 门宽度判断：低净空格（窄缝内部）不可规划穿越
            return step - 1
        # 目标对齐吸附（2026-08-10 修跳步粒度奇偶 bug）：开放区跳步恒取上限（2m），
        # 目标格不在跳步格点上永远落不上去 → A* 扩满 25 万格返回 None → 终点直奔瘫痪
        # （实测：看到球后 beeline 全失败，狗在门口风暴到超时）。走查跨过目标行/列时
        # 收步落在目标坐标上——先落目标行/列，再沿行/列落到目标格，A* 可精确到达。
        if tfx is not None and (nx == tfx or ny == tfy):
            return step
    return max_jump

_MAN5 = np.array([[abs(i-2)+abs(j-2) for j in range(5)] for i in range(5)], dtype=np.int8)

def wall_dist(vx, vy):
    """到最近感知墙的距离（格数）。热路径：局部 5×5 快速扫描，开阔地返回截断值 21。
    截断语义安全：jump_steps 大步跳有 traversable 逐格兜底（不会跳过墙）；
    惩罚 WALL_BUFFER_CELLS(20)-wd 在 wd≥20 时为 0；find_gates 过滤只需 >ROBOT_R-1(1)。
    （原版 WALL_SCAN_RADIUS=20 扫 41×41=1681 格——find_gates BFS 大量在路中间调用 → 爆炸）
    2026-08-09 性能：5×5 gget_plan 循环 → PG 切片一次判定（曼哈顿距离语义不变）。"""
    key = (vx, vy)
    if key in _wd: return _wd[key]
    if _pg_dirty[0]: _pg_ensure()
    if 2 <= vx < GRID_N-2 and 2 <= vy < GRID_N-2:
        w = PG[vx-2:vx+3, vy-2:vy+3] == WALL
        if w.any():
            best = int(_MAN5[w].min())
            _wd[key] = best
            return best
    else:   # 贴地图边（罕见）：保守逐格回退
        best = 999
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if gget_plan(vx+dx, vy+dy) == WALL:
                    d = abs(dx)+abs(dy)
                    if d < best: best = d
        if best < 999:
            _wd[key] = best
            return best
    _wd[key] = JUMP_1M + 1   # 截断：>JUMP_1M 即"开阔"（大步跳上限）
    return JUMP_1M + 1

def walkable(vx, vy):
    # 距感知墙 > KEEP_M 对应格数：A* 规划避开墙边禁入区（与 blocked 的 in_keepout 一致）
    return gget_plan(vx, vy) == FREE and wall_dist(vx, vy) > int(math.ceil(KEEP_M / VOXEL))

def traversable(vx, vy):
    """探索可通行：FREE 或 UNKNOWN 都行（WALL 不行）。
    核心：frontier 探索允许走向未知——A* 规划激进，执行层(Mover)实时避障兜底。
    真实机器人也是这么干的：未知区域可通行，撞到才知道有墙。
    """
    return gget_plan(vx, vy) != WALL

def plan_clear(vx, vy):
    """规划可通行（门宽度判断核心）：非墙 且 到最近感知墙净空 ≥PASS_CLEAR。
    UNKNOWN 乐观——只对**已知**墙量净空，未知区域仍可规划穿越（探索语义不变）；
    窄缝（宽 <2×PASS_CLEAR=1.2m）内没有任何格满足 → 对规划自动封闭。
    PASS_CLEAR=0 时退化为 traversable（A/B 开关）。"""
    if PASS_CLEAR <= 0:
        return gget_plan(vx, vy) != WALL
    if _dist_dirty[0]: _dist_ensure()
    return 0 <= vx < GRID_N and 0 <= vy < GRID_N and PASS[vx, vy]

def clear_ok(vx, vy):
    """门格净空判定：PASS_CLEAR>0 用净空场（≥0.6m，含旧 wall_dist>ROBOT_R=0.2m 语义）；
    =0 回退旧过滤（A/B 对照）。"""
    if PASS_CLEAR <= 0:
        return wall_dist(vx, vy) > ROBOT_R
    if _dist_dirty[0]: _dist_ensure()
    return 0 <= vx < GRID_N and 0 <= vy < GRID_N and PASS[vx, vy]

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

def _nearest_walkable(vx, vy, max_r=8, min_dist=0):
    """墙边脱困：从 (vx,vy) BFS 找最近的 walkable 格（机器人贴墙时跳不出 jump_steps）
    放宽到 traversable：机器人位置贴墙边但前方可能是未知区域，也能起步
    min_dist>0：要求距感知墙 ≥min_dist 格（HPA 起点需要 dist≥ROBOT_DIA=4，否则细层 A* 邻居全禁行）
    2026-08-10 门宽度判断：优先找满足规划净空（PASS）的格；找不到回退 traversable——
    狗被瞬态障碍标记围进低净空区时保证永远能起步（旧行为兜底，防起步死锁）。
    min_dist 判定改用 DIST 精确场（旧 wall_dist 5×5 窗口只能验证 ≤2 格，≥3 靠截断值 21 蒙混）。"""
    def _ok(nx, ny, strict):
        if strict:
            if _dist_dirty[0]: _dist_ensure()
            if not (0 <= nx < GRID_N and 0 <= ny < GRID_N) or not PASS[nx, ny]:
                return False
        elif not traversable(nx, ny):
            return False
        if min_dist == 0:
            return True
        if PASS_CLEAR > 0:
            if _dist_dirty[0]: _dist_ensure()
            return 0 <= nx < GRID_N and 0 <= ny < GRID_N and DIST[nx, ny] >= min_dist
        return wall_dist(nx, ny) >= min_dist
    for strict in ([True, False] if PASS_CLEAR > 0 else [False]):
        if _ok(vx, vy, strict):
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
                if _ok(nx, ny, strict):
                    return nx, ny
                q.append((nx, ny, dist+1))
    return None

def _open_frontier(fx, fy):
    """开阔前沿判定：格子的 UNKNOWN 邻居中，至少有一个其 5x5 邻域无墙。
    墙缝/墙根/角落后的未知射线永远扫不到（被墙挡）→ 不是门，是陷阱。
    5x5（0.5m 净空）：墙背面/墙根的未知格总挨着已标记的墙面格 → 杀掉。
    （2026-08-09 A/B：收窄到 3x3 后墙根伪门回归，混合场 bounce 93→1830，回退 5x5）
    2026-08-09 性能：PG 直读（原 9×gget_plan + 25×any 生成器是 find_gates 热点）。"""
    if _pg_dirty[0]: _pg_ensure()
    for _dy in (-1, 0, 1):
        for _dx in (-1, 0, 1):
            nx, ny = fx+_dx, fy+_dy
            if 0 <= nx < GRID_N and 0 <= ny < GRID_N and PG[nx, ny] == UNKNOWN:
                x0 = max(0, nx-2); y0 = max(0, ny-2)
                if not (PG[x0:nx+3, y0:ny+3] == WALL).any():
                    return True
    return False

def find_gates(fvx, fvy):
    # 起点放宽：机器人物理位置可能是 UNKNOWN（刚起步未扫描）或贴墙边，
    # 但已实际站在那，必须允许寻路，否则 find_gates 返回空 → 主循环死循环
    _pg_ensure()   # BFS 全程 PG 直读（2026-08-09 性能：原每格多次 gget_plan 函数调用）
    if PASS_CLEAR > 0: _dist_ensure()   # BFS/门验收全程 PASS 直读（门宽度判断）
    if not (0 <= fvx < GRID_N and 0 <= fvy < GRID_N) or PG[fvx, fvy] == WALL:
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
        if PG[cx, cy] == FREE:
            # 门=未知前沿（只认开阔前沿，墙缝/墙根未知不算——那是陷阱）。
            # 黑名单门（bounce 撞墙的）跳过
            # 门宽度判断（2026-08-10 主人指令）：clear_ok = 净空场 ≥PASS_CLEAR(0.6m)——
            # 窄缝（障碍↔墙 <1.2m）里的前沿格没有足够净空 → 识别为"过不去的缝"拒绝计数，
            # 不再是门（旧 wall_dist>ROBOT_R=0.2m 太弱：0.5m 级缝门把狗吸进 STOP/bounce 风暴）。
            # (cx,cy)!=(vx,vy)：起点自身格邻接未知不算门（否则 path=空 → bounce 死循环）
            if ((cx, cy) != (vx, vy) and (cx, cy) not in bad_gates and (cx, cy) not in dead_gates
                    and _open_frontier(cx, cy)):
                if clear_ok(cx, cy):
                    gates.append((cg, cx, cy))
                else:
                    _NARROW_REJ[0] += 1   # 窄缝门：能透过激光看到，但太窄过不去——识别并拒绝
        for dx, dy in [(0,-1),(0,1),(-1,0),(1,0)]:
            # 内联跳步走查：逐格前进，同时捕捉 FREE→UNKNOWN 过渡格——
            # 2m 大跳会飞越前沿格（中间格不进 visited，永远评不到门），
            # 实测 BFS 在虚空里漫游 5099 格却 0 门。过渡格直接补评入门。
            wd0 = wall_dist(cx, cy)
            if wd0 >= JUMP_1M:   max_jump = JUMP_1M
            elif wd0 >= JUMP_03: max_jump = JUMP_03
            else:                max_jump = JUMP_NEAR
            js = 0
            prev_free = None
            for s in range(1, max_jump + 1):
                mx, my = cx + dx*s, cy + dy*s
                if not (0 <= mx < GRID_N and 0 <= my < GRID_N):
                    break
                v = PG[mx, my]
                if v == WALL:
                    break
                # 门宽度判断：低净空格（窄缝内部/贴墙根）不走——缝后的区域再宽也不经此缝规划
                if PASS_CLEAR > 0 and not PASS[mx, my]:
                    break
                if v == UNKNOWN:
                    if prev_free is not None:
                        _fx, _fy, _fs = prev_free
                        if ((_fx, _fy) != (vx, vy) and (_fx, _fy) not in bad_gates and (_fx, _fy) not in dead_gates
                                and _open_frontier(_fx, _fy)):
                            if clear_ok(_fx, _fy):
                                gates.append((cg + _fs, _fx, _fy))
                                # 过渡格不在 visited/堆里，came_from 必须补录父节点——
                                # 否则 fine_path 回溯断链，路径只剩 1 格直冲穿墙（实测卡死根因）
                                if (_fx, _fy) not in came_from:
                                    came_from[(_fx, _fy)] = (cx, cy)
                            else:
                                _NARROW_REJ[0] += 1   # 窄缝过渡门——识别并拒绝
                        prev_free = None
                elif v == FREE:
                    prev_free = (mx, my, s)
                js = s
            if js < 1: continue
            nx, ny = cx + dx*js, cy + dy*js
            wd = wall_dist(nx, ny)
            penalty = max(0, WALL_BUFFER_CELLS - wd) * WALL_PENALTY
            # 走中间（KNOWN_MAP_MODE）：代价 + C/d²，贴墙爆炸、中线最低（Voronoi 骨架）
            if KNOWN_MAP_MODE:
                penalty += VORONOI_C / (max(1, wd) * max(1, wd))
            # UNKNOWN 格可通行但代价高（优先已知路，必要时才穿未知）
            # KNOWN_MAP_MODE：穿越未知无惩罚（已知地图模式要快速穿行，执行层实时避障）
            if PG[nx, ny] == UNKNOWN and not KNOWN_MAP_MODE:
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
    2026-08-09：质心→原始门格集合写入 _gate_cluster_cells——黑名单按门格集合
    拉黑（旧版只拉黑质心，而 find_gates 过滤的是原始门格，key 对不上 → 黑名单
    形同虚设，同一死门被无限重选；实测幽灵门 bounce 412 次卡 202s）。
    """
    _gate_cluster_cells.clear()
    if not gates:
        return []
    # 门少时（探索早期窄前沿）降低聚类阈值，避免全过滤
    # 2026-08-09：阈值降到 2-3——测距限界（30m）处射线稀疏，前沿天然是 1-3 格斑点簇，
    # 阈值 8 会把它们全过滤，保底又只选"最近门" → 系统性地把狗送进死胡同口袋角（实测）
    dyn_min = 2 if len(gates) < 40 else 3
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
        _gate_cluster_cells[(sx, sy)] = cluster
        regions.append((best_cg, sx, sy, len(cluster)))
    if not regions and gates:
        # 保底：全被过滤（斑点前沿）→ 原始门格全部作为 size=1 候选返回（按 cg 排序），
        # 让 pick_gate 评分选择——旧版只返回"最近门"，评分机制完全失效（实测狗被送进死角）
        for cg, vx, vy in gates[:50]:
            _gate_cluster_cells[(vx, vy)] = [(vx, vy)]
            regions.append((cg, vx, vy, 1))
    regions.sort(key=lambda r: r[0])
    return regions

def pick_gate(gates, mode="score", stuck=False, robot=(0, 0), fin=None, heading=None):
    if not gates: return None
    if stuck: return gates[0]
    if mode == "far": return gates[-1]
    if mode == "near": return gates[0]
    if mode == "mix":
        return gates[-1] if len(gates) >= MIX_THRESHOLD else gates[0]
    if mode == "score":
        # 门评分：方向推进 + 距离 + 大小。
        # 2026-08-09 去特权：fin=None 时不许用终点真值方向（旧版 advance 朝 FINISH
        # 直线投影，蛇形迷宫里正确出口常与终点方向垂直 → 被系统性拉进死胡同口袋，
        # 实测首个口袋卡 48s/105 次 bounce）。改为朝向保持（狗沿走廊继续走的自然
        # 倾向，无特权）；一旦相机看到终点（fin=finish_est），方向项才朝终点——
        # 那是狗亲眼所见，合法。
        bx, by = robot
        fx = fy = None
        if fin is not None:
            fx, fy = fin
        hx = hy = None
        if heading is not None:
            hx, hy = math.cos(heading), math.sin(heading)
        best = None; best_score = -1
        for g in gates:
            cg, gx, gy, size = g
            wx, wy = (gx+0.5)*VOXEL, (gy+0.5)*VOXEL
            d = math.hypot(wx - bx, wy - by)
            d = max(d, 1.0)
            advance = 0.0
            if fx is not None:
                # 向（已看到的）终点推进：目标-机器人 在 终点方向上的投影（归一化到 0~1）
                denom = math.hypot(fx-bx, fy-by)
                if denom > 1e-6:
                    adv = ((wx-bx)*(fx-bx) + (wy-by)*(fy-by)) / (denom * max(d, 0.01))
                    advance = max(0.0, min(1.0, adv))
            elif hx is not None:
                # 朝向保持：沿当前航向的门优先（走廊直行，减少横跳）
                adv = ((wx-bx)*hx + (wy-by)*hy) / max(d, 0.01)
                advance = max(0.0, min(1.0, adv))
            # （2026-08-10 实测回退：曾加 0.18 净空权重想让狗优先宽门避开边界缝探测——
            # 结果门选择系统性变差：seed7 28% 覆盖卡死未到达、seed23 碰撞 238。
            # 边界缝探测成本（每弯 1-2 次 ~10s）接受，记录于 square-maze 文档第九节）
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
def astar_to(fvx, fvy, tfx, tfy, goal_relax=False):
    ASTAR_CALLS[0] += 1
    if ASTAR_CALLS[0] % 100 == 1:
        print(f"  [A*] call#{ASTAR_CALLS[0]} from=({fvx},{fvy}) to=({tfx},{tfy})", flush=True)
    # 起点放宽：未知/已知都可起步（WALL 才拒绝），机器人物理位置贴墙边也必须能回溯寻路
    if gget_plan(fvx, fvy) == WALL:
        return None
    # 终点要求：探索模式要 walkable（已确认自由区）；KNOWN_MAP_MODE / goal_relax（终点直
    # 奔模式：相机看到的终点可能落在未知格）放宽到 traversable（未知也可达，执行层实时避障）
    if KNOWN_MAP_MODE or goal_relax:
        if not traversable(tfx, tfy):
            # 终点格被移动障碍的临时标记占住（几秒后障碍移走/射线清除会消退）→
            # 就近吸附到可达格，而不是直接失败（实测混合场 PLAN-FAIL 死循环：细层目标格
            # 在障碍标记团里，A* 扩 15 万格白费）
            _nsg = _nearest_walkable(tfx, tfy, max_r=20)
            if _nsg is None:
                return None
            tfx, tfy = _nsg
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
            js = jump_steps(cx, cy, dx, dy, tfx, tfy)
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
    # 障碍红色圆柱贴路面（中心 z = HF_SURF+半高1.0；旧硬编码 2.0 埋地里）
    # mocap="true"（2026-08-10 修视觉冻结 bug）：无关节静态 body 的 xpos 永远不随
    # obs_world 更新——移动障碍逻辑上在动（雷达/碰撞/DWA 都看到），但渲染画面 frozen
    # 在初始位置（实测录像里 10 个障碍全冻在 x=25 一列）。mocap 体每 tick 同步位置。
    OBS_XML = "".join(
        f'<body name="obs{i}" mocap="true" pos="{x:.1f} {y:.1f} {HF_SURF + 1.0:.2f}">'
        f'<geom type="cylinder" size="0.5 1.0" rgba="0.9 0.2 0.2 0.9" contype="0" conaffinity="0"/></body>'
        for i,(x,y) in enumerate(obs_world))
    FINISH_XML = f'<body mocap="true" pos="{FINISH[0]:.1f} {FINISH[1]:.1f} {HF_SURF + 1.5:.2f}"><geom type="sphere" size="1.5" rgba="0.2 1.0 0.2 0.8"/></body>'
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
    <material name="dog_mat" rgba="1 0.9 0.1 1" emission="1"/>
    {LM_ASSETS}
  </asset>
  <worldbody>
    <light pos="25 25 80" dir="0 0 -1" ambient="0.5 0.5 0.55" diffuse="0.9 0.9 0.95"/>
    <light pos="0 25 30" dir="0.3 0 -0.8" diffuse="0.3 0.3 0.35"/>
    <light pos="50 25 30" dir="-0.3 0 -0.8" diffuse="0.3 0.3 0.35"/>
    {FINISH_XML}{OBS_XML}
    <geom type="hfield" hfield="track" pos="25 25 0.0" rgba="0.55 0.6 0.65 1.0" friction="0 0 0" contype="0" conaffinity="0"/>
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
      <!-- 机器狗：水平胶囊（长轴沿局部 x = 前进方向 yaw），0.8m 长 × 0.4m 径，contype=0 纯算法控制 -->
      <!-- 亮黄 emissive：俯视视频清晰可见（emission=1 自发光不受光照影响） -->
      <geom type="capsule" fromto="-0.4 0 0 0.4 0 0" size="0.2" material="dog_mat" friction="0 0 0" contype="0" conaffinity="0"/>
      {CAM_XML}
    </body>
  </worldbody>
</mujoco>"""

class Mover:
    def __init__(self, m, d):
        self.m, self.d = m, d
        self.yaw = 0.0; self.speed = 0.0; self.bounce = 0
        self.pose = None   # (x,y) 里程计位姿覆盖（--odom 1 时决策用估计位姿；None=用真值）
        self.stuck_t = 0; self.stuck_x = 0.0; self.stuck_y = 0.0
        self.target = None            # 当前目标（GATE 方向），bounce 时优先朝向；None=尚未有目标（不用 FINISH 真值先验）
        self.escape_steps = 0   # bounce 逃生冷却：沿 escape_yaw 强制走 N 步（绕出墙角再回归路径）
        self.escape_yaw = 0.0
        self.need_replan = False  # 撞到新障碍（不在当前路径规划里）→ 主循环强制重规划
        self._last_bounce_step = -10**9   # bounce 节流：两次 bounce 至少 40 步
        self.dwa = None          # DWAAlgorithm 实例（args.obs_random>0 时创建）
        self.dwa_target = None   # DWA 决策结果 (v*, ω*)；None = 全碰撞 → 走原逻辑
        self.dwa_t = -10**9      # dwa_target 的咨询时刻（step）：None 只在"刚咨询过"时才=全碰撞
        self.omega = 0.0         # 当前角速度（DWA 动态窗口用）

    def _forward_clear(self, bx, by, yaw_ang):
        """沿 yaw 方向前瞻测距：返回前方最近障碍距离 (m)。
        判定 = 全局地图 blocked()（含机器人半径膨胀）**或** 滚动局部层新鲜命中——
        局部层是最近 3s 的激光直读（odom 增量误差毫米级），全局地图漂移错位时
        执行层依然看到真实障碍（ROS rolling obstacle layer 思想，2026-08-12 审核整改）。
        2026-08-09 性能：采样 0.05→0.1m（墙厚≥0.2m+keepout 邻格覆盖，0.1m 采样不会漏墙；
        原 80 次 blocked/步是执行层主成本）。"""
        for k in range(1, int(LOOKAHEAD / 0.1) + 1):
            px = bx + math.cos(yaw_ang) * 0.1 * k
            py = by + math.sin(yaw_ang) * 0.1 * k
            if blocked(px, py):
                return 0.1 * k
            _lx, _ly = int(px / VOXEL), int(py / VOXEL)
            if 0 <= _lx < GRID_N and 0 <= _ly < GRID_N and \
                    LOCAL_STAMP[_lx, _ly] > _scan_step[0] - LOCAL_WIN:
                return 0.1 * k
        return LOOKAHEAD

    def _pose_xy(self):
        return self.pose if self.pose is not None else (self.d.qpos[0], self.d.qpos[1])

    def step(self, tx, ty, step):
        bx, by = self._pose_xy()   # 决策位姿（里程计模式=估计值；否则=真值）
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
        # DWA 决策结果存在 → 用 ω* 转向（替代原 err 转向）
        err = 0.0   # 非 DWA 分支才计算（turn_limited 判定用）
        if self.dwa is not None and self.dwa_target is not None:
            self.yaw += self.dwa_target[1] * dt
        else:
            tgt_yaw = math.atan2(ty-by, tx-bx)
            err = (tgt_yaw-self.yaw+math.pi)%(2*math.pi)-math.pi
            dyaw = max(-YAW_RATE*dt, min(YAW_RATE*dt, err))
            self.yaw += dyaw
        self.omega = self.dwa_target[1] if (self.dwa is not None and self.dwa_target is not None) else 0.0
        # ── 大转向限速（DWA 思想：曲率越大速度越低）──
        # 目标方向偏离当前朝向 >57° 时限速 1.0——先转身再加速，防止直冲偏离（绑架随机起点 yaw=0 但目标在背后 170°）
        turn_limited = False
        if self.dwa is not None and self.dwa_target is not None:
            if abs(self.dwa_target[1]) > 1.0:   # DWA 模式：按 ω* 判断
                turn_limited = True
        elif abs(err) > 1.0:
            turn_limited = True
        # ── 前瞻测距：沿当前 yaw 方向量到障碍的距离 ──
        d_clear = self._forward_clear(bx, by, self.yaw)
        # ── 期望速度（限速）：目标距离速度 vs 制动约束，取小 ──
        # DWA 给的 v* 优先（已在速度空间采样里考虑障碍/目标），否则原目标距离速度
        if self.dwa is not None and self.dwa_target is not None:
            v_des = self.dwa_target[0]
        elif self.dwa is not None:
            # DWA 全碰撞（或尚未咨询）：原地等下个咨询窗口（10 步），障碍 1m/s 会自己走开——
            # 不能用"目标距离速度"盲目加速（DWA 刚判完全碰撞）；也不进 STOP 分支刷 bounce
            v_des = 0.0
        else:
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
        # ── 前方被堵且已停住 / DWA 全碰撞 → 预判转向，不碰撞 ──
        # 2026-08-09 修复：DWA-None 必须"新鲜"（≤LIDAR_TICK 前咨询的）且**正在移动**才算
        # 全碰撞事件——陈旧 None（DWA 只在路径跟随分支每 10 步咨询）会让 STOP 分支每步触发：
        # bounce 刷频 → need_replan → escape 被 _bounce 清零速度 → 永不移动的死锁
        # （实测移动障碍场 bounce 660 次锁死 (29.1,8.8)，DWA 说全碰撞但前瞻 4m 畅通）。
        # 静止+全碰撞：由上方 v_des=0 安静等待（障碍会走开），不再 bounce 刷屏
        # （实测 bounce#1900+ 锁死 (19.4,19.0)，障碍贴着狗但前方畅通）
        if (self.speed <= 0.05 and d_clear < STOP_MARGIN + 0.15) or (
                self.dwa is not None and self.dwa_target is None
                and step - self.dwa_t <= LIDAR_TICK and self.speed > 0.05):
            # 诊断日志用感知版"撞墙了吗"：狗脚下格有无 3s 内激光直接命中（局部层）。
            # （旧版 sample_hf 真值——只进日志不进决策，但审核口径下真值连日志也不该读）
            _svx, _svy = int(bx / VOXEL), int(by / VOXEL)
            _hit_wall = bool(LOCAL_STAMP[_svx, _svy] > _scan_step[0] - LOCAL_WIN) \
                if 0 <= _svx < GRID_N and 0 <= _svy < GRID_N else False
            # 被挡时前方 ~1.5m 内是否有障碍——纯感知：查雷达扫到的障碍格记忆 OBS_SEEN
            # （旧版查 obs_world 真值=特权）。放宽：狗在障碍 0.7m 外就会被 blocked 挡住，但距离判定>0.7 会漏
            _near_obs = False
            for _kk, _st in OBS_SEEN.items():
                if _scan_step[0] - _st > 50:   # 50 次扫描 ≈8s 前看到的才算"记得"
                    continue
                _cvx, _cvy = divmod(_kk, 4096)
                if math.hypot((_cvx+0.5)*VOXEL-bx, (_cvy+0.5)*VOXEL-by) < OBS_CLEAR + 0.8:
                    _near_obs = True
                    break
            if self.bounce % 5 == 0:
                print(f"  [STOP] bounce#{self.bounce} @({bx:.1f},{by:.1f}) d_clear={d_clear:.2f} wall={_hit_wall} obs={_near_obs}", flush=True)
            if _near_obs:
                # 撞到障碍（不在 HPA static 距离场里）→ 先 escape 走远离开障碍区（~2m），
                # escape 结束后主循环写安全圈 + 强制重规划绕行
                # （写圈时机：狗在圈外才有效——狗在圈内时身边格被排除，路径仍贴障碍 → 死循环）
                self.need_replan = True
                # 2026-08-09 修复：被障碍顶住时**每步 _bounce 是自杀**——bounce 把 yaw 转向
                # 目标对齐方向（= 顶着障碍的方向），qvel 清零，escape 永远起不来
                # （实测混合场 bounce#34325 锁死 (37.5,46.4)，障碍 0.1m 顶脸）。
                # 正确做法：不 bounce，只选一次"最远净空"逃逸方向然后让狗真的开走；
                # 移动障碍几秒后自己走开，固定障碍靠 escape+重规划绕行。
                if self.escape_steps <= 0:
                    # escape 方向 = 可走距离最大方向（全向扫描 d_clear 最大）——不是盲目远离障碍：
                    # 远离方向可能 1m 就撞墙（如 y=5 分界墙）→ escape 撞墙 bounce 死循环。
                    # 走最远方向逃出障碍区，重规划后回归路径。
                    best_esc_yaw, best_esc_d = None, -1.0
                    for _deg in range(0, 360, 10):
                        _cand = math.radians(_deg)
                        _d = self._forward_clear(bx, by, _cand)
                        if _d > best_esc_d:
                            best_esc_d, best_esc_yaw = _d, _cand
                    if best_esc_yaw is not None and best_esc_d > 0.5:
                        self.escape_yaw = best_esc_yaw
                        self.escape_steps = 150   # 0.75s 走 ~2m 离开障碍区
                    elif best_esc_yaw is not None:
                        # 局部被围（全向净空 <0.5m）：沿最远方向慢爬——爬出后写圈才有效
                        self.escape_yaw = best_esc_yaw
                        self.escape_steps = 60
                    if self.bounce % 10 == 0:
                        print(f"  [OBS-ESC2] esc_yaw={math.degrees(self.escape_yaw):.0f}° d={best_esc_d:.1f}m", flush=True)
            else:
                self._bounce(45, 120, step=step)
        # 卡死检测
        if step-self.stuck_t > STUCK_TIMEOUT:
            if math.hypot(bx-self.stuck_x, by-self.stuck_y) < STUCK_DIST_THRESH:
                self._bounce(90, 180)
            self.stuck_t = step; self.stuck_x = bx; self.stuck_y = by
        # 执行移动（沿轴向速度，物理 yaw 由 hinge 积分）
        vx = math.cos(self.yaw)*self.speed; vy = math.sin(self.yaw)*self.speed
        nx, ny = bx+vx*dt, by+vy*dt
        # 硬防穿墙/障碍：下一位置 blocked（中心0.2m圆触障碍）就不动——物理上不可能穿。
        # 2026-08-12：移动护卫叠加滚动局部层（新鲜直读命中格）——全局地图漂移时
        # 真墙/真障碍照样封死（实测只信地图会把狗嵌进墙体 5 万步碰撞）
        _nvx, _nvy = int(nx / VOXEL), int(ny / VOXEL)
        _local_hit = (0 <= _nvx < GRID_N and 0 <= _nvy < GRID_N
                      and LOCAL_STAMP[_nvx, _nvy] > _scan_step[0] - LOCAL_WIN)
        # bounce 决策已由上方 STOP 分支负责，这里只保证不移动（防低速漂移滑入）
        if blocked(nx, ny) or _local_hit:
            if blocked(bx, by) and self._forward_clear(bx, by, self.yaw) > 0.15:
                # 狗当前格已在禁入区（墙是狗停下后被后续扫描标到身边的）——此时 speed≈0
                # 导致 nx≈bx 永远 blocked → 永冻锁死（实测 ch9 (7.7,45.2) bounce#1530）。
                # 只有"还能动"才能开出去：朝净空方向允许爬出（≤1.5m/s——比移动障碍快，
                # 0.5m/s 会被 1m/s 障碍持续顶牛，实测 collision 800+），
                # 前方 0.15m 内也堵则维持不动（防继续深入墙体）
                self.speed = min(self.speed, 1.5)
                self.d.qvel[0] = math.cos(self.yaw)*self.speed
                self.d.qvel[1] = math.sin(self.yaw)*self.speed
                self.d.qvel[2] = 0
                self.d.qpos[2] = self.yaw
            else:
                self.speed = 0.0
                self.d.qvel[0] = 0; self.d.qvel[1] = 0; self.d.qvel[2] = 0
                # 从经验学习：撞到的未知格写回地图为 WALL（A* 下次就不会规划穿墙路径）
                # ⚠️ 必须**感知确认**（2026-08-12 去真值，代码审核整改）：
                # 旧版用 sample_hf 真值/obs_world 真值确认（特权）。现改为——
                #   ① 该格 ±0.3m 邻域内有激光**直接命中**记录（HIT_CONFIRMED：
                #      掠射填充/墙厚先验推断的格不算，防假墙自指扩散）；或
                #   ② 附近有近期雷达障碍记忆（OBS_SEEN，撞到的是障碍）
                # blocked 是纯感知判定，无确认就写会把"自己标的假墙"当挡 →
                # 假墙自指包围死循环（2.9,3.4 实测 40+ bounce）。
                gvx, gvy = int(nx/VOXEL), int(ny/VOXEL)
                if gget(gvx, gvy) == UNKNOWN:
                    _x0c, _x1c = max(0, gvx-3), min(GRID_N, gvx+4)
                    _y0c, _y1c = max(0, gvy-3), min(GRID_N, gvy+4)
                    _cfm = bool(HIT_CONFIRMED[_x0c:_x1c, _y0c:_y1c].any())
                    if not _cfm:
                        for _kk, _st in OBS_SEEN.items():
                            if _scan_step[0] - _st > 50:
                                continue
                            _cvx, _cvy = divmod(_kk, 4096)
                            if math.hypot((_cvx+0.5)*VOXEL-nx, (_cvy+0.5)*VOXEL-ny) < OBS_CLEAR + 0.5:
                                _cfm = True
                                break
                    if _cfm:
                        gset(gvx, gvy, WALL)
        else:
            self.d.qvel[0] = vx; self.d.qvel[1] = vy; self.d.qvel[2] = 0
            self.d.qpos[2] = self.yaw  # 控制 yaw 直接写回物理（滑动模型 friction=0，mj_step 保持）
        mujoco.mj_step(self.m, self.d)
        # 不再读回 qpos[2] 覆盖 self.yaw——yaw 是控制变量（旧代码读回导致每步转向被重置为初始 0）
        return True

    def _bounce(self, lo, hi, step=None):
        # bounce 节流（2026-08-09）：两次 bounce 至少隔 40 步（0.2s）——给狗时间真的朝新方向
        # 开出去；旧版每步都转，新方向走不出 1cm 就被重转，角点实测 bounce#1680 原地踏步
        # (45.2,45.1)。obs 逃逸路径不调 _bounce（不受影响）
        if step is not None:
            if step - self._last_bounce_step < 40:
                return
            self._last_bounce_step = step
        # 无条件计数+转向（防 escaping 卡死）；转向后 speed 已≈0，由 step() 重新加速
        self.bounce += 1
        if args.trail_every > 0:
            bounce_pts.append([round(self.d.qpos[0], 3), round(self.d.qpos[1], 3), self.bounce])
        if self.bounce % 5 == 0:
            print(f"  [BOUNCE] bounce#{self.bounce} @({self.d.qpos[0]:.1f},{self.d.qpos[1]:.1f})", flush=True)
        # 转向：先试目标方向（当前 GATE）的小角度偏转，再试随机（防墙边死循环）
        bx, by = self._pose_xy()
        if self.target is not None:
            tx, ty = self.target
        else:
            tx, ty = bx + math.cos(self.yaw) * 2.0, by + math.sin(self.yaw) * 2.0   # 无目标先验：朝当前朝向
        tgt_yaw = math.atan2(ty - by, tx - bx)
        _dbg = (self.bounce % 20 == 0)
        # 全向每 5° 扫（防稀疏角度漏掉窄缺口——迷宫段间墙缺口只有 ~10° 宽）
        # 评分：推进优先，但推进度差距小(<0.15)时距离只做轻 tie-break（防 Z 字斜穿震荡）
        candidates = []
        for deg in range(0, 360, 5):
            candidates.append(tgt_yaw + math.radians(deg))
        best_yaw, best_d, best_score = None, -1.0, -1e9
        for cand in candidates:
            d = self._forward_clear(bx, by, cand)
            # 净空 ≥0.8m 才可作逃逸方向：0.4m 是贴墙（STOP 阈值 0.55），选了走不动，
            # 实测目标对齐评分 dot*4 会把狗一遍遍送进贴墙方向（bounce 自维持死循环）
            if d < STOP_MARGIN + 0.4:
                continue
            dot = math.cos(cand - tgt_yaw)
            # 混合评分：推进优先但保留逃逸能力——dot*4 + 距离(截断2)
            # 平衡点：正前方被挡(d 小)时，侧向 dot=0.7 且 d=3m → 2.8+2=4.8 > 正前方 4+0.5=4.5
            # 但斜穿对面墙 dot=0.3,d=5 → 1.2+2=3.2 < 侧向贴墙走 dot=0.9,d=2 → 3.6+2=5.6
            score = dot * 4.0 + min(d, 2.0)
            if score > best_score:
                best_score, best_d, best_yaw = score, d, cand
        if best_yaw is not None and best_d >= STOP_MARGIN:
            if _dbg:
                print(f"  [BOUNCE-DBG] tgt_yaw={math.degrees(tgt_yaw):.0f}° best_yaw={math.degrees(best_yaw):.0f}° best_d={best_d:.2f} "
                      f"d_fwd={self._forward_clear(bx, by, self.yaw):.2f} yaw={math.degrees(self.yaw):.0f}° esc={self.escape_steps}", flush=True)
            self.yaw = best_yaw
            self.d.qpos[2] = best_yaw  # 同步物理 yaw，狗身体立即转到新方向
            self.d.qvel[:] = 0
            self.speed = 0.0
            # 逃生冷却：沿刚选的方向强制走 N 步（防 step() 又转向被挡目标 → bounce 死循环）
            # 撞障碍（need_replan 挂起）时保持 200（1s~2-3m）——走太长会撞别的墙
            if self.need_replan:
                self.escape_steps = max(self.escape_steps, 200)
            else:
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
    idx = np.nonzero(G != UNKNOWN)
    if idx[0].size == 0: return
    minx, maxx = int(idx[0].min()), int(idx[0].max())
    miny, maxy = int(idx[1].min()), int(idx[1].max())
    arr = G[minx:maxx+1, miny:maxy+1].T.copy()   # npz 格式 arr[vy, vx]（兼容旧档）
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
    """保存地图到文件（感知版，不许作弊）。
    只存 grid 雷达扫到的值（WALL/FREE）——不用 sample_hf 真值判结构墙。
    障碍残影：障碍格也会存为 WALL，KNOWN_MAP 阶段 scan 射线清除 live 层修正（感知建图的真实代价）。
    """
    idx = np.nonzero(G != UNKNOWN)
    if idx[0].size == 0: return
    minx, maxx = int(idx[0].min()), int(idx[0].max())
    miny, maxy = int(idx[1].min()), int(idx[1].max())
    arr = G[minx:maxx+1, miny:maxy+1].T.copy()   # 感知值（WALL/FREE），非真值；arr[vy, vx]
    np.savez(path, grid=arr, offset=(minx, miny), seed=FIXED_SEED)
    n_wall = int((arr == WALL).sum()); n_free = int((arr == FREE).sum())
    print(f"  [MAP] saved {n_free} FREE + {n_wall} WALL(感知) → {path}", flush=True)

def load_map(path):
    """加载旧地图为 static 层（numpy SG），开启 KNOWN_MAP_MODE（阶段2）"""
    global KNOWN_MAP_MODE
    if not os.path.exists(path):
        print(f"  [MAP] 警告: {path} 不存在", flush=True)
        return False
    data = np.load(path, allow_pickle=True)
    arr = data["grid"]; ox, oy = int(data["offset"][0]), int(data["offset"][1])
    SG.fill(0)
    h, w = arr.shape
    SG[ox:ox+w, oy:oy+h] = arr.T.astype(np.int8)   # 档格式 arr[vy, vx] → SG[vx, vy]
    _wd.clear()
    _pg_touch()
    KNOWN_MAP_MODE = True
    print(f"  [MAP] loaded {np.count_nonzero(SG)} cells from {path} (KNOWN_MAP_MODE)", flush=True)
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

_road_mask_cache = [None]   # 真值路面掩码缓存（coverage 评分用，500×500 bool）
def coverage_pct():
    """探索覆盖率（评分指标，可用真值）：真值路面内已探索格 / 真值路面格。
    2026-08-12 口径修复（代码审核：旧版 >104% 失真）——旧版分母排除障碍格、
    分子却含障碍残影/墙外涂抹格；现统一：分子分母都限定在真值路面掩码内。"""
    if _road_mask_cache[0] is None:
        rm = np.zeros((GRID_N, GRID_N), dtype=bool)
        for wy in range(0, 500):  # 50m / 0.1m
            for wx in range(0, 500):
                rm[wx, wy] = not is_obstacle_world((wx+0.5)*VOXEL, (wy+0.5)*VOXEL)
        _road_mask_cache[0] = rm
    rm = _road_mask_cache[0]
    road_total = int(rm.sum())
    explored = int(np.count_nonzero(((G == FREE) | (G == WALL)) & rm))
    return explored / road_total * 100 if road_total else 0

# ═══════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════

print(f"━━━ 萤火 Firefly v3 SLAM headless ━━━ {VOXEL}m 三级跳A* 模式={EXPLORE_MODE} seed={FIXED_SEED} ━━━", flush=True)


xml = build_xml()
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
# 障碍 mocap 体 id 表（视觉位置同步用；非 mocap 为 -1 跳过）
_OBS_MOCAP = []
for _i in range(len(obs_world)):
    _bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"obs{_i}")
    _OBS_MOCAP.append(int(m.body_mocapid[_bid]) if _bid >= 0 else -1)

def sync_obs_mocap():
    """把 obs_world 当前位置写进 mocap 体（渲染跟随逻辑位置）。contype=0 无物理副作用。"""
    for _i, (_ox, _oy) in enumerate(obs_world):
        if _i >= len(_OBS_MOCAP):
            break
        _mid = _OBS_MOCAP[_i]
        if _mid >= 0:
            d.mocap_pos[_mid] = (_ox, _oy, HF_SURF + 1.0)
if args.random_start:
    rs_x, rs_y = random_road_pos(FIXED_SEED + 1000, min_dist_from=5.0, from_pos=FINISH)
    d.qpos[0] = rs_x; d.qpos[1] = rs_y
    print(f"  [KIDNAP] 随机起点: ({rs_x:.1f}, {rs_y:.1f}) → 目标 {FINISH}", flush=True)
else:
    if args.target == "start":
        d.qpos[0] = 2.5; d.qpos[1] = 47.5   # 反向：起点=出口端，目标=入口端
    else:
        d.qpos[0]=2.5; d.qpos[1]=2.5
mujoco.mj_forward(m,d)

# 离屏渲染：Linux 优先 EGL；不可用（如 Windows 桌面）回退 GLFW 不可见窗口
os.makedirs(args.out_dir, exist_ok=True)
renderer = None
RENDER_OK = False
try:
    from mujoco import egl
    _ctx = egl.GLContext(640, 360)
    _ctx.make_current()
    renderer = mujoco.Renderer(m, 360, 640)   # 2026-08-09 性能：1280×720→640×360（mjr_render 30→~8ms/帧）
    RENDER_OK = True
    print("  [RENDER] EGL 离屏渲染 OK", flush=True)
except Exception as e:
    try:
        import glfw
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, 0)
        _glfw_win = glfw.create_window(640, 360, "offscreen", None, None)
        glfw.make_context_current(_glfw_win)
        renderer = mujoco.Renderer(m, 360, 640)
        RENDER_OK = True
        print(f"  [RENDER] GLFW 离屏渲染 OK（EGL 不可用: {e}）", flush=True)
    except Exception as e2:
        print(f"  [RENDER] 离屏渲染不可用: EGL={e} GLFW={e2}", flush=True)

# 俯视相机（--cam-elevation 控制俯角；MuJoCo elevation 负值=俯视，-60 = 上方60°）
_cam = None
CAM_ELEVATION = -60.0

def render_frame(step):
    """离屏渲染当前帧保存 PNG（俯视 60° 跟随狗）"""
    if renderer is None:
        return
    global _cam
    try:
        if _cam is None:
            _cam = mujoco.MjvCamera()
            _cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            _cam.azimuth = 90.0   # 从 +y 方向看：通道（x 向）横屏展示（azimuth=0 时通道竖屏）
            _cam.distance = 18.0
        _cam.elevation = CAM_ELEVATION
        _cam.lookat[:] = np.array([d.qpos[0], d.qpos[1], 1.0], dtype=np.float64)
        renderer.update_scene(d, _cam)
        img = renderer.render()
        Image.fromarray(img).save(os.path.join(args.out_dir, f"frame_{step:06d}.png"))
    except Exception as e:
        print(f"  [RENDER-ERR] {e}", flush=True)  # 不静默吞渲染失败

mv = Mover(m, d)
# 里程计（--odom 1，2026-08-12 起默认）：决策/建图只用带噪估计位姿；真值 d.qpos 只用于物理/碰撞/统计。
# 初始帧定义：狗以上电点为世界系原点（自己的坐标系自己做主，非特权）；
# 之后全靠带噪推算 + scan-matching（激光里程计）+ 二维码绝对修正收敛。
odom = None
matcher = None
tracker = None
if args.odom:
    # --odom-noise 是噪声量级档位（默认 0.05 = 四足 IMU+步态推算的 ~5%/s 水平），
    # 按 5% 档标定的各分量随档位等比缩放（白噪/线速偏差游走/陀螺偏置/转弯比例偏差）
    _s = args.odom_noise / 0.05
    odom = Odometry(d.qpos[0], d.qpos[1], 0.0, v_noise=0.01 * _s, w_noise=0.01 * _s,
                    rng=random.Random(FIXED_SEED + 55),
                    v_bias_rw=0.02 * _s, w_bias_rw=0.06 * _s, w_scale_rw=0.02 * _s)
    if args.match:
        matcher = ScanMatcher(voxel=VOXEL)
    print(f"  [ODOM] 里程计定位（默认无特权）：决策位姿=估计（噪声 {args.odom_noise*100:.0f}%/s 级+慢变偏差+陀螺漂移），"
          f"scan-matching={'开' if matcher else '关'} + 二维码标牌绝对修正", flush=True)
# 移动障碍感知跟踪器：DWA 运动预测的唯一障碍速度来源（替代真值 velocities，审核整改）
if args.obs_random > 0 or args.obs_mix > 0:
    tracker = ObstacleTracker()
_prev_yaw = 0.0
if args.obs_random > 0 or args.obs_mix > 0:
    mv.dwa = DWAAlgorithm(v_max=SPEED_MAX, w_max=YAW_RATE,
                          a_accel=A_ACCEL, a_decel=A_DECEL)
    print("  [DWA] 局部规划器已启用", flush=True)

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
    SG[:, :] = (arr_wall.T * WALL).astype(np.int8)   # arr_wall[vy, vx] → SG[vx, vy]
    _pg_touch()
    print(f"  [MAP] 原始 track_clean 地图就绪 (KNOWN_MAP_MODE, {np.count_nonzero(SG)} 墙格, MAX-pool 区域判定)", flush=True)

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
        # aruco=--landmarks：场景没放标牌时关掉 ArUco 金字塔（CPU 全尺度检测非常慢），终点球检测不受影响
        vis = VisionLandmark(m, d, renderer, cam_name="bot_cam", detect_every=40,
                             aruco=bool(args.landmarks))
        print("  [VISION] 视觉地标识别已启用 (GPU 32ms/帧, detect_every=40)", flush=True)
    except Exception as e:
        vis = None
        print(f"  [VISION] 视觉不可用: {e}", flush=True)
else:
    vis = None

step = 0; t0 = time.time()
last_mx = last_my = 0
path = None; path_idx = 0; _plan_bounce_base = 0
path_is_goal = False   # 当前路径是否终点直奔（是则必须真到达，不做 3m 提前消耗）
_last_gate_key = None; _last_gate_dist = None; _gate_stall = 0   # gate 无进展检测（假门黑名单）
_wedge = 0; _wedge_pos = None   # 墙角困住检测：连续强制重规划位置不动 → 回退路标出角落
_last_back_step = -1000         # BACK 限流（每次全图 A* 探路标，无路标可达时会烧穿计算）
gate = None; gates = []
no_gate_count = 0
wander = 0; last_dist = 999
milestones = []
start_pos = (d.qpos[0], d.qpos[1])
milestones.append((int(start_pos[0]/VOXEL), int(start_pos[1]/VOXEL)))
last_mx, last_my = milestones[0]
back_blacklist = set()  # BACK 过的路标索引（防死循环）
bad_gates = set()       # 死门黑名单（原始门格集合：bounce 撞墙/无进展的门——按格拉黑才拦得住）
from collections import deque as _deque
_bad_gates_order = _deque()   # 黑名单加入顺序（限量 FIFO 驱逐用：上限 1000 格）
dead_gates = set()      # 永久死门（同一区域无进展拉黑 ≥3 次——确认死胡同，BAD-CLEAR 也不复活）
_gate_bad_count = {}    # 区域(1m) → 拉黑次数

def bad_add(cells):
    """门格集合拉黑（限量 FIFO——防长时间风暴把本地前沿全拉黑）"""
    for c in cells:
        if c not in bad_gates:
            bad_gates.add(c)
            _bad_gates_order.append(c)
    while len(_bad_gates_order) > 1000:
        bad_gates.discard(_bad_gates_order.popleft())

def bad_add_counted(cells, gkey):
    """拉黑并计数：同一区域反复无进展 ≥3 次 → 永久死门（确认死胡同，不再浪费 bounce）"""
    bad_add(cells)
    _rk = (gkey[0] // 10, gkey[1] // 10)
    _gate_bad_count[_rk] = _gate_bad_count.get(_rk, 0) + 1
    if _gate_bad_count[_rk] >= 3:
        dead_gates.update(cells)
_gate_cluster_cells = {}  # 最近一次 find_gates 的聚类映射：质心(sx,sy) → 原始门格列表
finish_est = None       # 终点位置估计（相机看到绿球才有；无特权——狗亲眼所见）
_goal_close_since = [None]  # 直奔死锁保险：狗贴在估计点却判不了到达的起始步
_goal_blacklist = []        # 作废过的估计区域（5m 格，4 万步过期）——防同点再锁
_est_dbg = [None]           # 估计变更调试（打印旧→新）
finish_obs = []         # 最近 N 次终点观测（世界坐标），中位数滤波
finish_rays = []        # 最近 N 次观测射线 (x,y,方位)——三角定位用（方位角抗背景板遮挡）
finish_last_step = -1   # 上次观测步（去重）
finish_announced = False
finish_est_tri = False  # 当前估计是否来自良态三角定位（否则是尺寸法中位数，偏远不可做到达依据）
finish_area = 0         # 最近一帧终点球 blob 面积（px）——近距到达判定
finish_bottom = 0       # 最近一帧 blob 底行（px）
finish_wa = 0.0         # 最近一帧球的世界方位角（视觉视线核查用）

print(f"=== Firefly v3 headless start: seed={FIXED_SEED} max_steps={args.max_steps} ===", flush=True)

# 初始扫描：开机原地自旋一圈（真实机器人上电自检旋转建图）——
# 前方 180° 雷达只能看到半圆，不自旋身后永远 UNKNOWN → gate 把起点自身格当门 → 卡死
# 初始扫描：开机原地自旋一圈（真实机器人上电自检旋转建图）——
# 前方 180° 雷达只能看到半圆，不自旋身后永远 UNKNOWN → gate 把起点自身格当门 → 卡死
# 2026-08-12 去真值：自旋期间里程计同步积分（陀螺有噪声——转一圈下来 yaw 漂几度是
# 真实物理），写图用估计位姿；结束后**不再** odom.yaw=真值 同步（那是真值校准=作弊），
# 残余 yaw 误差由后续 scan-matching/二维码修正收敛（地图初期轻微涂抹会被射线清除翻新）。
_in_init_scan[0] = True
_spin_omega = 2 * math.pi / (INIT_SCAN_STEPS * m.opt.timestep)
for _ in range(INIT_SCAN_STEPS):
    bx, by = d.qpos[0], d.qpos[1]
    d.qpos[2] = 2 * math.pi * (_ / INIT_SCAN_STEPS)   # 匀速转一圈，每 LIDAR_TICK 扫当前朝向
    if odom is not None:
        odom.update(m.opt.timestep, 0.0, _spin_omega)   # 纯旋转积分（带噪）
        if _ % LIDAR_TICK == 0:
            scan(bx, by, d.qpos[2], odom.x, odom.y, odom.yaw)
    else:
        if _ % LIDAR_TICK == 0: scan(bx, by, d.qpos[2])
    mujoco.mj_step(m, d)
d.qpos[2] = mv.yaw   # 自旋结束归位（与 Mover 朝向一致，scan 继续用 mv 视角）
_in_init_scan[0] = False
if odom is not None:
    _prev_yaw = mv.yaw
    # 注意：mv.yaw 是控制系朝向（狗自己知道"我现在朝哪"=机身系定义），odom.yaw 保持自旋
    # 积分结果（含漂移）——两者差异就是 IMU 漂移，不许互相同步
print(f"  [OK] FREE={_count(FREE)} WALL={_count(WALL)}", flush=True)

frame_idx = 0
trail = []          # 轨迹记录：每 trail-every 步存 (step, x, y, yaw, bounce)
bounce_pts = []     # bounce 位置（分析卡点用）
while step < args.max_steps and time.time() - t0 < args.timeout:
    bx, by = d.qpos[0], d.qpos[1]   # 真值（仅物理/碰撞/统计/轨迹用）
    if odom is not None:
        # 里程计模式：积分上一步运动（带噪声），决策/建图/避障全用估计位姿
        _domega = (mv.yaw - _prev_yaw) / m.opt.timestep
        _prev_yaw = mv.yaw
        odom.update(m.opt.timestep, mv.speed, _domega)
        px, py, pyaw = odom.pose()
        mv.pose = (px, py)
    else:
        px, py, pyaw = bx, by, mv.yaw
    bx, by = px, py   # 下游全部是决策位姿（无特权）
    _dog_est[0], _dog_est[1] = bx, by   # DWA 局部层自清参照

    # 风暴检测（2026-08-12 实验后**移除**）：隔离本意是迷失期冻结建图防正反馈发散，
    # 实测反成死地——冻结把"射线清除自救"也冻了，地图不更新 → 狗永远逃不出 →
    # bounce 率降不下 → 永不解除（v13 全程锁死 4 万步实测）。教训：**恢复机制不能
    # 切断自身的逃生通道**。_STORM 恒 False（保留开关位防回归参考）。
    vx, vy = int(bx/VOXEL), int(by/VOXEL)
    if gget(vx, vy) == UNKNOWN:
        gset(vx, vy, FREE)

    # 巡逻障碍移动（每 tick 更新位置，1m/s 沿蛇形路径往返）
    if patrol_paths:
        update_patrol(m.opt.timestep)

    # 随机反弹障碍移动（每 tick 更新，1m/s 弹性反弹）
    if random_field is not None:
        update_random(m.opt.timestep)

    # 障碍视觉位置同步（mocap 体——无此调用渲染 frozen 在初始位置）
    if (patrol_paths or random_field is not None) and RENDER_OK and args.render_every > 0:
        sync_obs_mocap()

    # 路标放置
    if abs(vx-last_mx)+abs(vy-last_my) >= MILESTONE_STEP:
        if wall_dist(vx, vy) > CLEARANCE:
            milestones.append((vx, vy))
            last_mx, last_my = vx, vy
            save_state()

    if step % LIDAR_TICK == 0:
        # 物理投射用真值位姿（激光打真实世界）；写图用估计位姿（狗以为的自己在哪）；
        # scan 内部做 scan-matching 并回传修正后的估计位姿
        _ep = scan(d.qpos[0], d.qpos[1], mv.yaw, bx, by, pyaw)
        if _ep is not None:
            bx, by, pyaw = _ep
            vx, vy = int(bx/VOXEL), int(by/VOXEL)
            mv.pose = (bx, by)
            _dog_est[0], _dog_est[1] = bx, by
        # 移动障碍感知跟踪：本帧障碍命中点 → 聚类/关联/速度估计（DWA 运动预测用）。
        # 空帧也要 update（track 老化/丢弃依赖时间推进）
        if tracker is not None:
            tracker.update(OBS_PTS_LAST, step * m.opt.timestep)

    # B阶段：运行中障碍变化（--obs-reseed 指定步数换新障碍 seed）
    if args.obs_reseed > 0 and step == args.obs_reseed:
        old_obs = list(obs_world)
        obs_world = gen_obstacles(FIXED_SEED + 777)  # 新障碍位置（顶层代码直接赋值，全局生效）
        OBS_CELLS_VALID = False   # 障碍变了 → 重建格查表（懒加载）
        print(f"  [OBS] step={step} 障碍变化: {len(old_obs)}→{len(obs_world)} 个", flush=True)
        # 强制重规划（当前路径可能穿过新障碍）
        path = None; path_idx = 0; wander = 0; last_dist = 999
        _plan_bounce_base = mv.bounce
        stats["obs_changes"] = stats.get("obs_changes", 0) + 1

    # 视觉地标识别（相机帧 + ArUco；看到标牌记录唯一ID）+ 终点球检测
    if vis is not None:
        vis.scan_once(step)
        # 里程计绝对修正：二维码标牌位置已知（环境设施表）→ 由观测方位/距离反解狗位姿，拉回漂移
        # 风暴期不修正：迷失时 yaw 不可信，反解位姿是错的，强修=往正反馈里加注（实测发散主因之一）
        if odom is not None and vis.geo_obs and not _STORM[0]:
            for (_gs, _gidx, _gd, _gb) in vis.geo_obs:
                if _gidx in vis._pos_map():
                    _lx, _ly = vis._pos_map()[_gidx]
                    _wb = pyaw - _gb   # 图像右 = 世界 -y → 世界方位 = yaw - bearing
                    _ex = _lx - _gd * math.cos(_wb)
                    _ey = _ly - _gd * math.sin(_wb)
                    # 距离门控 [5,25]m：太近 yaw 误差被杠杆放大（est 绕标牌旋转），太远像素噪声大
                    if 5.0 <= _gd <= 25.0:
                        _w = max(0.2, min(0.8, 1.5 / max(_gd, 1.0)))   # 越近越可信
                        odom.correct(_ex, _ey, _w)
                        # 朝向修正：标牌近似正面（|bearing|<15°）→ 狗绝对方位 ≈ 标牌朝向反方向+bearing
                        # （偶通道标牌朝 -x、奇通道朝 +x——标牌朝向是环境先验）
                        if abs(_gb) < 0.26:
                            _exp_wb = 0.0 if (_gidx % 2 == 0 or _gidx == 9) else math.pi   # ch9 标牌改贴右墙（球遮挡），朝向同偶通道
                            _yaw_est = _exp_wb + _gb
                            _dyaw = (_yaw_est - odom.yaw + math.pi) % (2*math.pi) - math.pi
                            odom.yaw += _dyaw * 0.2
                        if odom.corrections % 10 == 1:
                            print(f"  [ODOM-FIX] 标牌#{_gidx} d={_gd:.1f}m → 修正后漂移 {odom.error(d.qpos[0], d.qpos[1]):.2f}m", flush=True)
            vis.geo_obs.clear()
        elif vis.geo_obs:
            vis.geo_obs.clear()   # 里程计关时也清空，防无限累积
        # 终点发现（无特权）：相机看到绿色终点球 → 方位+距离 → 世界坐标估计
        # 尺寸法距离在标牌背景板遮挡下系统性偏远（实测 3 倍）→ 多方位观测三角定位（方位角抗遮挡）
        if vis.finish_obs is not None and vis.finish_obs[4] != finish_last_step and not _STORM[0]:
            _br, _fd, _fa, _fbottom, finish_last_step = vis.finish_obs
            _wa = pyaw - _br   # 图像右 = 世界 -y（相机 right=-y）→ 世界方位 = yaw - bearing
            # 估计喂料统一门槛 fd≤25m：>25m 的帧球被部分遮挡/像素太少，针孔距离**和**
            # 质心方位角都不可信（实测 fd 30-80m 的垃圾帧把已收敛的 est (2.5,47.2)
            # 一步步拖到 35m 外）——远距帧只用于"看到球"的事实记录，不进任何估计
            if _fd > 25.0:
                pass   # 远距帧不进估计（finish_last_step 已更新=去重）
            else:
                finish_obs.append((bx + _fd*math.cos(_wa), by + _fd*math.sin(_wa)))
                finish_rays.append((bx, by, _wa))
            del finish_obs[:-12]
            del finish_rays[:-30]
            # 三角定位：各观测射线的法向方程 n·X=n·p 最小二乘（2x2）
            _tri = None
            # 三角定位病态防护：射线基线/角度太集中（同一视角反复看）时最小二乘会乱飞——
            # 实测 ch8 隔墙偷看时 2 条近重合射线把估计打到狗脚边 → 误判到达
            _base = 0.0; _spread = 0.0
            if len(finish_rays) >= 3:
                _rxs = [r[0] for r in finish_rays]; _rys = [r[1] for r in finish_rays]
                _ras = [r[2] for r in finish_rays]
                _base = max(max(_rxs)-min(_rxs), max(_rys)-min(_rys))
                _spread = max(_ras)-min(_ras)
            if len(finish_rays) >= 3 and _base > 3.0 and _spread > math.radians(15):
                _a00 = _a01 = _a11 = _b0 = _b1 = 0.0
                for _rx, _ry, _ra in finish_rays:
                    _nx, _ny = -math.sin(_ra), math.cos(_ra)
                    _a00 += _nx*_nx; _a01 += _nx*_ny; _a11 += _ny*_ny
                    _bb = _nx*_rx + _ny*_ry
                    _b0 += _nx*_bb; _b1 += _ny*_bb
                _det = _a00*_a11 - _a01*_a01
                if abs(_det) > 1e-6:
                    _tx2 = (_a11*_b0 - _a01*_b1) / _det
                    _ty2 = (_a00*_b1 - _a01*_b0) / _det
                    # 合理性：在地图内、离观测射线不远
                    if 0.0 <= _tx2 <= 50.0 and 0.0 <= _ty2 <= 50.0 and math.hypot(_tx2-bx, _ty2-by) < 80.0:
                        _tri = (_tx2, _ty2)
            if _tri is not None:
                # 终点估计锁定：良态三角一旦成立就别被后续远距噪声观测打飞（实测越界/错侧
                # 跳变会让 goal/gate 方向来回翻转，狗在墙根振荡）——首个良态三角、或与现估计
                # 相差 <5m、或现估计越界时才更新；狗到 12m 内允许重新三角（近距离修正锁定）
                if (finish_est is None or not finish_est_tri
                        or math.hypot(_tri[0]-finish_est[0], _tri[1]-finish_est[1]) < 5.0
                        or not (0.0 <= finish_est[0] <= 50.0 and 0.0 <= finish_est[1] <= 50.0)
                        or math.hypot(bx-finish_est[0], by-finish_est[1]) < 12.0):
                    finish_est = _tri
                finish_est_tri = True
            elif not finish_est_tri and _fd <= 25.0:
                # 尺寸法中位数（>25m 的针孔距离是噪声，不更新——远距只信三角定位的方位角）
                _oxs = sorted(p[0] for p in finish_obs)
                _oys = sorted(p[1] for p in finish_obs)
                finish_est = (_oxs[len(_oxs)//2], _oys[len(_oys)//2])
                finish_est_tri = False   # 尺寸法中位数——偏远不准，只用于大致方向/不做到达依据
            # 估计矛盾解锁（2026-08-09 混合场实测）：狗已到估计附近（<3.5m）但当前针孔
            # 距离明显更远（>6m）——估计被锁定在错点（直通道方位角展开不足 15° 无法重三角），
            # 不解锁狗会在错点"到达"（实测 19.46m 误判）。用当前观测点重估并解除三角锁定，
            # 让估计随接近逐步走向球（针孔在 <10m 无遮挡时可靠）。
            if (finish_est is not None and finish_est_tri
                    and math.hypot(bx-finish_est[0], by-finish_est[1]) < 3.5
                    and 6.0 < _fd <= 25.0):
                _nx2, _ny2 = bx + _fd*math.cos(_wa), by + _fd*math.sin(_wa)
                if 0.0 <= _nx2 <= 50.0 and 0.0 <= _ny2 <= 50.0:   # 越界观测不采纳（坏投影防 HPA 目标格非法）
                    finish_est = (_nx2, _ny2)
                    finish_est_tri = False
            # 作废区域拉黑：估计落进 4 万步内作废过的 5m 格 → 拒绝锁定（防假终点死锁复发）
            if finish_est is not None:
                _ek = (int(finish_est[0] // 5), int(finish_est[1] // 5))
                for _bk0, _bk1, _bt in _goal_blacklist:
                    if _bk0 == _ek[0] and _bk1 == _ek[1] and step - _bt < 40000:
                        finish_est = None; finish_est_tri = False
                        break
            # 估计变更调试：定位假终点锁定的凭据
            if finish_est is not None and _est_dbg[0] != finish_est:
                print(f"  [EST] step={step} est={_est_dbg[0]}→({finish_est[0]:.1f},{finish_est[1]:.1f}) "
                      f"tri={finish_est_tri} fd={_fd:.1f} pos=({bx:.1f},{by:.1f})", flush=True)
                _est_dbg[0] = finish_est
            if not finish_announced and finish_est is not None:
                finish_announced = True
                print(f"  [FINISH-SEEN] step={step} 首次看到终点! est=({finish_est[0]:.1f},{finish_est[1]:.1f}) pos=({bx:.1f},{by:.1f})", flush=True)
            finish_area = _fa   # 当前帧 blob 面积（近距到达判定用）
            finish_bottom = _fbottom   # 当前帧 blob 底行 px
            finish_wa = _wa     # 当前帧球的世界方位角（视觉视线核查用）

    # ── 视觉伺服直奔（终局，全局漂移免疫）：球新鲜可见（≤25m）且感知视线通畅 →
    # 直接把当前观测投影设为目标路径（相对方位/距离是精确的，全局位姿误差不相关），
    # 逐帧更新目标点收敛到球。实测 v12：漂移 14m 时 finish_est 错到对角，
    # 狗在真球 6m 外却因估计错误不触发到达——眼睛看到的球，直接走过去就是。
    if vis is not None and vis.finish_obs is not None and not _STORM[0]:
        _sbr, _sfd, _sfa, _sfb, _sstep = vis.finish_obs
        if step - _sstep < 40 and _sfd <= 25.0:
            _swa = pyaw - _sbr
            _spx, _spy = bx + _sfd * math.cos(_swa), by + _sfd * math.sin(_swa)
            if (0.0 <= _spx <= 50.0 and 0.0 <= _spy <= 50.0
                    and line_clear(vx, vy, int(_spx / VOXEL), int(_spy / VOXEL))):
                path = [(_spx, _spy)]; path_idx = 0; path_is_goal = True

    if (path is None or path_idx >= len(path) or (mv.need_replan and mv.escape_steps == 0)
            # bounce 过多 → 强制重规划（路径穿未知/贴墙时 path 有效但不推进 → 死循环；两种模式都需要）
            or (mv.bounce - _plan_bounce_base > 8)):
        # 2026-08-09 修复（重大）：路径耗尽（门 3m 即达消耗/pure pursuit 走完）→ 必须置 None，
        # 否则下面 find_gates/直奔分支全在 "if path is None" 下 → 跳过 → 狗拿着耗尽路径
        # 朝旧门格僵尸式踏步 → 贴墙 STOP/bounce，直到攒够 8 次 bounce 才触发强制重规划。
        # 实测每个门消耗后白打 ~8 次 bounce ≈16-25s（全程 270 bounce/728s 的主头）。
        if path is not None and path_idx >= len(path):
            path = None; path_idx = 0
        if mv.need_replan and mv.escape_steps == 0:
            if patrol_paths or random_field is not None:
                # ── 移动障碍（巡逻/随机反弹）：只写 live WALL（纯感知 OBS_SEEN 格，scan 射线自动清除旧位置），不写 static 不重建 ──
                # 障碍 1m/s 移动，static 圈会残留误导 HPA；live WALL 让 wall_fn 实时看到当前位置绕行
                for _kk, _st in OBS_SEEN.items():
                    if _scan_step[0] - _st > 50:
                        continue
                    _cvx, _cvy = divmod(_kk, 4096)
                    _ddog = math.hypot((_cvx+0.5)*VOXEL-bx, (_cvy+0.5)*VOXEL-by)
                    # 不写狗身边 0.3m 内的格：狗就站在那，不可能是障碍——
                    # 旧版 1.5m 内全写会把狗自己的格标 WALL → blocked 永冻 →
                    # 移动障碍继续逼近造成持续接触（实测 collision 800+）
                    if 0.3 < _ddog < OBS_CLEAR + 0.8:
                        gset(_cvx, _cvy, WALL)
                path = None; path_idx = 0; wander = 0; last_dist = 999
                mv.need_replan = False
            else:
                # 固定障碍：写 static 安全圈 + 重建 HPA（见下）
                pass
        if mv.need_replan and mv.escape_steps == 0 and not patrol_paths and random_field is None:
            # 狗已 escape 走远（≥2m）——现在写障碍安全圈 + 重规划绕行
            # 注意：①写圈不按距离过滤（escape 可能走 8m 远，距狗>2m 会漏写 → 路径又穿障碍）
            #      ②KNOWN_MAP 写 static_grid（永久）；探索模式写 LIVE grid——
            #        static 圈在探索模式会累积成"圈笼"把狗困死（实测 28 圈 1.1 万 bounce）；
            #        live 圈被射线清除维护：障碍在，射线必撞它（圈不消）；障碍移走，射线穿过即清除
            #      ③半径 0.8m：精确圆判定后无 0.283 冗余，物理边界 0.7 + 格偏移 0.1 即可——
            #        1.0m 太宽 → 5m 通道缝 1.1m < 执行层需求 1.6m → 挤不过死循环
            #      ④2026-08-09 纯感知：圆心 = OBS_SEEN 雷达障碍格簇心（可见面弧，偏近狗侧 ~0.3m），
            #        不用 obs_world 真值（特权）——0.8m 圈仍完整覆盖 0.5m 半径障碍
            obs_r_grid = int(math.ceil((OBS_CLEAR + 0.2) / VOXEL))   # 0.9m=9 格扫描范围
            _written = 0
            # 近期看到的障碍格聚簇（≤3 格相邻归一簇），簇心 = 感知障碍位置
            _clusters = []
            for _kk, _st in OBS_SEEN.items():
                if _scan_step[0] - _st > 50:
                    continue
                _c = divmod(_kk, 4096)
                for _cl in _clusters:
                    if any(abs(_c[0]-_q[0]) + abs(_c[1]-_q[1]) <= 3 for _q in _cl):
                        _cl.append(_c); break
                else:
                    _clusters.append([_c])
            for _cl in _clusters:
                ox = (sum(c[0] for c in _cl) / len(_cl) + 0.5) * VOXEL
                oy = (sum(c[1] for c in _cl) / len(_cl) + 0.5) * VOXEL
                # 狗还在障碍脸上（<1.2m）不写圈：写了会把狗自己围进圈笼（实测贴脸写圈
                # 反复累积成笼 3.5 万 bounce）——等爬远再写
                if math.hypot(bx-ox, by-oy) < 1.2:
                    continue
                cx0, cy0 = int(round(ox/VOXEL)), int(round(oy/VOXEL))   # round 防浮点 8.2/0.1=81.9999
                for dy in range(-obs_r_grid, obs_r_grid+1):
                    for dx in range(-obs_r_grid, obs_r_grid+1):
                        # 世界坐标判定：格中心距障碍 < 0.8m → WALL（执行层物理边界 0.7 + 格偏移 0.1）
                        wx2, wy2 = (cx0+dx+0.5)*VOXEL, (cy0+dy+0.5)*VOXEL
                        if math.hypot(wx2-ox, wy2-oy) < 0.8:
                            if KNOWN_MAP_MODE:
                                if 0 <= cx0+dx < GRID_N and 0 <= cy0+dy < GRID_N and SG[cx0+dx, cy0+dy] != WALL:
                                    SG[cx0+dx, cy0+dy] = WALL
                                    _wd.clear()
                                    _pg_touch()
                                    _written += 1
                            elif gget(cx0+dx, cy0+dy) != WALL:
                                gset(cx0+dx, cy0+dy, WALL)   # 探索模式：live 圈（射线维护）
                                _written += 1
            if _written:
                print(f"  [OBS-WALL] 写安全圈 {_written} 格 (狗已逃到 ({bx:.1f},{by:.1f}))", flush=True)
                # 重建 HPA：门网络/距离场是构建时的 static 快照（不含障碍圈）——
                # 不重建则粗层门可能被圈堵住 → plan 返回 None → 死循环。
                # 0.8m 圈 + 精确圆判定后重建可行（1.0m 圈 + 5×5 过保守时重建会把缝堵死 bounce 1741）
                if KNOWN_MAP_MODE and 'HPAStar' in dir():
                    try:
                        t_re = time.time()
                        hpa = HPAStar(_hpa_wall, verbose=False)
                        print(f"  [HPA] 重建 {time.time()-t_re:.1f}s ({_written} 格安全圈)", flush=True)
                    except Exception as e:
                        hpa = None
                        print(f"  [HPA] 重建失败: {e}", flush=True)
            path = None; path_idx = 0; wander = 0; last_dist = 999
            mv.need_replan = False   # 撞障碍重规划：HPA wall_fn 已看到 WALL，新路径绕行
        if KNOWN_MAP_MODE:
            # KNOWN_MAP_MODE：地图已知，HPA* 分层规划到终点（成熟算法，长距离规划消失）
            # 目标：优先用相机看到的终点估计（无特权）；没看到才用 FINISH（--known-raw/load-map 测试模式）
            # 2026-08-09 修复：finish_est 越界（坏观测/解锁重估投影出图）直接跳过本轮规划——
            # 旧版把越界 est 喂给 HPA → 目标格 (-1,10) 非法 → PLAN-FAIL 死循环
            _est_ok = (finish_est is not None and 0.0 <= finish_est[0] <= 50.0
                       and 0.0 <= finish_est[1] <= 50.0)
            _goal = finish_est if _est_ok else FINISH
            if hpa is not None:
                t_plan0 = time.time()
                # 狗贴墙（dist<ROBOT_DIA=6）时 HPA 起点周围全被膨胀禁行 → 先找 HPA 可达格（dist≥6）再规划
                _psx, _psy = vx, vy
                if hpa.dist[vy, vx] < 6:
                    _ns = _nearest_walkable(vx, vy, max_r=15, min_dist=6)
                    if _ns is not None:
                        _psx, _psy = _ns
                # 目标格被移动障碍临时标记占住 → 就近吸附可达格（标记随障碍移动/射线清除消退）
                # 2026-08-12：格坐标钳制图内——est/自定义目标贴地图边（如 x=50.0）时
                # int(50/0.1)=500 越界 → HPA 细层调试打印 IndexError 崩溃（实测）
                _gcx = min(GRID_N - 1, max(0, int(_goal[0]/VOXEL)))
                _gcy = min(GRID_N - 1, max(0, int(_goal[1]/VOXEL)))
                if not traversable(_gcx, _gcy):
                    _nsg = _nearest_walkable(_gcx, _gcy, max_r=20)
                    if _nsg is not None:
                        _gcx, _gcy = _nsg
                gp = hpa.plan(_psx, _psy, _gcx, _gcy, max_expand=150000)
                # HPA 失败（门网络不连通/细层堵）→ 全图 A* fallback（_nearest_walkable 起点放宽）
                if gp is None:
                    _ns2 = _nearest_walkable(vx, vy, max_r=15)
                    if _ns2 is not None:
                        gp = astar_to(_ns2[0], _ns2[1], _gcx, _gcy)
                dt_plan = time.time() - t_plan0
                stats["hpa_plan_ms"] = stats.get("hpa_plan_ms", 0) + dt_plan
                stats["hpa_plans"] = stats.get("hpa_plans", 0) + 1
                # 调试：检查新路径是否离障碍太近（穿安全圈）
                if gp:
                    _min_obs_d = 1e9
                    for _px, _py in gp:
                        for _ox, _oy in obs_world:
                            _d = math.hypot((_px+0.5)*VOXEL-_ox, (_py+0.5)*VOXEL-_oy)
                            if _d < _min_obs_d: _min_obs_d = _d
                    if step % 1000 == 0 or _min_obs_d < 0.8:
                        print(f"    [PATH] plan#{stats['hpa_plans']} len={len(gp)} min_obs_dist={_min_obs_d:.2f} pos=({bx:.1f},{by:.1f})", flush=True)
                path = [((px+0.5)*VOXEL, (py+0.5)*VOXEL) for px, py in gp] if gp else None
            else:
                path = astar_to(vx, vy, min(GRID_N - 1, max(0, int(_goal[0]/VOXEL))),
                                min(GRID_N - 1, max(0, int(_goal[1]/VOXEL))))
            path_is_goal = True   # KNOWN_MAP 直奔终点：必须真到达，不能提前消耗
            if path:
                path_idx = 0; wander = 0; last_dist = 999; no_gate_count = 0
                _plan_bounce_base = mv.bounce   # KNOWN_MAP：重规划成功 → 重置 bounce 基线
                stats["gates_selected"] += 1
                if step % 10000 == 0:
                    print(f"    [DECIDE] KNOWN_MAP path={len(path)} →{_goal} pos=({bx:.1f},{by:.1f})", flush=True)
            else:
                # 规划失败（新障碍挡住旧地图路径 / HPA/astar 不可达）→ 先避障，下轮重规划
                # bounce 走节流（step=step：40 步一次）——规划失败多是移动障碍临时标记，
                # 障碍移走/射线清除后即恢复，每步 bounce 只会原地刷屏
                path = None; path_idx = 0; no_gate_count += 1
                if no_gate_count % 5 == 1:
                    print(f"  [PLAN-FAIL] step={step} hpa={'OK' if hpa else 'None'} pos=({bx:.1f},{by:.1f})", flush=True)
                mv._bounce(60, 120, step=step)
        else:
            # 强制重规划：bounce 过多说明当前路径失效（偏离/被挡），换目标
            if path is not None and mv.bounce - _plan_bounce_base > 8:
                if gate is not None and len(gates) > 1:
                    # 按门格集合拉黑（旧版只拉黑质心格，find_gates 过滤的是原始门格 → 拦不住）
                    bad_add_counted(_gate_cluster_cells.get((gate[1], gate[2]), [(gate[1], gate[2])]), (gate[1], gate[2]))
                path = None; path_idx = 0; wander = 0; last_dist = 999
                _plan_bounce_base = mv.bounce
                # 墙角困住检测：连续 3 次强制重规划位置几乎不动 → 狗卡在角落里原地 bounce，
                # 换门没用（门没错，是角落出不去）→ 回退到最近可达路标，脱离角落再决策
                if _wedge_pos is not None and math.hypot(bx-_wedge_pos[0], by-_wedge_pos[1]) < 0.5:
                    _wedge += 1
                else:
                    _wedge = 0
                _wedge_pos = (bx, by)
                if _wedge >= 3 and len(milestones) > 1:
                    for _mi in range(len(milestones)-1, -1, -1):
                        # 回退必须真的离开角落：跳过 <1.5m 的路标（实测路标就在脚边时回退空转）
                        if math.hypot((milestones[_mi][0]+0.5)*VOXEL-bx, (milestones[_mi][1]+0.5)*VOXEL-by) < 1.5:
                            continue
                        _bp = astar_to(vx, vy, milestones[_mi][0], milestones[_mi][1])
                        if _bp:
                            path = _bp; path_idx = 0; wander = 0; last_dist = 999
                            path_is_goal = False
                            stats["backtracks"] += 1
                            print(f"  [WEDGE] [{step}] 墙角困住 → 回退路标#{_mi} 出角落", flush=True)
                            _wedge = 0
                            break
            # ── 终点直奔模式（无特权核心）：相机看到绿色终点球 → finish_est 已知 →
            # 直接 A* 过去（感知地图，目标放宽 traversable）。持续观测让估计收敛。
            # "每个门都可能是疑似终点"——一旦某扇门后确认是终点，探索就结束了。
            if path is None and finish_est is not None and 0.0 <= finish_est[0] <= 50.0 and 0.0 <= finish_est[1] <= 50.0:
                _tx, _ty = int(finish_est[0]/VOXEL), int(finish_est[1]/VOXEL)
                path = astar_to(vx, vy, _tx, _ty, goal_relax=True)
                path_is_goal = True   # 终点直奔路径：接近消耗不适用（终点必须真到达）
                if path:
                    path_idx = 0; wander = 0; last_dist = 999; no_gate_count = 0
                    _plan_bounce_base = mv.bounce
                    stats["gates_selected"] += 1
                    if step % 2000 == 0:
                        print(f"    [GOAL] step={step} →est=({finish_est[0]:.1f},{finish_est[1]:.1f}) path={len(path)} pos=({bx:.1f},{by:.1f})", flush=True)
            if path is None:
                gates, came_from = find_gates(vx, vy)
                # 黑名单限量+无门即清：长时间 bounce 风暴会把本地前沿全拉黑（实测 37000 bounce
                # 拉黑上千格 → 永远 0 门卡死）——0 门时清空黑名单（环境已变，旧禁令不再适用）
                if not gates and bad_gates:
                    print(f"  [BAD-CLEAR] step={step} 0 门且黑名单 {len(bad_gates)} 格 → 清空重试", flush=True)
                    bad_gates.clear(); _bad_gates_order.clear()
                    gates, came_from = find_gates(vx, vy)
                # 终点方向偏差只在估计"靠谱"时用（在地图内）：越界的坏估计会把门评分
                # 拉向穿墙方向，狗在墙根死循环（实测 ch8 隔墙偷看后卡 5.3 万步）
                _fin_bias = finish_est if (finish_est is not None and 0.0 <= finish_est[0] <= 50.0
                                           and 0.0 <= finish_est[1] <= 50.0) else None
                gate = pick_gate(gates, EXPLORE_MODE, stuck=(no_gate_count > 0),
                                 robot=(bx, by), fin=_fin_bias, heading=mv.yaw)
                if step % 10000 == 0:
                    print(f"    [DECIDE] step={step} gates={len(gates)} gate={'None' if gate is None else f'({gate[1]*VOXEL:.1f},{gate[2]*VOXEL:.1f})'} pos=({bx:.1f},{by:.1f})", flush=True)
                if gate is not None:
                    cg, gx, gy, _gsize = gate
                    path = fine_path(vx, vy, gx, gy, came_from)
                    path_is_goal = False
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
                        path_is_goal = False
                    if not path:
                        # 所有候选门都不可达：当前 gate 进黑名单（按门格集合），
                        # 不卡死，bounce 找路（下轮重试其他门 / 无门 → BACK 回路标）
                        bad_add(_gate_cluster_cells.get((gate[1], gate[2]), [(gate[1], gate[2])]))
                        path = None; path_idx = 0; wander = 0; last_dist = 999
                        no_gate_count += 1
                        mv._bounce(60, 120)
                    else:
                        path_idx = 0; wander = 0; last_dist = 999
                        no_gate_count = 0
                        _plan_bounce_base = mv.bounce   # gate 成功 → 重置 bounce 基线
                        stats["gates_selected"] += 1
                        # ── gate 无进展检测：同一门区域反复选中但狗距离不降 → 假门 → 门格集合拉黑
                        # 2026-08-09 修复：质心会随地图更新漂移，按"区域"（质心距 ≤0.5m）判同一门；
                        # 拉黑按门格集合（质心拉黑在 find_gates 的原始门格过滤上形同虚设）。
                        gkey = (gx, gy)
                        gd = math.hypot((gx+0.5)*VOXEL-bx, (gy+0.5)*VOXEL-by)
                        _same_gate = (_last_gate_key is not None and
                                      math.hypot(gx-_last_gate_key[0], gy-_last_gate_key[1]) <= 5)
                        if _same_gate and _last_gate_dist is not None:
                            if gd >= _last_gate_dist - 0.5:   # 没明显接近（0.5m 容差）
                                _gate_stall += 1
                            else:
                                _gate_stall = 0
                        else:
                            _gate_stall = 0
                        _last_gate_key = gkey; _last_gate_dist = gd
                        if _gate_stall > 8:
                            # 阈值 8（曾试 3 实测回退）：3 时移动障碍临时挡路也被误判"假门"
                            # → 好门被拉黑 → 狗乱闯遭遇战（碰撞 195/14）。探测成本主要靠
                            # pick_gate 净空加权前置避免（宽门优先），stall 只做兜底。
                            bad_add_counted(_gate_cluster_cells.get(gkey, [gkey]), gkey)
                            _gate_stall = 0
                            print(f"  [GATE-STALL] gate=({(gx+0.5)*VOXEL:.1f},{(gy+0.5)*VOXEL:.1f}) 无进展 → 门格集合黑名单(计数)", flush=True)
                        _nj = f" 窄缝拒×{_NARROW_REJ[0]}" if _NARROW_REJ[0] else ""
                        print(f"  [GATE] [{step}] →({(gx+0.5)*VOXEL:.1f},{(gy+0.5)*VOXEL:.1f}) path={len(path)} gates={len(gates)}{_nj}", flush=True)
                else:
                    no_gate_count += 1
                    # BACK 限流：每个路标一次全程 A*（25 万扩展上限），无路标可达时会烧穿计算
                    # （实测困点时 +100 次 A*/几秒 = 主循环卡死）——每次最多试 20 个最近路标 + 每 400 步最多一次
                    if no_gate_count > MAX_NO_GATE and len(milestones) > 1 and step - _last_back_step > 400:
                        _last_back_step = step
                        saved = False
                        for i in range(len(milestones)-2, max(len(milestones)-22, -1), -1):
                            if i in back_blacklist:
                                continue  # 已 BACK 过但没新门，跳过（防死循环）
                            mx, my = milestones[i]
                            bp = astar_to(vx, vy, mx, my)
                            if bp:
                                path = bp; path_idx = 0; wander = 0; last_dist = 999
                                path_is_goal = False
                                no_gate_count = 0
                                stats["backtracks"] += 1
                                back_blacklist.add(i)
                                print(f"  [BACK] [{step}] →路标#{i}", flush=True)
                                saved = True
                                break
                        if not saved and finish_est is not None:
                            # 探索完成：无新门且无法回溯，且终点已看到 → 感知地图转已知地图，直奔终点
                            # （探索→已知闭环：狗用雷达把地图画完了，剩下就是导航。真实机器人同理）
                            # 2026-08-09：终点没看到（finish_est None）不切——切了也不知道去哪（旧版用 FINISH 真值=特权）
                            print(f"  [EXPLORE-DONE] step={step} 无新门/BACK全失败 → 切 KNOWN_MAP 直奔终点估计", flush=True)
                            SG[:] = G
                            # 狗当前位置格强制 FREE（防假墙/经验墙把起点堵死 → KNOWN_MAP 无法规划）
                            SG[int(bx/VOXEL), int(by/VOXEL)] = FREE
                            _wd.clear()
                            _pg_touch()
                            KNOWN_MAP_MODE = True
                            no_gate_count = 0; path = None; path_idx = 0; wander = 0; last_dist = 999
                            # 重建 HPA（探索模式从未创建——只在 KNOWN_MAP_MODE 创建；这里首次 import+构建）
                            try:
                                sys.path.insert(0, os.path.join(PROJ, "scripts"))
                                from hpa_star import HPAStar, CELL_M as HPA_CELL_M
                                def _hpa_wall(vx, vy):
                                    return gget_plan(vx, vy) == WALL
                                hpa = HPAStar(_hpa_wall, verbose=False)
                                print(f"  [HPA] 探索完成重建 (t={time.time()-t0:.0f}s gates={sum(len(v) for v in hpa.gates.values())} 门)", flush=True)
                            except Exception as e:
                                hpa = None
                                print(f"  [HPA] 重建失败: {e}", flush=True)
                        elif not saved:
                            # 地图扫完了但终点还没看到：继续 bounce 走动找新视角（诚实失败路径，不用真值）
                            no_gate_count = 0
                            mv._bounce(60, 120)

    # 执行
    if step % 10000 == 0:
        print(f"    [EXEC] step={step} path={'None' if path is None else len(path)} idx={path_idx} no_gate={no_gate_count} pos=({d.qpos[0]:.1f},{d.qpos[1]:.1f})", flush=True)
    if path is not None and path_idx < len(path):
        # 接近门即消耗：门是"看的目标"不是"踩的目标"——狗到路径终点 3m 内，雷达（30m）
        # 已把门后区域垂直扫透（墙脸整段标 WALL、伪前沿门整片消解），就算到达。
        # 不强求踩门格（门格常贴墙根，硬踩必触发 STOP/bounce 风暴；1.8m 仍太近会贴墙）
        if len(path) > 1 and not path_is_goal and math.hypot(path[-1][0]-bx, path[-1][1]-by) < 3.0:
            path_idx = len(path)
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
        # 目标 = best_i 之后第一个 ≥ 前瞻距离 的点。
        # 自适应前瞻（2026-08-09）：贴墙/贴障碍时 3m 前瞻会抄近道穿墙角/障碍圈
        # （实测 1m 窄通道里 lookahead 切进障碍体 → STOP/bounce 风暴）——
        # 离墙越近前瞻越短（窄通道慢慢跟着路径走），开阔地保持 3m 满速
        _la = PATH_LOOKAHEAD
        _wdh = wall_dist(vx, vy)
        if _wdh < 12:
            _la = max(1.0, _wdh * 0.25)
        tx, ty = path[path_idx]
        look_target = None
        for _i in range(path_idx, len(path)):
            lx, ly = path[_i]
            if math.hypot(lx-bx, ly-by) >= _la:
                look_target = (lx, ly)
                break
        if look_target is not None:
            tx, ty = look_target
        # DWA 决策：每 LIDAR_TICK 步，用 lookahead 目标
        # obstacles_motion：感知跟踪器估计的障碍速度（ObstacleTracker，聚类+帧间关联+EMA），
        # 替代真值 velocities（2026-08-12 审核整改）——DWA 模拟时用未来位置判定（运动预测，
        # 替代盲目膨胀——障碍朝狗移动则提前避让，远离则不误判）
        if mv.dwa is not None and step % LIDAR_TICK == 0:
            mv.dwa_target = mv.dwa.choose_velocity(
                robot_pos=(bx, by), yaw=mv.yaw,
                v_now=mv.speed, w_now=mv.omega,
                target=(tx, ty),
                blocked_fn=lambda pt: blocked(pt[0], pt[1]),
                obstacles_motion=(random_field.velocities if (args.dwa_truth_vel and random_field is not None)
                                  else (tracker.moving() if tracker is not None else None)),
                blocked_batch=blocked_batch)
            mv.dwa_t = step   # 咨询时刻：dwa_target=None 只在新鲜时才判全碰撞（防陈旧 None 死锁）
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
                        path_is_goal = False
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
        # path 不可用（find_gates 空/回溯失败）：有靠谱终点估计朝估计走（越界坏估计会
        # 把狗带进墙根死循环，必须在地图内才用），否则朝当前朝向直走，
        # 运动学约束自己避障（不能只 _bounce 不 step——原地转向永不移动，stuck 检测也不触发）
        # 2026-08-09 去特权：不再用 FINISH 真值当兜底方向
        if finish_est is not None and 0.0 <= finish_est[0] <= 50.0 and 0.0 <= finish_est[1] <= 50.0:
            _fx, _fy = finish_est
        elif gate is not None:
            _fx, _fy = (gate[1]+0.5)*VOXEL, (gate[2]+0.5)*VOXEL
        else:
            _fx, _fy = bx + math.cos(mv.yaw)*2.0, by + math.sin(mv.yaw)*2.0
        # 无路径分支也要咨询 DWA（否则 dwa_target 停在陈旧 None → 被判全碰撞 → 死锁）；
        # 移动障碍在此分支同样需要运动预测避让（速度=感知跟踪估计，无真值）
        if mv.dwa is not None and step % LIDAR_TICK == 0:
            mv.dwa_target = mv.dwa.choose_velocity(
                robot_pos=(bx, by), yaw=mv.yaw,
                v_now=mv.speed, w_now=mv.omega,
                target=(_fx, _fy),
                blocked_fn=lambda pt: blocked(pt[0], pt[1]),
                obstacles_motion=(random_field.velocities if (args.dwa_truth_vel and random_field is not None)
                                  else (tracker.moving() if tracker is not None else None)),
                blocked_batch=blocked_batch)
            mv.dwa_t = step
        mv.step(_fx, _fy, step)

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

    # 到达判定（无特权）：①狗到"自己看到的终点估计"3.5m 内（三角定位已收敛）；
    # ②或当前帧针孔距离 <3m（球近在咫尺）**且感知地图视线可达**
    # （隔墙看到不算到达——首次实测狗在 ch8 隔墙看到球面积触发，真值还差 3.84m 且中间有墙）。
    # ②2026-08-09 修复：旧版用 blob 面积 >25000px 判近——那是 ch9 背景板遮挡球
    # （只剩 10-16% 面积）时代的标定；标牌贴墙后球无遮挡，14m 处面积就超 25000
    # （实测 14.01m 误判到达）→ 改用针孔距离（无遮挡时可靠），并要求 <3m
    # （360p 下针孔低估 ~17%：3.0 门槛 ≈ 真值 3.5m 内）。
    # 真值距离只进成绩单（物理事实评分，不参与决策，与碰撞统计同理）
    _arrived = False
    # ① 三角定位收敛 + 3.5m 内 + 面积确认 + **视觉视线核查**：沿最近观测方位走 dist(est)
    #    距离的点不能隔感知墙——隔墙看到球顶盖时该点落在墙后/墙上 → 不算到达（实测 ch8
    #    隔墙三角到错侧点 5.62m 判到达的漏网情形）；同空间看球则通畅
    if (finish_est is not None and finish_est_tri
            and math.hypot(bx-finish_est[0], by-finish_est[1]) < 3.5
            and finish_area > 12000
            and vis is not None and vis.finish_obs is not None
            and step - vis.finish_obs[4] < 200 and vis.finish_obs[1] < 4.5):   # 针孔交叉验证（防锁定错点到达）+ 200 步窗口（<2.5m 球出视野盲区）；阈值 4.5：近距 bob/偏帧使针孔读数偏大
        _dest = math.hypot(bx-finish_est[0], by-finish_est[1])
        _pwx = bx + _dest * math.cos(finish_wa)
        _pwy = by + _dest * math.sin(finish_wa)
        if line_clear(vx, vy, int(_pwx/VOXEL), int(_pwy/VOXEL)):
            _arrived = True
    elif (vis is not None and vis.finish_obs is not None
          and step - vis.finish_obs[4] < 200 and vis.finish_obs[1] < 4.0
          # 新鲜窗口 200 步（1s）：实测 360p 下球 <2.5m 会超出相机视野检测不到——
          # 狗从最后可见走到球前只需 <0.3s，窗口内到达判定仍有效；
          # 阈值 4.0：近距 bob/偏帧时 blob 缺块针孔偏远（3m 实测读 3.3+），3.0 会漏触发
          and finish_est is not None):
        _pwx = bx + 3.0 * math.cos(finish_wa)
        _pwy = by + 3.0 * math.sin(finish_wa)
        if line_clear(vx, vy, int(_pwx/VOXEL), int(_pwy/VOXEL)):
            _arrived = True
    if _arrived:
        _true_d = math.hypot(bx-FINISH[0], by-FINISH[1])
        print(f"\n  ★ ARRIVED! @({bx:.1f},{by:.1f}) step={step} ms={len(milestones)} 距真值终点={_true_d:.2f}m", flush=True)
        stats["arrived"] = True
        break

    # 直奔死锁保险（2026-08-12 实测纯墙场 est 锁到狗脚边 (49.2,47.6)，9 万步无进展）：
    # 狗贴在估计点 2.5m 内 >4000 步（20s）仍判不了到达 = 球根本不在那（坏估计）→
    # 作废估计 + 清空观测缓存，回 frontier 探索；该 5m 区域拉黑 4 万步防同点再锁。
    if finish_est is not None and math.hypot(bx-finish_est[0], by-finish_est[1]) < 2.5:
        if _goal_close_since[0] is None:
            _goal_close_since[0] = step
        elif step - _goal_close_since[0] > 4000:
            print(f"  [GOAL-INVALID] step={step} est=({finish_est[0]:.1f},{finish_est[1]:.1f}) "
                  f"贴估{step-_goal_close_since[0]}步未到达 → 作废回探索", flush=True)
            _goal_blacklist.append((int(finish_est[0]//5), int(finish_est[1]//5), step))
            finish_est = None; finish_est_tri = False
            finish_obs.clear(); finish_rays.clear()
            _est_dbg[0] = None
            path = None; path_idx = 0; wander = 0; last_dist = 999
            _goal_close_since[0] = None
    else:
        _goal_close_since[0] = None

    # 碰撞检测（主人要求碰撞=0）：真实几何——机器人中心进入障碍安全圈才算碰撞
    # is_obstacle_world 已含 OBS_CLEAR=0.7（障碍半径0.5+机器人半径0.2），中心<0.7m即碰撞
    if is_obstacle_world(d.qpos[0], d.qpos[1]):
        stats["collisions"] += 1
        stats.setdefault("collision_pos", []).append([round(d.qpos[0], 2), round(d.qpos[1], 2)])
        if stats["collisions"] == 1 or stats["collisions"] % 100 == 0:
            print(f"  [COLLISION] #{stats['collisions']} @({d.qpos[0]:.2f},{d.qpos[1]:.2f}) step={step}", flush=True)

    # 离屏渲染
    if RENDER_OK and args.render_every > 0 and step % args.render_every == 0:
        render_frame(step)

    # 进度日志
    if step % 20000 == 0:
        cov = coverage_pct()
        _drift = f" drift={odom.error(d.qpos[0], d.qpos[1]):.2f}m(评分)" if odom is not None else ""
        print(f"  ... step={step} F={_count(FREE)} W={_count(WALL)} ms={len(milestones)} cov={cov:.1f}% t={time.time()-t0:.0f}s pos=({d.qpos[0]:.1f},{d.qpos[1]:.1f}) yaw={math.degrees(d.qpos[2]):.0f}°{_drift}", flush=True)

# ── 收尾统计 ──
stats["steps"] = step
stats["time_sec"] = round(time.time() - t0, 2)
stats["bounces"] = mv.bounce
stats["collisions"] = stats.get("collisions", 0)
stats["narrow_rej"] = _NARROW_REJ[0]   # 门宽度判断：被拒绝的窄缝门计数（<2×PASS_CLEAR）
stats["pass_clear_m"] = PASS_CLEAR_M
if vis is not None:
    stats["landmarks_seen"] = vis.total_detected
    stats["landmarks_unique"] = len(vis.seen_ids)
    stats["landmark_channels"] = sorted(vis.seen_ids)
stats["milestones"] = len(milestones)
stats["final_pos"] = [round(d.qpos[0], 2), round(d.qpos[1], 2)]
stats["final_coverage"] = round(coverage_pct(), 2)
# 终点发现（无特权）：估计、首次看到步数、距真值距离（评分用真值，决策不用）
stats["finish_seen"] = finish_est is not None
if finish_est is not None:
    stats["finish_est"] = [round(finish_est[0], 2), round(finish_est[1], 2)]
    stats["finish_est_err"] = round(math.hypot(finish_est[0]-FINISH[0], finish_est[1]-FINISH[1]), 2)
stats["dist_to_true_finish"] = round(math.hypot(d.qpos[0]-FINISH[0], d.qpos[1]-FINISH[1]), 2)
stats["loc_mode"] = "odom" if odom is not None else "truth(DEBUG)"
if odom is not None:
    # 里程计漂移（评分用真值，决策不用）：最终估计位姿与真值的差距
    stats["odom_drift_final"] = round(odom.error(d.qpos[0], d.qpos[1]), 2)
    stats["odom_corrections"] = odom.corrections      # 二维码绝对修正次数
    stats["odom_bias_v_pct"] = round(odom.bias_v * 100, 2)
    stats["odom_bias_w_degs"] = round(math.degrees(odom.bias_w), 3)
if matcher is not None:
    stats["match_tries"] = matcher.matches
    stats["match_corrections"] = matcher.corrections   # scan-matching 激光里程计修正次数
if tracker is not None:
    stats["tracker_tracks"] = len(tracker.tracks)
stats["seed"] = FIXED_SEED
stats["free_cells"] = _count(FREE)
stats["wall_cells"] = _count(WALL)
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
