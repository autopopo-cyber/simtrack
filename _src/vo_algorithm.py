"""
VO避障算法 — 纯函数库 v5（零外部依赖）
剥离自 Marathongo tangent_arc_navigation，可脱离 MuJoCo 独立测试

v5 改进：
  1. 滞后阈值 — |d_left - d_right| < 3° 坚守当前侧（替代帧计数防抖）
  2. 宽扫兜底 — 常规 omega 找不到出路时扩大搜索
  3. 保留 DebounceState 兼容旧 API
"""
import math
from dataclasses import dataclass

# ── 常量 ──
ROBOT_RADIUS = 0.25
OBSTACLE_INFLATION = 0.10
ARC_SIM_STEPS = 80
TANGENT_OMEGA_RATIO = 0.05
MAX_SPEED = 2.0

# ── 滞后阈值 (v5 新增) ──
HYSTERESIS_DEG = 3.0
HYSTERESIS_RAD = math.radians(HYSTERESIS_DEG)

# ── 宽扫 (v5 新增) ──
WIDE_SWEEP_OMEGA = 4.0

# ── 防抖参数（保留兼容） ──
SIDE_STABILITY_FRAMES = 20
STUCK_DIST_THRESHOLD = 0.05
STUCK_FRAME_THRESHOLD = 100
BACKOFF_SPEED = 0.2
BACKOFF_PERTURB = math.radians(30)

# ── blocked 检测参数 ──
BLOCKED_RAY_COUNT = 36
BLOCKED_RANGE = 15.0
BLOCKED_PROXIMITY = None  # 实际检测用 2×ROBOT_RADIUS


# ═══════════════════════════════════════════
# NavStatus — 导航状态枚举
# ═══════════════════════════════════════════

class NavStatus:
    OK = "ok"              # 正常行驶
    STUCK = "stuck"        # 暂时卡住，正在逃逸
    BLOCKED = "blocked"    # 所有方向堵死，无法通过
    MAP_ERROR = "map_error" # 地图生成错误（如全封闭）


# ═══════════════════════════════════════════
# 角度工具
# ═══════════════════════════════════════════

def normalize_angle(a):
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def heading_diff(a, b):
    """两个航向之间的最小夹角（弧度，[0, π]）"""
    d = abs(normalize_angle(a - b))
    return min(d, 2*math.pi - d)


# ═══════════════════════════════════════════
# 碰撞检测
# ═══════════════════════════════════════════

def obstacle_clearance(obs_pos, obs_radius, point, clearance):
    """点到障碍物边缘的安全距离（负值=碰撞）"""
    d = math.hypot(point[0]-obs_pos[0], point[1]-obs_pos[1])
    return d - obs_radius - clearance

def simulate_arc_hit(robot_pos, start_heading, speed, omega, horizon,
                     obstacles, arc_side_sign, arc_sim_steps=ARC_SIM_STEPS):
    """
    模拟机器人沿圆弧轨迹运动，检测是否与障碍物碰撞。
    返回: (hit:bool, hit_time, hit_heading, hit_point)
    """
    dt = horizon / arc_sim_steps
    fwd = (math.cos(start_heading), math.sin(start_heading))
    left = (-fwd[1], fwd[0])
    arc_start = (
        robot_pos[0] - arc_side_sign * ROBOT_RADIUS * left[0],
        robot_pos[1] - arc_side_sign * ROBOT_RADIUS * left[1],
    )

    for i in range(1, arc_sim_steps + 1):
        t = dt * i
        if abs(omega) < 1e-6:
            px = arc_start[0] + speed * t * math.cos(start_heading)
            py = arc_start[1] + speed * t * math.sin(start_heading)
        else:
            h = start_heading + omega * t
            px = arc_start[0] + (speed/omega) * (math.sin(h) - math.sin(start_heading))
            py = arc_start[1] - (speed/omega) * (math.cos(h) - math.cos(start_heading))

        p = (px, py)
        for (ox, oy, orad) in obstacles:
            if obstacle_clearance((ox, oy), orad, p, OBSTACLE_INFLATION) <= 0:
                hit_heading = normalize_angle(start_heading + omega * t)
                return (True, t, hit_heading, p)

    return (False, horizon, start_heading, None)


# ═══════════════════════════════════════════
# 切线搜索
# ═══════════════════════════════════════════

def find_hit_tangent(robot_pos, start_heading, speed, horizon, omega_max,
                     sign, obstacles, fallback_heading):
    """从 omega_max 往 0 扫，找第一个命中障碍物的圆弧切线方向"""
    step = max(1e-4, abs(omega_max) * TANGENT_OMEGA_RATIO)
    omega = omega_max
    while omega >= 0:
        w = sign * omega
        hit, _, hit_heading, _ = simulate_arc_hit(
            robot_pos, start_heading, speed, w, horizon, obstacles, sign
        )
        if hit:
            return normalize_angle(hit_heading)
        omega -= step
    return normalize_angle(fallback_heading)


def pick_best_heading(desired_heading, left_heading, right_heading,
                      debounce_state=None):
    """
    选左右切线中最接近目标方向的。
    双保险：① 滞后阈值 — 两侧成本差 < HYSTERESIS_RAD 坚守当前侧；
           ② 帧计数门控 — 未达稳定帧数不允许切侧（防快速振荡）。
    返回: (heading, side: +1=左/-1=右)
    """
    d_left = heading_diff(left_heading, desired_heading)
    d_right = heading_diff(right_heading, desired_heading)

    best_side = +1 if d_left <= d_right else -1

    if debounce_state and debounce_state.last_side != 0:
        # 保险①：滞后阈值 —— 两侧太近就守住
        if abs(d_left - d_right) < HYSTERESIS_RAD:
            return debounce_state.last_heading, debounce_state.last_side
        # 保险②：帧计数门控 —— 想切侧但不够稳定就拦住
        if best_side != debounce_state.last_side:
            if not debounce_state.is_side_stable():
                return debounce_state.last_heading, debounce_state.last_side

    if best_side == +1:
        return left_heading, +1
    else:
        return right_heading, -1


# ═══════════════════════════════════════════
# 防抖状态机（保留兼容）
# ═══════════════════════════════════════════

@dataclass
class DebounceState:
    """跟踪方位选择稳定性（v5：滞后阈值下仅存少量辅助字段）"""
    last_heading: float = 0.0
    last_side: int = 0           # +1=左, -1=右, 0=直行/未知
    side_stable_count: int = 0   # 保留兼容
    stuck_count: int = 0         # 由外部管理
    last_dist_to_goal: float = float('inf')  # 由外部管理

    def is_side_stable(self) -> bool:
        return self.side_stable_count >= SIDE_STABILITY_FRAMES

    def is_stuck(self) -> bool:
        return self.stuck_count >= STUCK_FRAME_THRESHOLD


# ═══════════════════════════════════════════
# blocked / MAP_ERROR 检测 (v6)
# ═══════════════════════════════════════════

def _ray_obstacle_distance(robot_pos, ray_dir, obstacles, max_range):
    """单条射线到最近障碍物的距离（无命中返回 max_range）"""
    px, py = robot_pos
    dx, dy = ray_dir
    best_t = max_range

    for (ox, oy, orad) in obstacles:
        # 射线-圆交点: |P + t*D - C| = R
        fx = px - ox
        fy = py - oy
        b = dx * fx + dy * fy
        c = fx * fx + fy * fy - orad * orad

        disc = b * b - c
        if disc < 0:
            continue

        sqrt_disc = math.sqrt(disc)
        t1 = -b - sqrt_disc
        t2 = -b + sqrt_disc

        if t1 >= 0 and t1 < best_t:
            best_t = t1
        elif t1 < 0 and t2 >= 0:
            # 射线起点在障碍物内部 → 距离为 0
            best_t = 0.0
        elif t2 >= 0 and t2 < best_t:
            best_t = t2

    return best_t


def check_blocked(robot_pos, obstacles, forward_heading=None,
                  max_range=BLOCKED_RANGE, num_rays=BLOCKED_RAY_COUNT,
                  robot_radius=ROBOT_RADIUS):
    """
    用 LiDAR 扫描 360°，判定无路可走的状态。

    返回 NavStatus:
      - BLOCKED: 所有方向 2×robot_radius 内都有障碍物
      - MAP_ERROR: 180° 正面全部有障碍物（道路设计问题）
      - OK: 存在可通过方向

    forward_heading: 前进方向航向（弧度），用于 MAP_ERROR 判定。
                     None 时仅检测 BLOCKED。
    """
    proximity = 2.0 * robot_radius
    all_blocked_close = True    # 所有方向 proximity 内都堵
    forward_all_blocked = True  # 180° 正面全堵
    any_forward_clear = False

    for i in range(num_rays):
        angle = 2.0 * math.pi * i / num_rays
        ray_dir = (math.cos(angle), math.sin(angle))

        hit_dist = _ray_obstacle_distance(
            robot_pos, ray_dir, obstacles, max_range
        )

        if hit_dist >= proximity:
            all_blocked_close = False

        if forward_heading is not None:
            # 判断该射线是否在 180° 正面扇区内
            angle_diff = heading_diff(angle, forward_heading)
            if angle_diff <= math.pi / 2.0:
                if hit_dist >= max_range:
                    forward_all_blocked = False
                    any_forward_clear = True

    if all_blocked_close:
        return NavStatus.BLOCKED

    if forward_heading is not None and forward_all_blocked:
        return NavStatus.MAP_ERROR

    return NavStatus.OK


# ═══════════════════════════════════════════
# 主入口：选最优航向
# ═══════════════════════════════════════════

def choose_optimal_heading(robot_pos, robot_speed, target_pos, obstacles,
                           debounce: DebounceState | None = None,
                           wide_sweep: bool = False):
    """
    左右两侧各找切线方向，选最接近目标方向的。
    有防抖状态时加入侧切换滞回（v5：滞后阈值）。
    返回: (heading, avoiding:bool)
    """
    desired_heading = math.atan2(
        target_pos[1] - robot_pos[1],
        target_pos[0] - robot_pos[0]
    )
    horizon = 3.0
    omega_max = WIDE_SWEEP_OMEGA if wide_sweep else 2.5

    # 直线路径无障碍 → 直行
    hit_straight, _, _, _ = simulate_arc_hit(
        robot_pos, desired_heading, robot_speed, 0.0, horizon, obstacles, 0
    )
    if not hit_straight:
        if debounce:
            debounce.last_heading = desired_heading
            debounce.last_side = 0
        return desired_heading, False

    # 左右切线
    left_heading = find_hit_tangent(
        robot_pos, desired_heading, robot_speed, horizon, omega_max,
        +1, obstacles, desired_heading
    )
    right_heading = find_hit_tangent(
        robot_pos, desired_heading, robot_speed, horizon, omega_max,
        -1, obstacles, desired_heading
    )

    chosen, side = pick_best_heading(desired_heading, left_heading, right_heading, debounce)

    if debounce:
        # 帧计数追踪：同侧累计，切侧清零
        if side == debounce.last_side and side != 0:
            debounce.side_stable_count += 1
        elif side != debounce.last_side:
            debounce.side_stable_count = 0
        debounce.last_heading = chosen
        debounce.last_side = side

    return chosen, True
