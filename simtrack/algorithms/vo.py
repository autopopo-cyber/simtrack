"""
VO 避障算法 (Velocity Obstacle 简化版)

左右两侧各找切线方向，选最接近目标方向的一条。
带滞后阈值防抖 + 宽扫兜底 + blocked 检测。

源自朗毅 VO 算法，纯 Python 实现，零外部依赖。
"""

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

from simtrack.algorithms.base import AvoidanceAlgorithm, AvoidanceResult


# ── 常量 ──
ARC_SIM_STEPS = 80
TANGENT_OMEGA_RATIO = 0.05
HYSTERESIS_DEG = 3.0
HYSTERESIS_RAD = math.radians(HYSTERESIS_DEG)
WIDE_SWEEP_OMEGA = 4.0
SIDE_STABILITY_FRAMES = 20


# ═══════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════

def _normalize_angle(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _heading_diff(a: float, b: float) -> float:
    d = abs(_normalize_angle(a - b))
    return min(d, 2 * math.pi - d)


def _obstacle_clearance(obs_pos, obs_radius, point, clearance):
    d = math.hypot(point[0] - obs_pos[0], point[1] - obs_pos[1])
    return d - obs_radius - clearance


def _simulate_arc_hit(robot_pos, start_heading, speed, omega, horizon,
                      obstacles, arc_side_sign, arc_sim_steps=ARC_SIM_STEPS,
                      robot_radius=0.25, inflation=0.10):
    """模拟圆弧轨迹，检测碰撞。"""
    dt = horizon / arc_sim_steps
    fwd = (math.cos(start_heading), math.sin(start_heading))
    left = (-fwd[1], fwd[0])
    arc_start = (
        robot_pos[0] - arc_side_sign * robot_radius * left[0],
        robot_pos[1] - arc_side_sign * robot_radius * left[1],
    )

    for i in range(1, arc_sim_steps + 1):
        t = dt * i
        if abs(omega) < 1e-6:
            px = arc_start[0] + speed * t * math.cos(start_heading)
            py = arc_start[1] + speed * t * math.sin(start_heading)
        else:
            h = start_heading + omega * t
            px = arc_start[0] + (speed / omega) * (math.sin(h) - math.sin(start_heading))
            py = arc_start[1] - (speed / omega) * (math.cos(h) - math.cos(start_heading))

        p = (px, py)
        for (ox, oy, orad) in obstacles:
            if _obstacle_clearance((ox, oy), orad, p, inflation) <= 0:
                return (True, t, _normalize_angle(start_heading + omega * t), p)

    return (False, horizon, start_heading, None)


def _find_hit_tangent(robot_pos, start_heading, speed, horizon, omega_max,
                      sign, obstacles, fallback_heading,
                      robot_radius=0.25, inflation=0.10):
    """从 omega_max 扫到 0，找第一个碰撞切线。"""
    step = max(1e-4, abs(omega_max) * TANGENT_OMEGA_RATIO)
    omega = omega_max
    while omega >= 0:
        w = sign * omega
        hit, _, hit_heading, _ = _simulate_arc_hit(
            robot_pos, start_heading, speed, w, horizon, obstacles, sign,
            robot_radius=robot_radius, inflation=inflation,
        )
        if hit:
            return _normalize_angle(hit_heading)
        omega -= step
    return _normalize_angle(fallback_heading)


def _pick_best_heading(desired_heading, left_heading, right_heading,
                       last_side=0, last_heading=0.0, stable_count=0):
    """滞后阈值 + 帧计数门控选择切线侧。"""
    d_left = _heading_diff(left_heading, desired_heading)
    d_right = _heading_diff(right_heading, desired_heading)
    best_side = +1 if d_left <= d_right else -1

    if last_side != 0:
        if abs(d_left - d_right) < HYSTERESIS_RAD:
            return last_heading, last_side
        if best_side != last_side and stable_count < SIDE_STABILITY_FRAMES:
            return last_heading, last_side

    if best_side == +1:
        return left_heading, +1
    return right_heading, -1


def _ray_obstacle_distance(robot_pos, ray_dir, obstacles, max_range):
    """单射线→最近障碍物距离。"""
    px, py = robot_pos
    dx, dy = ray_dir
    best_t = max_range
    for (ox, oy, orad) in obstacles:
        fx, fy = px - ox, py - oy
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
        elif t1 < 0 <= t2 and 0 < best_t:
            best_t = 0.0
        elif t2 >= 0 and t2 < best_t:
            best_t = t2
    return best_t


# ═══════════════════════════════════════════
# VOAlgorithm
# ═══════════════════════════════════════════

class VOAlgorithm(AvoidanceAlgorithm):
    """VO 避障算法。

    左右两侧各搜索切线方向，选择最接近目标方向的一条。
    带滞后阈值防抖 + 宽扫兜底 + blocked 检测。

    参数:
        max_speed:      最大速度 (m/s, 默认 2.0)
        robot_radius:   机器人半径 (m, 默认 0.25)
        inflation:      障碍物膨胀半径 (m, 默认 0.10)
        horizon:        前视时间 (s, 默认 3.0)
        omega_max:      最大搜索角速度 (rad/s, 默认 2.5)
        avoid_speed_ratio: 避障时速度降低比例 (默认 0.6)
    """

    def __init__(
        self,
        max_speed: float = 2.0,
        robot_radius: float = 0.25,
        inflation: float = 0.10,
        horizon: float = 3.0,
        omega_max: float = 2.5,
        avoid_speed_ratio: float = 0.6,
    ):
        super().__init__(max_speed, robot_radius)
        self.inflation = inflation
        self.horizon = horizon
        self.omega_max = omega_max
        self.avoid_speed_ratio = avoid_speed_ratio

        # 内部状态
        self._last_heading = 0.0
        self._last_side = 0
        self._stable_count = 0

    def choose_heading(
        self,
        robot_pos: Tuple[float, float],
        robot_speed: float,
        target_pos: Tuple[float, float],
        obstacles: List[Tuple[float, float, float]],
    ) -> AvoidanceResult:
        """选择最优航向。"""
        desired = math.atan2(
            target_pos[1] - robot_pos[1],
            target_pos[0] - robot_pos[0],
        )

        # 直线路径无障碍 → 直行
        hit_straight, _, _, _ = _simulate_arc_hit(
            robot_pos, desired, robot_speed, 0.0, self.horizon,
            obstacles, 0, robot_radius=self.robot_radius,
            inflation=self.inflation,
        )
        if not hit_straight:
            self._last_heading = desired
            self._last_side = 0
            return AvoidanceResult(
                heading=desired,
                speed=self.max_speed,
                avoiding=False,
            )

        # 左右切线
        left_h = _find_hit_tangent(
            robot_pos, desired, robot_speed, self.horizon, self.omega_max,
            +1, obstacles, desired, robot_radius=self.robot_radius,
            inflation=self.inflation,
        )
        right_h = _find_hit_tangent(
            robot_pos, desired, robot_speed, self.horizon, self.omega_max,
            -1, obstacles, desired, robot_radius=self.robot_radius,
            inflation=self.inflation,
        )

        chosen, side = _pick_best_heading(
            desired, left_h, right_h,
            self._last_side, self._last_heading, self._stable_count,
        )

        # 帧计数更新
        if side == self._last_side and side != 0:
            self._stable_count += 1
        elif side != self._last_side:
            self._stable_count = 0
        self._last_heading = chosen
        self._last_side = side

        return AvoidanceResult(
            heading=chosen,
            speed=self.max_speed * self.avoid_speed_ratio,
            avoiding=True,
        )

    # ── 工具: blocked 检测 ──

    def check_blocked(self, robot_pos, obstacles, forward_heading=None,
                      num_rays=36, max_range=15.0):
        """LiDAR 扫描 360°，判定 blocked / MAP_ERROR。

        Returns:
            "ok" | "blocked" | "map_error"
        """
        proximity = 2.0 * self.robot_radius
        all_close = True
        forward_blocked = True

        for i in range(num_rays):
            angle = 2.0 * math.pi * i / num_rays
            ray_dir = (math.cos(angle), math.sin(angle))
            hit_dist = _ray_obstacle_distance(robot_pos, ray_dir, obstacles, max_range)

            if hit_dist >= proximity:
                all_close = False

            if forward_heading is not None:
                angle_diff = _heading_diff(angle, forward_heading)
                if angle_diff <= math.pi / 2.0 and hit_dist >= max_range:
                    forward_blocked = False

        if all_close:
            return "blocked"
        if forward_heading is not None and forward_blocked:
            return "map_error"
        return "ok"
