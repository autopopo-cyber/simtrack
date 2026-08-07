"""DWA 局部规划器 (Dynamic Window Approach)

速度空间采样：动态窗口（加速度约束）内采样 (v, ω) 组合，
逐条模拟圆弧轨迹检测碰撞，按 heading/clearance/velocity/smoothness 加权评分选最优。

- 碰撞轨迹硬排除
- 全部碰撞 → 返回 None（调用方触发 _bounce 兜底）
- 第一版不做障碍运动预测（狗 4m/s vs 障碍 1m/s，0.05s 重决策足够）
"""

import math
import numpy as np


class DWAAlgorithm:
    def __init__(
        self,
        v_max: float = 4.0,
        w_max: float = 1.5,          # YAW_RATE
        a_accel: float = 5.0,        # A_ACCEL
        a_decel: float = 8.0,        # A_DECEL
        a_w: float = 10.0,           # 角加速度上限
        T: float = 0.05,             # 决策周期（LIDAR_TICK × timestep）
        horizon: float = 1.5,        # 轨迹模拟时长 (s)
        dt_sample: float = 0.05,     # 模拟步长 (s)
        n_v: int = 7,
        n_w: int = 11,
        w_heading: float = 0.6,
        w_clearance: float = 0.25,
        w_velocity: float = 0.1,
        w_smooth: float = 0.05,
        clearance_max: float = 2.0,  # 归一化上限 (m)
        stop_margin: float = 0.4,    # 停车安全余量（低于此距离的轨迹直接排除）
        pred_t: float = 0.5,         # 障碍运动预测时间上限 (s)：障碍每 1s 20% 变向+撞墙反弹，
                                     # 长期预测不可靠——只外推 0.5s（决策周期+反应时间）
    ):
        self.v_max = v_max
        self.w_max = w_max
        self.a_accel = a_accel
        self.a_decel = a_decel
        self.a_w = a_w
        self.T = T
        self.horizon = horizon
        self.dt_sample = dt_sample
        self.n_v = n_v
        self.n_w = n_w
        self.w_heading = w_heading
        self.w_clearance = w_clearance
        self.w_velocity = w_velocity
        self.w_smooth = w_smooth
        self.clearance_max = clearance_max
        self.stop_margin = stop_margin
        self.pred_t = pred_t

    def choose_velocity(self, robot_pos, yaw, v_now, w_now, target, blocked_fn,
                        obstacles_motion=None):
        """返回最优 (v*, ω*)；全部轨迹碰撞 → None。

        Args:
            robot_pos: (x, y) 当前位置
            yaw:       当前朝向 (rad)
            v_now:     当前线速度 (m/s)
            w_now:     当前角速度 (rad/s)
            target:    (tx, ty) lookahead 目标
            blocked_fn: callable(point) -> bool，判定点是否被堵（静态：墙 + 障碍当前位置）
            obstacles_motion: 可选 [(ox, oy, vx, vy, r), ...] 移动障碍（速度向量 + 半径），
                轨迹模拟时用障碍**未来位置**判定——补偿 1m/s 障碍运动，不做盲目膨胀
        """
        # ① 动态窗口
        # v_lo 强制为 0：全速接近障碍时必须能选低速/停车轨迹（否则窗口全高速 → 全碰撞）
        v_lo = 0.0
        v_hi = min(self.v_max, v_now + self.a_accel * self.T)
        w_lo = max(-self.w_max, w_now - self.a_w * self.T)
        w_hi = min(self.w_max, w_now + self.a_w * self.T)
        if v_hi - v_lo < 1e-6:
            v_hi = v_lo + 1e-6
        if w_hi - w_lo < 1e-6:
            w_hi = w_lo + 1e-6

        target_angle = math.atan2(target[1] - robot_pos[1], target[0] - robot_pos[0])

        vs = np.linspace(v_lo, v_hi, self.n_v)
        ws = np.linspace(w_lo, w_hi, self.n_w)

        best = None
        best_score = -1e18
        for v in vs:
            for w in ws:
                # ③ 轨迹模拟
                traj = self._simulate(robot_pos, yaw, v, w)
                # 碰撞检测 + 最近障碍距离（静态 blocked_fn + 移动障碍未来位置）
                hit, min_clear = self._check_collision(traj, blocked_fn, obstacles_motion)
                if hit or min_clear < self.stop_margin:
                    continue
                # ④ 评分
                score = self._score(traj, robot_pos, v, w, target_angle, min_clear, v_now, w_now)
                if score > best_score:
                    best_score = score
                    best = (float(v), float(w))
        return best

    def _simulate(self, robot_pos, yaw, v, w):
        """模拟圆弧轨迹，返回 [(x, y), ...]"""
        pts = []
        x, y, h = robot_pos[0], robot_pos[1], yaw
        dt = self.dt_sample
        n = max(1, int(self.horizon / dt))
        for _ in range(n):
            h += w * dt
            x += v * math.cos(h) * dt
            y += v * math.sin(h) * dt
            pts.append((x, y))
        return pts

    def _check_collision(self, traj, blocked_fn, obstacles_motion=None):
        """返回 (hit, min_clear)。采样点之间插值细化检测（0.1m 子步长）。

        obstacles_motion: [(ox, oy, vx, vy, r), ...]——模拟时障碍按速度线性外推，
        在 t 时刻用 (ox+vx·min(t,pred_t), oy+vy·min(t,pred_t)) 判定（pred_t 截断：
        障碍会变向/反弹，长期预测不可靠），狗半径 0.2m 计入。
        """
        min_clear = 1e18
        prev = None
        for i, pt in enumerate(traj):
            t = (i + 1) * self.dt_sample
            tp = min(t, self.pred_t)   # 预测截断
            if blocked_fn(pt):
                return True, min_clear
            if obstacles_motion:
                for ox, oy, vx, vy, r in obstacles_motion:
                    ex, ey = ox + vx * tp, oy + vy * tp   # 障碍未来位置（截断）
                    if math.hypot(pt[0]-ex, pt[1]-ey) < r + 0.2:   # 障碍半径 + 狗半径
                        return True, min_clear
            if prev is not None:
                for t2 in np.linspace(0.05, 1.0, 5):
                    ip = (prev[0] + (pt[0]-prev[0])*t2, prev[1] + (pt[1]-prev[1])*t2)
                    if blocked_fn(ip):
                        return True, min_clear
                    if obstacles_motion:
                        tt = min(t - self.dt_sample * (1.0 - t2), self.pred_t)
                        for ox, oy, vx, vy, r in obstacles_motion:
                            ex, ey = ox + vx * tt, oy + vy * tt
                            if math.hypot(ip[0]-ex, ip[1]-ey) < r + 0.2:
                                return True, min_clear
            prev = pt
        if traj:
            min_clear = self.horizon
        return False, min_clear

    def _score(self, traj, robot_pos, v, w, target_angle, min_clear, v_now, w_now):
        """加权评分。返回值越大越好。"""
        # heading：轨迹终点方向（相对机器人）vs 目标方向（cos 相似度 → [-1,1]）
        end = traj[-1]
        end_angle = math.atan2(end[1] - robot_pos[1], end[0] - robot_pos[0]) if traj else 0.0
        heading = math.cos(end_angle - target_angle)
        # clearance：归一化 [0,1]
        clearance = min(min_clear / self.clearance_max, 1.0)
        # velocity：归一化 [0,1]
        velocity = v / self.v_max if self.v_max > 0 else 0.0
        # smoothness：与当前状态接近 [0,1]
        dv = abs(v - v_now) / self.v_max if self.v_max > 0 else 0.0
        dw = abs(w - w_now) / (2 * self.w_max) if self.w_max > 0 else 0.0
        smooth = 1.0 - (dv + dw) / 2.0
        return (self.w_heading * heading
                + self.w_clearance * clearance
                + self.w_velocity * velocity
                + self.w_smooth * smooth)
