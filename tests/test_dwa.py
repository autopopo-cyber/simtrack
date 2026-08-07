"""DWA 局部规划器单元测试 — 纯 Python (不依赖 MuJoCo)"""
import math
import pytest
from simtrack.algorithms.dwa import DWAAlgorithm


def _blocked_none(*args):
    return False


def test_no_obstacle_goes_fast_forward():
    """无障碍 → 选窗口内最大速度 + 零角速度（加速度约束：v = a_accel*T）"""
    dwa = DWAAlgorithm()
    v, w = dwa.choose_velocity(
        robot_pos=(0.0, 0.0), yaw=0.0, v_now=0.0, w_now=0.0,
        target=(10.0, 0.0), blocked_fn=_blocked_none)
    assert v == pytest.approx(dwa.a_accel * dwa.T, abs=1e-6)   # 0.25
    assert abs(w) < 1e-6


def test_obstacle_in_front_avoid():
    """正前方 3m 障碍 → 选非零角速度（绕行）"""
    def blocked(pt):
        # 半径 0.7 的圆形障碍（中心 3,0）
        return math.hypot(pt[0]-3.0, pt[1]-0.0) < 0.7
    dwa = DWAAlgorithm()
    v, w = dwa.choose_velocity(
        robot_pos=(0.0, 0.0), yaw=0.0, v_now=2.0, w_now=0.0,
        target=(10.0, 0.0), blocked_fn=blocked)
    assert w != pytest.approx(0.0, abs=1e-6)   # 有转向


def test_all_collision_returns_none():
    """全部轨迹都碰撞（四面墙）→ 返回 None（触发 _bounce 兜底）"""
    def blocked_all(pt):
        return True
    dwa = DWAAlgorithm()
    result = dwa.choose_velocity(
        robot_pos=(0.0, 0.0), yaw=0.0, v_now=0.0, w_now=0.0,
        target=(10.0, 0.0), blocked_fn=blocked_all)
    assert result is None


def test_dynamic_window_limits():
    """动态窗口：加速度限制下 v 不能瞬间跳到 v_max"""
    dwa = DWAAlgorithm(a_accel=1.0, a_decel=1.0, T=0.1)
    v, _ = dwa.choose_velocity(
        robot_pos=(0.0, 0.0), yaw=0.0, v_now=0.0, w_now=0.0,
        target=(10.0, 0.0), blocked_fn=_blocked_none)
    assert v <= 0.1 + 1e-6   # a_accel*T = 1.0*0.1


def test_smoothness_penalty():
    """smoothness：无障碍时速度平稳（在窗口内不跳变）"""
    dwa = DWAAlgorithm()
    v, w = dwa.choose_velocity(
        robot_pos=(0.0, 0.0), yaw=0.0, v_now=2.0, w_now=0.0,
        target=(10.0, 0.0), blocked_fn=_blocked_none)
    assert 0.0 <= v <= dwa.v_max + 1e-6
    assert abs(w) < 1e-6


def test_heading_relative_to_robot_pos():
    """heading 必须相对机器人位置（不在原点时目标方向仍正确）——回归：世界原点参照系 bug"""
    dwa = DWAAlgorithm()
    # 机器人在 (45, 8)，目标在正前方 (50, 8) → 应选正前方向（ω≈0）
    v, w = dwa.choose_velocity(
        robot_pos=(45.0, 8.0), yaw=0.0, v_now=2.0, w_now=0.0,
        target=(50.0, 8.0), blocked_fn=_blocked_none)
    assert abs(w) < 1e-6, f"目标在正前方应直行, w={w}"
    # 机器人在 (45, 8)，目标在左后方 (42, 7.6) → 应转向（ω≠0）
    v2, w2 = dwa.choose_velocity(
        robot_pos=(45.0, 8.0), yaw=0.0, v_now=2.0, w_now=0.0,
        target=(42.0, 7.6), blocked_fn=_blocked_none)
    assert abs(w2) > 1e-4, f"目标在左后方应转向, w2={w2}"
