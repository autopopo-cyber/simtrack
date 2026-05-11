"""VO 避障算法单元测试 — 纯 Python (不依赖 MuJoCo)"""

import math
from simtrack.algorithms.vo import VOAlgorithm
from simtrack.algorithms.base import AvoidanceResult


def test_straight_line_no_obstacles():
    """无障碍 → 直行"""
    algo = VOAlgorithm(max_speed=2.0)
    result = algo.choose_heading(
        robot_pos=(0.0, 0.0),
        robot_speed=2.0,
        target_pos=(10.0, 0.0),
        obstacles=[],
    )
    assert isinstance(result, AvoidanceResult)
    assert result.avoiding is False
    assert abs(result.heading) < 0.01  # 正前方


def test_single_obstacle_avoid():
    """前方有障碍 → 避障"""
    algo = VOAlgorithm(max_speed=2.0)
    result = algo.choose_heading(
        robot_pos=(0.0, 0.0),
        robot_speed=2.0,
        target_pos=(10.0, 0.0),
        obstacles=[(3.0, 0.0, 0.5)],  # 正前方 3m 处 0.5m 半径障碍
    )
    assert result.avoiding is True
    assert result.speed < 2.0  # 避障速度降低


def test_avoidance_speed_ratio():
    """避障时速度 = max_speed * avoid_speed_ratio"""
    algo = VOAlgorithm(max_speed=3.0, avoid_speed_ratio=0.5)
    result = algo.choose_heading(
        robot_pos=(0.0, 0.0),
        robot_speed=3.0,
        target_pos=(10.0, 0.0),
        obstacles=[(3.0, 0.0, 0.5)],
    )
    assert abs(result.speed - 1.5) < 0.01


def test_left_right_obstacle():
    """单侧有障碍 → 偏向另一侧"""
    algo = VOAlgorithm(max_speed=2.0)
    # 障碍在左边
    result = algo.choose_heading(
        robot_pos=(0.0, 0.0),
        robot_speed=2.0,
        target_pos=(10.0, 0.0),
        obstacles=[(3.0, 0.5, 0.5)],
    )
    # 应偏向右边 (正航向)
    assert result.heading < 0 or abs(result.heading) < 0.5


def test_check_blocked_ok():
    """无障碍 → OK"""
    algo = VOAlgorithm()
    status = algo.check_blocked((0, 0), [])
    assert status == "ok"


def test_check_blocked_surrounded():
    """被障碍包围 → blocked"""
    algo = VOAlgorithm(robot_radius=0.25)
    # 8 个障碍物围住 (距离 < 2*robot_radius = 0.5m)
    obstacles = []
    for i in range(8):
        a = 2 * math.pi * i / 8
        obstacles.append((0.4 * math.cos(a), 0.4 * math.sin(a), 0.2))
    status = algo.check_blocked((0, 0), obstacles)
    assert status == "blocked"
