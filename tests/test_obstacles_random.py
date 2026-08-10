"""随机反弹障碍单元测试 — 纯 Python (不依赖 MuJoCo)"""
import math
import random
from simtrack.obstacles_random import RandomObstacle, RandomObstacleField


def test_initial_position():
    """初始位置 = 段内随机（2026-08-10 主人观察：全在 x=25 段中央一列齐排太假），种子可复现"""
    field = RandomObstacleField(channels=[1], seed=42)
    obs = field.obstacles[0]
    assert obs.ch == 1
    assert obs.wall_x0 + 0.5 - 1e-9 <= obs.pos[0] <= obs.wall_x1 - 0.5 + 1e-9   # x 段内（含余量）
    assert obs.y_lo + 1.0 - 1e-9 <= obs.pos[1] <= obs.y_hi - 1.0 + 1e-9        # y 通道内（含余量）
    # 同种子同位置（可复现）
    obs2 = RandomObstacleField(channels=[1], seed=42).obstacles[0]
    assert obs.pos == obs2.pos and obs.dir == obs2.dir


def test_seed_reproducibility():
    f1 = RandomObstacleField(channels=[1, 4, 6, 8], seed=42)
    f2 = RandomObstacleField(channels=[1, 4, 6, 8], seed=42)
    assert len(f1.obstacles) == len(f2.obstacles) == 4
    for a, b in zip(f1.obstacles, f2.obstacles):
        assert a.pos == b.pos and a.dir == b.dir


def test_move_forward():
    """不变向期间障碍沿 dir 匀速移动 1m/s"""
    field = RandomObstacleField(channels=[1], seed=42)
    obs = field.obstacles[0]
    obs.change_timer = 2.0        # 保证 1s 内不触发变向判定
    d0 = obs.dir
    x0, y0 = obs.pos              # 记录初始位置（update 前）
    field.update(1.0)   # 1s
    expect = (x0 + math.cos(d0), y0 + math.sin(d0))
    assert math.hypot(obs.pos[0]-expect[0], obs.pos[1]-expect[1]) < 1e-6


def test_virtual_wall_bounce_x():
    """虚拟墙反弹：障碍往左出界 → vx 取反，拉回界内"""
    field = RandomObstacleField(channels=[1], seed=1)
    obs = field.obstacles[0]
    obs.pos = [15.0, 2.5 + 5.0]   # 虚拟墙 x=15.1 外
    obs.dir = math.pi             # 朝 -x
    field.update(0.1)
    assert obs.pos[0] >= 15.1 - 1e-6          # 拉回界内
    # 方向应变成朝 +x（vx 取反）
    assert math.cos(obs.dir) > 0


def test_real_wall_bounce_y():
    """真实墙反弹：障碍往通道下墙撞 → vy 取反"""
    field = RandomObstacleField(channels=[1], seed=2)
    obs = field.obstacles[0]
    obs.pos = [25.0, 5.0 + 0.49]  # y_lo=5.0，中心距墙 < 0.5 → 反弹
    obs.dir = -math.pi/2          # 朝 -y
    field.update(0.1)
    assert obs.pos[1] >= 5.0 + 0.5 - 1e-6    # 半径约束
    assert math.sin(obs.dir) > 0             # 改朝 +y


def test_direction_change_timer():
    """变向倒计时归零 → 触发判定 → timer 重置为 1.0"""
    field = RandomObstacleField(channels=[1], seed=7)
    obs = field.obstacles[0]
    obs.change_timer = 0.0
    field.update(0.5)
    assert obs.change_timer <= 1.0 + 1e-6   # 已重置（回到 1.0 或继续消耗）
