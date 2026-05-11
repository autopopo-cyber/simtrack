"""
ObstacleGenerator — 沿赛道中轴线随机障碍物生成

每次调用生成全新随机布局，杜绝算法"记住"障碍物位置。

用法:
    from simtrack.obstacles import ObstacleGenerator
    gen = ObstacleGenerator(center_line, cum_dists, total_len)
    obstacles, waypoints = gen.generate()

可调参数:
    spacing_range — 沿赛道距离间距 (默认 4-8m)
    lateral_range — 横向偏移范围 (默认 0.5-4.5m)
    start_clear / end_clear — 起终点无障碍区 (默认 5m)
"""

import math
import random
import numpy as np


class ObstacleGenerator:
    """沿赛道中轴线随机障碍物生成器。

    从起点+start_clear 处开始，每次滚两个骰子:
        1. 沿赛道距离增量 (spacing_range)
        2. 横向偏移 (lateral_range, 随机左/右)
    直到终点-lateral_range 处停止。

    每次实例化 (默认无 seed) 都产生不同布局。
    """

    def __init__(
        self,
        center_line: list,
        cum_dists: list = None,
        total_len: float = None,
        spacing_range: tuple = (4.0, 8.0),
        lateral_range: tuple = (0.5, 4.5),
        start_clear: float = 5.0,
        end_clear: float = 5.0,
        seed: int = None,
    ):
        """初始化障碍物生成器。

        Args:
            center_line: 赛道中心线 [(x,y), ...]
            cum_dists: 中心线累积距离 [d0, d1, ...] (可选，自动计算)
            total_len: 赛道总长 (可选，自动计算)
            spacing_range: (min, max) 沿赛道距离间距 (米)
            lateral_range: (min, max) 横向偏移 (米)
            start_clear: 起点无障碍区 (米)
            end_clear: 终点无障碍区 (米)
            seed: 随机种子 (None=每次不同)
        """
        self.center_line = center_line
        self.spacing_range = spacing_range
        self.lateral_range = lateral_range
        self.start_clear = start_clear
        self.end_clear = end_clear
        self.seed = seed
        self.rng = random.Random(seed)

        # 计算累积距离 (如果未提供)
        if cum_dists is not None:
            self.cum_dists = cum_dists
        else:
            self.cum_dists = [0.0]
            for i in range(1, len(center_line)):
                p0, p1 = center_line[i - 1], center_line[i]
                self.cum_dists.append(
                    self.cum_dists[-1] + math.hypot(p1[0] - p0[0], p1[1] - p0[1])
                )

        self.total_len = total_len or self.cum_dists[-1]

    def generate(self) -> list:
        """生成障碍物列表。

        Returns:
            list[tuple]: [(x, y), ...] 障碍物世界坐标
        """
        obstacles = []
        d = self.start_clear
        safe_end = self.total_len - self.end_clear

        while d < safe_end - 1.0:
            inc = round(self.rng.uniform(*self.spacing_range), 1)
            offset = round(self.rng.uniform(*self.lateral_range), 1)
            side = self.rng.choice([-1, 1])

            idx = np.searchsorted(self.cum_dists, d)
            if idx >= len(self.center_line) - 1:
                break

            # 该点法线方向
            p0, p1 = self.center_line[idx], self.center_line[idx + 1]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            mag = math.hypot(dx, dy)
            if mag < 0.001:
                d += inc
                continue

            nx = -dy / mag
            ny = dx / mag

            ox = p0[0] + side * offset * nx
            oy = p0[1] + side * offset * ny
            obstacles.append((ox, oy))
            d += inc

        return obstacles

    def to_mujoco_xml(self, obstacles, obs_radius=0.3, obs_height=0.3,
                      rgba="0.9 0.3 0.3 0.8"):
        """将障碍物列表转为 MuJoCo XML body 字符串。

        Args:
            obstacles: [(x,y), ...]
            obs_radius: 障碍物圆柱半径 (米)
            obs_height: 障碍物圆柱半高 (米)
            rgba: 颜色

        Returns:
            str: MuJoCo XML body 块
        """
        lines = []
        for i, (ox, oy) in enumerate(obstacles):
            lines.append(
                f'<body name="o{i}" pos="{ox:.1f} {oy:.1f} {obs_height}">'
                f'<geom type="cylinder" size="{obs_radius} {obs_height}" '
                f'rgba="{rgba}"/></body>'
            )
        return "\n".join(lines)


# ── 自测 ──
def _self_test():
    """纯 Python 自测"""
    # 生成假中心线 (直道)
    cx = [(float(i), 0.0) for i in np.arange(0, 100, 0.5)]
    gen = ObstacleGenerator(cx, spacing_range=(4, 8), lateral_range=(0.5, 4.5), seed=42)
    obs = gen.generate()

    assert 10 <= len(obs) <= 25, f"障碍物数量异常: {len(obs)}"
    for ox, oy in obs:
        assert 5 <= ox <= 95, f"障碍物超出起点/终点范围: ({ox}, {oy})"
        assert 0.5 <= abs(oy) <= 4.5, f"横向偏移超出范围: ({ox}, {oy})"

    # 验证无种子每次不同
    gen2 = ObstacleGenerator(cx, spacing_range=(4, 8), lateral_range=(0.5, 4.5))
    obs2 = gen2.generate()
    obs3 = gen2.__class__(cx, spacing_range=(4, 8), lateral_range=(0.5, 4.5)).generate()
    assert obs2 != obs3 or len(obs2) != len(obs3), \
        "无 seed 的两次生成应该不同 (概率性, 重试即可)"

    print(f"  ✓ 自测通过: {len(obs)} 个障碍物")
    return True


if __name__ == "__main__":
    _self_test()
