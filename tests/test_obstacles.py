"""ObstacleGenerator 单元测试"""

import numpy as np
from simtrack.obstacles import ObstacleGenerator


def _make_straight_line(n=200):
    return [(float(i), 0.0) for i in np.arange(0, n * 0.5, 0.5)]


def test_basic_generation():
    """基本障碍物生成"""
    cl = _make_straight_line(200)
    gen = ObstacleGenerator(cl, spacing_range=(4, 8), lateral_range=(0.5, 4.5), seed=42)
    obs = gen.generate()

    assert 10 <= len(obs) <= 30, f"数量异常: {len(obs)}"
    for ox, oy in obs:
        assert 5 <= ox <= 95, f"超出范围: ({ox}, {oy})"
        assert 0.5 <= abs(oy) <= 4.5, f"横向: ({ox}, {oy})"


def test_randomness():
    """无 seed 时每次不同"""
    cl = _make_straight_line(200)
    gen1 = ObstacleGenerator(cl, spacing_range=(4, 8), lateral_range=(0.5, 4.5))
    obs1 = gen1.generate()
    gen2 = ObstacleGenerator(cl, spacing_range=(4, 8), lateral_range=(0.5, 4.5))
    obs2 = gen2.generate()

    assert obs1 != obs2, "两次生成应不同"


def test_seed_reproducibility():
    """相同 seed 相同输出"""
    cl = _make_straight_line(200)
    gen1 = ObstacleGenerator(cl, seed=42)
    obs1 = gen1.generate()
    gen2 = ObstacleGenerator(cl, seed=42)
    obs2 = gen2.generate()

    assert obs1 == obs2


def test_to_mujoco_xml():
    """XML 输出格式"""
    cl = _make_straight_line(200)
    gen = ObstacleGenerator(cl, seed=42)
    obs = gen.generate()
    xml = gen.to_mujoco_xml(obs)

    assert "<body" in xml
    assert "<geom type=\"cylinder\"" in xml
    assert xml.count("<body") == len(obs)


def test_density_tuning():
    """密度参数影响障碍物数量"""
    cl = _make_straight_line(400)  # 200m 赛道

    # 稀疏: 10-15m 间距
    gen_sparse = ObstacleGenerator(cl, spacing_range=(10, 15), seed=42)
    n_sparse = len(gen_sparse.generate())

    # 密集: 3-5m 间距
    gen_dense = ObstacleGenerator(cl, spacing_range=(3, 5), seed=42)
    n_dense = len(gen_dense.generate())

    assert n_dense > n_sparse, f"密集={n_dense} 应 > 稀疏={n_sparse}"
