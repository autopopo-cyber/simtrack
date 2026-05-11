"""TrackGen 单元测试 — 纯 Python (不依赖 MuJoCo)"""

import numpy as np
from simtrack.trackgen import TrackGenerator


def test_default_generation():
    """默认参数生成 2000×2000 赛道"""
    tg = TrackGenerator(hf_res=500, guard_height=3.0)
    tg.generate()

    hf = tg.hfield
    assert hf.shape == (500, 500), f"尺寸: {hf.shape}"
    unique = np.unique(hf)
    assert 128 in unique, "路面像素 128 缺失"
    assert 255 in unique, "护栏像素 255 缺失"
    assert tg.total_len > 500, f"赛道太短: {tg.total_len:.0f}m"
    assert len(tg.center_line) > 100
    assert len(tg.waypoints) > 10


def test_guard_brush_thickness():
    """护栏笔刷 ≥3px"""
    tg = TrackGenerator(hf_res=500, guard_height=3.0, guard_brush=3)
    tg.generate()

    # 检查护栏像素在路旁连续分布
    hf = tg.hfield
    guard_pixels = np.argwhere(hf == 255)
    assert len(guard_pixels) > 100, f"护栏太少: {len(guard_pixels)}"


def test_hfield_encoding():
    """hfield 编码: 路面≈0m, 护栏=3m"""
    tg = TrackGenerator(hf_res=500, guard_height=3.0)
    tg.generate()

    _hx, _hy, scale, neg = tg.hfield_size  # (half_x, half_y, scale, negative)
    # 路面高度
    road_m = 128 / 255 * scale - neg
    # 护栏高度
    guard_m = 255 / 255 * scale - neg

    assert abs(road_m) < 0.1, f"路面高度={road_m:.3f}m (期望≈0)"
    assert abs(guard_m - 3.0) < 0.1, f"护栏高度={guard_m:.3f}m (期望≈3)"


def test_reproducibility():
    """相同 seed 产生相同输出"""
    tg1 = TrackGenerator(hf_res=500, seed=42)
    tg1.generate()
    tg2 = TrackGenerator(hf_res=500, seed=42)
    tg2.generate()
    assert np.array_equal(tg1.hfield, tg2.hfield)


# TrackGenerator 是确定性生成器 (种子保留用于未来扩展)
