#!/usr/bin/env python3
"""终点发现（无特权）：前置相机帧 → 绿色终点球检测 → 方位角 + 距离估计。

主人要求（2026-08-09）：狗不能作弊用 FINISH 真值坐标；终点球是场景里的
物理实体（绿色球，半径 1.5m），相机看到 → 狗知道那就是终点。

检测原理：
- 场景里唯一的绿色物体就是终点球（墙灰蓝、障碍红、狗黄、标牌黑白）
- HSV 阈值太依赖 cv2，这里直接 RGB 通道判定：G 显著高于 R/B
- 最大连通块 → 质心像素 → 方位角；块半径 → 距离（针孔相机模型）

相机模型（bot_cam，fovy=45°，1280x720）：
    fy = (H/2) / tan(fovy/2) ≈ 869 px
    bearing = atan((cx - W/2) / fy)   # 相对相机正前方，右为正（图像坐标）
    dist    = R_true * fy / r_px       # R_true=1.5m 球半径
"""
import math
import numpy as np

try:
    from scipy import ndimage
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

FINISH_SPHERE_R = 1.5   # 终点球半径 (m)，与 build_xml 的 FINISH_XML size="1.5" 一致
MIN_AREA_PX = 24        # 最小块面积（去噪）：30m 处球约 43px 半径、面积约 5800px


def detect_finish(img, fovy_deg=45.0):
    """检测绿色终点球。返回 (bearing_rad, dist_m, area_px, bottom_row) 或 None。

    bearing: 相对相机正前方的方位角，图像右侧为正（弧度）。
    dist:    估计距离（m）。远处球小，精度随距离下降，靠近后收敛。
    bottom_row: 块最底行像素行号——球落地时贴地，近距无遮挡时底行在画面下半部；
    隔墙只能看到球顶盖（底行≈地平线）→ 到达判定用（防隔墙误判）。
    """
    if img is None or not _SCIPY_OK:
        return None
    r = img[:, :, 0].astype(np.int16)
    g = img[:, :, 1].astype(np.int16)
    b = img[:, :, 2].astype(np.int16)
    mask = (g > 120) & (g - r > 40) & (g - b > 40)
    if not mask.any():
        return None
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    i = int(np.argmax(sizes)) + 1
    area = float(sizes[i - 1])
    if area < MIN_AREA_PX:
        return None
    ys, xs = np.nonzero(lab == i)
    h, w = img.shape[:2]
    cx = float(xs.mean())
    fy = (h / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    bearing = math.atan((cx - w / 2.0) / fy)
    r_px = math.sqrt(area / math.pi)
    dist = FINISH_SPHERE_R * fy / max(r_px, 1.0)
    return bearing, dist, area, float(ys.max())
