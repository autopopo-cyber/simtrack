"""
simtrack.map — 地图模块 (track_clean.png)

固定地图: 50×50m 迷宫, 1m 墙, 2000×2000 hfield.
提供: 坐标系转换、hfield采样、碰撞检测、中心线路点生成。
"""
import math, os
import numpy as np
import cv2

# ── 常量 (matches maze_coords.py) ──
MAZE_SIZE = 50.0
WORLD_SIZE = 100.0
SCALE = WORLD_SIZE / MAZE_SIZE  # = 2.0
HF_RES = 2000
PIX_PER_M = HF_RES / MAZE_SIZE  # = 40
ROAD_PIX = 128
WALL_PIX = 255

MAP_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "confirmed", "track_clean.png")

def load():
    """加载 hfield 图像，返回 numpy 数组和参数"""
    hf = cv2.imread(MAP_PATH, cv2.IMREAD_GRAYSCALE)
    if hf is None:
        raise FileNotFoundError(f"地图文件未找到: {MAP_PATH}")
    return hf

# ── 坐标转换 ──

def maze_to_world(mx, my):
    return (mx * SCALE, my * SCALE)

def world_to_maze(wx, wy):
    return (wx / SCALE, wy / SCALE)

def maze_to_pixel(mx, my):
    return (int(mx * PIX_PER_M), int(my * PIX_PER_M))

# ── hfield 采样 ──

def sample(hf, wx, wy):
    """世界坐标 → hfield像素值, OOB返回-1"""
    mx, my = wx / SCALE, wy / SCALE
    px = int(mx * PIX_PER_M)
    py = HF_RES - 1 - int(my * PIX_PER_M)
    if 0 <= px < HF_RES and 0 <= py < HF_RES:
        return int(hf[py, px])
    return -1

def is_road(hf, wx, wy):
    """世界坐标是否在路面上"""
    return sample(hf, wx, wy) == ROAD_PIX

# ── 碰撞检测 ──

def detect_collision(hf, wx, wy, radius=0.6):
    """在(wx,wy)半径内采样，非128=撞墙"""
    for dy in np.arange(-radius, radius + 0.01, 0.15):
        max_dx = np.sqrt(max(0, radius**2 - dy**2))
        for dx in np.arange(-max_dx, max_dx + 0.01, 0.15):
            if sample(hf, wx + dx, wy + dy) != ROAD_PIX:
                return True
    return False

# ── 检查点 (世界坐标) ──

CHECKPOINTS_MAZE = [
    (3, 3), (47, 5), (3, 10), (47, 15), (3, 20),
    (47, 25), (3, 30), (47, 35), (3, 40), (47, 45), (3, 48),
]

def get_checkpoints_world():
    return [maze_to_world(x, y) for x, y in CHECKPOINTS_MAZE]

# ── 中心线路点生成 ──

def generate_centerline(hf, spacing_m=4.0):
    """从 hfield 图像提取中心线，间隔 spacing_m 米采样路点。

    沿 y 轴扫描每一行，找到路面(像素128)的中点作为中心线。
    返回世界坐标系的中心线路点列表 [(wx,wy), ...]。
    """
    waypoints = []
    
    # 从迷宫坐标 y=2 到 y=49，每隔 spacing_m/SCALE 米采样
    maze_spacing = spacing_m / SCALE  # 世界间距→迷宫间距
    
    for my in np.arange(2.5, 48.5, maze_spacing):
        # 扫描这一行找路面区域
        row_px = HF_RES - 1 - int(my * PIX_PER_M)
        if row_px < 0 or row_px >= HF_RES:
            continue
        
        road_cols = []
        for px in range(HF_RES):
            if hf[row_px, px] == ROAD_PIX:
                road_cols.append(px)
        
        if len(road_cols) < 10:
            continue
        
        # 找连续段的中点
        segments = []
        start = road_cols[0]
        for j in range(1, len(road_cols)):
            if road_cols[j] - road_cols[j-1] > 3:  # 间隙>3px=新段
                segments.append((start, road_cols[j-1]))
                start = road_cols[j]
        segments.append((start, road_cols[-1]))
        
        # 取最宽段的中点
        best = max(segments, key=lambda s: s[1]-s[0])
        mid_px = (best[0] + best[1]) // 2
        mx = mid_px / PIX_PER_M
        wx, wy = maze_to_world(mx, my)
        waypoints.append((wx, wy))
    
    return waypoints
