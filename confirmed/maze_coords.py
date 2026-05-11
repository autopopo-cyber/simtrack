"""
迷宫坐标系转换模块
===================
迷宫系 (maze):  0-50m, 入口(3,3), 出口(3,48)
世界系 (world): MuJoCo全局坐标, hfield跨度0-100m
PNG像素系:      2000×2000, 左侧/底部对应迷宫0
高度编码:       pixel→h = (pixel/255)*4.0 - 2.0, 路面128→0m, 墙255→2m

所有导航算法统一使用此模块, 保证坐标一致。
"""

# ── 常量 ──
MAZE_SIZE = 50.0      # 迷宫边长(m)
WORLD_SIZE = 100.0     # 世界跨度(m), = MAZE_SIZE * 2
SCALE = WORLD_SIZE / MAZE_SIZE  # = 2.0
HF_RES = 2000          # PNG分辨率
PIX_PER_M = HF_RES / MAZE_SIZE  # = 40

SCALE_H = 4.0         # hfield高度缩放
NEG_H = 2.0            # hfield高度偏移

CHECKPOINTS_MAZE = [
    (3, 3),            # start
    (47, 5),           # CP1
    (3, 10),           # CP2
    (47, 15),          # CP3
    (3, 20),           # CP4
    (47, 25),          # CP5
    (3, 30),           # CP6
    (47, 35),          # CP7
    (3, 40),           # CP8
    (47, 45),          # CP9
    (3, 48),           # CP10 (终点)
]

# ── 坐标转换 ──

def maze_to_world(mx, my=None):
    """迷宫坐标 → 世界坐标 (×2)"""
    if my is None:
        return (mx[0] * SCALE, mx[1] * SCALE)
    return (mx * SCALE, my * SCALE)

def world_to_maze(wx, wy=None):
    """世界坐标 → 迷宫坐标 (/2)"""
    if wy is None:
        return (wx[0] / SCALE, wx[1] / SCALE)
    return (wx / SCALE, wy / SCALE)

def maze_to_pixel(mx, my=None):
    """迷宫坐标 → PNG像素(左下原点)"""
    if my is None:
        return (mx[0] * PIX_PER_M, mx[1] * PIX_PER_M)
    return (int(mx * PIX_PER_M), int(my * PIX_PER_M))

def pixel_to_maze(px, py):
    """PNG像素 → 迷宫坐标"""
    return (px / PIX_PER_M, py / PIX_PER_M)

def height_to_pixel(h):
    """物理高度(m) → hfield像素值"""
    return int((h + NEG_H) / SCALE_H * 255)

def pixel_to_height(p):
    """hfield像素值 → 物理高度(m)"""
    return p / 255.0 * SCALE_H - NEG_H

# ── 导航用 ──

def get_checkpoints_world():
    """返回世界坐标系的检查点列表"""
    return [maze_to_world(x, y) for x, y in CHECKPOINTS_MAZE]

def get_checkpoints_maze():
    """返回迷宫坐标系的检查点列表"""
    return list(CHECKPOINTS_MAZE)

# ── 自测 ──
if __name__ == "__main__":
    print("迷宫↔世界: (3,3) →", maze_to_world(3,3))
    print("世界↔迷宫: (6,6) →", world_to_maze(6,6))
    print("迷宫↔像素: (3,3) →", maze_to_pixel(3,3))
    print("高度128→", pixel_to_height(128), "m")
    print("高度255→", pixel_to_height(191), "m")
    print("1m→pixel:", height_to_pixel(1.0))
    wps = get_checkpoints_world()
    print(f"Checkpoints: {len(wps)}个, start={wps[0]}, end={wps[-1]}")
