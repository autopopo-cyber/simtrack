#!/usr/bin/env python3
"""
maze_gen.py — 为 MuJoCo + ROS2 SLAM/Nav2 生成干净的迷宫高度图。

坐标系约定（消除旧版 y 翻转混乱）：
  世界系：x 向右, y 向上, 原点 (0,0) = 迷宫左下角
  图像系：col = x * PX_PER_M,  row = (MAZE_H - y) * PX_PER_M
          （row 0 = 图像顶部 = 世界 y 最大处，标准图像约定）
  起点在 (1.5, 1.5) 朝 +x，靠近原点，无偏移混乱。

输出：
  confirmed/maze20.png   — 高度图 PNG（road=128, wall=255）
  MuJoCo hfield 直接引用此文件（value=高度，255=高墙 128=地面）

用法：python -m simtrack.maze_gen
"""
import os
import numpy as np
from PIL import Image, ImageDraw

# ── 迷宫参数 ──
MAZE_W = 20.0          # 宽 (m), x ∈ [0, 20]
MAZE_H = 20.0          # 高 (m), y ∈ [0, 20]
PX_PER_M = 50          # 分辨率：2cm/像素 → 1000×1000（雷达步进 0.02m，够精细）
WALL_T = 0.3           # 墙厚 (m)，雷达清晰可见、狗（0.4m 宽）过不去
ROAD_VAL = 128         # 地面像素值（MuJoCo hfield 中等高度）
WALL_VAL = 255         # 墙像素值（MuJoCo hfield 最高）
START = (1.5, 1.5)     # 起点世界坐标（左下角内侧）
START_YAW = 0.0        # 起点朝向（弧度，0=朝+x）

IMG_W = int(MAZE_W * PX_PER_M)
IMG_H = int(MAZE_H * PX_PER_M)

# ── 墙段定义：((x1,y1), (x2,y2)) 世界坐标 ──
# 设计目标：有拐角（SLAM 特征）、有回路（回环修正）、走廊宽 ≥3m（狗/Nav2 舒适）
WALLS = [
    # 外边界
    ((0, 0), (MAZE_W, 0)),
    ((MAZE_W, 0), (MAZE_W, MAZE_H)),
    ((MAZE_W, MAZE_H), (0, MAZE_H)),
    ((0, MAZE_H), (0, 0)),
    # 内部墙——创造回路 + 房间
    # 中央方块（绕一圈走 = 回环）
    ((6, 6), (14, 6)),
    ((14, 6), (14, 14)),
    ((14, 14), (6, 14)),
    ((6, 14), (6, 6)),
    # 外圈走廊中的隔断（增加特征，不完全封死）
    ((3, 3), (3, 10)),       # 左下竖墙（上方留口 → 走廊连通）
    ((17, 10), (17, 17)),    # 右上竖墙（下方留口）
]


def world_to_pixel(x, y):
    """世界坐标 → 图像像素 (col, row)。"""
    col = int(x * PX_PER_M)
    row = IMG_H - 1 - int(y * PX_PER_M)
    return col, row


def generate():
    """生成高度图 numpy 数组 (IMG_H, IMG_W)，road=128, wall=255。"""
    img = Image.new("L", (IMG_W, IMG_H), ROAD_VAL)  # 全地面
    draw = ImageDraw.Draw(img)
    wt_px = max(1, int(WALL_T * PX_PER_M))  # 墙厚像素

    for (x1, y1), (x2, y2) in WALLS:
        c1, r1 = world_to_pixel(x1, y1)
        c2, r2 = world_to_pixel(x2, y2)
        # ImageDraw.line 的 width 是居中加粗；用矩形更精确
        if x1 == x2:  # 竖墙
            c = c1
            draw.rectangle([c - wt_px // 2, min(r1, r2),
                             c + wt_px // 2, max(r1, r2)], fill=WALL_VAL)
        elif y1 == y2:  # 横墙
            r = r1
            draw.rectangle([min(c1, c2), r - wt_px // 2,
                             max(c1, c2), r + wt_px // 2], fill=WALL_VAL)
        else:
            draw.line([c1, r1, c2, r2], fill=WALL_VAL, width=wt_px)

    arr = np.array(img)
    return arr


def save(arr, path):
    """保存高度图 PNG。"""
    Image.fromarray(arr).save(path)
    wall_pct = 100.0 * (arr == WALL_VAL).sum() / arr.size
    print(f"  生成: {path}  {arr.shape[1]}×{arr.shape[0]}px  "
          f"墙{wall_pct:.1f}%  分辨率{PX_PER_M}px/m({1000/PX_PER_M:.0f}mm/px)")
    print(f"  迷宫: {MAZE_W}×{MAZE_H}m  起点: ({START[0]}, {START[1]}) 朝{START_YAW}rad")
    print(f"  坐标系: 原点(0,0)=左下角, x→右 y→上")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "..", "confirmed", "maze20.png")
    out = os.path.normpath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    arr = generate()
    save(arr, out)
