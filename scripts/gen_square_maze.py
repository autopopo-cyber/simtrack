"""生成方角弯道版蛇形迷宫（主人指令 2026-08-09）：
去掉 U 型弯 45° 喇叭口斜墙，转弯处改成方角 `[ ---` 形状（分隔墙方头 + 直角转弯区）。
同时消除斜边与边界墙围出的口袋死角（探索陷阱）和亚格子薄斜墙（幽灵门历史根源）。

输出：
  confirmed/track_clean.png     碰撞真值图 2000×2000（路=128 墙=191）
  confirmed/track_500_bin.png   渲染用降分辨率二值图 500×500（4×4 块 MAX-pool，无抗锯齿）

几何（与原图一致，仅斜边处方角化）：
  - 边界围墙 4px：x∈[0,0.075] / [49.9,50]，y∈[0,0.1] / [49.9,50]
  - 分隔墙 y=5k（k=1..9），厚 4px（0.1m）：
      k 奇数（5,15,25,35,45）：右端开口（x=50 转弯）→ 墙 x∈[0,45]
      k 偶数（10,20,30,40）  ：左端开口（x=0 转弯） → 墙 x∈[5,50]
  - 无任何 45° 斜墙 → 转弯区为 4.9m×5m 矩形开口，通道 5m 宽不变
"""
import os
import numpy as np
from PIL import Image

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = 2000
PIX_PER_M = 40
ROAD, WALL = 128, 191

def px(m):
    """世界米 → 像素坐标（x 轴：col=x*40；y 轴：row0=y=50m 顶部，row=1999-y*40）"""
    return int(m * PIX_PER_M)

def main():
    hf = np.full((RES, RES), ROAD, dtype=np.uint8)

    def fill_rect(x0, x1, y0, y1, val=WALL):
        """填世界矩形 [x0,x1]×[y0,y1]（含端点）"""
        c0, c1 = px(x0), min(px(x1), RES - 1)
        r0, r1 = max(1999 - px(y1), 0), 1999 - px(y0)   # row0=y 大
        hf[r0:r1 + 1, c0:c1 + 1] = val

    # ── 边界围墙 4px ──
    fill_rect(0.0, 0.075, 0.0, 50.0)      # 左 x=0
    fill_rect(49.9, 50.0, 0.0, 50.0)      # 右 x=50
    fill_rect(0.0, 50.0, 0.0, 0.1)        # 底 y=0
    fill_rect(0.0, 50.0, 49.9, 50.0)      # 顶 y=50

    # ── 分隔墙 y=5k（厚 0.1m=4px，方头，无斜边）──
    for k in range(1, 10):
        y = k * 5.0
        if k % 2 == 1:
            fill_rect(0.0, 45.0, y - 0.05, y + 0.05)   # 奇数：右端开口 x∈(45,49.9)
        else:
            fill_rect(5.0, 50.0, y - 0.05, y + 0.05)   # 偶数：左端开口 x∈(0.075,5)

    out_clean = os.path.join(PROJ, "confirmed", "track_clean.png")
    Image.fromarray(hf).save(out_clean)
    print(f"  [MAP] {out_clean}  墙像素 {np.count_nonzero(hf == WALL)}")

    # ── 渲染图 500×500：4×4 块 MAX-pool（块内任一为墙 → 墙；与 algo3 known-raw 判定一致）──
    bin_wall = (hf != ROAD).astype(np.uint8)
    pooled = bin_wall.reshape(500, 4, 500, 4).max(axis=(1, 3))
    render = np.where(pooled, WALL, ROAD).astype(np.uint8)
    out_render = os.path.join(PROJ, "confirmed", "track_500_bin.png")
    Image.fromarray(render).save(out_render)
    print(f"  [MAP] {out_render}  墙像素 {np.count_nonzero(render == WALL)}")

    # ── 连通性验证：BFS 起点 (2.5,2.5) → 终点 (2.5,47.5)（0.1m 格）──
    gy = (hf[::4, ::4] != ROAD)  # 500×500 快速近似（行=y 翻转前）
    grid = pooled.reshape(500, 500).T[:, ::-1]  # [vx,vy]，gy=0 → y=0m
    from collections import deque
    sx, sy = 25, 25
    tx, ty = 25, 475
    assert grid[sx, sy] == 0 and grid[tx, ty] == 0, "起点/终点在墙里？"
    seen = np.zeros_like(grid, dtype=bool)
    seen[sx, sy] = True
    q = deque([(sx, sy)])
    while q:
        cx, cy = q.popleft()
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < 500 and 0 <= ny < 500 and not seen[nx, ny] and grid[nx, ny] == 0:
                seen[nx, ny] = True
                q.append((nx, ny))
    ok = seen[tx, ty]
    print(f"  [MAP] 连通性: 起点→终点 {'CONNECTED' if ok else 'BLOCKED'}，可达格 {np.count_nonzero(seen)}")
    assert ok, "起点到终点不连通！"

if __name__ == "__main__":
    main()
