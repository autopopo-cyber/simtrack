#!/usr/bin/env python3
"""分析 track_clean.png 迷宫真值结构：通道/分界墙/缺口位置与宽度。

世界坐标: 50x50m, hfield 2000x2000 (40px/m), 路面=128, 墙!=128。
世界 (wx,wy) -> 像素 (px=int(wx*40), py=1999-int(wy*40))。
"""
import numpy as np
from PIL import Image

hf = np.array(Image.open("confirmed/track_clean.png"))
H, W = hf.shape
PIX_PER_M = W // 50
print(f"map {W}x{H}, {PIX_PER_M}px/m, unique={np.unique(hf)[:10]}")

def is_road(wx, wy):
    px, py = int(wx * PIX_PER_M), H - 1 - int(wy * PIX_PER_M)
    if 0 <= px < W and 0 <= py < H:
        return hf[py, px] == 128
    return False

# 1) 每通道中心线 y=2.5+5k 的通行情况（x 从 0 到 50，0.1m 采样）
print("\n== 通道中心线通行性 (x: road 区间) ==")
for ch in range(10):
    y = 2.5 + ch * 5.0
    segs = []
    in_seg = False
    for xi in range(0, 501):
        x = xi * 0.1
        r = is_road(x, y)
        if r and not in_seg:
            start = x; in_seg = True
        elif not r and in_seg:
            segs.append((start, round(x - 0.1, 1))); in_seg = False
    if in_seg: segs.append((start, 50.0))
    print(f"ch{ch} y={y:4.1f}: {segs}")

# 2) 分界墙 y=5k (k=1..9) 上的缺口（沿墙线采样 road 段 = 开口）
print("\n== 分界墙缺口 (沿 y=5k 线找 road 开口) ==")
for k in range(1, 10):
    y = k * 5.0
    segs = []
    in_seg = False
    for xi in range(0, 501):
        x = xi * 0.1
        r = is_road(x, y)
        if r and not in_seg:
            start = x; in_seg = True
        elif not r and in_seg:
            segs.append((round(start, 1), round(x - 0.1, 1))); in_seg = False
    if in_seg: segs.append((round(start, 1), 50.0))
    print(f"y={y:4.1f}: 开口 {segs}  宽 {[round(b-a+0.1,1) for a,b in segs]}")

# 3) U 型弯开口：每个分界墙两端，垂直方向通道连通（x 固定，沿 y 穿过墙线）
print("\n== U 型弯垂直通道（关键 x 列沿 y 的 road 区间）==")
for x in [1.0, 2.0, 2.5, 3.0, 4.0, 46.0, 47.0, 47.5, 48.0, 49.0]:
    segs = []
    in_seg = False
    for yi in range(0, 501):
        y = yi * 0.1
        r = is_road(x, y)
        if r and not in_seg:
            start = y; in_seg = True
        elif not r and in_seg:
            segs.append((round(start, 1), round(y - 0.1, 1))); in_seg = False
    if in_seg: segs.append((round(start, 1), 50.0))
    n = len(segs)
    print(f"x={x:4.1f}: {n} 段  {segs if n <= 12 else str(segs[:12])+'...'}")

# 4) 左端斜边：ch0/ch1 之间 (y=5 墙左端) 和 ch1/ch2 (y=10 墙) 附近的局部结构
print("\n== 左端斜边局部图 (x=0..8, y=3..12, 0.2m/字符, #=墙 .=路) ==")
for yi in range(60, 14, -1):
    y = yi * 0.2
    row = ""
    for xi in range(0, 41):
        x = xi * 0.2
        row += "." if is_road(x, y) else "#"
    print(f"y={y:5.1f} {row}")

# 5) 全图连通性 (BFS on 0.1m grid, 4-conn) + 各通道可达性
print("\n== 0.1m 栅格 BFS 连通性（含 0.2m 机器人半径膨胀两档）==")
VOX = 0.1
G = 500
# 栅格化：4x4 px 块任一墙 -> 墙 (与 known-raw MAX-pool 一致)
bin_wall = (hf != 128).astype(np.uint8).reshape(500, 4, 500, 4).max(axis=(1, 3))
bin_wall = bin_wall[::-1, :]  # flip y
for inflate in (0, 2):  # 0 和 0.2m 膨胀
    w = bin_wall.copy()
    if inflate:
        from scipy import ndimage  # 可能没装
        w = ndimage.binary_dilation(w, iterations=inflate).astype(np.uint8)
    seen = np.zeros((500, 500), dtype=bool)
    sx, sy = 25, 25  # (2.5,2.5)
    if w[sy, sx]:
        print(f"inflate={inflate}: 起点是墙!")
        continue
    q = [(sx, sy)]; seen[sy, sx] = True
    n = 0
    while q:
        cx, cy = q.pop()
        n += 1
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = cx+dx, cy+dy
            if 0 <= nx < 500 and 0 <= ny < 500 and not w[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((nx, ny))
    goal_ok = seen[475, 25]
    free_total = int((~w.astype(bool)).sum())
    print(f"inflate={inflate}格: 起点可达 {n} 格 / 自由 {free_total} 格 ({n/max(free_total,1)*100:.1f}%), 终点(2.5,47.5)可达={goal_ok}")
    # 各通道中心可达
    reach = [f"ch{k}={'Y' if seen[int((2.5+k*5)/VOX), int(25/VOX)] else 'N'}" for k in range(10)]
    print("   ", " ".join(reach))
