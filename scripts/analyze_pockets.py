#!/usr/bin/env python3
"""细查左端口袋（y=15 墙 ch2→ch3 左开口）与对角盖板的结构。"""
import numpy as np
from PIL import Image

hf = np.array(Image.open("confirmed/track_clean.png"))
H, W = hf.shape
P = W // 50

def is_road(wx, wy):
    px, py = int(wx * P), H - 1 - int(wy * P)
    if 0 <= px < W and 0 <= py < H:
        return hf[py, px] == 128
    return False

print("== y=15 墙左端口袋 (x=0..10, y=12..19, 0.2m/字符) ==")
for yi in range(95, 59, -1):
    y = yi * 0.2
    row = ""
    for xi in range(0, 51):
        x = xi * 0.2
        row += "." if is_road(x, y) else "#"
    print(f"y={y:5.1f} {row}")

print("\n== 右端口袋对照 (y=10 墙, x=40..50, y=7..14) ==")
for yi in range(70, 34, -1):
    y = yi * 0.2
    row = ""
    for xi in range(200, 251):
        x = xi * 0.2
        row += "." if is_road(x, y) else "#"
    print(f"y={y:5.1f} {row}")

# 口袋是否死胡同：口袋区 (x 0.5-4, y 14-17) 每个 free 格 BFS 看能否到 ch3 主区 (y>17)
print("\n== 口袋连通性：从 (1.0,15.0) BFS 能否到 (1.0,18.0)（ch3 主区）==")
from collections import deque
def bfs_reach(p0, p1, res=0.05):
    n = int(50 / res)
    def cell(p): return (int(p[0]/res), int(p[1]/res))
    def road(c): return is_road((c[0]+0.5)*res, (c[1]+0.5)*res)
    c0, c1 = cell(p0), cell(p1)
    if not road(c0) or not road(c1): return f"端点非路: {road(c0)},{road(c1)}"
    seen = {c0}; q = deque([c0])
    while q:
        c = q.popleft()
        if c == c1: return "连通"
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nc = (c[0]+dx, c[1]+dy)
            if 0 <= nc[0] < n and 0 <= nc[1] < n and nc not in seen and road(nc):
                seen.add(nc); q.append(nc)
    return f"不连通 (BFS {len(seen)} 格)"
print("  (1.0,15.0)->(1.0,18.0):", bfs_reach((1.0,15.0),(1.0,18.0)))
print("  (2.0,15.0)->(2.0,18.0):", bfs_reach((2.0,15.0),(2.0,18.0)))
print("  (47.0,11.0)->(47.0,9.0):", bfs_reach((47.0,11.0),(47.0,9.0)))
