#!/usr/bin/env python3
"""复现"幽灵门"泄漏：从口袋内按 algo3_headless.scan() 同样的逻辑打雷达，
看 FREE 是否泄漏到盖板对角线另一侧（ch1 主区）形成幽灵门。"""
import math
import numpy as np
from PIL import Image

hf = np.array(Image.open("confirmed/track_clean.png"))
H, W = hf.shape
P = W // 50
VOXEL = 0.1
LIDAR_RANGE = 30.0
LIDAR_STEPS = int(LIDAR_RANGE / VOXEL)
LIDAR_RAYS = 360
ROAD_PIX = 128

def sample_hf(wx, wy):
    px, py = int(wx * P), H - 1 - int(wy * P)
    if 0 <= px < W and 0 <= py < H:
        return int(hf[py, px])
    return -1

def is_obstacle_world(wx, wy):
    return sample_hf(wx, wy) != ROAD_PIX

UNKNOWN, FREE, WALL = 0, 1, 2
grid = {}

def gset(vx, vy, v):
    if grid.get((vx, vy), UNKNOWN) != v:
        grid[(vx, vy)] = v

def gget(vx, vy):
    return grid.get((vx, vy), UNKNOWN)

def scan(bx, by, yaw_ang, fov_deg=180.0):
    fov = math.radians(fov_deg)
    for a in np.linspace(yaw_ang - fov/2, yaw_ang + fov/2, LIDAR_RAYS):
        ca, sa = math.cos(a), math.sin(a)
        pvx, pvy = int(bx/VOXEL), int(by/VOXEL)
        for si in range(1, LIDAR_STEPS+1):
            wx, wy = bx + ca*si*VOXEL, by + sa*si*VOXEL
            vx, vy = int(wx/VOXEL), int(wy/VOXEL)
            if is_obstacle_world(wx, wy):
                gset(vx, vy, WALL)
                if gget(pvx, pvy) == UNKNOWN:
                    gset(pvx, pvy, FREE)
                break
            if gget(vx, vy) != FREE:
                gset(vx, vy, FREE)
            pvx, pvy = vx, vy

# 狗卡在 y=5 墙左口袋 (1.4,6.2)，扫 360°（bounce 时朝向变化，近似全向累计）
bx, by = 1.4, 6.2
for deg in range(0, 360, 30):
    scan(bx, by, math.radians(deg))

# 检查 ch1 主区 (y=7.2..9.5, x=0.5..5) 的 FREE 泄漏
print("== 口袋上方 ch1 区感知图 (x=0..6, y=5..10, 0.2m/字符: .=FREE #=WALL ' '=UNKNOWN) ==")
for yi in range(50, 24, -1):
    y = yi * 0.2
    row = ""
    for xi in range(0, 31):
        x = xi * 0.2
        v = gget(int(x/VOXEL), int(y/VOXEL))
        row += "." if v == FREE else ("#" if v == WALL else " ")
    print(f"y={y:5.1f} |{row}|")

# 幽灵门统计：ch1 区 (y>7.2) 内 FREE 且邻接 UNKNOWN 的格
leak_gates = 0
leak_free = 0
for gy in range(72, 98):
    for gx in range(3, 60):
        if gget(gx, gy) == FREE:
            leak_free += 1
            if any(gget(gx+dx, gy+dy) == UNKNOWN for dx in (-1,0,1) for dy in (-1,0,1)):
                leak_gates += 1
print(f"\nch1 区泄漏: FREE={leak_free} 格, 其中幽灵门(邻接UNKNOWN)={leak_gates} 格")
