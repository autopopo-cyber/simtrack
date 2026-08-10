#!/usr/bin/env python3
"""隔离测试向量化 scan：从 (2.5,2.5) 扫描，验证感知图正确性 + 计时。"""
import math, time, sys, os
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
MAP = "confirmed/track_clean.png"
hf = np.array(Image.open(MAP))
HF_RES = hf.shape[0]
PIX_PER_M = 40
ROAD_PIX = 128
VOXEL = 0.1
LIDAR_RANGE = 30.0
LIDAR_RAYS = 360
SCAN_STEP = 0.025
_hf_bin = hf != ROAD_PIX
_scan_k = np.arange(1, int(LIDAR_RANGE / SCAN_STEP) + 1, dtype=np.float64) * SCAN_STEP
OBS_CLEAR = 0.7
obs_world = []

UNKNOWN, FREE, WALL = 0, 1, 2
grid = {}
def gget(vx, vy): return grid.get((vx, vy), UNKNOWN)
def gset(vx, vy, v): grid[(vx, vy)] = v

def scan(bx, by, yaw_ang, fov_deg=180.0):
    fov_rad = math.radians(fov_deg)
    angles = yaw_ang + np.linspace(-fov_rad/2, fov_rad/2, LIDAR_RAYS)
    cos_a = np.cos(angles); sin_a = np.sin(angles)
    k = _scan_k
    xs = bx + cos_a[:, None] * k[None, :]
    ys = by + sin_a[:, None] * k[None, :]
    px = (xs * PIX_PER_M).astype(np.int32)
    py = HF_RES - 1 - (ys * PIX_PER_M).astype(np.int32)
    inb = (px >= 0) & (px < HF_RES) & (py >= 0) & (py < HF_RES)
    pxc = np.clip(px, 0, HF_RES-1); pyc = np.clip(py, 0, HF_RES-1)
    wall = (_hf_bin[pyc, pxc]) & inb
    hit_any = wall | (~inb)
    R, S = hit_any.shape
    first = np.argmax(hit_any, axis=1)
    has = hit_any[np.arange(R), first]
    stop = np.where(has, first, S)
    cx = (xs / VOXEL).astype(np.int32)
    cy = (ys / VOXEL).astype(np.int32)
    key = cx * 4096 + cy
    free_mask = np.arange(S)[None, :] < stop[:, None]
    free_key = np.unique(key[free_mask])
    hi = np.minimum(first, S - 1)
    wall_key = np.unique(key[np.arange(R), hi][has & inb[np.arange(R), hi]])
    wall_set = set(wall_key.tolist())
    n_f = n_w = 0
    for kk in free_key:
        if kk in wall_set: continue
        vx, vy = divmod(int(kk), 4096)
        if gget(vx, vy) != FREE: gset(vx, vy, FREE); n_f += 1
    for kk in wall_key:
        vx, vy = divmod(int(kk), 4096)
        if gget(vx, vy) != WALL: gset(vx, vy, WALL); n_w += 1
    return n_f, n_w

# 起点自旋扫描（模拟 INIT_SCAN）
bx, by = 2.5, 2.5
t0 = time.time()
for deg in range(0, 360, 18):
    scan(bx, by, math.radians(deg))
t_scan = (time.time()-t0)/20
print(f"20 次全向扫描 平均 {t_scan*1000:.1f}ms/次")

print("\n== 起点感知图 (x=0..10, y=0..10, 0.2m/字符 .=FREE #=WALL ' '=UNK D=dog) ==")
for yi in range(50, -1, -1):
    y = yi*0.2
    row = ""
    for xi in range(0, 51):
        x = xi*0.2
        v = gget(int(x/VOXEL), int(y/VOXEL))
        c = "." if v==FREE else ("#" if v==WALL else " ")
        if abs(x-bx)<0.11 and abs(y-by)<0.11: c = "D"
        row += c
    print(f"y={y:5.1f} |{row}|")

# 计时细分
t0=time.time(); scan(2.5,2.5,0.0); t1=time.time()
print(f"\n单次 scan: {(t1-t0)*1000:.1f}ms (含 dict 写入)")
# 再扫一次（无新格，纯计算开销）
t0=time.time(); scan(2.5,2.5,0.0); t1=time.time()
print(f"单次 scan（重复）: {(t1-t0)*1000:.1f}ms")
