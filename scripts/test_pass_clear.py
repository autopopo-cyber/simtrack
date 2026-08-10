"""门宽度判断（净空场 PASS）静态几何验证 —— 2026-08-10

不跑仿真，直接在真值地图上验证门宽度判断的两个性质：
①连通性：PASS（净空≥0.6m 可规划格）从起点到终点**全图连通**——净空场没把迷宫封死；
②窄缝封闭：混合场 9 个弯道障碍（贴外墙 x=48.5/1.5，1.0m 窄缝）与外墙之间的缝
  **全部对规划关闭**（栅栏陷阱识别——能透过激光但过不去的缝不是路）；
③宽路不误伤：直道中线、转弯内侧走线保持开放。

用法：.venv/Scripts/python scripts/test_pass_clear.py
退出码 0=全过，1=有失败项。
"""
import sys
import numpy as np
import cv2
from PIL import Image

VOXEL = 0.1
GRID_N = 500
PASS_CLEAR = 6.0          # 格 = 0.6m（与 algo3_headless.PASS_CLEAR_M 对齐）
ROAD_PIX = 128

# ── 真值地图 → 0.1m 栅格（4×4 MAX-pool：任一真值像素是墙则格为墙）──
hf = np.array(Image.open("confirmed/track_clean.png").convert("L"))
assert hf.shape == (2000, 2000), hf.shape
wall_px = hf != ROAD_PIX
# 图像 row0 = y=50m 顶部 → 翻到 G[x, y]（gy=0 = y=0m）
wall_px = wall_px[::-1, :]
WALLG = wall_px.reshape(GRID_N, 4, GRID_N, 4).any(axis=(1, 3)).T   # [x, y]

def to_cell(wx, wy):
    return int(wx / VOXEL), int(wy / VOXEL)

def paint_disk(G, ox, oy, r=0.5):
    """障碍物理盘标墙（保守：整个盘，不只是感知面弧）"""
    cx, cy = to_cell(ox, oy)
    R = int(np.ceil(r / VOXEL))
    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            wx, wy = (cx+dx+0.5)*VOXEL, (cy+dy+0.5)*VOXEL
            if (wx-ox)**2 + (wy-oy)**2 < r*r:
                if 0 <= cx+dx < GRID_N and 0 <= cy+dy < GRID_N:
                    G[cx+dx, cy+dy] = True

# ── 混合场弯道障碍（贴外墙，与 algo3_headless.init_mix_obstacles 同几何）──
bends = [(48.5, k*5.0) if k % 2 == 1 else (1.5, k*5.0) for k in range(1, 10)]
# 注意：init_mix_obstacles 会用 sample_hf/_obs_hits_wall 过滤嵌墙的；这里全部保留做几何验证
G = WALLG.copy()
for ox, oy in bends:
    paint_disk(G, ox, oy)

# ── 感知变体：墙厚先验 +2 格（狗实际感知到的墙比真值厚 0.2m）──
G_perc = cv2.dilate(G.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

def compute_pass(G):
    DIST = cv2.distanceTransform((~G).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return DIST >= PASS_CLEAR, DIST

def reachable_from(PASS, sx, sy):
    """PASS 格 4 连通洪水填充"""
    seen = np.zeros_like(PASS)
    if not PASS[sx, sy]:
        return seen
    q = [(sx, sy)]
    seen[sx, sy] = True
    while q:
        cx, cy = q.pop()
        for dx, dy in ((0,1),(0,-1),(1,0),(-1,0)):
            nx, ny = cx+dx, cy+dy
            if 0 <= nx < GRID_N and 0 <= ny < GRID_N and PASS[nx, ny] and not seen[nx, ny]:
                seen[nx, ny] = True
                q.append((nx, ny))
    return seen

fails = []

for name, Gx in (("真值", G), ("感知(+2格墙厚先验)", G_perc)):
    PASS, DIST = compute_pass(Gx)
    print(f"\n══ {name}地图 ══ PASS 可规划格 {int(PASS.sum())} / 开放格 {int((~Gx).sum())}")

    # ① 连通性：起点 (2.5,2.5) → 终点 (2.5,47.5) 邻域
    sx, sy = to_cell(2.5, 2.5)
    fx, fy = to_cell(2.5, 47.5)
    reach = reachable_from(PASS, sx, sy)
    ok = reach[fx-5:fx+6, fy-35:fy+6].any()   # 终点 3.5m 到达圈内有可达格
    print(f"  ①连通性 起点→终点3.5m圈: {'PASS' if ok else 'FAIL'}")
    if not ok: fails.append(f"{name}: 连通性")

    # ② 窄缝封闭：每个弯道障碍与外墙的 1.0m 缝（x∈[49,50] 或 [0,1]，|y-oy|≤0.4m）
    n_closed = 0
    for ox, oy in bends:
        slit_x0, slit_x1 = (49.05, 49.95) if ox > 25 else (0.05, 0.95)
        closed = True
        for wx in np.arange(slit_x0, slit_x1, 0.1):
            for wy in np.arange(oy-0.4, oy+0.41, 0.1):
                cx, cy = to_cell(wx, wy)
                if not Gx[cx, cy] and PASS[cx, cy]:   # 物理自由但规划可穿 → 缝没封住
                    closed = False
        n_closed += closed
    print(f"  ②窄缝封闭 {n_closed}/9: {'PASS' if n_closed == 9 else 'FAIL'}")
    if n_closed != 9: fails.append(f"{name}: 窄缝 {n_closed}/9")

    # ③ 宽路不误伤：直道中线 + 转弯内侧走线开放
    mid_ok = all(PASS[to_cell(25.0, 2.5+5*k)] for k in range(10))
    inner_ok = True
    for ox, oy in bends:
        ix = 46.5 if ox > 25 else 3.5     # 转弯内侧走线（远离贴墙障碍一侧）
        if not PASS[to_cell(ix, oy)]:
            inner_ok = False
    print(f"  ③直道中线 10/10: {'PASS' if mid_ok else 'FAIL'}  转弯内侧 9/9: {'PASS' if inner_ok else 'FAIL'}")
    if not mid_ok: fails.append(f"{name}: 直道中线")
    if not inner_ok: fails.append(f"{name}: 转弯内侧")

print()
if fails:
    print("[FAIL]", fails)
    sys.exit(1)
print("[OK] 门宽度判断静态验证全过：窄缝封闭 + 全图连通 + 宽路不误伤")
