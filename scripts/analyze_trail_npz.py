#!/usr/bin/env python3
"""分析 trail npz：时间花在哪（口袋绕行/卡点/bounce 分布）"""
import numpy as np, sys, math

path = sys.argv[1] if len(sys.argv) > 1 else "scans/trail_seed7_steps41327_b106.npz"
d = np.load(path)
trail = d["trail"]  # step, x, y, yaw_deg, bounce
bp = d["bounce_pts"]

print(f"trail={len(trail)} bounces_at={len(bp)}")
t0, t1 = int(trail[0,0]), int(trail[-1,0])
print(f"steps {t0}..{t1}  physics={t1*0.005:.0f}s")

# 每通道(y带)停留步数：y 带 = 2.5+5k ±2.5
print("\n== 各通道停留 (按 y 分带, 步数/物理秒) ==")
bands = {}
for s, x, y, yaw, b in trail:
    ch = int((y - 0.0) // 5)
    bands.setdefault(ch, [0, 0.0, 1e9, -1e9])
    bands[ch][0] += 1
    bands[ch][2] = min(bands[ch][2], s)
    bands[ch][3] = max(bands[ch][3], s)
for ch in sorted(bands):
    n, _, s0, s1 = bands[ch]
    print(f"y∈[{ch*5:4.1f},{ch*5+5:4.1f}): {n*20:6d} 步 ~{n*20*0.005:5.0f}s  首次步{s0:6.0f} 末次步{s1:6.0f}")

# 找"长时间不动"段（卡死）：连续 trail 点位置变化 <0.5m 且跨度 >2000 步
print("\n== 卡死段 (位移<1m 持续>1000步=5s) ==")
i = 0
segs = []
while i < len(trail) - 50:
    x0, y0, s0 = trail[i,1], trail[i,2], trail[i,0]
    j = i + 50
    while j < len(trail):
        if math.hypot(trail[j,1]-x0, trail[j,2]-y0) > 1.0:
            break
        j += 1
    dur = trail[j-1,0] - s0
    if dur > 1000:
        segs.append((int(s0), int(trail[j-1,0]), round(x0,1), round(y0,1), int(dur)))
    i = max(j, i+1)
for s0, s1, x, y, dur in segs[:20]:
    print(f"  步{s0:6d}-{s1:6d} ({dur*0.005:5.0f}s) @({x},{y})")

# bounce 位置分布
if len(bp):
    print("\n== bounce 位置聚类 ==")
    pts = bp[:, :2]
    used = np.zeros(len(pts), bool)
    cl = []
    for i in range(len(pts)):
        if used[i]: continue
        m = np.hypot(pts[:,0]-pts[i,0], pts[:,1]-pts[i,1]) < 1.5
        cl.append((pts[m,0].mean(), pts[m,1].mean(), int(m.sum())))
        used |= m
    cl.sort(key=lambda c: -c[2])
    for x, y, n in cl[:15]:
        print(f"  ({x:5.1f},{y:5.1f}) x{n}")

# 路径总长 & 重复
if len(trail) > 1:
    dist = sum(math.hypot(trail[i,0+1]-trail[i-1,1], trail[i,2]-trail[i-1,2]) for i in range(1, len(trail)))
    print(f"\n轨迹总长 ~{dist:.0f}m (理论蛇形 476m), 物理 {t1*0.005:.0f}s, 均速 {dist/max(t1*0.005,1):.2f}m/s")
