#!/usr/bin/env python3
"""profile 单段 A*：50格路径为什么 450ms？"""
import time, sys, os, math
import numpy as np
import cProfile, pstats, io
sys.path.insert(0, 'test_scripts')
import importlib.util
spec = importlib.util.spec_from_file_location('a3', 'test_scripts/algo3_headless.py')
a3 = importlib.util.module_from_spec(spec)
sys.modules['a3'] = a3
sys.argv = ['x', '--no-obs', '1', '--load-map', 'scans/full_map.npz', '--timeout', '1',
            '--vision', '0', '--landmarks', '0', '--render-every', '0', '--max-steps', '1']
try:
    spec.loader.exec_module(a3)
except SystemExit:
    pass

H = W = 500
wall_mask = np.zeros((H, W), dtype=bool)
for (vx, vy), val in a3.static_grid.items():
    if val == 2 and 0 <= vy < H and 0 <= vx < W:
        wall_mask[vy, vx] = True
INF = 10**6
d = np.where(wall_mask, 0, INF).astype(np.int32)
for i in range(1, H):
    di, d1 = d[i], d[i-1]
    for j in range(1, W):
        di[j] = min(di[j], d1[j]+1, di[j-1]+1)
for i in range(H-2, -1, -1):
    di, d1 = d[i], d[i+1]
    for j in range(W-2, -1, -1):
        di[j] = min(di[j], d1[j]+1, di[j+1]+1)
a3._DIST_FIELD = d
def fast_wd(vx, vy):
    if 0 <= vy < H and 0 <= vx < W:
        return int(d[vy, vx])
    return 999
a3.wall_dist = fast_wd
a3._wd.clear()

# 单段：起点(47.5,2.5)→(2.5,7.5) 50格
pr = cProfile.Profile()
pr.enable()
t0 = time.time()
seg = a3.astar_to(475, 25, 25, 75)
t1 = time.time()
pr.disable()
print(f"段1 A*: {t1-t0:.3f}s path={len(seg) if seg else 0}")
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(15)
print(s.getvalue())
