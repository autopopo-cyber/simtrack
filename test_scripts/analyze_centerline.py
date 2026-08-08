"""分析已知地图寻路路径是否"走中间"：跑阶段2（--load-map）时收集路径点，统计 wall_dist。

用法：阶段2 跑完后分析日志中的 [GATE]/[EXEC] 路径，或独立加载地图 + A* 规划一段路径验证。
"""
import sys, os, math
sys.path.insert(0, "/home/qin/workspace/simtrack")
os.chdir("/home/qin/workspace/simtrack")

import numpy as np
from PIL import Image

# 复用 algo3 的地图/规划（不跑主循环）
src = open("test_scripts/algo3_headless.py").read()
code = src[:src.index("# ═══════════════════════════════════════════\n# 主入口")]
# 屏蔽 argparse 解析（exec 的代码会 parse_args）
import sys
_saved_argv = sys.argv
sys.argv = ["algo3_core"]
ns = {}
ns["__file__"] = "test_scripts/algo3_headless.py"
import contextlib, io
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(code, "algo3_core", "exec"), ns)
sys.argv = _saved_argv

load_map = ns["load_map"]
gget_plan = ns["gget_plan"]
wall_dist = ns["wall_dist"]
VOXEL = ns["VOXEL"]
astar_to = ns["astar_to"]

def analyze_path(map_path, start, goal):
    """加载地图，A* 规划 start→goal，统计路径 wall_dist"""
    if not load_map(map_path):
        print(f"加载失败: {map_path}")
        return
    path = astar_to(int(start[0]/VOXEL), int(start[1]/VOXEL),
                    int(goal[0]/VOXEL), int(goal[1]/VOXEL))
    if not path:
        print("A* 无路径!")
        return
    print(f"路径长度: {len(path)} 点, start={start} → goal={goal}")
    ds = []
    for wx, wy in path:
        d = wall_dist(int(wx/VOXEL), int(wy/VOXEL))
        ds.append(d)
    ds = np.array(ds)
    print(f"wall_dist: 平均={ds.mean():.2f} 中位={np.median(ds):.2f} 最小={ds.min():.1f} 最大={ds.max():.1f}")
    # 通道宽 5m = 50 格，中线 wall_dist 应 ~25 格
    mid_ratio = ds.mean() / 25.0
    print(f"居中率: {mid_ratio*100:.0f}% (100%=完美中线)")
    # 前 20 点路径
    print("前 20 点:", [(round(wx,1), round(wy,1), int(d)) for (wx,wy),d in zip(path[:20], ds[:20])])
    return path, ds

if __name__ == "__main__":
    map_path = sys.argv[1] if len(sys.argv) > 1 else "scans/map_seed7.npz"
    start = (2.5, 2.5)
    goal = (2.5, 47.5)
    if len(sys.argv) > 3:
        start = (float(sys.argv[2]), float(sys.argv[3]))
    if len(sys.argv) > 5:
        goal = (float(sys.argv[4]), float(sys.argv[5]))
    analyze_path(map_path, start, goal)
