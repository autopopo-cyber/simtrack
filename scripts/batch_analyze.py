#!/usr/bin/env python3
"""批量窄门实验分析：房间停留/通过时间统计 + 窄门专项 + 慢点检测。

输入：results/batch1/seed<k>_traj.csv（t,true_x,true_y,… 5Hz）+ seed<k>_meta.json
      （grid_x/grid_y 抖动网格线, path_rooms BFS 路径, narrow_door 窄门两侧房间）
输出：控制台报告 + results/batch1/_report.json

指标定义：
  - 房间归属：bisect 进抖动网格线（精确，不用 floor(x/5) 近似）
  - 每房通过时间 = 首次进入该房 → 首次进入路径下一房（含房内探索+找门+过门）
  - 房内"移动/停滞"分解：相邻样本位移>0.1m/0.2s 算移动
  - 慢点 = 通过时间 > SLOW_S（默认120s）的房间
"""
import csv
import glob
import json
import math
import os
from bisect import bisect_right

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(REPO, "results", "batch1")
SLOW_S = 120.0


def room_of(x, y, gx, gy):
    c = max(0, min(len(gx) - 2, bisect_right(gx, x) - 1))
    r = max(0, min(len(gy) - 2, bisect_right(gy, y) - 1))
    return (r, c)


def load_seed(traj_path, meta_path):
    meta = json.load(open(meta_path, encoding="utf-8"))
    rows = []
    with open(traj_path) as f:
        for r in csv.reader(f):
            if r and r[0] != "t" and r[1] and r[2]:
                rows.append((float(r[0]), float(r[1]), float(r[2])))
    rows.sort(key=lambda x: x[0])
    return rows, meta


def analyze_seed(seed, rows, meta):
    gx, gy = meta["grid_x"], meta["grid_y"]
    path = [tuple(rc) for rc in meta["path_rooms"]]
    narrow = [tuple(rc) for rc in (meta.get("narrow_door") or [])]
    dt = 0.2  # 5Hz

    # 每样本房间 + 速度
    samples = []
    for i, (t, x, y) in enumerate(rows):
        if i > 0:
            pt, px, py = rows[i - 1]
            v = math.hypot(x - px, y - py) / max(t - pt, 1e-3)
        else:
            v = 0.0
        samples.append((t, x, y, room_of(x, y, gx, gy), v))

    # 房间停留（累计）+ 移动/停滞分解
    dwell, dwell_move = {}, {}
    for t, x, y, rc, v in samples:
        dwell[rc] = dwell.get(rc, 0.0) + dt
        if v > 0.5:  # 0.5 m/s ≈ 巡航一半以上算移动
            dwell_move[rc] = dwell_move.get(rc, 0.0) + dt

    # 首次进入时间
    first_t = {}
    for t, x, y, rc, v in samples:
        if rc not in first_t:
            first_t[rc] = t

    # 路径上每房通过时间（首进本房 → 首进下一路径房）
    transits = []
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        if a in first_t and b in first_t:
            transits.append({"room": a, "next": b, "t_in": first_t[a],
                             "dur": first_t[b] - first_t[a],
                             "dwell": dwell.get(a, 0.0),
                             "is_narrow": a in narrow or b in narrow})

    crossed_narrow = all(r in first_t for r in narrow) if narrow else None
    reached = sum(1 for r in path if r in first_t)
    total_t = rows[-1][0] - rows[0][0] if len(rows) > 1 else 0.0
    return {"seed": seed, "transits": transits, "dwell": dwell, "dwell_move": dwell_move,
            "first_t": first_t, "path": path, "narrow": narrow,
            "crossed_narrow": crossed_narrow, "reached_rooms": reached,
            "total_t": total_t, "n_path": len(path)}


def main():
    results = []
    for traj in sorted(glob.glob(os.path.join(IN, "seed*_traj.csv"))):
        seed = int(os.path.basename(traj).split("_")[0].replace("seed", ""))
        meta_p = os.path.join(IN, "seed%d_meta.json" % seed)
        if not os.path.exists(meta_p):
            continue
        try:
            rows, meta = load_seed(traj, meta_p)
            if len(rows) < 50:
                print("seed%d 样本太少(%d)跳过" % (seed, len(rows)))
                continue
            results.append(analyze_seed(seed, rows, meta))
        except Exception as e:
            print("seed%d 分析失败: %r" % (seed, e))

    if not results:
        print("无数据")
        return

    print("=" * 72)
    print("每种子概况")
    print("=" * 72)
    all_transits = []
    for R in results:
        n_slow = sum(1 for x in R["transits"] if x["dur"] > SLOW_S)
        print("seed%-3d 路径%d房 到达%d/%d 耗时%.0fs 通过%d房 慢房%d个 窄门%s" % (
            R["seed"], R["n_path"], R["reached_rooms"], R["n_path"], R["total_t"],
            len(R["transits"]), n_slow,
            "过✓" if R["crossed_narrow"] else ("未到" if R["crossed_narrow"] is None else "未过✗")))
        all_transits.extend(x | {"seed": R["seed"]} for x in R["transits"])

    durs = sorted(x["dur"] for x in all_transits)
    nd = [x for x in all_transits if x["is_narrow"]]
    od = [x for x in all_transits if not x["is_narrow"]]

    def stats(v):
        if not v:
            return "n=0"
        v = sorted(v)
        med = v[len(v) // 2]
        p90 = v[min(len(v) - 1, int(len(v) * 0.9))]
        return "n=%d mean=%.0fs med=%.0fs p90=%.0fs min=%.0fs max=%.0fs" % (
            len(v), sum(v) / len(v), med, p90, v[0], v[-1])

    print()
    print("=" * 72)
    print("房间通过时间（含房内探索+找门+过门）")
    print("=" * 72)
    print("全部:  ", stats(durs))
    print("普通房:", stats([x["dur"] for x in od]))
    print("窄门房:", stats([x["dur"] for x in nd]))
    if durs:
        slow = [x for x in all_transits if x["dur"] > SLOW_S]
        tot = sum(durs)
        print()
        print("慢点(>%ds): %d 个房次, 占总通过时间 %.0f%%" % (
            SLOW_S, len(slow), 100.0 * sum(x["dur"] for x in slow) / tot if tot else 0))
        for x in sorted(slow, key=lambda x: -x["dur"])[:15]:
            tag = " [窄门!]" if x["is_narrow"] else ""
            print("  seed%-3d 房间(%d,%d)→(%d,%d) %.0fs 停留%.0fs%s" % (
                x["seed"], *x["room"], *x["next"], x["dur"], x["dwell"], tag))
        fast = sorted(all_transits, key=lambda x: x["dur"])[:5]
        print("最快5房次: ", ", ".join("seed%d(%d,%d) %.0fs" % (x["seed"], *x["room"], x["dur"]) for x in fast))
        # 占比分段
        for lo, hi in ((0, 30), (30, 60), (60, 120), (120, 10 ** 9)):
            n = sum(1 for d in durs if lo <= d < hi)
            label = ("<%ds" % hi) if hi < 10 ** 9 else (">%ds" % lo)
            print("  %s: %d 房次 (%.0f%%), 时间占比 %.0f%%" % (
                label, n, 100.0 * n / len(durs),
                100.0 * sum(d for d in durs if lo <= d < hi) / tot))

    out = [{"seed": R["seed"], "reached": R["reached_rooms"], "n_path": R["n_path"],
            "total_t": R["total_t"], "crossed_narrow": R["crossed_narrow"],
            "transits": R["transits"]} for R in results]
    json.dump(out, open(os.path.join(IN, "_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n明细已写 %s" % os.path.join(IN, "_report.json"))


if __name__ == "__main__":
    main()
