#!/usr/bin/env python3
"""两批实验对比 + 成功标准判定（调研文档 §五 验证协议）。

用法: compare_batches.py [基线目录] [新目录]   （缺省 batch1 batch2）

成功标准（§五）：
  ① >120s 慢房时间占比 < 3%
  ② 步进失败跳过事件 < 2 次（runner 日志统计，12 种子合计）
  ③ 完全到达种子数 ≥ 4
"""
import glob
import json
import math
import os
import re
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(dirn):
    rep = os.path.join(REPO, "results", dirn, "_report.json")
    return {R["seed"]: R for R in json.load(open(rep, encoding="utf-8"))}


def trans_stats(R):
    ts = sorted(x["dur"] for x in R["transits"])
    if not ts:
        return None
    return {"n": len(ts), "mean": statistics.mean(ts), "med": statistics.median(ts),
            "p90": ts[max(0, int(0.9 * len(ts)) - 1)], "max": max(ts),
            "slow": sum(1 for d in ts if d > 120)}


def log_events(dirn):
    """runner 日志里的事件计数（跳过/超时/拉黑/脱离/失败）。"""
    ev = {"skip": 0, "timeout": 0, "blacklist": 0, "escape": 0, "fail": 0}
    for p in glob.glob(os.path.join(REPO, "results", dirn, "seed*_runner.log")):
        txt = open(p, encoding="utf-8").read()
        txt = txt.replace("\n", "")  # tmux 80列折行重组
        ev["skip"] += len(re.findall(r"跳过该航点", txt))
        ev["timeout"] += len(re.findall(r"进度超时", txt))
        ev["blacklist"] += len(re.findall(r"拉黑 \(", txt))
        ev["escape"] += len(re.findall(r"换向脱离：", txt))
        ev["fail"] += len(re.findall(r"失败\(status=", txt))
    return ev


def main():
    A, B = (sys.argv[1:3] + ["batch1", "batch2"])[:2]
    ra, rb = load(A), load(B)
    common = sorted(set(ra) & set(rb))
    print("对比: 基线 %s (%d种子有效) vs 新 %s (%d种子有效)，共同种子 %d"
          % (A, len(ra), B, len(rb), len(common)))

    def agg(rr, seeds):
        rows = [trans_stats(rr[s]) for s in seeds]
        rows = [r for r in rows if r]
        allt = [x["dur"] for s in seeds for x in rr[s]["transits"]]
        slow_t = sum(d for d in allt if d > 120)
        return {
            "transits": len(allt), "mean": statistics.mean(allt) if allt else 0,
            "med": statistics.median(allt) if allt else 0,
            "p90": sorted(allt)[max(0, int(0.9 * len(allt)) - 1)] if allt else 0,
            "max": max(allt) if allt else 0,
            "slow_n": sum(1 for d in allt if d > 120),
            "slow_share": 100.0 * slow_t / sum(allt) if allt else 0,
            "arrived": sum(1 for s in seeds if rr[s]["reached"] >= rr[s]["n_path"]),
            "reached_sum": sum(rr[s]["reached"] for s in seeds),
            "narrow_ok": sum(1 for s in seeds if rr[s]["crossed_narrow"] is True),
        }

    aa, bb = agg(ra, common), agg(rb, common)
    print("\n%-22s %10s %10s %s" % ("指标(共同种子)", A, B, "判定"))
    print("-" * 66)
    for k, label, fmt, better in (
            ("arrived", "完全到达种子数", "%.0f", "up"),
            ("reached_sum", "到达房间总和", "%.0f", "up"),
            ("narrow_ok", "窄门通过种子数", "%.0f", "up"),
            ("mean", "房均通过 mean(s)", "%.1f", "down"),
            ("med", "中位(s)", "%.0f", "down"),
            ("p90", "p90(s)", "%.0f", "down"),
            ("max", "最差房(s)", "%.0f", "down"),
            ("slow_n", ">120s 慢房房次", "%.0f", "down"),
            ("slow_share", ">120s 时间占比(%)", "%.2f", "down")):
        va, vb = aa[k], bb[k]
        mark = ""
        if better == "up":
            mark = "✓" if vb > va else ("✗" if vb < va else "=")
        else:
            mark = "✓" if vb < va else ("✗" if vb > va else "=")
        print("%-22s %10s %10s  %s" % (label, fmt % va, fmt % vb, mark))

    print("\n每种子明细 (reached/n_path):")
    for s in common:
        da = "%d/%d" % (ra[s]["reached"], ra[s]["n_path"])
        db = "%d/%d" % (rb[s]["reached"], rb[s]["n_path"])
        na = trans_stats(ra[s]); nb = trans_stats(rb[s])
        fa = "慢%d" % na["slow"] if na else "-"
        fb = "慢%d" % nb["slow"] if nb else "-"
        print("  seed%-3d 基线 %-8s %-4s | 新 %-8s %-4s %s" % (
            s, da, fa, db, fb,
            "↑" if rb[s]["reached"] > ra[s]["reached"] else ("↓" if rb[s]["reached"] < ra[s]["reached"] else "")))

    for d in (A, B):
        print("\n%s runner 日志事件: %s" % (d, log_events(d)))
    print("\n成功标准判定（新批）: >120s时间占比 %.2f%% (目标<3%%) %s | 跳过事件 %d (目标<2) %s | 到达 %d (目标≥4) %s"
          % (bb["slow_share"], "✓" if bb["slow_share"] < 3 else "✗",
             log_events(B)["skip"], "✓" if log_events(B)["skip"] < 2 else "✗",
             bb["arrived"], "✓" if bb["arrived"] >= 4 else "✗"))


if __name__ == "__main__":
    main()
