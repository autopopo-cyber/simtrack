#!/usr/bin/env python3
"""analyze_drift.py — 分析漂移实验 CSV，出轨迹图 + 误差曲线 + 统计。

对比三条轨迹（ground truth / 原始漂移里程计 / slam 修正后），量化 slam_toolbox
把足式机器人式里程计漂移压下去了多少。

用法：
  python analyze_drift.py _traj.csv [out_prefix]
输出：
  <prefix>_traj.png   三轨迹叠加图
  <prefix>_error.png  位置误差随时间曲线
  + 终端打印 max/mean/rms/endpoint 误差与修正倍数
"""
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def pil_traj_plot(tx, ty, ox, oy, sx, sy, out):
    """matplotlib 不可用时的 PIL 轨迹图（黑白红蓝三线）。"""
    from PIL import Image, ImageDraw
    W = H = 900
    margin = 50
    xs = np.concatenate([tx, ox, sx])
    ys = np.concatenate([ty, oy, sy])
    xs = xs[np.isfinite(xs)]
    ys = ys[np.isfinite(ys)]
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    span = max(xmax - xmin, ymax - ymin, 1.0) * 1.1
    cx = (xmax + xmin) / 2
    cy = (ymax + ymin) / 2
    xmin, xmax = cx - span / 2, cx + span / 2
    ymin, ymax = cy - span / 2, cy + span / 2

    def pj(x, y):
        px = margin + (x - xmin) / (xmax - xmin) * (W - 2 * margin)
        # y 翻转（图像 row 向下）
        py = (H - margin) - (y - ymin) / (ymax - ymin) * (H - 2 * margin)
        return px, py

    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    # 边框 + 网格
    d.rectangle([margin, margin, W - margin, H - margin], outline=(180, 180, 180))

    def line(a, color, w):
        pts = [pj(x, y) for x, y in zip(a[0], a[1]) if np.isfinite(x) and np.isfinite(y)]
        for i in range(len(pts) - 1):
            d.line([pts[i], pts[i + 1]], fill=color, width=w)

    line((ox, oy), (220, 0, 0), 2)      # odom red
    line((sx, sy), (0, 0, 200), 2)      # slam blue
    line((tx, ty), (0, 0, 0), 3)        # true black (last, on top)
    # start marker
    sp = pj(float(tx[0]), float(ty[0]))
    d.ellipse([sp[0] - 7, sp[1] - 7, sp[0] + 7, sp[1] + 7],
              fill=(0, 255, 0), outline=(0, 0, 0))
    d.text((margin + 5, 8), "black=true  red=raw odom  blue=slam-corrected  green=start",
           fill=(0, 0, 0))
    img.save(out)


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else "_traj.csv"
    prefix = sys.argv[2] if len(sys.argv) > 2 else "drift_analysis"

    d = np.genfromtxt(csv, delimiter=",", names=True, filling_values=np.nan)

    def col(name):
        return np.where(np.isfinite(d[name]), d[name], np.nan)

    t = col("t")
    tx, ty = col("true_x"), col("true_y")
    ox, oy = col("odom_x"), col("odom_y")
    sx, sy = col("slam_x"), col("slam_y")

    odom_err = np.hypot(ox - tx, oy - ty)
    slam_err = np.hypot(sx - tx, sy - ty)

    def stats(e):
        e = e[np.isfinite(e)]
        if len(e) == 0:
            return None
        return dict(max=float(np.nanmax(e)), mean=float(np.nanmean(e)),
                    rms=float(np.sqrt(np.nanmean(e ** 2))), n=len(e))

    def last(a):
        f = a[np.isfinite(a)]
        return float(f[-1]) if len(f) else float("nan")

    # 行程（真值轨迹总长）
    seg = np.hypot(np.diff(tx), np.diff(ty))
    path = float(np.nansum(seg))
    dur = float(np.nanmax(t) - np.nanmin(t))

    os_ = stats(odom_err)
    ss_ = stats(slam_err)
    odom_end = last(odom_err)
    slam_end = last(slam_err)

    print("=" * 64)
    print("DRIFT EXPERIMENT RESULTS")
    print("=" * 64)
    print("duration: %.1fs   path traveled: %.1fm   avg speed: %.2fm/s"
          % (dur, path, path / dur if dur else 0))
    if os_:
        print("raw odom drift : max=%.2fm  mean=%.2fm  rms=%.2fm  endpoint=%.2fm"
              % (os_["max"], os_["mean"], os_["rms"], odom_end))
    if ss_:
        print("slam-corrected : max=%.2fm  mean=%.2fm  rms=%.2fm  endpoint=%.2fm"
              % (ss_["max"], ss_["mean"], ss_["rms"], slam_end))
    if np.isfinite(slam_end) and slam_end > 1e-3 and np.isfinite(odom_end):
        print("endpoint correction: %.1fx  (odom %.2fm -> slam %.2fm)"
              % (odom_end / slam_end, odom_end, slam_end))
    if path > 0:
        pe = 100 * odom_end / path if np.isfinite(odom_end) else float("nan")
        ps = 100 * slam_end / path if np.isfinite(slam_end) else float("nan")
        print("endpoint drift %% of path:  raw odom %.1f%%   slam %.1f%%" % (pe, ps))
    print("=" * 64)

    # ── 轨迹图 ──
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.plot(tx, ty, "k-", lw=2.5, label="true (ground truth)", zorder=3)
        ax.plot(ox, oy, "r-", lw=1.3, label="raw odom (drifted)", alpha=0.9)
        ax.plot(sx, sy, "b-", lw=1.3, label="slam-corrected", alpha=0.9)
        ax.plot(tx[0], ty[0], "o", color="lime", ms=12, mec="k", label="start", zorder=4)
        ax.set_aspect("equal")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_title("Trajectory: true vs raw-odom vs slam-corrected")
        fig.savefig(prefix + "_traj.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        # 误差随时间
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(t, odom_err, "r-", lw=1.2, label="raw odom error")
        ax.plot(t, slam_err, "b-", lw=1.2, label="slam-corrected error")
        ax.set_xlabel("time (s)"); ax.set_ylabel("position error (m)")
        ax.legend(loc="best"); ax.grid(alpha=0.3)
        ax.set_title("Position error over time")
        fig.savefig(prefix + "_error.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        print("saved %s_traj.png, %s_error.png" % (prefix, prefix))
    else:
        pil_traj_plot(tx, ty, ox, oy, sx, sy, prefix + "_traj.png")
        print("(matplotlib 不可用，用 PIL 出轨迹图) saved %s_traj.png" % prefix)


if __name__ == "__main__":
    main()
