"""轨迹耗时分析：时间花在哪个速度段/哪个通道/bounce 在哪。
用法: python scripts/analyze_speed.py scans/trail_seedXXX.npz
"""
import sys
import numpy as np

def main(path):
    d = np.load(path)
    tr = d["trail"]   # [step, x, y, yaw_deg, bounce]
    bp = d["bounce_pts"] if "bounce_pts" in d else None
    dt = (tr[1, 0] - tr[0, 0]) * 0.005   # 记录间隔(步)→物理秒
    pos = tr[:, 1:3]
    seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    spd = seg / dt
    t_total = (tr[-1, 0] - tr[0, 0]) * 0.005
    print(f"总物理时间 {t_total:.0f}s, 轨迹点 {len(tr)}, 里程 {seg.sum():.0f}m, 均速 {seg.sum()/t_total:.2f}m/s")
    bands = [(0, 0.3), (0.3, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 10.0)]
    print("速度段耗时分布：")
    for lo, hi in bands:
        m = (spd >= lo) & (spd < hi)
        print(f"  v {lo:.1f}-{hi:.1f}: {m.sum()*dt:6.0f}s ({m.sum()*dt/t_total*100:4.1f}%)  里程 {seg[m].sum():5.0f}m")
    # 通道耗时（y//5）
    print("各通道耗时：")
    ch = (tr[:-1, 1+1] // 5).astype(int)
    for c in range(10):
        m = ch == c
        if m.any():
            print(f"  ch{c}: {m.sum()*dt:6.0f}s  里程 {seg[m].sum():5.0f}m")
    # bounce 位置聚簇
    if bp is not None and len(bp):
        b = np.array(bp)
        print(f"bounce 共 {len(b)} 次，按通道分布：")
        bch = (b[:, 1] // 5).astype(int)
        for c in range(10):
            n = (bch == c).sum()
            if n: print(f"  ch{c}: {n} 次")
        # 相邻 bounce 间隔
        if len(b) > 2:
            print(f"  前 5 个 bounce 位置: {b[:5, :2].tolist()}")

if __name__ == "__main__":
    main(sys.argv[1])
