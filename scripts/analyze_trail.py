#!/usr/bin/env python3
"""分析 trail npz：通道进度 / bounce 分布 / 分段速度"""
import numpy as np, sys, math, os

def channel_of(y):
    """通道号：y=2.5+5k → k"""
    return int(round((y - 2.5) / 5.0))

def main(path):
    d = np.load(path, allow_pickle=True)
    tr = d['trail']
    bp = d['bounce_pts']
    print(f"=== {os.path.basename(path)} ===")
    print(f"seed={d['seed']} mode={d['mode']} lidar_rays={d['lidar_rays']} tick={d['lidar_tick']} no_obs={d['no_obs']}")
    print(f"轨迹点 {len(tr)}  bounce {len(bp)}")
    if len(tr) < 2:
        print("轨迹太短"); return

    steps = tr[:,0]; xs = tr[:,1]; ys = tr[:,2]; bounces = tr[:,4]
    T = steps[-1] - steps[0]
    dt = T / len(tr) if len(tr) > 1 else 1
    dur = steps[-1] / max(1, (steps[1]-steps[0]) if len(steps)>1 else 1)  # 模拟步数/实际
    print(f"总步数 {int(steps[-1])}  总位移 {math.hypot(xs[-1]-xs[0], ys[-1]-ys[0]):.1f}m")

    # 通道轨迹（按时间）
    chs = [channel_of(y) for y in ys]
    ch_visits = []
    cur = chs[0]; start_i = 0
    for i, c in enumerate(chs):
        if c != cur:
            ch_visits.append((cur, start_i, i, steps[start_i], steps[i-1]))
            cur = c; start_i = i
    ch_visits.append((cur, start_i, len(chs), steps[start_i], steps[-1]))
    print("\n通道访问（按时间）:")
    for ch, i0, i1, s0, s1 in ch_visits:
        dseg = math.hypot(xs[i1-1]-xs[i0], ys[i1-1]-ys[i0])
        seg_s = (s1 - s0) / max(1, (steps[1]-steps[0]))  # 步数→墙钟s 粗略（假设每步~0.005s?）
        # 用 (steps[i1]-steps[i0]) 步数 → 时间 = 步数/步速，步速从总时间推
        print(f"  通道{ch}: 步 {int(s0)}→{int(s1)} ({s1-s0:.0f}步) 位移 {dseg:.1f}m")

    # bounce 分布
    if len(bp):
        print(f"\nbounce {len(bp)} 个，位置分布:")
        for b in bp[:40]:
            print(f"  bounce#{int(b[2])} @({b[0]:.1f},{b[1]:.1f}) ch{channel_of(b[1])}")
        # 按通道聚类
        from collections import Counter
        c = Counter(channel_of(b[1]) for b in bp)
        print("  按通道:", dict(sorted(c.items())))

    # 步速统计
    last_step_t = None; speeds = []
    for i in range(1, len(tr)):
        ds = math.hypot(xs[i]-xs[i-1], ys[i]-ys[i-1])
        speeds.append(ds)
    sp = np.array(speeds)
    print(f"\n步距统计: 平均 {sp.mean()*10:.2f} cm/步(轨迹间隔) 中位 {np.median(sp)*10:.2f}cm 零位移步占比 {(sp<0.02).mean()*100:.1f}%")
    # bounce 时刻的零位移
    zb = (sp < 0.02).sum()
    print(f"零位移步（原地转向/停车）: {zb} / {len(sp)} = {zb/len(sp)*100:.1f}%")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
        print()
