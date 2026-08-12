"""移动障碍感知跟踪器 —— 替代 DWA 直接读障碍真值速度（代码审核 2026-08-11 指出）。

现实情况：真实狗不知道障碍的速度向量，只能从连续激光帧里**看**出来。
本模块 = 极简多目标跟踪（雷达命中点 → 聚类 → 帧间关联 → 速度滤波）：

- 每次 scan 把"命中障碍体"的点集（估计系世界坐标）喂进来
- 距离聚类（1.2m 链式连通）得到障碍簇 → 质心 = 障碍位置观测
  （0.5m 半径盘的可见弧面，质心偏近狗侧 ~0.25m——是固定偏差，
   不影响速度估计；位置偏差由 DWA 的 r+0.3m 排斥余量覆盖）
- 簇与已有 track 最近邻关联（门限 1.5m：障碍 1m/s × 短暂离开视野可接受）
- 速度 = 位移/Δt 的指数平滑（EMA 0.5）——抗单帧噪声
- 2s 没再见到的 track 丢弃；静止 track（|v|<0.2）不喂 DWA
  （固定障碍由地图 WALL 格 + blocked() 静态兜底，不需要运动预测）

ROS 对应物：costmap_2d 的 obstacle layer 标记/清除 + 行人跟踪（如 leg_tracker）。
"""
import math


class ObstacleTracker:
    def __init__(self, cluster_link=1.2, min_cluster_pts=3, assoc_gate=1.5,
                 ema_v=0.5, drop_after=2.0, moving_thresh=0.2, fresh_for_dwa=0.6):
        self.cluster_link = cluster_link
        self.min_cluster_pts = min_cluster_pts
        self.assoc_gate = assoc_gate
        self.ema_v = ema_v
        self.drop_after = drop_after
        self.moving_thresh = moving_thresh
        self.fresh_for_dwa = fresh_for_dwa
        self.tracks = []   # dict: x, y, vx, vy, last_t, hits
        self.now = 0.0

    def update(self, pts, now):
        """pts: [(x, y), ...] 本次扫描的障碍命中点（估计系）；now: 仿真秒"""
        dt_since = now - self.now
        self.now = now
        # ── 聚类：链式距离连通 ──
        clusters = []
        for px, py in pts:
            for cl in clusters:
                if any(math.hypot(px - qx, py - qy) <= self.cluster_link for qx, qy in cl):
                    cl.append((px, py))
                    break
            else:
                clusters.append([(px, py)])
        cents = []
        for cl in clusters:
            if len(cl) >= self.min_cluster_pts:
                cents.append((sum(p[0] for p in cl) / len(cl),
                              sum(p[1] for p in cl) / len(cl)))
        # ── 关联 + 更新 ──
        used = set()
        for cx, cy in cents:
            best_i, best_d = None, self.assoc_gate
            for i, tr in enumerate(self.tracks):
                if i in used:
                    continue
                dd = math.hypot(cx - tr["x"], cy - tr["y"])
                if dd < best_d:
                    best_d, best_i = dd, i
            if best_i is None:
                self.tracks.append({"x": cx, "y": cy, "vx": 0.0, "vy": 0.0,
                                    "last_t": now, "hits": 1})
                used.add(len(self.tracks) - 1)
            else:
                tr = self.tracks[best_i]
                used.add(best_i)
                dt = now - tr["last_t"]
                if dt > 1e-3:
                    vmx = (cx - tr["x"]) / dt
                    vmy = (cy - tr["y"]) / dt
                    # 离谱速度（>3m/s，关联错误/两障碍串簇）不采纳进滤波
                    if vmx * vmx + vmy * vmy < 9.0:
                        tr["vx"] += (vmx - tr["vx"]) * self.ema_v
                        tr["vy"] += (vmy - tr["vy"]) * self.ema_v
                tr["x"], tr["y"] = cx, cy
                tr["last_t"] = now
                tr["hits"] += 1
        # ── 丢弃陈旧 track ──
        self.tracks = [tr for tr in self.tracks if now - tr["last_t"] <= self.drop_after]

    def moving(self, radius=0.5):
        """DWA 运动预测输入：[(x, y, vx, vy, r), ...] —— 纯感知估计，无真值。
        只给"确认在动"（观测≥3 次、估计速度≥阈值、最近 0.6s 内仍可见）的 track。"""
        out = []
        for tr in self.tracks:
            if tr["hits"] < 3 or self.now - tr["last_t"] > self.fresh_for_dwa:
                continue
            if math.hypot(tr["vx"], tr["vy"]) < self.moving_thresh:
                continue
            out.append((tr["x"], tr["y"], tr["vx"], tr["vy"], radius))
        return out
