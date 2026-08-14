#!/usr/bin/env python3
"""探针 v2：把相关匹配器本体当探针，验证 /map y 行约定 + 验证"对自建图重定位"可行性。

从 /odom（漂移里程计）初始位姿出发，对 /map（slam 自建图）做两层相关搜索：
  粗 ±1.5m/±20°(11³) → 细 ±0.2m/±4°(7³)
评分 = scan 端点落在墙格(data>=65)的个数，分别用 flip / no-flip 两种行约定。
输出：各约定最佳得分、匹配位姿、与真值/里程计的偏差。
正确约定：得分显著>0 且匹配位姿在真值附近（差 ≈ 地图局部变形，~1-2m 内）。
"""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped


def yaw_from_q(q):
    return 2.0 * math.atan2(q.z, q.w)


class Probe(Node):
    def __init__(self):
        super().__init__("probe_map_match")
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        scan_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=10)
        self.create_subscription(OccupancyGrid, "/map", self.map_cb, map_qos)
        self.create_subscription(LaserScan, "/scan", self.scan_cb, scan_qos)
        self.create_subscription(Odometry, "/odom", self.odom_cb, 10)
        self.create_subscription(PoseStamped, "/true_pose", self.true_cb, 10)
        self.map = self.scan = self.odom = self.truep = None

    def map_cb(self, m): self.map = m
    def scan_cb(self, s): self.scan = s
    def odom_cb(self, o): self.odom = o
    def true_cb(self, p): self.truep = p

    def run_match(self):
        m, s, o, p = self.map, self.scan, self.odom, self.truep
        res = m.info.resolution
        w, h = m.info.width, m.info.height
        ox, oy = m.info.origin.position.x, m.info.origin.position.y
        wall = (np.array(m.data, dtype=np.int16).reshape(h, w) >= 65)
        n_wall = int(wall.sum())

        yaw = yaw_from_q(o.pose.pose.orientation)
        ix, iy = o.pose.pose.position.x, o.pose.pose.position.y

        a = np.arange(len(s.ranges)) * s.angle_increment + s.angle_min
        r = np.array(s.ranges)
        ok = np.isfinite(r) & (r > 0.1) & (r < s.range_max)
        r, a = r[ok].astype(np.float32), a[ok].astype(np.float32)
        if len(r) < 30 or n_wall < 300:
            print("数据不足: rays=%d 墙格=%d" % (len(r), n_wall))
            return

        def score_noflip(x, y, tyaw):
            wx = x + r * np.cos(a + tyaw); wy = y + r * np.sin(a + tyaw)
            col = ((wx - ox) / res).astype(np.int32)
            row = ((wy - oy) / res).astype(np.int32)
            inb = (col >= 0) & (col < w) & (row >= 0) & (row < h)
            c2 = np.clip(col, 0, w - 1); r2 = np.clip(row, 0, h - 1)
            return int((wall[r2, c2] & inb).sum())

        def score_flip(x, y, tyaw):
            wx = x + r * np.cos(a + tyaw); wy = y + r * np.sin(a + tyaw)
            col = ((wx - ox) / res).astype(np.int32)
            row = (h - 1) - ((wy - oy) / res).astype(np.int32)
            inb = (col >= 0) & (col < w) & (row >= 0) & (row < h)
            c2 = np.clip(col, 0, w - 1); r2 = np.clip(row, 0, h - 1)
            return int((wall[r2, c2] & inb).sum())

        def search(score_fn):
            best, bs = (ix, iy, yaw), -1
            for dyaw in np.linspace(-0.35, 0.35, 11):
                for dx in np.linspace(-1.5, 1.5, 11):
                    for dy in np.linspace(-1.5, 1.5, 11):
                        v = score_fn(ix + dx, iy + dy, yaw + dyaw)
                        if v > bs: bs, best = v, (ix + dx, iy + dy, yaw + dyaw)
            bx, by, byaw = best
            for dyaw in np.linspace(-0.07, 0.07, 7):
                for dx in np.linspace(-0.2, 0.2, 7):
                    for dy in np.linspace(-0.2, 0.2, 7):
                        v = score_fn(bx + dx, by + dy, byaw + dyaw)
                        if v > bs: bs, best = v, (bx + dx, by + dy, byaw + dyaw)
            return best, bs

        import time
        tx, ty = p.pose.position.x, p.pose.position.y
        tyaw_t = yaw_from_q(p.pose.orientation)
        print("地图: %dx%d res=%.3f origin=(%.2f,%.2f) 墙格=%d" % (w, h, res, ox, oy, n_wall))
        print("初始(漂移odom)=(%.2f,%.2f,%.1f°) 真值=(%.2f,%.2f,%.1f°) odom-真值偏差=%.2fm" % (
            ix, iy, math.degrees(yaw), tx, ty, math.degrees(tyaw_t),
            math.hypot(ix - tx, iy - ty)))
        for name, fn in (("no-flip", score_noflip), ("flip", score_flip)):
            t0 = time.time()
            (mx, my, myaw), sc = search(fn)
            dt = time.time() - t0
            print("%-8s 得分 %d/%d (%.0f%%)  位姿=(%.2f,%.2f,%.1f°)  距真值=%.2fm 距odom=%.2fm  [%.2fs]" % (
                name, sc, len(r), 100.0 * sc / len(r), mx, my, math.degrees(myaw),
                math.hypot(mx - tx, my - ty), math.hypot(mx - ix, my - iy), dt))


def main():
    rclpy.init()
    node = Probe()
    import time
    t0 = time.time()
    while time.time() - t0 < 30:
        rclpy.spin_once(node, timeout_sec=1000)
        if node.map is not None and node.scan is not None and node.odom is not None and node.truep is not None:
            node.run_match()
            break
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
