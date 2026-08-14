#!/usr/bin/env python3
"""record_traj.py — 漂移实验数据采集：记录三条轨迹到 CSV。

订阅：
  /odom       (nav_msgs/Odometry)      — sim_bridge 发的漂移里程计（错的）
  /true_pose  (geometry_msgs/PoseStamped) — 仿真真值（ground truth）
  TF map→base_footprint                — slam_toolbox 修正后的位姿（对的）

输出 CSV 列：t, true_x,true_y,true_yaw, odom_x,odom_y,odom_yaw, slam_x,slam_y,slam_yaw
  slam_* 在 slam 还没激活/TF 未就绪时为空。

用法（远程，需 ROS env）：
  python3 record_traj.py [duration_sec=120] [out.csv=_traj.csv]
"""
import sys
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import tf2_ros


def yaw_of(z, w):
    """纯偏航四元数 → 偏航角（rad）。"""
    return 2.0 * math.atan2(z, w)


class Recorder(Node):
    def __init__(self, out_path, duration):
        super().__init__("record_traj")
        self.out_path = out_path
        self.duration = duration
        self.odom = None
        self.true = None
        self.create_subscription(Odometry, "/odom", self._odom_cb, 10)
        self.create_subscription(PoseStamped, "/true_pose", self._true_cb, 10)
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf, self)
        self.f = open(out_path, "w")
        self.f.write("t,true_x,true_y,true_yaw,odom_x,odom_y,odom_yaw,"
                     "slam_x,slam_y,slam_yaw\n")
        self.t0 = self.get_clock().now().nanoseconds * 1e-9
        self.create_timer(0.2, self._tick)
        self.get_logger().info("recording -> %s for %.0fs" % (out_path, duration))

    def _odom_cb(self, m):
        self.odom = m

    def _true_cb(self, m):
        self.true = m

    def _tick(self):
        t = self.get_clock().now().nanoseconds * 1e-9 - self.t0
        row = ["%.2f" % t]
        if self.true is not None:
            p = self.true.pose.position
            o = self.true.pose.orientation
            row += ["%.3f" % p.x, "%.3f" % p.y, "%.3f" % yaw_of(o.z, o.w)]
        else:
            row += ["", "", ""]
        if self.odom is not None:
            p = self.odom.pose.pose.position
            o = self.odom.pose.pose.orientation
            row += ["%.3f" % p.x, "%.3f" % p.y, "%.3f" % yaw_of(o.z, o.w)]
        else:
            row += ["", "", ""]
        # slam 修正后位姿 = TF map→base_footprint
        slam = ["", "", ""]
        try:
            tr = self.tf_buf.lookup_transform(
                "map", "base_footprint", Time(), Duration(seconds=0.05))
            p = tr.transform.translation
            o = tr.transform.rotation
            slam = ["%.3f" % p.x, "%.3f" % p.y, "%.3f" % yaw_of(o.z, o.w)]
        except Exception:
            pass
        row += slam
        self.f.write(",".join(row) + "\n")
        self.f.flush()


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
    out = sys.argv[2] if len(sys.argv) > 2 else "/home/qin/simtrack/_traj.csv"
    rclpy.init()
    n = Recorder(out, duration)
    t0 = time.time()
    try:
        while time.time() - t0 < duration and rclpy.ok():
            rclpy.spin_once(n, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    n.f.close()
    n.destroy_node()
    rclpy.shutdown()
    print("DONE -> %s" % out)


if __name__ == "__main__":
    main()
