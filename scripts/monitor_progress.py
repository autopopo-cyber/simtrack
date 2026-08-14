"""轻量进度监控：每 15s 记录 true/slam 位姿 + 到终点距离到 _progress.log。
检测到 dist_goal<1.0 写 GOAL_REACHED，slam_err>15 写 MAP_LOST。
远程后台跑：python3 monitor_progress.py &
"""
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped
import tf2_ros

GOAL = (47.5, 47.5)


class Mon(Node):
    def __init__(self):
        super().__init__("progress_mon")
        self.true = None
        self.create_subscription(PoseStamped, "/true_pose", lambda m: setattr(self, "true", m), 10)
        self.tf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tf, self)
        self.t0 = time.time()
        self.f = open("/home/qin/simtrack/_progress.log", "w")
        self.reached = False
        self.create_timer(15.0, self._tick)

    def _tick(self):
        if self.true is None:
            return
        t = time.time() - self.t0
        p = self.true.pose.position
        dg = math.hypot(p.x - GOAL[0], p.y - GOAL[1])
        sx = sy = serr = float("nan")
        try:
            tr = self.tf.lookup_transform("map", "base_footprint", Time(),
                                          Duration(seconds=0.1))
            sx = tr.transform.translation.x
            sy = tr.transform.translation.y
            serr = math.hypot(sx - p.x, sy - p.y)
        except Exception:
            pass
        line = "t=%.0f true=(%.1f,%.1f) slam=(%.1f,%.1f) slam_err=%.2f dist_goal=%.1f" % (
            t, p.x, p.y, sx, sy, serr, dg)
        self.f.write(line + "\n")
        self.f.flush()
        if not self.reached and dg < 1.0:
            self.f.write("GOAL_REACHED at t=%.0f\n" % t)
            self.f.flush()
            self.reached = True
        if serr > 15.0:
            self.f.write("MAP_LOST_WARN slam_err=%.1f\n" % serr)
            self.f.flush()


rclpy.init()
n = Mon()
try:
    while rclpy.ok():
        rclpy.spin_once(n, timeout_sec=0.5)
except KeyboardInterrupt:
    pass
n.f.close()
n.destroy_node()
rclpy.shutdown()
