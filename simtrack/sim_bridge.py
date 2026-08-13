#!/usr/bin/env python3
"""
sim_bridge.py — ROS2 节点，把 MuJoCo SimBackend 包装成标准 ROS topic。

发布：
  /scan   (sensor_msgs/LaserScan)  — 360° 激光，frame=laser
  /odom   (nav_msgs/Odometry)      — 轮式里程计，frame=odom → base_link
  /tf     odom→base_link（动态）+ base_link→laser（静态）
  /clock  (rosgraph_msgs/Clock)    — 仿真时钟（use_sim_time 的主时钟）

订阅：
  /cmd_vel (geometry_msgs/Twist)   — Nav2 速度指令 → SimBackend

本节点是仿真时钟的主人（use_sim_time=false 对自己，发 /clock 给别人）。
sim_bridge 不使用 sim time（避免鸡生蛋），用 wall-clock 跑 timer。

用法：
  source /opt/ros/jazzy/setup.bash
  python3 -m simtrack.sim_bridge
"""
import math
import sys
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, TransformStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from rosgraph_msgs.msg import Clock
from builtin_interfaces.msg import Time

# 确保 simtrack 包可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simtrack.sim_server import SimBackend


def _stamp(sim_time):
    """sim_time（秒，float）→ builtin_interfaces/Time。"""
    secs = int(sim_time)
    nsecs = int((sim_time - secs) * 1e9)
    t = Time(sec=secs, nanosec=nsecs)
    return t


class SimBridge(Node):
    def __init__(self):
        super().__init__("sim_bridge")
        self.get_logger().info("sim_bridge starting…")

        # ── SimBackend ──
        self.sim = SimBackend(
            start=(1.5, 1.5, 0.0),
            lidar_rays=360,
            lidar_fov_deg=360,
            lidar_range=15.0,
            timestep=0.01,   # 100Hz 物理
            use_mujoco_viewer=False,
        )
        self.physics_dt = 0.01
        self.sim_time = 0.0          # 仿真秒（从 0 开始）
        self._cmd = (0.0, 0.0)       # (vx_body, vyaw) 最新速度指令

        # ── QoS ──
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=10,
        )
        clock_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # /clock 标准 QoS
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )

        # ── Publisher ──
        self.scan_pub = self.create_publisher(LaserScan, "/scan", sensor_qos)
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.clock_pub = self.create_publisher(Clock, "/clock", clock_qos)

        # ── TF ──
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self._publish_static_tf()

        # ── Subscriber ──
        self.create_subscription(Twist, "/cmd_vel", self._cmd_cb, 10)

        # ── Timer：物理 100Hz，发布 10Hz ──
        self.physics_timer = self.create_timer(self.physics_dt, self._physics_cb)
        self.pub_timer = self.create_timer(0.1, self._publish_cb)

        self.get_logger().info(
            f"sim_bridge ready: maze={self.sim.hf_w}×{self.sim.hf_h}px "
            f"({self.sim.px_per_m}px/m), lidar={self.sim.lidar_rays}rays/"
            f"{self.sim.lidar_range}m, physics={self.physics_dt}s"
        )

    # ──────────────────────────────────────────
    def _now(self):
        """wall-clock 时间戳（不用 sim_time，避免 /clock 同步问题）。"""
        return self.get_clock().now().to_msg()

    def _publish_static_tf(self):
        """静态 TF：base_footprint→base_link（零偏移）+ base_footprint→laser。
        base_footprint 给 slam_toolbox，base_link 给 Nav2——两者都要。"""
        stamp = self._now()
        # base_footprint → base_link（Nav2 local_costmap 要 base_link）
        t1 = TransformStamped()
        t1.header.stamp = stamp
        t1.header.frame_id = "base_footprint"
        t1.child_frame_id = "base_link"
        t1.transform.rotation.w = 1.0
        # base_footprint → laser（scan frame）
        t2 = TransformStamped()
        t2.header.stamp = stamp
        t2.header.frame_id = "base_footprint"
        t2.child_frame_id = "laser"
        t2.transform.translation.z = 0.5
        t2.transform.rotation.w = 1.0
        self.static_broadcaster.sendTransform(t1)
        self.static_broadcaster.sendTransform(t2)

    def _cmd_cb(self, msg: Twist):
        self._cmd = (msg.linear.x, msg.angular.z)

    # ──────────────────────────────────────────
    def _physics_cb(self):
        """100Hz：推进物理 + 发 /clock。"""
        vx, vyaw = self._cmd
        self.sim.set_cmd_vel(vx, vyaw)
        self.sim.step()
        self.sim_time += self.physics_dt
        # 发时钟
        cmsg = Clock()
        cmsg.clock = _stamp(self.sim_time)
        self.clock_pub.publish(cmsg)

    def _publish_cb(self):
        """10Hz：发 TF（先）+ /odom + /scan（后）——TF 先发确保 slam 能查到。"""
        stamp = self._now()

        # ── 位姿 ──
        x, y, yaw = self.sim.get_true_pose()
        half = yaw * 0.5
        sz, cz = math.sin(half), math.cos(half)

        # ── TF odom → base_footprint（先发！）──
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_footprint"
        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.rotation.z = sz
        tf.transform.rotation.w = cz
        self.tf_broadcaster.sendTransform(tf)

        # ── /odom ──
        vx_body = self.sim.vx * math.cos(yaw) + self.sim.vy * math.sin(yaw)
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_footprint"
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation.z = sz
        odom.pose.pose.orientation.w = cz
        odom.twist.twist.linear.x = vx_body
        odom.twist.twist.angular.z = self.sim.vyaw
        self.odom_pub.publish(odom)

        # ── /scan（最后发——TF 已在 buffer 里）──
        ranges, angles = self.sim.get_scan()
        scan = LaserScan()
        scan.header.stamp = stamp
        scan.header.frame_id = "laser"
        scan.angle_min = float(angles[0])
        scan.angle_max = float(angles[-1])
        scan.angle_increment = float(angles[1] - angles[0])
        scan.time_increment = 0.0
        scan.scan_time = 0.1
        scan.range_min = 0.02
        scan.range_max = self.sim.lidar_range
        scan.ranges = [float(r) for r in ranges]
        self.scan_pub.publish(scan)


def main():
    rclpy.init()
    node = SimBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.sim.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
