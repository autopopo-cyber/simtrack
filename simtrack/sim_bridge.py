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
import json

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, TransformStamped, PoseStamped
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


def _wrap_angle(a):
    """把角度归一到 (-pi, pi]。"""
    return (a + math.pi) % (2 * math.pi) - math.pi


class OdometryDrift:
    """足式机器人式里程计漂移模型——模拟轮速计/腿足里程计的真实误差（足式比轮式差）。

    模型：
      - 前进速度有 ±scale_pct 的**尺度偏置**（符号随机，种子可复现）→ 行程累积漂移
      - 偏航率有**常值偏置** yaw_bias（陀螺漂移，~0.5°/s 量级）→ 横向误差 ∝ 行程，最难修
      - 两者叠加高斯噪声

    用法：每物理步喂"真实达成的机身速度"，输出漂移后的里程计位姿。
    真值仍在 SimBackend 内部不变，供 /true_pose 测量对比。

    典型量级（scale_pct=0.05, yaw_bias≈0.5°/s）：走 30m 行程后原始里程计位置
    可漂移数米——这正是足式机器人没有外部定位时的真实处境。
    """

    def __init__(self, start, scale_pct=0.05, yaw_bias_deg=0.5,
                 noise_v=0.01, noise_w=0.01, seed=42):
        self.rng = np.random.default_rng(seed)
        # 尺度偏置：固定符号（一次抽样，整跑可复现），代表里程计标定误差
        self.scale_f = 1.0 + self.rng.uniform(-scale_pct, scale_pct)
        # 偏航率常值偏置（陀螺零漂）
        self.yaw_bias = self.rng.uniform(-1.0, 1.0) * math.radians(yaw_bias_deg)
        self.nv = noise_v      # 速度噪声 std (m/s)
        self.nw = noise_w      # 角速度噪声 std (rad/s)
        self.x, self.y, self.yaw = start

    def step(self, vx_body, vyaw, dt):
        """用真实机身速度推进漂移里程计一个 dt。"""
        vx_m = vx_body * self.scale_f + float(self.rng.normal(0, self.nv))
        vw_m = vyaw + self.yaw_bias + float(self.rng.normal(0, self.nw))
        self.yaw = _wrap_angle(self.yaw + vw_m * dt)
        self.x += vx_m * math.cos(self.yaw) * dt
        self.y += vx_m * math.sin(self.yaw) * dt

    def bias_str(self):
        return "scale_f=%.3f yaw_bias=%+.2f°/s" % (
            self.scale_f, math.degrees(self.yaw_bias))


class SimBridge(Node):
    def __init__(self):
        super().__init__("sim_bridge")
        self.get_logger().info("sim_bridge starting…")

        # ── SimBackend ──
        # 迷宫由环境变量 MAZE 选择（loop20 / rooms5x5 / rooms10x10），文件名 confirmed/maze_<name>.png
        proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        maze_name = os.environ.get("MAZE", "loop20")
        maze_path = os.path.join(proj, "confirmed", "maze_%s.png" % maze_name)
        if not os.path.exists(maze_path):
            self.get_logger().warn("MAZE=%s 的文件 %s 不存在，回退 maze_loop20.png"
                                   % (maze_name, maze_path))
            maze_path = os.path.join(proj, "confirmed", "maze_loop20.png")
            maze_name = "loop20"

        # 起点/尺寸读 maze_<name>.meta.json sidecar（maze_gen 生成）。
        # 不同房间尺寸起点不同：3m房→(1.5,1.5)，5m房→(2.5,2.5)。无 sidecar 则回退 (1.5,1.5)。
        meta_path = os.path.join(proj, "confirmed", "maze_%s.meta.json" % maze_name)
        sx, sy, syaw = 1.5, 1.5, 0.0
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                s = meta.get("start", [1.5, 1.5])
                sx, sy = float(s[0]), float(s[1])
                syaw = float(meta.get("start_yaw", 0.0))
                g = meta.get("goal")
                self.get_logger().info(
                    "迷宫元数据 {}: start=({:.1f},{:.1f}) goal={} size={:.0f}x{:.0f}m".format(
                        maze_name, sx, sy,
                        tuple(round(v, 1) for v in g) if g else None,
                        meta.get("w", 0), meta.get("h", 0)))
            except Exception as e:
                self.get_logger().warn("读 %s 失败 (%s)，回退 start=(1.5,1.5)" % (meta_path, e))

        self.sim = SimBackend(
            maze_path=maze_path,
            start=(sx, sy, syaw),
            lidar_rays=360,
            lidar_fov_deg=360,
            lidar_range=15.0,
            timestep=0.01,   # 100Hz 物理
            use_mujoco_viewer=False,
            px_per_m=50,
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

        # ── 里程计漂移（可选，足式机器人式）：env ODOM_DRIFT_PCT>0 开启 ──
        # 开启后 /odom 与 TF(odom→base) 发的是漂移位姿；真值仍在内部，发 /true_pose 供测量。
        # ODOM_DRIFT_PCT 是百分比（5 = 5%），内部转成小数。
        drift_pct = float(os.environ.get("ODOM_DRIFT_PCT", "0"))
        if drift_pct > 0:
            self.drift = OdometryDrift(
                start=(sx, sy, syaw),
                scale_pct=drift_pct / 100.0,
                yaw_bias_deg=float(os.environ.get("ODOM_DRIFT_YAW_BIAS_DEG", "0.5")),
                noise_v=float(os.environ.get("ODOM_DRIFT_NOISE_V", "0.01")),
                noise_w=float(os.environ.get("ODOM_DRIFT_NOISE_W", "0.01")),
                seed=int(os.environ.get("ODOM_DRIFT_SEED", "42")),
            )
            self.true_pub = self.create_publisher(PoseStamped, "/true_pose", 10)
            self.get_logger().info(
                "里程计漂移 ON: %s —— /odom 为漂移估计，/true_pose 发真值供测量"
                % self.drift.bias_str())
        else:
            self.drift = None
            self.true_pub = None
        self._prev_true = (self.sim.x, self.sim.y, self.sim.yaw)

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
        # 推进漂移里程计：用"真实达成的机身速度"（真值位姿差分），代表里程计实测
        if self.drift is not None:
            tx, ty, tyaw = self.sim.get_true_pose()
            dt = self.physics_dt
            ptx, pty, ptyaw = self._prev_true
            # 世界位移 → 机身前进分量（狗只前进不横移）
            vx_body = ((tx - ptx) * math.cos(tyaw) + (ty - pty) * math.sin(tyaw)) / dt
            vyaw_real = _wrap_angle(tyaw - ptyaw) / dt
            self.drift.step(vx_body, vyaw_real, dt)
            self._prev_true = (tx, ty, tyaw)
        # 发时钟
        cmsg = Clock()
        cmsg.clock = _stamp(self.sim_time)
        self.clock_pub.publish(cmsg)

    def _publish_cb(self):
        """10Hz：发 TF（先）+ /odom + /scan（后）——TF 先发确保 slam 能查到。

        漂移开启时：TF/odom 用**漂移位姿**（错的），/true_pose 发**真值**（测量用），
        /scan 仍用真值扫描——让 slam_toolbox 拿"错的里程计 + 对的激光"去修正。
        """
        stamp = self._now()

        # ── 位姿：真值 + 漂移里程计 ──
        tx, ty, tyaw = self.sim.get_true_pose()
        if self.drift is not None:
            ox, oy, oyaw = self.drift.x, self.drift.y, self.drift.yaw
        else:
            ox, oy, oyaw = tx, ty, tyaw
        ohalf = oyaw * 0.5
        osz, ocz = math.sin(ohalf), math.cos(ohalf)

        # ── TF odom → base_footprint（先发！用漂移位姿）──
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = "odom"
        tf.child_frame_id = "base_footprint"
        tf.transform.translation.x = ox
        tf.transform.translation.y = oy
        tf.transform.rotation.z = osz
        tf.transform.rotation.w = ocz
        self.tf_broadcaster.sendTransform(tf)

        # ── /odom（漂移位姿；twist 用真值速度——Nav2 预测用，不影响建图）──
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_footprint"
        odom.pose.pose.position.x = ox
        odom.pose.pose.position.y = oy
        odom.pose.pose.orientation.z = osz
        odom.pose.pose.orientation.w = ocz
        odom.twist.twist.linear.x = self.sim.vx * math.cos(tyaw) + self.sim.vy * math.sin(tyaw)
        odom.twist.twist.angular.z = self.sim.vyaw
        self.odom_pub.publish(odom)

        # ── /true_pose（漂移开启时发，纯测量诊断，frame=true 防被误用）──
        if self.true_pub is not None:
            tp = PoseStamped()
            tp.header.stamp = stamp
            tp.header.frame_id = "true"
            tp.pose.position.x = tx
            tp.pose.position.y = ty
            thalf = tyaw * 0.5
            tp.pose.orientation.z = math.sin(thalf)
            tp.pose.orientation.w = math.cos(thalf)
            self.true_pub.publish(tp)

        # ── /scan（最后发——TF 已在 buffer 里；始终用真值扫描）──
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
