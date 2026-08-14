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
from nav_msgs.msg import Odometry, OccupancyGrid
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
            # 真实 Unitree L2（A2 前后双装）：10m@10%反射/30m@90%，精度±3cm。
            # LIDAR_RANGE=10 + LIDAR_NOISE_M=0.03 = 保守设计点（暗面家具全场景）
            lidar_range=float(os.environ.get("LIDAR_RANGE", "15.0")),
            range_noise_m=float(os.environ.get("LIDAR_NOISE_M", "0.0")),
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
            f"{self.sim.lidar_range}m(noise±{self.sim.range_noise_m * 100:.0f}cm), "
            f"physics={self.physics_dt}s"
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

        # 周期性 LiDAR 重定位（env CORRECT_PERIOD_S，默认 30s；0=关）——航向/位置不裸积分，
        # 每隔 N 秒拿当前 scan 在已知迷宫上 scan-match 估出真实位姿，把漂移里程计重置回去。
        # 参考图选择（env CORRECT_REF）：
        #   true = 真迷宫高度图（仿真特权，仅实验上限参考——真机没有真图）
        #   map  = slam_toolbox 自建 /map（诚实版=土法 AMCL：反复经过已建图房间即可重锚，
        #          行约定 no-flip 已由 scripts/probe_map_convention.py 实测验证）
        self._correct_ref = os.environ.get("CORRECT_REF", "true").lower()
        self._correct_period = float(os.environ.get("CORRECT_PERIOD_S", "30")) if self.drift is not None else 0.0
        self._last_correct = 0.0
        self._map_wall = None      # (wall_bool, res, w, h, ox, oy, n_wall)
        if self.drift is not None and self._correct_period > 0:
            if self._correct_ref == "map":
                map_qos = QoSProfile(
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,  # slam 的 /map 是锁存的
                    history=HistoryPolicy.KEEP_LAST, depth=1)
                self.create_subscription(OccupancyGrid, "/map", self._map_cb, map_qos)
            self.create_timer(1.0, self._correction_cb)
            self.get_logger().info(
                "🛰 周期 LiDAR 重定位 ON：每 %.0fs 对%s scan-match 修正里程计漂移"
                % (self._correct_period,
                   "自建/map" if self._correct_ref == "map" else "真迷宫图(仿真特权)"))

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

    def _map_cb(self, msg: OccupancyGrid):
        """缓存 slam 自建图（墙格掩膜）。data>=65 为墙；-1 未知/0 自由自动排除。"""
        info = msg.info
        wall = np.array(msg.data, dtype=np.int16).reshape(info.height, info.width) >= 65
        self._map_wall = (wall, info.resolution, info.width, info.height,
                          info.origin.position.x, info.origin.position.y,
                          int(wall.sum()))

    def _match_score_map(self, r, a, x, y, yaw):
        """候选位姿下 scan 端点落在自建图墙格的个数。行约定 no-flip（探针实测验证）。"""
        wall, res, w, h, ox, oy, _ = self._map_wall
        wx = x + r * np.cos(a + yaw)
        wy = y + r * np.sin(a + yaw)
        col = ((wx - ox) / res).astype(np.int32)
        row = ((wy - oy) / res).astype(np.int32)
        inb = (col >= 0) & (col < w) & (row >= 0) & (row < h)
        col_c = np.clip(col, 0, w - 1)
        row_c = np.clip(row, 0, h - 1)
        return int((wall[row_c, col_c] & inb).sum())

    def _scan_match_map(self, r, a, ix, iy, iyaw):
        """相关匹配：从漂移 odom 初始位姿出发，对自建图两层搜索（粗±1.5m/±20° → 细±0.2m/±4°）。"""
        best, best_score = (ix, iy, iyaw), -1
        for dyaw in np.linspace(-0.35, 0.35, 11):
            for dx in np.linspace(-1.5, 1.5, 11):
                for dy in np.linspace(-1.5, 1.5, 11):
                    s = self._match_score_map(r, a, ix + dx, iy + dy, iyaw + dyaw)
                    if s > best_score:
                        best_score, best = s, (ix + dx, iy + dy, iyaw + dyaw)
        bx, by, byaw = best
        for dyaw in np.linspace(-0.07, 0.07, 7):
            for dx in np.linspace(-0.2, 0.2, 7):
                for dy in np.linspace(-0.2, 0.2, 7):
                    s = self._match_score_map(r, a, bx + dx, by + dy, byaw + dyaw)
                    if s > best_score:
                        best_score, best = s, (bx + dx, by + dy, byaw + dyaw)
        return best, best_score

    def _correction_cb(self):
        """周期性 LiDAR 重定位：scan-match 当前 scan 到参考图（真迷宫 或 slam 自建图），把漂移里程计重置回去。"""
        if self.drift is None:
            return
        if self.sim_time - self._last_correct < self._correct_period:
            return
        if self._correct_ref == "map":
            # 自建图未就绪（太年轻/墙格太少）→ 不更新 _last_correct，1s 后重试
            if self._map_wall is None or self._map_wall[-1] < 300:
                return
        self._last_correct = self.sim_time
        ranges, angles = self.sim.get_scan()
        valid = np.isfinite(ranges) & (ranges < self.sim.lidar_range)
        r = ranges[valid].astype(np.float32)
        a = angles[valid].astype(np.float32)
        if self._correct_ref == "map":
            (lx, ly, lyaw), score = self._scan_match_map(
                r, a, self.drift.x, self.drift.y, self.drift.yaw)
            # 得分太低=当前区域建图质量差，匹配不可信 → 拒绝修正（保持原 odom）
            if score < 40:
                self.get_logger().warn(
                    "🛰(map) @t=%.0fs 匹配得分 %d/%d 过低，跳过本次修正（该区域建图不足）"
                    % (self.sim_time, score, len(r)))
                return
        else:
            (lx, ly, lyaw), score = self.sim.scan_match(
                ranges, angles, self.drift.x, self.drift.y, self.drift.yaw)
        dpos = math.hypot(lx - self.drift.x, ly - self.drift.y)
        dyaw = (lyaw - self.drift.yaw + math.pi) % (2 * math.pi) - math.pi
        self.drift.x, self.drift.y, self.drift.yaw = lx, ly, lyaw
        self.get_logger().info(
            "🛰(%s) LiDAR重定位 @t=%.0fs：修正 pos %.2fm yaw %+.1f° (命中 %d/%d rays)"
            % ("map" if self._correct_ref == "map" else "true",
               self.sim_time, dpos, math.degrees(dyaw), score, len(r)))

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
