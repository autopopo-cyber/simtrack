#!/usr/bin/env python3
"""goal_runner.py — 沿房间路径依次发 NavigateToPose，演示全程自主导航到终点。

路径来自迷宫生成时的 BFS 房间连通图，作为"楼层图先验"提供（类似真实部署有建筑平面图/
拓扑地图）。低层建图(slam_toolbox)+规划控制(Nav2)+漂移修正完全自主——这个节点只做
"按顺序报下一个房间中心"。

健壮性：
  - 发航点前先确认该格在 SLAM 地图上已 free（狗的 15m 激光能提前扫到相邻房间），
    没 free 就等地图长大，避免往未知区发不可达目标。
  - 已在 2m 内的航点自动跳过（狗可能已越过）。
  - 失败重试 2 次（给地图长大/Nav2 恢复机会），仍失败则跳过下一站继续。

用法（远程，需 ROS env，drift/slam/nav2 已起）：
  python3 -m simtrack.goal_runner
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
import tf2_ros

# rooms10x10 seed42 的 BFS 房间路径质心（世界坐标 = map 帧；IMU 漂移下 slam_err~0.1m 可忽略）
WAYPOINTS = [
    (2.5, 2.5), (2.5, 7.5), (7.5, 7.5), (12.5, 7.5), (12.5, 2.5),
    (17.5, 2.5), (22.5, 2.5), (27.5, 2.5), (32.5, 2.5), (37.5, 2.5),
    (42.5, 2.5), (47.5, 2.5), (47.5, 7.5), (47.5, 12.5), (47.5, 17.5),
    (47.5, 22.5), (47.5, 27.5), (47.5, 32.5), (47.5, 37.5), (47.5, 42.5),
    (47.5, 47.5),
]


class GoalRunner(Node):
    def __init__(self):
        super().__init__("goal_runner")
        self.nav = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.tf_buf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tf_buf, self)
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(OccupancyGrid, "/map", self._map_cb, map_qos)
        self.latest_map = None
        self.idx = 0          # 下一个要去的航点
        self.busy = False     # 是否有 goal 在途
        self.retries = 0
        self._ff_done = False  # 是否已快进到离狗最近的航点（重启续跑用）
        self.create_timer(2.0, self._tick)
        self.get_logger().info("goal_runner: %d 个航点，终点 (47.5,47.5)" % len(WAYPOINTS))

    def _map_cb(self, msg):
        self.latest_map = msg

    def _robot_pose(self):
        try:
            tf = self.tf_buf.lookup_transform("map", "base_link", rclpy.time.Time())
            return (tf.transform.translation.x, tf.transform.translation.y)
        except Exception:
            return None

    def _free_in_map(self, wx, wy):
        if self.latest_map is None:
            return False
        info = self.latest_map.info
        col = int((wx - info.origin.position.x) / info.resolution)
        row = int((wy - info.origin.position.y) / info.resolution)
        if not (0 <= col < info.width and 0 <= row < info.height):
            return False
        return self.latest_map.data[row * info.width + col] == 0

    def _step_target(self, robot, wx, wy, max_step=3.0, grid=0.2):
        """沿 robot→waypoint 射线，返回地图上已知 free 的最远点（≤max_step）。
        逐 0.2m 探：让狗始终往"已探明 free"的方向走，狗一动地图就长大，下一步能走更远——
        解决 slam 关键帧式建图下"静止狗地图不长大、远航点永不 free"的冷启动死锁。"""
        rx, ry = robot
        dx, dy = wx - rx, wy - ry
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return None
        ux, uy = dx / dist, dy / dist
        best = None
        d = grid
        while d <= min(dist, max_step):
            px, py = rx + ux * d, ry + uy * d
            if self._free_in_map(px, py):
                best = (px, py)
                d += grid
            else:
                break
        if best is None or math.hypot(best[0] - rx, best[1] - ry) < 0.5:
            return None
        return best

    def _tick(self):
        if self.busy:
            return
        if self.idx >= len(WAYPOINTS):
            return
        if not self.nav.wait_for_server(timeout_sec=0.5):
            return
        robot = self._robot_pose()
        if robot is None:
            return
        # 首次：快进到离狗最近的航点（runner 重启时狗可能已在路径中段，避免回退到起点）
        if not self._ff_done:
            self.idx = min(range(len(WAYPOINTS)),
                           key=lambda i: math.hypot(WAYPOINTS[i][0] - robot[0],
                                                     WAYPOINTS[i][1] - robot[1]))
            self._ff_done = True
            self.get_logger().info("续跑：从最近航点 #%d (%.1f,%.1f) 开始（狗在 %.1f,%.1f）"
                                   % (self.idx, WAYPOINTS[self.idx][0],
                                      WAYPOINTS[self.idx][1], robot[0], robot[1]))
        wx, wy = WAYPOINTS[self.idx]
        # 到达当前航点（1.5m 内）→ 下一站
        if math.hypot(wx - robot[0], wy - robot[1]) < 1.5:
            self.get_logger().info("  ✅ 航点 #%d/%d (%.1f,%.1f) 达成"
                                   % (self.idx, len(WAYPOINTS) - 1, wx, wy))
            self.idx += 1
            self.retries = 0
            if self.idx >= len(WAYPOINTS):
                self.get_logger().info("🏁🏁 到达终点！(21 房间全程完成)")
            return
        # 航点已探明 free → 直接发 Nav2，让 NavFn 自己绕门规划（解决狗偏离门轴时直线撞墙）
        if self._free_in_map(wx, wy):
            d = math.hypot(wx - robot[0], wy - robot[1])
            self.get_logger().info("→ 航点 #%d/%d (%.1f,%.1f) 直冲 距离=%.1fm"
                                   % (self.idx, len(WAYPOINTS) - 1, wx, wy, d))
            self._send(wx, wy)
            return
        # 航点未探明（冷启动/探索在前）→ 沿射线找已知 free 最远点逐步推进
        tgt = self._step_target(robot, wx, wy)
        if tgt is None:
            self.get_logger().info("航点 #%d (%.1f,%.1f)：前方待探明，等地图…"
                                   % (self.idx, wx, wy), throttle_duration_sec=5.0)
            return
        d = math.hypot(tgt[0] - robot[0], tgt[1] - robot[1])
        self.get_logger().info("→ 航点 #%d/%d (%.1f,%.1f) 步进至 (%.1f,%.1f) +%.1fm"
                               % (self.idx, len(WAYPOINTS) - 1, wx, wy, tgt[0], tgt[1], d))
        self._send(tgt[0], tgt[1])

    def _send(self, wx, wy):
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(wx)
        goal.pose.pose.position.y = float(wy)
        goal.pose.pose.orientation.w = 1.0
        self.busy = True
        fut = self.nav.send_goal_async(goal)
        fut.add_done_callback(lambda f: self._resp(f, wx, wy))

    def _resp(self, future, wx, wy):
        gh = future.result()
        if gh is None or not gh.accepted:
            self.get_logger().warn("航点 (%.1f,%.1f) 被拒" % (wx, wy))
            self._advance_or_retry(wx, wy, ok=False)
            return
        res = gh.get_result_async()
        res.add_done_callback(lambda f: self._result(f, wx, wy))

    def _result(self, future, wx, wy):
        wrap = future.result()
        ok = wrap is not None and wrap.status == GoalStatus.STATUS_SUCCEEDED
        self._advance_or_retry(wx, wy, ok)

    def _advance_or_retry(self, wx, wy, ok):
        self.busy = False
        if ok:
            self.retries = 0
            # 步进目标到达，不推进 idx——_tick 会据 1.5m 判定是否到达航点并推进
        else:
            self.retries += 1
            if self.retries >= 4:
                self.get_logger().warn("  步进 (%.1f,%.1f) 失败 %d 次，跳过该航点"
                                       % (wx, wy, self.retries))
                self.retries = 0
                self.idx += 1
            else:
                self.get_logger().warn("  步进 (%.1f,%.1f) 失败，重试 %d"
                                       % (wx, wy, self.retries))


def main():
    rclpy.init()
    n = GoalRunner()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
