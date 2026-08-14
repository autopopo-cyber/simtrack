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

    def _step_target(self, robot, wx, wy, max_step=3.0, grid=0.2, min_step=0.5):
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
        if best is None or math.hypot(best[0] - rx, best[1] - ry) < min_step:
            return None
        return best

    def _fan_step(self, robot, wx, wy):
        """直线步进被墙挡时的兜底：朝航点方向 ±15°..±90° 扇形扫，返回首个能走 ≥0.8m 的
        已知 free 步进点。效果=贴墙滑行，直到门口出现在步进方向上。
        没有 it 会死锁：slam 关键帧式建图下静止狗地图不长大，"等地图"永远等不到。"""
        base = math.atan2(wy - robot[1], wx - robot[0])
        for off in (15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90):
            ang = base + math.radians(off)
            probe = (robot[0] + math.cos(ang) * 3.0,
                     robot[1] + math.sin(ang) * 3.0)
            tgt = self._step_target(robot, probe[0], probe[1], min_step=0.8)
            if tgt is not None:
                return tgt
        return None

    def _route_step(self, robot, wx, wy, pad=3.0, max_expand=40000):
        """A* 规划 dog→waypoint：free=1、unknown=8（可穿但贵）、墙=∞。
        门已知 → 路径自然穿门（free 便宜）；门未知 → 路径乐观穿 unknown——包括**地图数组
        边界外**（数组外=从未观测=unknown，不是错误；航点常在地图外）。
        子目标 = 路径上"从狗出发连续 free 段"的最后一格（=前沿）：NavFn 对它必成功，
        狗推进后地图长大、下个 tick 重规划。比"closest-to-waypoint"目标强：后者在墙前
        有鞍点（绕行两侧到航点直线距离几乎相等），狗会钉在离航点最近的墙点上来回蹭。"""
        if self.latest_map is None:
            return None
        import heapq
        info = self.latest_map.info
        res, W, H = info.resolution, info.width, info.height
        ox, oy = info.origin.position.x, info.origin.position.y
        data = self.latest_map.data
        sc, sr = int((robot[0] - ox) / res), int((robot[1] - oy) / res)
        gc, gr = int((wx - ox) / res), int((wy - oy) / res)
        if not (0 <= sc < W and 0 <= sr < H):
            return None
        p = int(pad / res)
        c0, c1 = min(sc, gc) - p, max(sc, gc) + p   # 盒子不裁剪到地图内——数组外=unknown
        r0, r1 = min(sr, gr) - p, max(sr, gr) + p

        def cost(c, r):
            if not (0 <= c < W and 0 <= r < H):
                return 8.0                    # 数组外：从未观测 → unknown
            v = data[r * W + c]
            if v == 0:
                return 1.0
            if v < 0:
                return 8.0                    # unknown：可穿但贵
            return None                       # 墙

        def h(c, r):
            return math.hypot(c - gc, r - gr)

        openq = [(h(sc, sr), 0.0, sc, sr)]
        came, gsc = {}, {(sc, sr): 0.0}
        found, n_exp = False, 0
        while openq and n_exp < max_expand:
            _, g, c, r = heapq.heappop(openq)
            if (c, r) == (gc, gr):
                found = True
                break
            n_exp += 1
            for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nc, nr = c + dc, r + dr
                if not (c0 <= nc <= c1 and r0 <= nr <= r1):
                    continue
                w = cost(nc, nr)
                if w is None:
                    continue
                ng = g + w
                if ng < gsc.get((nc, nr), 1e18):
                    gsc[(nc, nr)] = ng
                    came[(nc, nr)] = (c, r)
                    heapq.heappush(openq, (ng + h(nc, nr), ng, nc, nr))
        if not found:
            return None
        # 回溯路径（goal→start），再从 start 向前找"连续 free 段"末尾作子目标
        path = [(gc, gr)]
        while path[-1] != (sc, sr):
            path.append(came[path[-1]])
        path.reverse()
        # 狗贴墙时自身格可能读作占据（A* 不穿墙，脏格最多只有起点这一格）→ 跳过
        start_i = 0 if data[path[0][1] * W + path[0][0]] == 0 else 1
        last_free = start_i
        for i in range(start_i, len(path)):
            c, r = path[i]
            if data[r * W + c] == 0:
                last_free = i
            else:
                break
        c, r = path[last_free]
        px, py = ox + (c + 0.5) * res, oy + (r + 0.5) * res
        if math.hypot(px - robot[0], py - robot[1]) < 0.8:
            # 连续 free 段太短=狗已在前沿边缘：允许子目标越入 unknown 1.5m
            # （NavFn 对近距离 unknown 目标没问题；卡死的是远距离全程 unknown）
            if last_free + 1 < len(path):
                nc, nr = path[last_free + 1]
                v = data[nr * W + nc] if 0 <= nc < W and 0 <= nr < H else -1
                if v < 0:
                    qx = px + 0.75 * (nc - c)
                    qy = py + 0.75 * (nr - r)
                    if math.hypot(qx - robot[0], qy - robot[1]) >= 0.8:
                        return (qx, qy)
            return None   # 前方是墙不是 unknown → 交给扇形兜底
        return (px, py)

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
        # 航点未探明（冷启动/探索在前）→ 两级推进：A* 找门/前沿 > 扇形兜底
        tgt = self._route_step(robot, wx, wy)
        how = "route" if tgt is not None else None
        if tgt is None:
            tgt = self._fan_step(robot, wx, wy)
            how = "fan" if tgt is not None else None
        if tgt is None:
            self.get_logger().info("航点 #%d (%.1f,%.1f)：前方待探明，等地图…"
                                   % (self.idx, wx, wy), throttle_duration_sec=5.0)
            return
        d = math.hypot(tgt[0] - robot[0], tgt[1] - robot[1])
        self.get_logger().info("→ 航点 #%d/%d (%.1f,%.1f) %s (%.1f,%.1f) +%.1fm"
                               % (self.idx, len(WAYPOINTS) - 1, wx, wy,
                                  {"route": "A*推进至", "fan": "扇形绕行至"}[how],
                                  tgt[0], tgt[1], d))
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
