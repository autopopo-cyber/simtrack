#!/usr/bin/env python3
"""
firefly_explorer.py — 自主探索 ROS2 节点（"萤火V3引擎"）。

订阅 slam_toolbox 的 /map（OccupancyGrid），自己找 frontier（free/unknown 边界），
挑最优一个，用 NavigateToPose 让 Nav2 走过去；到了/失败后再找下一个。
无人给航点，全自主探索整张地图。

frontier 检测移植自 algo3_headless.py 的 find_gates / _open_frontier / cluster_gates，
但寻路/控制全交给 Nav2——这里只做"选点"：
  1. free 格中至少有一个"开阔未知"邻居 → frontier 格（_open_frontier 的等价：未知邻居
     的 5×5 邻域无墙，墙缝/墙根后的未知射线永远扫不到，不是门是陷阱）。
  2. 8-连通聚类成 region（cluster_gates 等价），动态 min_size 防过滤死。
  3. 打分 = size（信息增益） - α·distance，跳过失败黑名单。
  4. 发 frontier 质心（snap 到实际 free 格）的 NavigateToPose。
  5. 成功→看新地图找下一个；失败/超时→拉黑该区域找下一个；无 frontier→探索完成。

用法：
  source /opt/ros/jazzy/setup.bash
  python3 -m simtrack.firefly_explorer
"""
import math
import sys
import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
import tf2_ros


class FireflyExplorer(Node):
    IDLE = 0
    NAVIGATING = 1

    def __init__(self):
        super().__init__("firefly_explorer")
        # ── 参数 ──
        self.declare_parameter("min_cluster", 3)        # 最小簇格数（稀疏前沿会动态下调）
        self.declare_parameter("robot_radius", 0.4)     # 机器人半径（m），宽缝判定用
        self.declare_parameter("nav_timeout", 90.0)     # 单点导航超时（s）
        self.declare_parameter("blacklist_r", 2.5)      # 失败区域拉黑半径（m）
        self.declare_parameter("dist_weight", 0.15)     # 距离项权重（越大越偏近）
        self.declare_parameter("min_goal_dist", 0.35)   # 离机器人太近(<此值)的 frontier 跳过：机器人在它上面时 Nav2 秒成功→重选→忙循环
        self.declare_parameter("wall_clear_cells", 1)   # 开阔判定：未知邻居的 ±N 格内无墙（slam_toolbox 0.05m/格，1=0.05m 轻过滤；0=纯 Yamauchi）

        self.min_cluster = self.get_parameter("min_cluster").value
        self.robot_radius = self.get_parameter("robot_radius").value
        self.nav_timeout = self.get_parameter("nav_timeout").value
        self.blacklist_r = self.get_parameter("blacklist_r").value
        self.dist_weight = self.get_parameter("dist_weight").value
        self.min_goal_dist = self.get_parameter("min_goal_dist").value
        self.wall_clear_cells = self.get_parameter("wall_clear_cells").value

        # ── 目标导向模式（可选）：env GOAL_X/GOAL_Y 设定终点 → firefly 改为冲终点 ──
        # 打分从"信息增益-距离"变成"朝终点贪心"(min d + goal_weight·dg)，size 仅作同距 tiebreaker。
        # 终点格在 SLAM 地图上变为 free 后，直发 NavigateToPose 冲终点。
        self.goal_mode = "GOAL_X" in os.environ and "GOAL_Y" in os.environ
        if self.goal_mode:
            self.declare_parameter("goal_weight", float(os.environ.get("GOAL_WEIGHT", "1.0")))
            self.goal_weight = self.get_parameter("goal_weight").value
            self.goal = (float(os.environ["GOAL_X"]), float(os.environ["GOAL_Y"]))
            self.goal_reached = False
            self._goal_retry_after = 0.0      # 终点直冲失败后的冷却（避免死循环）
        else:
            self.goal = None
            self.goal_weight = 0.0
            self.goal_reached = False
            self._goal_retry_after = 0.0

        # ── 状态 ──
        self.state = self.IDLE
        self._completed = False   # 探索是否已完成（完成后不再每帧重判，避免日志刷屏）
        self.latest_map = None
        self.blacklist = []          # [(wx, wy)] 失败区域中心（世界坐标）
        self.current_goal = None     # (wx, wy) 当前导航目标
        self.current_gh = None       # 当前 goal handle（看门狗取消用）
        self.nav_start = None        # 导航开始时刻

        # ── TF（取 map→base_link 机器人位姿）──
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Nav2 动作客户端 ──
        self.nav = ActionClient(self, NavigateToPose, "navigate_to_pose")

        # ── 订阅 /map（slam_toolbox 发，QoS 须匹配：transient_local + reliable）──
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, depth=1,
        )
        self.create_subscription(OccupancyGrid, "/map", self._map_cb, map_qos)

        # ── 看门狗：导航超时取消 + 拉黑 ──
        self.create_timer(2.0, self._watchdog)

        self._stats = {"sent": 0, "ok": 0, "fail": 0}
        if self.goal_mode:
            self.get_logger().info("firefly_explorer ready 🏁目标导向：终点=(%.1f,%.1f) goal_weight=%.2f"
                                   % (self.goal[0], self.goal[1], self.goal_weight))
        else:
            self.get_logger().info("firefly_explorer ready，等待 /map…")

    # ════════════════════════════════════════════
    # 地图回调（驱动整个探索循环）
    # ════════════════════════════════════════════
    def _map_cb(self, msg: OccupancyGrid):
        self.latest_map = msg
        if self._completed:
            return   # 已完成，不再每帧重判（否则日志被"探索结束"刷屏）
        if self.state == self.IDLE:
            self._try_explore()

    def _try_explore(self):
        if self.latest_map is None:
            return
        if self._completed:
            return
        if not self.nav.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("Nav2 navigate_to_pose 未就绪，等待…", throttle_duration_sec=5.0)
            return
        robot = self._robot_pose()
        if robot is None:
            self.get_logger().warn("查不到 map→base_link TF，等待…", throttle_duration_sec=5.0)
            return

        # 目标导向：终点格已在 SLAM 地图上变为 free 且过了冷却 → 直冲终点
        if (self.goal_mode and not self.goal_reached
                and self.get_clock().now().nanoseconds * 1e-9 > self._goal_retry_after
                and self._goal_is_free(self.latest_map)):
            dg = math.hypot(self.goal[0] - robot[0], self.goal[1] - robot[1])
            self.get_logger().info("🏁 终点已探明(map free)，直冲 (%.1f,%.1f) 距离=%.1fm"
                                   % (self.goal[0], self.goal[1], dg))
            self._send_goal(self.goal[0], self.goal[1], is_goal=True)
            return

        frontiers = self._detect_frontiers(self.latest_map)
        if not frontiers:
            self._completed = True
            self.get_logger().info(
                "✅ 无 frontier —— 探索完成！(sent=%d ok=%d fail=%d)"
                % (self._stats["sent"], self._stats["ok"], self._stats["fail"]))
            return

        target = self._pick_best(frontiers, robot)
        if target is None:
            self._completed = True
            self.get_logger().info(
                "所有 frontier 已拉黑，探索结束。(sent=%d ok=%d fail=%d)"
                % (self._stats["sent"], self._stats["ok"], self._stats["fail"]))
            return

        wx, wy, size = target["wx"], target["wy"], target["size"]
        d = math.hypot(wx - robot[0], wy - robot[1])
        self.get_logger().info(
            "→ 选 frontier #%d  size=%d  目标=(%.1f,%.1f)  距离=%.1fm  剩余=%d"
            % (target["id"], size, wx, wy, d, len(frontiers) - 1))
        self._send_goal(wx, wy)

    # ════════════════════════════════════════════
    # frontier 检测（numpy 向量化，移植 algo3 的 gate 逻辑）
    # ════════════════════════════════════════════
    def _detect_frontiers(self, msg: OccupancyGrid):
        info = msg.info
        W, H = info.width, info.height
        res = info.resolution
        g = np.array(msg.data, dtype=np.int8).reshape((H, W))
        free = (g == 0)
        unknown = (g == -1)
        wall = (g == 100)
        if not unknown.any():
            return []

        # 开阔未知：未知格中，5×5（=2格半径）邻域内无墙的。
        # 墙缝/墙根后的未知永远扫不到（被墙挡），不是门是陷阱 → 杀掉。
        wc = self.wall_clear_cells
        wall_near = wall.copy()
        for _ in range(wc):
            wall_near = self._dilate4(wall_near)
        good_unknown = unknown & ~wall_near

        # frontier = free 格，4-邻接至少一个 good_unknown
        near_good = self._dilate4(good_unknown)
        frontier = free & near_good
        if not frontier.any():
            return []

        # 聚类（8-连通）
        clusters = self._cluster8(frontier)
        # 动态 min_size：前沿格少（探索早期/斑点前沿）就降低阈值，防全过滤卡死。
        # 注：曾试过目标模式硬性 min=10 过滤碎 frontier，但会把探索早期的门洞 frontier
        # 一起杀掉（只剩起始大环），狗秒到就地宣布完成——已回退为动态阈值。
        dyn_min = 2 if len(clusters) < 40 else max(self.min_cluster, 3)

        result = []
        for i, comp in enumerate(clusters):
            if len(comp) < dyn_min:
                continue
            # 质心（像素，行/列），snap 到簇内最近格（保证是 free frontier 格）
            cy = sum(c[0] for c in comp) / len(comp)
            cx = sum(c[1] for c in comp) / len(comp)
            best = min(comp, key=lambda c: (c[0]-cy)**2 + (c[1]-cx)**2)
            # 宽度判定：簇内最远两点的米距 ≥ 机器人直径（够狗挤过去）
            # 近似用簇格数×分辨率——大簇天然够宽，小簇才需查
            w_m = math.sqrt(len(comp)) * res
            wx = info.origin.position.x + (best[1] + 0.5) * res
            wy = info.origin.position.y + (best[0] + 0.5) * res
            result.append({
                "id": i, "row": best[0], "col": best[1],
                "wx": wx, "wy": wy, "size": len(comp), "width_m": w_m,
            })
        return result

    @staticmethod
    def _dilate4(mask):
        """4-连通膨胀一格（numpy 平移实现，无 cv2/scipy 依赖）。"""
        out = mask.copy()
        out[1:, :]  |= mask[:-1, :]
        out[:-1, :] |= mask[1:, :]
        out[:, 1:]  |= mask[:, :-1]
        out[:, :-1] |= mask[:, 1:]
        return out

    @staticmethod
    def _cluster8(mask):
        """8-连通连通域（BFS，对稀疏 frontier 掩膜很快）。返回 list of [(row,col),...]。"""
        ys, xs = np.where(mask)
        cells = list(zip(ys.tolist(), xs.tolist()))
        cset = set(cells)
        seen = set()
        clusters = []
        for c in cells:
            if c in seen:
                continue
            stack = [c]; seen.add(c); comp = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        nb = (y+dy, x+dx)
                        if nb in cset and nb not in seen:
                            seen.add(nb); stack.append(nb)
            clusters.append(comp)
        return clusters

    # ════════════════════════════════════════════
    # 打分选最优
    # ════════════════════════════════════════════
    def _pick_best(self, frontiers, robot):
        rx, ry = robot
        best, best_score = None, -1e18
        for f in frontiers:
            if self._is_blacklisted(f["wx"], f["wy"]):
                continue
            d = math.hypot(f["wx"] - rx, f["wy"] - ry)
            if d < self.min_goal_dist:
                continue   # 机器人就在这 frontier 上：Nav2 秒成功→重选→忙循环，跳过
            if self.goal_mode and not self.goal_reached:
                # 目标导向：min(到机器人d + goal_weight·到终点dg)，size 仅作同距 tiebreaker
                dg = math.hypot(f["wx"] - self.goal[0], f["wy"] - self.goal[1])
                score = 0.005 * f["size"] - (d + self.goal_weight * dg)
            else:
                score = f["size"] - self.dist_weight * (d + 1e-3)
            if score > best_score:
                best_score = score; best = f
        return best

    def _is_blacklisted(self, wx, wy):
        for bx, by in self.blacklist:
            if math.hypot(wx - bx, wy - by) < self.blacklist_r:
                return True
        return False

    def _blacklist(self, wx, wy):
        if not self._is_blacklisted(wx, wy):
            self.blacklist.append((wx, wy))
            self.get_logger().info("  ⛔ 拉黑区域 (%.1f,%.1f) r=%.1f  累计=%d"
                                   % (wx, wy, self.blacklist_r, len(self.blacklist)))

    # ════════════════════════════════════════════
    # 发目标 / 结果处理
    # ════════════════════════════════════════════
    def _send_goal(self, wx, wy, is_goal=False):
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(wx)
        goal.pose.pose.position.y = float(wy)
        goal.pose.pose.orientation.w = 1.0
        self.state = self.NAVIGATING
        self.current_goal = (wx, wy)
        self.current_is_goal = is_goal
        self.nav_start = self.get_clock().now()
        self._goal_seq = (getattr(self, "_goal_seq", 0) or 0) + 1   # 单调目标序号（竞态守卫）
        seq = self._goal_seq
        self._stats["sent"] += 1
        fut = self.nav.send_goal_async(goal, feedback_callback=self._fb_cb)
        fut.add_done_callback(lambda f: self._goal_resp_cb(f, wx, wy, seq, is_goal))

    def _fb_cb(self, feedback):
        # 距离剩余（Nav2 NavigateToPose feedback 有 distance_remaining）
        try:
            d = feedback.feedback.distance_remaining
            self.get_logger().debug("  …距离剩余 %.2fm" % d)
        except Exception:
            pass

    def _goal_resp_cb(self, future, gx, gy, seq, is_goal):
        # 竞态守卫：若此目标已被看门狗取代（seq 过期），丢弃结果
        if seq != self._goal_seq:
            self.get_logger().debug("  丢弃过期目标响应 (seq=%d)" % seq)
            return
        gh = future.result()
        if gh is None or not gh.accepted:
            self.get_logger().warn("  Nav2 拒绝目标，拉黑后重选")
            if not is_goal:
                self._blacklist(gx, gy)
            else:
                self._goal_retry_after = self.get_clock().now().nanoseconds * 1e-9 + 15.0
            self.current_gh = None
            self.current_goal = None
            self.state = self.IDLE
            self._try_explore()
            return
        self.current_gh = gh
        res_fut = gh.get_result_async()
        res_fut.add_done_callback(lambda f: self._result_cb(f, gx, gy, seq, is_goal))

    def _result_cb(self, future, gx, gy, seq, is_goal):
        if seq != self._goal_seq:
            # 看门狗已发新目标——旧结果丢弃，不碰状态/统计
            return
        wrap = future.result()
        ok = wrap is not None and wrap.status == GoalStatus.STATUS_SUCCEEDED
        if ok:
            self._stats["ok"] += 1
            if is_goal:
                self.goal_reached = True
                self._completed = True
                self.get_logger().info(
                    "🏁🏁 到达终点 (%.1f,%.1f)！(sent=%d ok=%d fail=%d)"
                    % (gx, gy, self._stats["sent"], self._stats["ok"], self._stats["fail"]))
                self.current_gh = None
                self.current_goal = None
                self.state = self.IDLE
                return
            self.get_logger().info("  ✅ 到达 (%.1f,%.1f)" % (gx, gy))
        else:
            self._stats["fail"] += 1
            if is_goal:
                # 终点直冲失败：多半是路径还没探通——冷却 15s 用 frontier 推进，不拉黑终点
                self._goal_retry_after = self.get_clock().now().nanoseconds * 1e-9 + 15.0
                self.get_logger().warn("  ❌ 终点直冲失败 status=%s，冷却 15s 改用 frontier"
                                       % (wrap.status if wrap else None))
            else:
                self.get_logger().warn("  ❌ 失败 status=%s，拉黑" % (wrap.status if wrap else None))
                self._blacklist(gx, gy)
        self.current_gh = None
        self.current_goal = None
        self.state = self.IDLE
        # 立刻用最新地图找下一个
        self._try_explore()

    def _watchdog(self):
        if self.state != self.NAVIGATING:
            return
        if (self.get_clock().now() - self.nav_start).nanoseconds * 1e-9 > self.nav_timeout:
            is_goal = getattr(self, "current_is_goal", False)
            if is_goal:
                # 终点直冲超时：不拉黑终点，冷却 15s 改用 frontier
                self._goal_retry_after = self.get_clock().now().nanoseconds * 1e-9 + 15.0
                self.get_logger().warn("  ⏱ 终点直冲超时(%.0fs)，冷却改用 frontier" % self.nav_timeout)
            else:
                self.get_logger().warn("  ⏱ 导航超时(%.0fs)，取消+拉黑" % self.nav_timeout)
                self._blacklist(*self.current_goal)
            self._cancel_current()
            self.current_goal = None
            self.state = self.IDLE
            # 推进序号，使在途的旧结果回调失效
            self._goal_seq = (self._goal_seq or 0) + 1
            self._try_explore()

    def _cancel_current(self):
        # 取消动作（尽力而为，不阻塞）
        gh = self.current_gh
        if gh is not None:
            try:
                gh.cancel_goal_async()
            except Exception:
                pass
            self.current_gh = None

    # ════════════════════════════════════════════
    # 工具
    # ════════════════════════════════════════════
    def _robot_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
            return (tf.transform.translation.x, tf.transform.translation.y)
        except Exception:
            return None

    def _goal_is_free(self, msg: OccupancyGrid):
        """终点格在 SLAM 地图上是否为 free(==0)。OccupancyGrid data 行主序 idx=row*width+col。"""
        info = msg.info
        col = int((self.goal[0] - info.origin.position.x) / info.resolution)
        row = int((self.goal[1] - info.origin.position.y) / info.resolution)
        if not (0 <= col < info.width and 0 <= row < info.height):
            return False
        return msg.data[row * info.width + col] == 0


def main():
    rclpy.init()
    node = FireflyExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("firefly_explorer 收到 Ctrl-C，退出。统计: %s" % node._stats)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
