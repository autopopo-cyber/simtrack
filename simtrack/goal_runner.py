#!/usr/bin/env python3
"""goal_runner.py — 沿房间路径依次发 NavigateToPose，演示全程自主导航到终点。

路径来自迷宫生成时的 BFS 房间连通图，作为"楼层图先验"提供（类似真实部署有建筑平面图/
拓扑地图）。低层建图(slam_toolbox)+规划控制(Nav2)+漂移修正完全自主——这个节点只做
"按顺序报下一个房间中心"。

健壮性（三层失败处理，仿 explore_lite + 我们自己的补丁，见调研文档 §三/§五）：
  1. 事前：发航点前先确认该格在 SLAM 地图上已 free；未 free 走 A* 子目标推进/扇形兜底。
  2. 失败拉黑换路：同一子目标失败 2 次 → 拉黑该点 0.5m 半径（TTL 60s，explore_lite 无
     TTL 的缺陷我们补上）→ A* 重规划自动绕开给备选子目标。**子目标失败永不弃站**——
     旧版"4 次失败跳过整航点"是批量实验里 2% 房次吃 10% 时间的元凶。
  3. 进度超时兜底：Nav2 不报错但原地门舞（基线最惨 222s）→ 30s 无推进就取消在途目标。
     推进的度量是两级：直线距离快速路径 + A* 路线剩余长度二次确认——迷宫绕行时直线
     变大而路线变短，只用直线会把有效绕行误杀（v4.0 实测教训）。
     超时**条件拉黑**（v4.4：拉黑后 A* 验证仍有路才保留，封死唯一通路立即回滚——
     v4.1 seed6 封门雪球 / v4.3 脱离机动把狗扔出 15m 外，两个极端都实测否决）。
     进度基线属航点不属单目标（防扇形流浪每 6-16s 换目标重置计时器的检测盲区）。
     拉黑也用于 ABORTED 2 次的确定坏目标。同目标活跃期绝不重发。
  4. 航点预算兜底：单航点累计 240s 未达成 → 跳站（防拉黑/TTL 振荡活锁）。

用法（远程，需 ROS env，drift/slam/nav2 已起）：
  python3 -m simtrack.goal_runner
"""
import json
import math
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
import tf2_ros

# 兜底：rooms10x10 seed42 的 BFS 房间路径质心。正常应从 maze meta.json 读数据驱动航点表
# （maze_gen 生成时写入 waypoints=path_rooms 各房间中心）。
_FALLBACK_WAYPOINTS = [
    (2.5, 2.5), (2.5, 7.5), (7.5, 7.5), (12.5, 7.5), (12.5, 2.5),
    (17.5, 2.5), (22.5, 2.5), (27.5, 2.5), (32.5, 2.5), (37.5, 2.5),
    (42.5, 2.5), (47.5, 2.5), (47.5, 7.5), (47.5, 12.5), (47.5, 17.5),
    (47.5, 22.5), (47.5, 27.5), (47.5, 32.5), (47.5, 37.5), (47.5, 42.5),
    (47.5, 47.5),
]

# ── 失败处理参数（调研文档 §三/§五 + v4.0/v4.1 实测迭代）──
FAIL_BL_AFTER = 2        # 同一子目标 ABORTED 几次后拉黑（确定坏目标才拉黑）
BL_RADIUS_M = 0.5        # 拉黑半径
BL_KEY_M = 0.25          # 黑名单格粒度（比半径细，保证圆边采样）
BL_TTL_S = 60.0          # 拉黑存活期：到期自动解禁（explore_lite 永久拉黑的缺陷）
PROGRESS_TIMEOUT_S = 30.0  # 无推进超过此值 → 取消在途目标（门舞兜底）
PROGRESS_STEP_M = 0.5      # 距离至少缩短这么多才算"有推进"（噪声滤波）
WP_BUDGET_S = 240.0        # 单航点累计耗时上限（防活锁；基线最惨门舞房=222s）
FORCE_RELEASE_S = 920.0    # 在途目标超过此值强制复位 busy（bt_navigator 自身 900s 超时）


def _load_waypoints():
    """数据驱动航点：env MAZE → confirmed/maze_<name>.meta.json 的 waypoints。
    网格抖动后房间中心随 seed 变化，硬编码表只对 seed42 有效。"""
    name = os.environ.get("MAZE", "")
    proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meta_path = os.path.join(proj, "confirmed", "maze_%s.meta.json" % name)
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        wps = [(float(x), float(y)) for x, y in meta.get("waypoints", [])]
        if len(wps) >= 2:
            return wps, meta
    except Exception:
        pass
    return list(_FALLBACK_WAYPOINTS), None


WAYPOINTS, META = _load_waypoints()


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
        self._ff_done = False  # 是否已快进到离狗最近的航点（重启续跑用）
        # 失败处理状态
        self.fails = {}       # 目标位置键 → 连续失败次数（只统计未被拉黑处理的目标）
        self.blacklist = {}   # (kx,ky) 黑名单格 → 解禁时刻(s)
        self.gh = None        # 在途 goal handle（进度超时/预算超时要 cancel）
        self.kind = ""        # 在途目标类型 "wp"(航点直冲) / "step"(A*或扇形子目标)
        self.last_goal = None # 在途目标坐标 (x,y)
        self.prog_best = None  # 到当前航点历史最近直线距离（进度检测快速路径）
        self.prog_route = None  # A* 路线剩余长度（绕行时的正确进度度量）
        self.prog_t = 0.0      # 上次有推进的时刻
        self.sent_t = 0.0      # 当前目标发出时刻
        self.wp_t0 = None      # 当前航点开始时刻（预算兜底）
        self._timeout_cancel = False  # 本次取消是进度超时主动发起（回调里不再计数）
        self.create_timer(2.0, self._tick)
        g = WAYPOINTS[-1]
        src = "meta(MAZE=%s)" % os.environ.get("MAZE", "?") if META else "硬编码兜底"
        self.get_logger().info("goal_runner: %d 个航点（%s），终点 (%.1f,%.1f)"
                               % (len(WAYPOINTS), src, g[0], g[1]))

    # ── 基础工具 ──

    def _now(self):
        return self.get_clock().now().nanoseconds / 1e9

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

    # ── 黑名单（0.25m 格采样 0.5m 半径圆，TTL 过期自动解禁）──

    def _bl_add(self, x, y, reason, cond_robot=None, cond_wp=None):
        """拉黑 (x,y) 周边 BL_RADIUS_M 圆内：所有中心落在圆内的 BL_KEY_M 格。
        条件拉黑（cond_robot/cond_wp 给出时）：拉黑后验证 A* 仍有路到航点，无路立即回滚
        ——v4.1 seed6 教训：树状迷宫的门没有替代路，封门=A*无路=雪球跳站。"""
        now = self._now()
        exp = now + BL_TTL_S
        k0x, k1x = int((x - BL_RADIUS_M) / BL_KEY_M), int((x + BL_RADIUS_M) / BL_KEY_M)
        k0y, k1y = int((y - BL_RADIUS_M) / BL_KEY_M), int((y + BL_RADIUS_M) / BL_KEY_M)
        added = {}
        for kx in range(k0x, k1x + 1):
            for ky in range(k0y, k1y + 1):
                cx, cy = (kx + 0.5) * BL_KEY_M, (ky + 0.5) * BL_KEY_M
                if math.hypot(cx - x, cy - y) <= BL_RADIUS_M:
                    k = (kx, ky)
                    if self.blacklist.get(k, 0.0) <= now:   # 只记录真新增/真过期刷新
                        added[k] = self.blacklist.get(k)
                    self.blacklist[k] = exp
        if cond_robot is not None and self._route_len(cond_robot, cond_wp[0], cond_wp[1]) is None:
            for k, old in added.items():
                if old is None:
                    self.blacklist.pop(k, None)
                else:
                    self.blacklist[k] = old
            self.get_logger().warn("  ⛔ 拉黑 (%.1f,%.1f) 会封死唯一通路，回滚（原地重试）"
                                   % (x, y))
            return False
        self.get_logger().warn("  ⛔ 拉黑 (%.1f,%.1f) r=%.1fm 60s（%s）→ 重规划绕行"
                               % (x, y, BL_RADIUS_M, reason))
        return True

    def _bl_blocked(self, x, y):
        exp = self.blacklist.get((int(x / BL_KEY_M), int(y / BL_KEY_M)), 0.0)
        return exp > self._now()

    def _bl_prune(self):
        t = self._now()
        for k in [k for k, exp in self.blacklist.items() if exp <= t]:
            del self.blacklist[k]

    # ── 子目标计算 ──

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
        已知 free 步进点（黑名单内的不算——否则超时拉黑后扇形又把同一个点递回来）。
        效果=贴墙滑行，直到门口出现在步进方向上。
        没有 it 会死锁：slam 关键帧式建图下静止狗地图不长大，"等地图"永远等不到。"""
        base = math.atan2(wy - robot[1], wx - robot[0])
        for off in (15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90):
            ang = base + math.radians(off)
            probe = (robot[0] + math.cos(ang) * 3.0,
                     robot[1] + math.sin(ang) * 3.0)
            tgt = self._step_target(robot, probe[0], probe[1], min_step=0.8)
            if tgt is not None and not self._bl_blocked(tgt[0], tgt[1]):
                return tgt
        return None

    def _route_step(self, robot, wx, wy, pad=3.0, max_expand=40000):
        """A* 规划 dog→waypoint：free=1、unknown=8（可穿但贵）、墙=∞、黑名单格=墙。
        门已知 → 路径自然穿门（free 便宜）；门未知 → 路径乐观穿 unknown——包括**地图数组
        边界外**（数组外=从未观测=unknown，不是错误；航点常在地图外）。
        子目标 = 路径上"从狗出发连续 free 段"的最后一格（=前沿）：规划器对它必成功，
        狗推进后地图长大、下个 tick 重规划。比"closest-to-waypoint"目标强：后者在墙前
        有鞍点（绕行两侧到航点直线距离几乎相等），狗会钉在离航点最近的墙点上来回蹭。
        返回 (子目标, 路线全长m)；找不到路返回 (None, None)。"""
        path = self._astar(robot, wx, wy, pad, max_expand)
        if path is None:
            return None, None
        res = self.latest_map.info.resolution
        robot_ = robot
        # 狗贴墙时自身格可能读作占据（A* 不穿墙，脏格最多只有起点这一格）→ 跳过
        start_i = 0 if self.latest_map.data[path[0][1] * self.latest_map.info.width
                                             + path[0][0]] == 0 else 1
        last_free = start_i
        for i in range(start_i, len(path)):
            c, r = path[i]
            if self.latest_map.data[r * self.latest_map.info.width + c] == 0:
                last_free = i
            else:
                break
        c, r = path[last_free]
        ox, oy = self.latest_map.info.origin.position.x, self.latest_map.info.origin.position.y
        px, py = ox + (c + 0.5) * res, oy + (r + 0.5) * res
        if math.hypot(px - robot_[0], py - robot_[1]) < 0.8:
            # 连续 free 段太短=狗已在前沿边缘：允许子目标越入 unknown 1.5m
            # （近距 unknown 目标规划无压力；卡死的是远距全程 unknown）
            if last_free + 1 < len(path):
                nc, nr = path[last_free + 1]
                W, H = self.latest_map.info.width, self.latest_map.info.height
                v = self.latest_map.data[nr * W + nc] if 0 <= nc < W and 0 <= nr < H else -1
                if v < 0:
                    qx = px + 0.75 * (nc - c)
                    qy = py + 0.75 * (nr - r)
                    if math.hypot(qx - robot_[0], qy - robot_[1]) >= 0.8:
                        return (qx, qy), len(path) * res
            return None, len(path) * res   # 前方是墙不是 unknown → 交给扇形兜底
        return (px, py), len(path) * res

    def _route_len(self, robot, wx, wy):
        """A* 路线剩余长度（m）。进度超时的正确度量：迷宫里绕行时直线距离会变大，
        路线长度单调递减——用直线会把有效绕行误判成卡死。"""
        path = self._astar(robot, wx, wy)
        if path is None:
            return None
        return len(path) * self.latest_map.info.resolution

    def _astar(self, robot, wx, wy, pad=3.0, max_expand=40000, allow_nudge=True):
        """A* 主体的独立函数：返回 (start→goal) 网格路径或 None。"""
        if self.latest_map is None:
            return None
        import heapq
        info = self.latest_map.info
        res, W, H = info.resolution, info.width, info.height
        ox, oy = info.origin.position.x, info.origin.position.y
        data = self.latest_map.data
        t = self._now()
        blocked = set(k for k, exp in self.blacklist.items() if exp > t)
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
                w = 1.0
            elif v < 0:
                w = 8.0                       # unknown：可穿但贵
            else:
                return None                   # 墙
            if blocked and (int((ox + (c + 0.5) * res) / BL_KEY_M),
                            int((oy + (r + 0.5) * res) / BL_KEY_M)) in blocked:
                return None                   # 拉黑区：当墙，逼 A* 绕行
            return w

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
            # 目标格被（变形）地图封进墙里 → 找最近 free 格代打（房间中心离墙≥1.5m，
            # 挪 1-2m 就能出墙）。v4.2 seed1 教训：不挪就 A* 永久 None → 扇形流浪 →
            # 240s 预算烧完跳站，连续两个航点就这么废掉。
            # allow_nudge=False 时禁止再递归：代打格也找不到=目标被墙围死，
            # 不禁止会无限递归 RecursionError 炸掉整个节点（v4.4 seed2 实测暴毙）。
            if not allow_nudge:
                return None
            nud = self._nearest_free_cell(gc, gr)
            if nud is None:
                return None
            nc_, nr_ = nud
            self.get_logger().info(
                "  航点格在图上为墙(地图变形)，A* 改打最近 free 格 (%.1f,%.1f)"
                % (ox + (nc_ + 0.5) * res, oy + (nr_ + 0.5) * res),
                throttle_duration_sec=30.0)
            return self._astar(robot, ox + (nc_ + 0.5) * res,
                               oy + (nr_ + 0.5) * res, pad, max_expand, allow_nudge=False)
        # 回溯路径（goal→start）
        path = [(gc, gr)]
        while path[-1] != (sc, sr):
            path.append(came[path[-1]])
        path.reverse()
        return path

    def _nearest_free_cell(self, gc, gr, max_ring=20):
        """绕 (gc,gr) 一圈圈扩，返回最近的 free 格（找不到返回 None）。"""
        if self.latest_map is None:
            return None
        W, H = self.latest_map.info.width, self.latest_map.info.height
        data = self.latest_map.data

        def ok(c, r):
            return 0 <= c < W and 0 <= r < H and data[r * W + c] == 0

        for ring in range(1, max_ring):
            for c in range(gc - ring, gc + ring + 1):
                if ok(c, gr - ring) or ok(c, gr + ring):
                    return (c, gr - ring) if ok(c, gr - ring) else (c, gr + ring)
            for r in range(gr - ring + 1, gr + ring):
                if ok(gc - ring, r) or ok(gc + ring, r):
                    return (gc - ring, r) if ok(gc - ring, r) else (gc + ring, r)
        return None

    # ── 主循环 ──

    def _tick(self):
        if self.idx >= len(WAYPOINTS):
            return
        self._bl_prune()
        if self.busy:
            self._watch_progress()
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
            self.wp_t0 = self._now()
            self.get_logger().info("续跑：从最近航点 #%d (%.1f,%.1f) 开始（狗在 %.1f,%.1f）"
                                   % (self.idx, WAYPOINTS[self.idx][0],
                                      WAYPOINTS[self.idx][1], robot[0], robot[1]))
        if self.wp_t0 is None:
            self.wp_t0 = self._now()
        # 航点预算兜底：超时直接跳站（防拉黑/TTL 振荡活锁——A* 和扇形都返回 None 时
        # 狗会原地"等地图"直到 TTL 解禁，240s 上限保证最惨也就损失一个房间）
        if self._now() - self.wp_t0 > WP_BUDGET_S:
            self.get_logger().warn("  ⏰ 航点 #%d (%.1f,%.1f) 超 %.0fs 预算，跳过该航点"
                                   % (self.idx, WAYPOINTS[self.idx][0],
                                      WAYPOINTS[self.idx][1], WP_BUDGET_S))
            self.idx += 1
            self.wp_t0 = self._now()
            self.prog_best = None
            self.prog_route = None
            return
        wx, wy = WAYPOINTS[self.idx]
        # 到达当前航点（1.5m 内）→ 下一站
        if math.hypot(wx - robot[0], wy - robot[1]) < 1.5:
            self.get_logger().info("  ✅ 航点 #%d/%d (%.1f,%.1f) 达成"
                                   % (self.idx, len(WAYPOINTS) - 1, wx, wy))
            self.idx += 1
            self.wp_t0 = self._now()
            self.prog_best = None
            self.prog_route = None
            if self.idx >= len(WAYPOINTS):
                self.get_logger().info("🏁🏁 到达终点！(全程完成)")
            return
        # 航点已探明 free 且未被拉黑 → 直接发 Nav2 让它自己绕门规划（解决狗偏离门轴时
        # 直线撞墙）。拉黑中的航点强制走 A* 换个进近方向，别再从老路直冲
        if self._free_in_map(wx, wy) and not self._bl_blocked(wx, wy):
            d = math.hypot(wx - robot[0], wy - robot[1])
            self.get_logger().info("→ 航点 #%d/%d (%.1f,%.1f) 直冲 距离=%.1fm"
                                   % (self.idx, len(WAYPOINTS) - 1, wx, wy, d))
            self._send(wx, wy, "wp")
            return
        # 航点未探明（冷启动/探索在前）→ 两级推进：A* 找门/前沿 > 扇形兜底
        tgt, rlen = self._route_step(robot, wx, wy)
        how = "route" if tgt is not None else None
        if rlen is not None:
            self.prog_route = rlen
        if tgt is None:
            tgt = self._fan_step(robot, wx, wy)
            how = "fan" if tgt is not None else None
        if tgt is None:
            self.get_logger().info("航点 #%d (%.1f,%.1f)：前方待探明，等地图…（黑名单 %d 格）"
                                   % (self.idx, wx, wy, len(self.blacklist)),
                                   throttle_duration_sec=5.0)
            return
        d = math.hypot(tgt[0] - robot[0], tgt[1] - robot[1])
        self.get_logger().info("→ 航点 #%d/%d (%.1f,%.1f) %s (%.1f,%.1f) +%.1fm"
                               % (self.idx, len(WAYPOINTS) - 1, wx, wy,
                                  {"route": "A*推进至", "fan": "扇形绕行至"}[how],
                                  tgt[0], tgt[1], d))
        self._send(tgt[0], tgt[1], "step")

    def _watch_progress(self):
        """在途目标监视（explore_lite 式进度超时，v4.1 修正度量）：
        快路径=直线距离；直线 30s 不缩 → 查 A* 路线剩余长度（迷宫绕行时直线变大、
        路线变短，用直线会把有效绕行误判卡死——v4.0 在 seed3 实测连打 6 次误杀）。
        两者都无推进才判死：取消在途目标 + 拉黑该子目标。另含 lost-callback 强制复位。"""
        now = self._now()
        if now - self.sent_t > FORCE_RELEASE_S:
            self.get_logger().warn("在途目标 %.0fs 无回调，强制复位 busy" % (now - self.sent_t))
            self.busy = False
            self.gh = None
            return
        robot = self._robot_pose()
        if robot is None:
            return
        wx, wy = WAYPOINTS[min(self.idx, len(WAYPOINTS) - 1)]
        d = math.hypot(wx - robot[0], wy - robot[1])
        if self.prog_best is None or d < self.prog_best - PROGRESS_STEP_M:
            self.prog_best = d
            self.prog_t = now
            return
        if now - self.prog_t <= PROGRESS_TIMEOUT_S:
            return
        # 直线停滞：用路线长度二次确认（绕行中路线仍在缩 = 活着）
        rl = self._route_len(robot, wx, wy)
        if rl is not None:
            if self.prog_route is None or rl < self.prog_route - PROGRESS_STEP_M:
                self.prog_route = rl
                self.prog_t = now
                self.get_logger().info("  ↩ 绕行中（直线 %.1fm 停滞，路线余 %.1fm 在缩）"
                                       % (d, rl), throttle_duration_sec=20.0)
                return
            self.prog_route = rl
        self.get_logger().warn("⏱ 航点 #%d 进度超时 %.0fs（直线 %.1fm 路线 %s 均无推进）→ 取消在途%s"
                               % (self.idx, now - self.prog_t, d,
                                  "%.1fm" % self.prog_route if self.prog_route else "?",
                                  "并条件拉黑子目标" if self.kind == "step" else ""))
        self.prog_t = now          # 重置基准：一个停滞窗口只报一次（v4.0 每 2s 重复报）
        self.prog_route = None
        self._timeout_cancel = True
        if self.kind == "step" and self.last_goal is not None:
            self._bl_add(self.last_goal[0], self.last_goal[1], "进度超时",
                         cond_robot=robot, cond_wp=(wx, wy))
        if self.gh is not None:
            self.gh.cancel_goal_async()

    # ── 目标收发与失败处理 ──

    def _send(self, wx, wy, kind):
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(wx)
        goal.pose.pose.position.y = float(wy)
        goal.pose.pose.orientation.w = 1.0
        self.busy = True
        self.kind = kind
        self.last_goal = (wx, wy)
        self.sent_t = self._now()
        # 注意：prog_best/prog_route/prog_t **不重置**——进度基线属于航点而非单个目标。
        # v4.2 seed1 教训：扇形流浪每 6-16s 换目标，若每次发目标都重置 30s 计时器，
        # 换目标churn就永远凑不满超时窗口，卡滞检测全盲。基线只在航点推进/真实推进时重置。
        fut = self.nav.send_goal_async(goal)
        fut.add_done_callback(lambda f: self._resp(f, wx, wy))

    def _resp(self, future, wx, wy):
        gh = future.result()
        if gh is None or not gh.accepted:
            self.get_logger().warn("目标 (%.1f,%.1f) 被拒" % (wx, wy))
            self._advance_or_retry(wx, wy, ok=False, status=-1, ec=0)
            return
        self.gh = gh
        res = gh.get_result_async()
        res.add_done_callback(lambda f: self._result(f, wx, wy))

    def _result(self, future, wx, wy):
        wrap = future.result()
        ok = wrap is not None and wrap.status == GoalStatus.STATUS_SUCCEEDED
        res = getattr(wrap, "result", None)
        ec = getattr(res, "error_code", 0) or 0
        self._advance_or_retry(wx, wy, ok,
                               wrap.status if wrap is not None else -1, ec)

    def _advance_or_retry(self, wx, wy, ok, status=-1, ec=0):
        """失败分流：进度超时取消的（已拉黑）只复位；子目标失败 2 次拉黑换路；
        航点直冲失败 4 次跳站（老逻辑保留，直冲失败=该房间确实过不去）。"""
        self.busy = False
        self.gh = None
        if ok:
            return
        if self._timeout_cancel:
            # 我们主动取消的：拉黑已在 _watch_progress 里做过了，这里只复位
            self._timeout_cancel = False
            self.prog_best = None
            self.prog_route = None
            self.prog_t = self._now()
            return
        key = (round(wx, 1), round(wy, 1))
        n = self.fails.get(key, 0) + 1
        self.fails[key] = n
        if self.kind == "step":
            if n >= FAIL_BL_AFTER:
                self._bl_add(wx, wy, "失败%d次" % n)
                self.fails[key] = 0
            else:
                self.get_logger().warn("  步进 (%.1f,%.1f) 失败(status=%d ec=%d)，重试"
                                       % (wx, wy, status, ec))
        else:
            if n >= 4:
                self.get_logger().warn("  航点 (%.1f,%.1f) 直冲失败 %d 次(status=%d ec=%d)，跳过该航点"
                                       % (wx, wy, n, status, ec))
                self.fails[key] = 0
                self.idx += 1
                self.wp_t0 = self._now()
                self.prog_best = None
                self.prog_route = None
            else:
                self.get_logger().warn("  航点直冲 (%.1f,%.1f) 失败(status=%d ec=%d)，重试 %d"
                                       % (wx, wy, status, ec, n))


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
