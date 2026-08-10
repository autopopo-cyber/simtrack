"""随机反弹障碍 — 小球弹性反弹模型

主人需求（2026-08-07）：
- 每段 20m 长 × 5m 宽（通道真实宽度），x∈[15,35]，两端虚拟墙内收 10cm
- 障碍 1m/s，每满 1s 20% 概率随机改变方向（uniform 0~2π）
- 撞真实墙（y 边界）/虚拟墙（x 边界）→ 镜面反射（分量取反）
- 机器狗无视虚拟墙（虚拟墙只约束障碍中心）
"""

import math
import random


class RandomObstacle:
    """单个随机反弹障碍。pos/dir 是可变状态，ch 标识所在通道。"""

    def __init__(self, ch, x0=15.0, x1=35.0, y_lo=None, y_hi=None,
                 speed=1.0, radius=0.5, rng=None):
        self.ch = ch
        self.x0, self.x1 = x0, x1
        self.y_lo = y_lo if y_lo is not None else ch * 5.0
        self.y_hi = y_hi if y_hi is not None else (ch + 1) * 5.0
        self.speed = speed
        self.radius = radius
        # 虚拟墙：名义 20m 段内收 10cm
        self.wall_x0 = x0 + 0.1
        self.wall_x1 = x1 - 0.1
        self.rng = rng if rng is not None else random
        # 初始：段内随机（2026-08-10 主人观察：旧版全在 x=25 段中央，一列齐排太假）——种子可复现
        self.pos = [self.rng.uniform(self.wall_x0 + 0.5, self.wall_x1 - 0.5),
                    self.rng.uniform(self.y_lo + 1.0, self.y_hi - 1.0)]
        self.dir = self.rng.uniform(0.0, 2 * math.pi)
        self.change_timer = 1.0

    def update(self, dt):
        """步进 dt 秒：变向判定 → 移动 → 真实墙/虚拟墙反弹"""
        # ① 变向：每满 1s，20% 概率随机新方向
        self.change_timer -= dt
        if self.change_timer <= 0.0:
            self.change_timer = 1.0
            if self.rng.random() < 0.2:
                self.dir = self.rng.uniform(0.0, 2 * math.pi)
        # ② 移动
        self.pos[0] += math.cos(self.dir) * self.speed * dt
        self.pos[1] += math.sin(self.dir) * self.speed * dt
        # ③ 真实墙反弹（y 边界，半径约束）
        if self.pos[1] < self.y_lo + self.radius:
            self.pos[1] = self.y_lo + self.radius
            self.dir = self._reflect(self.dir, axis="y")
        if self.pos[1] > self.y_hi - self.radius:
            self.pos[1] = self.y_hi - self.radius
            self.dir = self._reflect(self.dir, axis="y")
        # ④ 虚拟墙反弹（x 边界，只约束中心）
        if self.pos[0] < self.wall_x0:
            self.pos[0] = self.wall_x0
            self.dir = self._reflect(self.dir, axis="x")
        if self.pos[0] > self.wall_x1:
            self.pos[0] = self.wall_x1
            self.dir = self._reflect(self.dir, axis="x")

    @staticmethod
    def _reflect(dir_angle, axis):
        """镜面反射：撞 x 边界 → vx 取反（cos 取反）；撞 y 边界 → vy 取反（sin 取反）"""
        vx, vy = math.cos(dir_angle), math.sin(dir_angle)
        if axis == "x":
            vx = -vx
        else:
            vy = -vy
        return math.atan2(vy, vx)


class RandomObstacleField:
    """管理多个反弹区段（每段 1 个随机障碍）。

    x0/x1：每段的 x 向活动范围（虚拟墙）。2026-08-09 混合场：直道障碍 x∈[4.5,45]
    （不超出本通道直道段，不进转弯开口）。"""

    def __init__(self, channels=(1, 4, 6, 8), seed=None, x0=15.0, x1=35.0):
        self._rng = random.Random(seed)
        self.obstacles = [RandomObstacle(ch, x0=x0, x1=x1, rng=self._rng) for ch in channels]

    def update(self, dt):
        for obs in self.obstacles:
            obs.update(dt)

    @property
    def positions(self):
        """[(x, y), ...] 供 obs_world 同步（兼容现有 blocked()）"""
        return [tuple(obs.pos) for obs in self.obstacles]

    @property
    def velocities(self):
        """[(ox, oy, vx, vy, r), ...] 供 DWA 障碍运动预测"""
        return [(obs.pos[0], obs.pos[1],
                 math.cos(obs.dir) * obs.speed, math.sin(obs.dir) * obs.speed,
                 obs.radius) for obs in self.obstacles]


def mix_bend_positions(seed):
    """混合场弯道固定障碍位置（2026-08-10 主人指令：弯道内随机分布，不再刚性一列贴 x=48.5/1.5）。

    每弯 1 个：右弯（奇数分隔墙，开口 x∈[45,50]）x∈[47.6,48.6]；
    左弯（偶数分隔墙，开口 x∈[0,5]）x∈[1.4,2.4]；y=5k±1.5（弯道口袋内）。
    范围核算（r=0.5 不嵌墙、转弯走线保持 ≥1.9m）：
    - 随机范围必须**贴外墙半侧**：障碍摆到口袋中间（如 x=46.9）会把转弯区分成两条
      ~1.3-2.0m 的紧缝，执行层穿缝 bounce 风暴（实测 404 bounce/301s）——
      障碍的几何意义是"逼狗走外侧弯"，不是"把弯道切成迷宫"。
    - 贴外墙极限 48.6+0.5=49.1 < 50（留 0.9m 缝 → 净空场封闭，狗不走）
    - 最靠内 47.6：外墙侧走线 1.9m / 分隔墙端侧 2.1m，两侧都留得出舒适通道
    - y 极限 5k±2.0（含半径）远在上下隔壁墙（5k±5）内
    种子可复现（seed7 = 回归/录像用）。"""
    rng = random.Random(seed)
    bends = []
    for k in range(1, 10):
        if k % 2 == 1:
            x = rng.uniform(47.6, 48.6)   # 右 U 型弯，贴外侧墙一带
        else:
            x = rng.uniform(1.4, 2.4)     # 左 U 型弯，贴外侧墙一带
        y = k * 5.0 + rng.uniform(-1.5, 1.5)
        bends.append((x, y))
    return bends
