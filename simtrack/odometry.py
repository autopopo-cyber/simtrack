"""里程计（只能估算）：积分运动指令 + 噪声，不用物理真值。

主人要求（2026-08-09）：狗内置里程计只能估算位置；二维码标牌（位置已知的环境
设施）作为绝对定位修正——看到标牌 N → 知道自己在通道 N 的哪个位置附近 →
纠正里程计漂移。这是真实机器人的标准做法（航位推算 + 地标绝对修正）。

模型：
- 每步积分：x += v·cos(yaw)·dt·(1+εv)，yaw += ω·dt·(1+εω)
- εv：线速度比例噪声（轮滑/地面不平），εω：角速度比例噪声
- 另加微小零均值随机游走（模拟抖动）
- correct()：地标绝对观测 → 指数加权拉回（权重 = 观测可信度）
"""
import math
import random


class Odometry:
    def __init__(self, x, y, yaw, v_noise=0.02, w_noise=0.01, rng=None):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v_noise = v_noise      # 线速度比例噪声（2% = 每走 50m 漂 1m 量级）
        self.w_noise = w_noise      # 角速度比例噪声
        self.rng = rng if rng is not None else random.Random(0)
        self.corrections = 0        # 地标修正次数（统计）

    def update(self, dt, v, omega):
        """按指令速度积分（真实狗用的是轮速计/IMU，这里指令速度≈轮速）"""
        vn = v * (1.0 + self.rng.gauss(0.0, self.v_noise))
        wn = omega * (1.0 + self.rng.gauss(0.0, self.w_noise))
        self.yaw += wn * dt
        self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi
        self.x += vn * math.cos(self.yaw) * dt
        self.y += vn * math.sin(self.yaw) * dt

    def correct(self, wx, wy, weight=0.7):
        """地标绝对修正：观测到狗在 (wx,wy) → 指数加权拉回，yaw 不动（yaw 噪声小）"""
        self.x += (wx - self.x) * weight
        self.y += (wy - self.y) * weight
        self.corrections += 1

    def pose(self):
        return self.x, self.y, self.yaw

    def error(self, tx, ty):
        """相对真值的误差（仅评分/分析用，决策不可用）"""
        return math.hypot(self.x - tx, self.y - ty)
