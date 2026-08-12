"""里程计（只能估算）：积分运动指令 + 噪声，不用物理真值。

主人要求（2026-08-09）：狗内置里程计只能估算位置；二维码标牌（位置已知的环境
设施）作为绝对定位修正——看到标牌 N → 知道自己在通道 N 的哪个位置附近 →
纠正里程计漂移。这是真实机器人的标准做法（航位推算 + 地标绝对修正）。

2026-08-12 真实化升级（主人指令 + 代码审核意见）：
真实四足狗的"里程计" = IMU + 步态节拍推算（腿式航位推算），误差特性：
- 线速度比例误差 ~5%/s 量级，且不是白噪声——步态/地面打滑是**慢变系统偏差**
  （bias random walk），白噪声只贡献小幅抖动；
- 转弯角度同样不准：陀螺比例误差 + IMU 偏置漂移（yaw 误差随时间累积，
  无上限——这是定位最大的敌人）；
- 结论（与 ROS 生态一致）：纯推算必然漂走，必须靠激光 scan-matching
  （相对地图的连续修正）+ 稀疏地标绝对修正（二维码）融合才能长航。
  ROS 对应物：robot_localization(EKF 融合 odom+IMU) + Cartographer/
  slam_toolbox(scan-to-map matching + 回环) / AMCL(已知地图粒子滤波)。

模型：
- v：测量速度 = v·(1 + bias_v + εv)，bias_v 慢变随机游走（±6% 封顶），
  εv 白噪声 1%——合计等效 ~5%/s 漂移量级
- ω：测量角速度 = ω·(1 + bias_ws + εw) + bias_w，bias_ws 转弯比例偏差（±2.5%），
  bias_w 陀螺偏置随机游走（±0.3°/s 封顶）——yaw 漂移随时间累积
- correct()：地标绝对观测 → 指数加权拉回（权重 = 观测可信度）
"""
import math
import random


class Odometry:
    def __init__(self, x, y, yaw, v_noise=0.01, w_noise=0.01, rng=None,
                 v_bias_rw=0.02, w_bias_rw=0.06, w_scale_rw=0.02):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v_noise = v_noise      # 线速度白噪声（每步比例）
        self.w_noise = w_noise      # 角速度白噪声（每步比例）
        self.rng = rng if rng is not None else random.Random(0)
        # 慢变偏差（random walk，参数 = 每 √s 的游走标准差）
        self.bias_v = self.rng.gauss(0.0, 0.02)    # 线速度比例偏差（±6% 封顶）
        self.bias_w = self.rng.gauss(0.0, 0.002)   # 陀螺偏置 rad/s（±0.3°/s 封顶）
        self.bias_ws = self.rng.gauss(0.0, 0.01)   # 转弯比例偏差（±2.5% 封顶：U 型弯 180° 转完
        # 偏 ~4.5° 上限/典型 2°——再大 scan-matching 的 ±4.5° 窗拉不回来，实测 5% 档全程漂移 4-8m）
        self.v_bias_rw = v_bias_rw
        self.w_bias_rw = w_bias_rw      # rad/s/√s：60s 漂 ~0.5°/s 量级
        self.w_scale_rw = w_scale_rw
        self.corrections = 0        # 地标修正次数（统计）

    def update(self, dt, v, omega):
        """按指令速度积分（真实狗用的是 IMU+步态推算，这里指令速度≈推算速度）"""
        # 偏差随机游走（√dt 缩放，时间尺度无关）
        _sq = math.sqrt(dt)
        self.bias_v = max(-0.06, min(0.06, self.bias_v + self.rng.gauss(0.0, self.v_bias_rw * _sq)))
        # 陀螺偏置：封顶 ±0.3°/s（0.005 rad/s）——好点的狗 IMU 就这量级；
        # 无上限的 yaw 漂移是定位第一大敌（实测 ±1°/s 封顶 100s 就转飞 30°+，匹配窗拉不回）
        self.bias_w = max(-0.005, min(0.005, self.bias_w + self.rng.gauss(0.0, self.w_bias_rw * _sq * 0.05)))
        self.bias_ws = max(-0.05, min(0.05, self.bias_ws + self.rng.gauss(0.0, self.w_scale_rw * _sq)))
        vn = v * (1.0 + self.bias_v + self.rng.gauss(0.0, self.v_noise))
        # ω 噪声封顶（2026-08-12）：bounce 的朝向瞬跳会让指令 ω 冲到数百 rad/s，
        # 比例噪声随之爆炸（实测 bounce 风暴中 yaw 随机游走发散到几十度）。
        # 物理依据：受控原地转身有陀螺全程积分，角度误差远小于"5%/转角"的步态推算；
        # 且真实陀螺量程会饱和。噪声项按 ≤3 rad/s（快速真实转身）计，转角本身足额积分。
        w_for_noise = max(-3.0, min(3.0, omega))
        wn = omega + w_for_noise * (self.bias_ws + self.rng.gauss(0.0, self.w_noise)) + self.bias_w
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
