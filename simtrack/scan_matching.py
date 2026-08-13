"""Scan-to-map 相关匹配（激光里程计修正）——ROS Cartographer 思想的简化版。

为什么需要（2026-08-12，主人指令 + 代码审核）：
真实四足狗的里程计（IMU+步态推算）误差 ~5%/s 且含慢变偏差与陀螺漂移，
纯推算几十米就漂出走廊宽度；ROS 生态的标准解法是
robot_localization(EKF) + Cartographer/slam_toolbox 的 **scan-to-map matching**：
把当前激光帧的命中点与已建地图做相关匹配，求出位姿修正量，连续压住漂移。
（长直走廊沿墙方向无特征 → 该方向匹配退化不可观测，这是已知物理特性；
 我们的对策 = 二维码标牌绝对修正 + 直道特征障碍，见 README。）

实现：暴力相关搜索（Ceres 实时匹配的粗网格版）：
- 输入：当前帧近距墙命中点（估计系）+ 膨胀墙掩码（命中点允许 ±1 格量化误差）
- 搜索 (dx, dy, δyaw) 粗网格 → 细网格，得分 = 命中点落墙比例
- 零偏移优先（得分差距 <2% 不修正）——长直道退化方向上匹配得分平坦，
  零偏移优先保证狗不被噪声拖来拖去；只有证据明确才修正
- 修正限幅 + 增益 <1（指数收敛，防跳变振荡）
"""
import math
import numpy as np


class ScanMatcher:
    def __init__(self, voxel=0.1,
                 dx_max=0.3, dx_step=0.1, dx_refine=0.05,
                 dyaw_max_deg=3.0, dyaw_step_deg=3.0,
                 gain=0.8, min_pts=40, min_score=0.40,
                 max_dx=0.25, max_dyaw_deg=2.0):
        self.voxel = voxel
        self.dx_max = dx_max
        self.dx_step = dx_step
        self.dx_refine = dx_refine
        self.dyaw_max = math.radians(dyaw_max_deg)
        self.dyaw_step = math.radians(dyaw_step_deg)
        self.gain = gain
        self.min_pts = min_pts      # 近距命中点少于此数 = 证据不足（新区域/开阔地）
        self.min_score = min_score  # 最佳得分低于此 = 匹配不上（地图没内容），不修正
        self.max_dx = max_dx        # 单次平移修正限幅 (m)
        self.max_dyaw = math.radians(max_dyaw_deg)  # 单次角度修正限幅 (rad)
        self.matches = 0            # 尝试次数
        self.corrections = 0        # 实际修正次数
        self.last_identity = 1.0    # 最近一次零偏移对齐分（漂移检测用，1=完美对齐）
        self.last_best = 1.0        # 最近一次最优偏移对齐分

    def match(self, pts, ox, oy, wall_dil):
        """求位姿修正量。

        Args:
            pts: (N,2) float 当前帧墙命中点（估计系世界坐标，用未修正的估计位姿算的）
            ox, oy: 估计狗位（命中点围绕此原点旋转）
            wall_dil: (GRID_N, GRID_N) uint8 膨胀后的感知墙掩码（1=墙或墙邻格）
        Returns:
            (dx, dy, dyaw, score) 修正量（世界系）或 None（证据不足/零偏移已最优）
        """
        self.matches += 1
        n = len(pts)
        if n < self.min_pts:
            return None
        rx = pts[:, 0] - ox
        ry = pts[:, 1] - oy
        W = wall_dil.shape[0]
        min_pts = self.min_pts

        def _score(dx, dy, da):
            if da != 0.0:
                c, s = math.cos(da), math.sin(da)
                tx = rx * c - ry * s + (ox + dx)
                ty = rx * s + ry * c + (oy + dy)
            else:
                tx = rx + (ox + dx)
                ty = ry + (oy + dy)
            gx = (tx / self.voxel).astype(np.int32)
            gy = (ty / self.voxel).astype(np.int32)
            ok = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < W)
            if ok.sum() < min_pts:
                return 0.0
            return float(wall_dil[gx[ok], gy[ok]].mean())

        # ── 粗搜索：δyaw × (dx, dy) 网格 ──
        da_list = np.arange(-self.dyaw_max, self.dyaw_max + 1e-9, self.dyaw_step)
        d_list = np.arange(-self.dx_max, self.dx_max + 1e-9, self.dx_step)
        best = (0.0, 0.0, 0.0)
        best_score = _score(0.0, 0.0, 0.0)
        zero_score = best_score
        for da in da_list:
            for dx in d_list:
                for dy in d_list:
                    if da == 0.0 and dx == 0.0 and dy == 0.0:
                        continue
                    s = _score(dx, dy, da)
                    if s > best_score:
                        best_score = s
                        best = (dx, dy, da)
        # ── 细搜索：最佳平移附近 ±step 细化（角度保持）──
        if best != (0.0, 0.0, 0.0):
            bx, by, ba = best
            for dx in np.arange(bx - self.dx_step, bx + self.dx_step + 1e-9, self.dx_refine):
                for dy in np.arange(by - self.dx_step, by + self.dx_step + 1e-9, self.dx_refine):
                    s = _score(dx, dy, ba)
                    if s > best_score:
                        best_score = s
                        best = (dx, dy, ba)
        # ── 采纳门控：得分可信 且 显著优于零偏移（1%）──
        # （零偏移优先 = 长直道退化方向防噪声拖动；细网格 0.1m 让真实小误差可被表达——
        #   粗 0.2m 网格会把 <0.1m 的漂移挡在采纳门外，实测修正率 4.5% 压不住漂）
        # last_identity = 零偏移对齐分（当前激光帧在估计位姿处贴地图墙的程度）。
        # 持续低 = 漂移已超匹配窗（P0-4 漂移失控检测信号）——每次 match 都更新，供调用方 EMA 跟踪。
        self.last_identity = zero_score
        self.last_best = best_score
        if best_score < self.min_score or best_score - zero_score < 0.01:
            return None
        dx, dy, da = best
        dx = max(-self.max_dx, min(self.max_dx, dx)) * self.gain
        dy = max(-self.max_dx, min(self.max_dx, dy)) * self.gain
        da = max(-self.max_dyaw, min(self.max_dyaw, da)) * self.gain
        self.corrections += 1
        return (dx, dy, da, best_score)
