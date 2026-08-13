#!/usr/bin/env python3
"""
sim_server.py — MuJoCo 仿真后端，为 ROS2 sim_bridge 提供干净的仿真 API。

架构：
  MuJoCo 场景（狗体 + 迷宫可视化）管理位姿；解析式 heightfield 射线产生激光。
  狗是运动学滑动体（slide x/y + hinge yaw, contype=0 无物理碰撞），
  速度直接写 qvel，yaw 直接写 qpos——跟 algo3_headless 一致。

坐标系（maze_gen.py 约定）：
  世界 (0,0) = 迷宫左下角，x→右 y→上
  雷达机身系：角度 0 = 前方(+x)，CCW 正，linspace(-fov/2, +fov/2, n_rays)

用法：
  python -m simtrack.sim_server                  # 自测：viewer 开车，打印 scan
  from simtrack.sim_server import SimBackend     # 被 sim_bridge 导入
"""
import math
import os
import time

import numpy as np
from PIL import Image

# MuJoCo 可选（自测模式才需要；sim_bridge 在 ROS 环境里也 import）
try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


class SimBackend:
    """MuJoCo 仿真后端——sim_bridge 的地基。

    所有方法都是同步的、无副作用的（除 step/set_cmd_vel 改变内部状态）。
    sim_bridge 节点在 ROS spin 循环里调 step() + get_scan() + get_true_pose()。
    """

    def __init__(self, maze_path=None, start=(1.5, 1.5, 0.0),
                 lidar_rays=360, lidar_fov_deg=360, lidar_range=15.0,
                 timestep=0.005, use_mujoco_viewer=False, px_per_m=50):
        """
        Args:
            maze_path: 高度图 PNG 路径（默认 confirmed/maze_loop20.png）
            start: (x, y, yaw) 起点位姿
            lidar_rays: 射线数（360 = 1°间隔）
            lidar_fov_deg: 视场角（360=全向，180=前半圆）
            lidar_range: 最大探测距离 (m)
            timestep: 物理步长 (s)
            use_mujoco_viewer: 自测用，开 MuJoCo 窗口
            px_per_m: 高度图分辨率（像素/米），maze_gen 统一 50。不再从图宽硬算，
                      否则非 20m 宽的迷宫（如 rooms5x5=15m）会算错射线尺度。
        """
        # ── 路径 ──
        proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if maze_path is None:
            maze_path = os.path.join(proj, "confirmed", "maze_loop20.png")
        self.maze_path = maze_path

        # ── 加载高度图（射线用）──
        hf = np.array(Image.open(maze_path))
        self._hf_bin = hf != 128           # True = 墙
        self.hf_h, self.hf_w = hf.shape    # 图像尺寸（像素）
        self.px_per_m = px_per_m           # 50 px/m（maze_gen 统一），不再 hf_w//20
        self.scan_step = 1.0 / self.px_per_m  # 0.02m = 1 像素

        # ── 雷达参数 ──
        self.lidar_rays = lidar_rays
        self.lidar_fov = math.radians(lidar_fov_deg)
        self.lidar_range = lidar_range
        self._scan_k = np.arange(1, int(lidar_range / self.scan_step) + 1,
                                 dtype=np.float32) * self.scan_step

        # ── 位姿状态 ──
        self.x, self.y, self.yaw = start
        self.vx, self.vy, self.vyaw = 0.0, 0.0, 0.0
        self.timestep = timestep

        # ── MuJoCo 场景 ──
        self.m = self.d = None
        self.viewer = None
        if HAS_MUJOCO:
            self._init_mujoco(use_mujoco_viewer)

    # ──────────────────────────────────────────────
    # MuJoCo 场景
    # ──────────────────────────────────────────────
    def _build_xml(self):
        """构建 MuJoCo 场景 XML（狗体 + 迷宫 hfield 可视化）。"""
        maze_abs = os.path.abspath(self.maze_path)
        sx, sy, syaw = self.x, self.y, self.yaw
        # hfield half-extent = 迷宫半宽（10m for 20m maze）
        hw = self.hf_w / self.px_per_m / 2.0  # 10.0
        return f"""<mujoco>
  <compiler angle="radian"/>
  <option timestep="{self.timestep}"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <asset>
    <hfield name="maze" size="{hw} {hw} 4.0 2.0" file="{maze_abs}"/>
  </asset>
  <worldbody>
    <light pos="{hw} {hw} 30" dir="0 0 -1" diffuse="0.9 0.9 0.95" ambient="0.4 0.4 0.45"/>
    <geom type="hfield" hfield="maze" pos="{hw} {hw} 0" rgba="0.55 0.6 0.65 1"
          friction="0 0 0" contype="0" conaffinity="0"/>
    <body name="bot" pos="{sx} {sy} 2.5">
      <joint type="slide" axis="1 0 0" damping="0"/>
      <joint type="slide" axis="0 1 0" damping="0"/>
      <joint name="yaw" type="hinge" axis="0 0 1" damping="0"/>
      <geom type="capsule" fromto="-0.4 0 0 0.4 0 0" size="0.2"
            rgba="1 0.9 0.1 1" contype="0" conaffinity="0"/>
      <site name="lidar" pos="0 0 0.5" size="0.05"/>
    </body>
  </worldbody>
</mujoco>"""

    def _init_mujoco(self, use_viewer):
        xml = self._build_xml()
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.d.qpos[0] = self.x
        self.d.qpos[1] = self.y
        self.d.qpos[2] = self.yaw
        mujoco.mj_forward(self.m, self.d)
        if use_viewer:
            import mujoco.viewer as mv
            self.viewer = mv.launch_passive(self.m, self.d)

    # ──────────────────────────────────────────────
    # 控制
    # ──────────────────────────────────────────────
    def set_cmd_vel(self, vx_body, vyaw):
        """设置速度（机身系）。
        Args:
            vx_body: 前进速度 (m/s, 机身系 +x)
            vyaw: 转向角速度 (rad/s)
        """
        self.vyaw = vyaw
        # 机身系 → 世界系
        self.vx = vx_body * math.cos(self.yaw)
        self.vy = vx_body * math.sin(self.yaw)

    def _is_free(self, wx, wy):
        """世界坐标是否在自由空间（非墙、非出界）。"""
        col = int(wx * self.px_per_m)
        row = self.hf_h - 1 - int(wy * self.px_per_m)
        if not (0 <= col < self.hf_w and 0 <= row < self.hf_h):
            return False  # 出界 = 不通行
        return not self._hf_bin[row, col]

    def _check_collision(self, x, y, yaw):
        """检测机器狗足印（0.8m×0.4m 胶囊）是否碰墙。"""
        c, s = math.cos(yaw), math.sin(yaw)
        for dx, dy in [(0, 0), (0.3, 0), (-0.3, 0), (0, 0.25), (0, -0.25)]:
            if not self._is_free(x + dx * c - dy * s, y + dx * s + dy * c):
                return True
        return False

    def step(self):
        """推进一个物理步（timestep 秒）。碰到墙停住（不穿墙）。"""
        # 更新 yaw（转向不受墙限制）
        self.yaw += self.vyaw * self.timestep
        self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi
        # 试探性新位置
        new_x = self.x + self.vx * self.timestep
        new_y = self.y + self.vy * self.timestep
        # 碰撞检测：碰墙就不动（原地转向仍允许）
        if self._check_collision(new_x, new_y, self.yaw):
            self.vx, self.vy = 0.0, 0.0
        else:
            self.x, self.y = new_x, new_y
        # 写入 MuJoCo（qpos 直接设，qvel=0 防 mj_step 二次积分）
        if self.d is not None:
            self.d.qpos[0] = self.x
            self.d.qpos[1] = self.y
            self.d.qpos[2] = self.yaw
            self.d.qvel[0] = 0.0
            self.d.qvel[1] = 0.0
            mujoco.mj_step(self.m, self.d)
        if self.viewer is not None:
            self.viewer.sync()

    # ──────────────────────────────────────────────
    # 感知
    # ──────────────────────────────────────────────
    def get_true_pose(self):
        """真值位姿 (x, y, yaw)——sim_bridge 发 /odom 或调试用。"""
        return self.x, self.y, self.yaw

    def get_scan(self):
        """解析式 heightfield 射线扫描。
        Returns:
            ranges: (R,) float32, 命中距离（m），无命中=inf
            angles: (R,) float32, 机身系角度（rad），0=前方 CCW 正
        """
        rel = np.linspace(-self.lidar_fov / 2, self.lidar_fov / 2,
                          self.lidar_rays, dtype=np.float32)
        angles = self.yaw + rel
        cos_a = np.cos(angles).astype(np.float32)
        sin_a = np.sin(angles).astype(np.float32)

        # 射线上采样点世界坐标 (R, S)
        xs = cos_a[:, None] * self._scan_k[None, :] + np.float32(self.x)
        ys = sin_a[:, None] * self._scan_k[None, :] + np.float32(self.y)

        # 世界 → 图像像素
        col = (xs * self.px_per_m).astype(np.int32)
        row = self.hf_h - 1 - (ys * self.px_per_m).astype(np.int32)
        inb = (col >= 0) & (col < self.hf_w) & (row >= 0) & (row < self.hf_h)
        col_c = np.clip(col, 0, self.hf_w - 1)
        row_c = np.clip(row, 0, self.hf_h - 1)

        wall = self._hf_bin[row_c, col_c] & inb       # 真墙命中
        hit_any = wall | (~inb)                         # 出界也算终止
        R, S = hit_any.shape
        idx = np.arange(R)
        first = np.argmax(hit_any, axis=1)              # 首个终止下标
        has = hit_any[idx, first]
        ranges = np.where(has & inb[idx, np.minimum(first, S - 1)],
                          first.astype(np.float32) * self.scan_step,
                          np.float32(np.inf))
        return ranges, rel  # rel = 机身系角度

    def get_scan_polar(self):
        """便捷：返回 (ranges, angles) 同 get_scan。"""
        return self.get_scan()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


# ═══════════════════════════════════════════════
# 自测：开车走迷宫，打印 scan
# ═══════════════════════════════════════════════
def _selftest():
    import sys
    use_viewer = "--viewer" in sys.argv
    sim = SimBackend(use_mujoco_viewer=use_viewer)
    print(f"SimBackend ready: maze={sim.hf_w}x{sim.hf_h}px "
          f"({sim.px_per_m}px/m), lidar={sim.lidar_rays}rays/{sim.lidar_range}m")

    # 前进 3 秒（0.3 m/s），每秒打印 scan 摘要
    sim.set_cmd_vel(0.3, 0.0)
    steps_per_sec = int(1.0 / sim.timestep)
    for sec in range(5):
        for _ in range(steps_per_sec):
            sim.step()
        x, y, yaw = sim.get_true_pose()
        ranges, angles = sim.get_scan()
        valid = ranges[ranges < sim.lidar_range]
        nearest = valid.min() if len(valid) else float("inf")
        # 前方 90° 扇区
        fwd_mask = np.abs(angles) < math.pi / 4
        fwd = ranges[fwd_mask]
        fwd_min = fwd[fwd < sim.lidar_range].min() if np.any(fwd < sim.lidar_range) else float("inf")
        print(f"  t={sec+1}s  pos=({x:.2f},{y:.2f}) yaw={math.degrees(yaw):.0f}°  "
              f"scan: {len(valid)}/{sim.lidar_rays} hit  nearest={nearest:.2f}m  "
              f"fwd_min={fwd_min:.2f}m")

    # 右转 90° 后再前进
    print("\n右转 90°...")
    sim.set_cmd_vel(0.0, -0.5)
    for _ in range(int(math.pi / 2 / 0.5 / sim.timestep)):
        sim.step()
    sim.set_cmd_vel(0.3, 0.0)
    for sec in range(3):
        for _ in range(steps_per_sec):
            sim.step()
        x, y, yaw = sim.get_true_pose()
        ranges, angles = sim.get_scan()
        fwd_mask = np.abs(angles) < math.pi / 4
        fwd = ranges[fwd_mask]
        fwd_min = fwd[fwd < sim.lidar_range].min() if np.any(fwd < sim.lidar_range) else float("inf")
        print(f"  t={sec+6}s  pos=({x:.2f},{y:.2f}) yaw={math.degrees(yaw):.0f}°  fwd_min={fwd_min:.2f}m")

    sim.close()
    print("\n自测完成。")


if __name__ == "__main__":
    _selftest()
