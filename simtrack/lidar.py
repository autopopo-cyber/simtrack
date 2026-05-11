"""
LidarSensor — 多线激光雷达模块 (10Hz, 点数/线数可调)

独立于算法和模型运行，每 10Hz 刷新点云。
通过 MuJoCo mj_ray 进行真实射线扫描 (非上帝视角)。

用法:
    from simtrack.lidar import LidarSensor
    lidar = LidarSensor(model, data, rays=120, lines=3, range_m=15.0, hz=10)
    # 每 10Hz:
    lidar.update(robot_x, robot_y, robot_yaw)
    clusters = lidar.cluster(grid_size=1.0, min_hits=3)

可调参数:
    rays             — 每线水平射线数 (默认 120)
    lines            — 垂直线层数 (默认 3, 模拟 16 线雷达简配)
    range_m          — 最大探测距离 (默认 15m)
    elevation_range  — 俯仰角范围 (默认 ±2°)
    hz               — 更新频率 (默认 10Hz)
"""

import math
import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None


class LidarSensor:
    """10Hz 多线激光雷达。

    使用 MuJoCo mj_ray 进行真实物理射线扫描，
    自动过滤自身 (dist < 0.25m) 和地面以下 (z < 0.1m) 的点。

    雷达可挂在 MuJoCo site 上 (通过 site_name) 或手动指定坐标。
    推荐挂在机器人顶部的 site 上以避开自身体积。
    """

    def __init__(
        self,
        model,
        data,
        site_name: str = None,
        position: tuple = None,
        rays: int = 120,
        lines: int = 3,
        range_m: float = 15.0,
        elevation_range: float = 2.0,
        hz: int = 10,
        min_dist: float = 0.25,
        min_z: float = 0.1,
    ):
        """初始化激光雷达。

        Args:
            model, data: MuJoCo 模型和数据
            site_name:   雷达挂载 site 名称 (None=手动指定坐标)
            position:    雷达坐标 (x,y,z), site_name=None 时使用
            rays:        每线水平射线数 (默认 120)
            lines:       垂直线层数 (默认 3)
            range_m:     最大探测距离 (默认 15m)
            elevation_range: 俯仰角范围 ±° (默认 ±2°)
            hz:          更新频率 (默认 10Hz)
            min_dist:    最小有效距离 (过滤自身, 默认 0.25m)
            min_z:       最小有效高度 (过滤地面, 默认 0.1m)
        """
        if mujoco is None:
            raise ImportError("mujoco 未安装")

        self.model = model
        self.data = data
        self.site_id = None
        self.position = position
        self.rays = rays
        self.lines = lines
        self.range_m = range_m
        self.hz = hz
        self.min_dist = min_dist
        self.min_z = min_z

        if site_name is not None:
            self.site_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, site_name
            )

        self.sim_dt = model.opt.timestep
        self.step_interval = int(1.0 / hz / self.sim_dt)

        # 预计算俯仰角
        if lines == 1:
            self.elevations = [0.0]
        else:
            half_range = math.radians(elevation_range / 2)
            self.elevations = np.linspace(-half_range, half_range, lines)

        self._last_points = []
        self._last_hit_count = 0
        self._gid = np.array([-1], np.int32)

    # ── 主更新 ──

    def update(self, robot_x: float, robot_y: float, robot_yaw: float) -> list:
        """执行一次射线扫描 (10Hz 调用)。

        通常只在 step_interval 整数倍时调用。
        射线扫描开销较大 (~120×3×模型复杂度)，不应每帧调用。

        Args:
            robot_x, robot_y: 机器人世界坐标
            robot_yaw: 机器人朝向 (弧度)

        Returns:
            list[tuple]: 点云 [(x,y,z), ...]
        """
        if self.site_id is not None:
            pos = self.data.site_xpos[self.site_id].copy()
        elif self.position is not None:
            pos = np.array(self.position, np.float64)
        else:
            pos = np.array([robot_x, robot_y, 0.3], np.float64)

        points = []
        total_hits = 0

        for elev in self.elevations:
            cos_elev = math.cos(elev)
            sin_elev = math.sin(elev)

            for i in range(self.rays):
                a = robot_yaw + 2 * math.pi * i / self.rays
                dw = np.array([
                    math.cos(a) * cos_elev,
                    math.sin(a) * cos_elev,
                    sin_elev,
                ])

                dist = mujoco.mj_ray(
                    self.model, self.data, pos, dw,
                    None, 1, -1, self._gid,
                )

                if self._gid[0] >= 0 and 0 < dist < self.range_m:
                    hit = pos + dw * dist
                    if hit[2] > self.min_z and dist > self.min_dist:
                        points.append((float(hit[0]), float(hit[1]), float(hit[2])))
                        total_hits += 1

        self._last_points = points
        self._last_hit_count = total_hits
        return points

    # ── 属性 ──

    @property
    def points(self):
        """最新点云 [(x,y,z), ...]"""
        return self._last_points

    @property
    def points_2d(self):
        """最新点云 2D [(x,y), ...]"""
        return [(p[0], p[1]) for p in self._last_points]

    @property
    def hit_count(self):
        """最新扫描命中点数"""
        return self._last_hit_count

    # ── 聚类 ──

    def cluster(self, grid_size: float = 1.0, min_hits: int = 3) -> list:
        """简单网格聚类 → 障碍物中心列表。

        将 2D 点云按 grid_size 网格分组，每组点数 ≥ min_hits
        则视为一个障碍物，返回其中心坐标和半径。

        Args:
            grid_size: 网格边长 (米, 默认 1.0)
            min_hits: 每组最小点数 (默认 3)

        Returns:
            list[tuple]: [(cx, cy, radius), ...]
        """
        pts = self.points_2d
        if len(pts) < min_hits:
            return []

        grid = {}
        for px, py in pts:
            gx = int(px / grid_size)
            gy = int(py / grid_size)
            key = (gx, gy)
            if key not in grid:
                grid[key] = []
            grid[key].append((px, py))

        obstacles = []
        for (_gx, _gy), cpts in grid.items():
            if len(cpts) >= min_hits:
                xs = [p[0] for p in cpts]
                ys = [p[1] for p in cpts]
                cx = np.mean(xs)
                cy = np.mean(ys)
                r = max(
                    math.hypot(cx - xs[j], cy - ys[j]) for j in range(len(cpts))
                ) + 0.1
                obstacles.append((float(cx), float(cy), min(r, 0.5)))

        return obstacles
