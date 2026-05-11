"""
雷达模块 — 多种可替换实现

接口:
    update(robot_x, robot_y, robot_yaw) → list[tuple]  # [(x,y,z), ...]
    hit_count → int
    step_interval → int

实现:
    MultiLineLidar   — 水平多线 (均匀角间距, 可调线数)
    SphereSampler    — Fibonacci球面均匀采样
"""

import math
import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None


def fibonacci_sphere(n: int) -> np.ndarray:
    """Fibonacci球面均匀采样 n 个方向 (n×3)"""
    idx = np.arange(n, dtype=np.float64)
    phi = np.arccos(1 - 2 * (idx + 0.5) / n)
    theta = np.pi * (1 + 5**0.5) * idx
    return np.column_stack([
        np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(phi),
    ])


class _BaseRadar:
    """雷达基类 — 共享 mj_ray 调用逻辑"""

    def __init__(self, model, data, site_name=None, position=None,
                 range_m=15.0, hz=10, min_z=0.05, min_dist=0.25):
        self.model = model
        self.data = data
        self.site_id = None
        self.position = position
        self.range_m = range_m
        self.hz = hz
        self.min_z = min_z
        self.min_dist = min_dist

        if site_name is not None:
            self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                             site_name)

        self.sim_dt = model.opt.timestep
        self.step_interval = int(1.0 / hz / self.sim_dt)
        self._last_points = []
        self._gid = np.array([-1], np.int32)
        self._gg = np.ones(6, dtype=np.uint8) * 255

    def _get_pos(self, rx, ry):
        if self.site_id is not None:
            return self.data.site_xpos[self.site_id].copy()
        if self.position is not None:
            return np.array(self.position, np.float64)
        return np.array([rx, ry, 0.3], np.float64)

    def _cast(self, pos, dw):
        self._gid[0] = -1
        dist = mujoco.mj_ray(self.model, self.data, pos, dw,
                             self._gg, 1, -1, self._gid)
        if self._gid[0] >= 0 and 0 < dist < self.range_m:
            hit = pos + dw * dist
            if hit[2] > self.min_z and dist > self.min_dist:
                return (float(hit[0]), float(hit[1]), float(hit[2]))
        return None

    @property
    def points(self):
        return self._last_points

    @property
    def hit_count(self):
        return len(self._last_points)


class MultiLineLidar(_BaseRadar):
    """水平多线雷达 — 均匀角间距 + 可调线数。

    参数:
        rays: 每线水平射线数 (默认120)
        lines: 垂直线层数 (默认3)
        elevation_range: 俯仰角总范围 ±° (默认2)
    """

    def __init__(self, model, data, rays=120, lines=3,
                 elevation_range=2.0, **kwargs):
        super().__init__(model, data, **kwargs)
        self.rays = rays
        self.lines = lines

        if lines == 1:
            self.elevations = [0.0]
        else:
            half = math.radians(elevation_range / 2)
            self.elevations = np.linspace(-half, half, lines)

    def update(self, rx, ry, yaw=0.0):
        pos = self._get_pos(rx, ry)
        pts = []
        for elev in self.elevations:
            cos_e = math.cos(elev)
            sin_e = math.sin(elev)
            for i in range(self.rays):
                a = yaw + 2 * math.pi * i / self.rays
                dw = np.array([math.cos(a) * cos_e,
                               math.sin(a) * cos_e,
                               sin_e])
                hit = self._cast(pos, dw)
                if hit:
                    pts.append(hit)
        self._last_points = pts
        return pts


class SphereSampler(_BaseRadar):
    """Fibonacci球面均匀采样雷达 — 全方向均匀覆盖。

    参数:
        n_rays: 总射线数 (默认2000, 均匀分布在4π球面上)
    """

    def __init__(self, model, data, n_rays=2000, **kwargs):
        super().__init__(model, data, **kwargs)
        self.directions = fibonacci_sphere(n_rays)

    def update(self, rx, ry, yaw=0.0):
        pos = self._get_pos(rx, ry)
        pts = []
        for dw in self.directions:
            hit = self._cast(pos, dw)
            if hit:
                pts.append(hit)
        self._last_points = pts
        return pts


# ── 自测 ──
if __name__ == "__main__":
    if mujoco is None:
        print("SKIP: mujoco not installed")
    else:
        # 快速射线验证
        xml = """<mujoco>
        <option timestep="0.008"/>
        <worldbody>
        <geom type="plane" size="10 10 0.05"/>
        <body pos="3 0 0.5"><geom type="sphere" size="0.5"/></body>
        <body name="r" pos="0 0 0.5"><site name="top" pos="0 0 0.5"/></body>
        </worldbody></mujoco>"""
        import tempfile, os
        p = os.path.join(tempfile.gettempdir(), "radar_test.xml")
        with open(p, "w") as f:
            f.write(xml)
        m = mujoco.MjModel.from_xml_path(p)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)

        for name, cls, kw in [
            ("MultiLine(120r×3l)", MultiLineLidar, {"rays": 120, "lines": 3}),
            ("MultiLine(240r×1l)", MultiLineLidar, {"rays": 240, "lines": 1}),
            ("Sphere(500r)", SphereSampler, {"n_rays": 500}),
        ]:
            r = cls(m, d, site_name="top", range_m=10, **kw)
            r.update(0, 0)
            print(f"{name}: {r.hit_count}点")
        print("OK")
