#!/usr/bin/env python3
"""独立雷达模块 — 10Hz 多线lidar 点云点数线数可调"""
import math, numpy as np, mujoco

class LidarSensor:
    """
    10Hz多线激光雷达传感器。
    
    参数:
        model, data: MuJoCo模型和数据
        site_name:   雷达挂载site名称（None=用坐标直接算）
        position:    雷达坐标 (x,y,z)，site_name为None时使用
        rays:        每线射线数（默认120）
        lines:       扫描线层数（默认3，模拟16线雷达简配）
        range_m:     最大探测距离（默认15m）
        elevation_range: 俯仰角范围（默认±2°，窄带扫描地面障碍物）
        hz:          更新频率（默认10Hz）
    """
    
    def __init__(self, model, data, site_name=None, position=None,
                 rays=120, lines=3, range_m=15.0, elevation_range=2.0, hz=10):
        self.model = model
        self.data = data
        self.site_id = None
        self.position = position
        
        if site_name is not None:
            self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        
        self.rays = rays          # 每线水平射线数
        self.lines = lines        # 垂直线层数
        self.range_m = range_m    # 最大距离
        self.hz = hz
        self.sim_dt = model.opt.timestep
        self.step_interval = int(1.0 / hz / self.sim_dt)  # 多少sim步更新一次
        
        # 预计算各线的俯仰角
        if lines == 1:
            self.elevations = [0.0]  # 单线水平
        else:
            half_range = math.radians(elevation_range / 2)
            self.elevations = np.linspace(-half_range, half_range, lines)
        
        self._last_points = []   # 最新点云 [(x,y,z),...]
        self._last_hit_count = 0
        self._gid = np.array([-1], np.int32)
        
    def update(self, robot_x, robot_y, robot_yaw):
        """10Hz更新：从MuJoCo做射线扫描，填充点云"""
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
                    sin_elev
                ])
                
                dist = mujoco.mj_ray(
                    self.model, self.data, pos, dw,
                    None, 1, -1, self._gid
                )
                
                if self._gid[0] >= 0 and 0 < dist < self.range_m:
                    hit = pos + dw * dist
                    # 过滤自身和地面以下的点
                    if hit[2] > 0.1 and dist > 0.25:
                        points.append((float(hit[0]), float(hit[1]), float(hit[2])))
                        total_hits += 1
        
        self._last_points = points
        self._last_hit_count = total_hits
        return points
    
    @property
    def points(self):
        """最新点云 [(x,y,z),...]"""
        return self._last_points
    
    @property
    def points_2d(self):
        """最新点云 仅xy [(x,y),...]"""
        return [(p[0], p[1]) for p in self._last_points]
    
    @property
    def hit_count(self):
        return self._last_hit_count
    
    def cluster(self, grid_size=1.0, min_hits=3):
        """简单网格聚类 → 障碍物中心列表 [(cx,cy,r),...]"""
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
        for (gx, gy), cpts in grid.items():
            if len(cpts) >= min_hits:
                xs = [p[0] for p in cpts]
                ys = [p[1] for p in cpts]
                cx = np.mean(xs)
                cy = np.mean(ys)
                r = max(math.hypot(cx - xs[j], cy - ys[j]) for j in range(len(cpts))) + 0.1
                obstacles.append((float(cx), float(cy), min(r, 0.5)))
        
        return obstacles


# ── 自测 ──
if __name__ == "__main__":
    import time
    from xml.etree import ElementTree as ET
    
    # 简单测试场景
    scene = """<mujoco>
    <option timestep="0.008"/>
    <worldbody>
    <geom type="plane" size="20 20 0.05"/>
    <body name="r" pos="0 0 0.5">
    <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
    <joint name="x" type="slide" axis="1 0 0"/>
    <joint name="y" type="slide" axis="0 1 0"/>
    <site name="lidar" pos="0 0 0.3" size="0.02"/>
    </body>
    <body pos="3 0 0.2"><geom type="cylinder" size="0.3 0.2"/></body>
    <body pos="0 5 0.2"><geom type="cylinder" size="0.2 0.2"/></body>
    </worldbody></mujoco>"""
    
    with open("/tmp/lidar_test.xml", "w") as f:
        f.write(scene)
    
    m = mujoco.MjModel.from_xml_path("/tmp/lidar_test.xml")
    d = mujoco.MjData(m)
    d.qpos[0:2] = [0, 0]
    
    lidar = LidarSensor(m, d, site_name="lidar", rays=36, lines=1, range_m=10)
    
    for step in range(100):
        mujoco.mj_step(m, d)
        if step % lidar.step_interval == 0:
            pts = lidar.update(0, 0, 0)
    
    print(f"Rays: {36}, Points: {lidar.hit_count}")
    print(f"Clusters: {len(lidar.cluster(grid_size=1.0, min_hits=2))}")
    print("OK")
