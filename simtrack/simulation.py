"""
仿真运行器 — 组件化赛道避障仿真

组装 地图/障碍物/雷达/算法/模型 五大组件，
运行 MuJoCo 闭环仿真并输出日志。

用法:
    from simtrack import Simulation
    sim = Simulation(
        track_hfield="/tmp/track_hd.png",
        algorithm=VOAlgorithm(max_speed=3.0),
        lidar_rays=240,   # 推荐 240+ 保证墙壁检测
    )
    sim.run()
"""

import sys
import time
import math
import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    mujoco = None

from simtrack.obstacles import ObstacleGenerator
from simtrack.lidar import LidarSensor
from simtrack.algorithms.base import AvoidanceAlgorithm
from simtrack.algorithms.vo import VOAlgorithm
from simtrack.models.cylinder import build_cylinder_scene


class Simulation:
    """模块化赛道避障仿真。

    五组件可替换:
        track_hfield    — hfield PNG 路径
        obstacle_gen    — 障碍物生成器 (默认: 沿赛道中轴线)
        lidar           — 激光雷达 (默认: 240射线/3线/15m)
        algorithm       — 避障算法 (默认: VO)
        model_builder   — 场景 XML 构建器 (默认: 圆柱体)

    使用步骤:
        1. sim = Simulation(...)
        2. sim.setup()     # 生成障碍物 + 构建 MuJoCo 场景
        3. sim.run()       # 运行仿真 (含 viewer)
    """

    def __init__(
        self,
        track_hfield: str = "/tmp/track_hd.png",
        obstacle_gen: ObstacleGenerator = None,
        lidar: LidarSensor = None,
        algorithm: AvoidanceAlgorithm = None,
        lidar_rays: int = 240,
        lidar_lines: int = 3,
        lidar_range: float = 15.0,
        lidar_hz: int = 10,
        max_speed: float = 2.0,
        start_x: float = 5.0,
        start_y: float = 45.0,
        robot_radius: float = 0.25,
        obs_radius: float = 0.3,
        obs_height: float = 0.3,
        log_path: str = "/tmp/simtrack.log",
        seed: int = None,
    ):
        """初始化仿真。

        Args:
            track_hfield: hfield PNG 路径
            obstacle_gen: 障碍物生成器 (None=默认)
            lidar: 雷达实例 (None=默认 240射线/3线/15m/10Hz)
            algorithm: 避障算法 (None=VOAlgorithm)
            lidar_rays: 默认雷达射线数
            lidar_lines: 默认雷达线数
            lidar_range: 默认雷达距离
            lidar_hz: 默认雷达频率
            max_speed: 最大速度
            start_x, start_y: 起点
            robot_radius: 机器人半径
            obs_radius: 障碍物半径
            obs_height: 障碍物半高
            log_path: 日志路径
            seed: 随机种子 (None=每次不同)
        """
        self.track_hfield = track_hfield
        self._obstacle_gen = obstacle_gen
        self._lidar = lidar
        self._algorithm = algorithm or VOAlgorithm(max_speed=max_speed)
        self.lidar_rays = lidar_rays
        self.lidar_lines = lidar_lines
        self.lidar_range = lidar_range
        self.lidar_hz = lidar_hz
        self.max_speed = max_speed
        self.start_x = start_x
        self.start_y = start_y
        self.robot_radius = robot_radius
        self.obs_radius = obs_radius
        self.obs_height = obs_height
        self.log_path = log_path
        self.seed = seed

        self.model = None
        self.data = None
        self.log = None

    def log(self, msg):
        print(msg, flush=True)
        if self.log:
            self.log.write(msg + "\n")
            self.log.flush()

    def setup(self):
        """构建场景: 生成障碍物 → 构建 XML → 加载 MuJoCo 模型。"""
        self.log = open(self.log_path, "w")

        # 默认障碍物生成器 (需要中心线数据 — 从 hfield 无法反推，
        # 所以这里用硬编码的赛道参数。用户可传入自己的 obstacle_gen。)
        if self._obstacle_gen is None:
            # 构建与 trackgen 匹配的中心线
            center_line = self._make_default_centerline()
            self._obstacle_gen = ObstacleGenerator(
                center_line, seed=self.seed,
                spacing_range=(4, 8), lateral_range=(0.5, 4.5),
            )

        obstacles = self._obstacle_gen.generate()
        self.log(f"Obstacles: {len(obstacles)}")

        obstacles_xml = self._obstacle_gen.to_mujoco_xml(
            obstacles, obs_radius=self.obs_radius, obs_height=self.obs_height,
        )

        scene_xml = build_cylinder_scene(
            hfield_path=self.track_hfield,
            start_x=self.start_x, start_y=self.start_y,
            robot_radius=self.robot_radius,
            obs_radius=self.obs_radius,
            obs_height=self.obs_height,
            obstacles_xml=obstacles_xml,
        )

        import tempfile
        xml_path = tempfile.mktemp(suffix=".xml")
        with open(xml_path, "w") as f:
            f.write(scene_xml)

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[0:2] = [self.start_x, self.start_y]

        # 雷达
        if self._lidar is None:
            self._lidar = LidarSensor(
                self.model, self.data, site_name="lidar_top",
                rays=self.lidar_rays, lines=self.lidar_lines,
                range_m=self.lidar_range, hz=self.lidar_hz,
            )

        self.log(
            f"Lidar: {self._lidar.rays}r x {self._lidar.lines}l, "
            f"{self._lidar.range_m}m, {self._lidar.hz}Hz, "
            f"update every {self._lidar.step_interval} steps"
        )

        return self

    def run(self, headless: bool = False):
        """运行仿真闭环。

        Args:
            headless: True=无窗模式 (需 Xvfb)
        """
        if self.model is None:
            self.setup()

        # Waypoints (与中心线匹配)
        waypoints = self._make_default_waypoints()
        track_len = sum(
            math.hypot(waypoints[i][0] - waypoints[i-1][0],
                        waypoints[i][1] - waypoints[i-1][1])
            for i in range(1, len(waypoints))
        )
        self.log(f"Track: {track_len:.0f}m, {len(waypoints)}wp")

        m, d = self.model, self.data
        lidar = self._lidar
        algo = self._algorithm

        cnt = 0
        cw = 0
        goal = waypoints[0]
        vc = self.max_speed
        coll = 0
        decision_every = lidar.step_interval

        if headless:
            # 无窗模式
            t0 = time.time()
            while True:
                bx, by = float(d.qpos[0]), float(d.qpos[1])
                yaw = float(d.qpos[2])

                if cnt % decision_every == 0:
                    lidar.update(bx, by, yaw)
                    obstacles = lidar.cluster()
                    dg = math.hypot(bx - goal[0], by - goal[1])

                    result = algo.choose_heading(
                        (bx, by), vc, goal, obstacles,
                    )
                    yaw = result.heading
                    vc = result.speed

                    if dg < 1.5 and cw < len(waypoints) - 1:
                        cw += 1
                        goal = waypoints[cw]
                        self.log(
                            f"[{cnt*m.opt.timestep:.0f}s] WP{cw} "
                            f"pts={lidar.hit_count} obs={len(obstacles)}"
                        )

                    if cw >= len(waypoints) - 1 and dg < 2.0:
                        elapsed = cnt * m.opt.timestep
                        self.log(
                            f"ARRIVED sim={elapsed:.0f}s "
                            f"avg={track_len/elapsed:.1f}m/s coll={coll}"
                        )
                        break

                for i in range(d.ncon):
                    if d.contact[i].dist < -0.01:
                        coll += 1
                        break

                d.qvel[0] = vc * math.cos(yaw)
                d.qvel[1] = vc * math.sin(yaw)
                d.qvel[2] = 0
                mujoco.mj_step(m, d)
                cnt += 1

                if cnt % 500 == 0:
                    self.log(
                        f"[{cnt*m.opt.timestep:.0f}s] ({bx:.1f},{by:.1f}) "
                        f"wp{cw} v={vc:.1f} pts={lidar.hit_count} "
                        f"obs={len(obstacles)} coll={coll}"
                    )

                if cnt * m.opt.timestep > 600:
                    self.log("TIMEOUT")
                    break
        else:
            with mujoco.viewer.launch_passive(m, d) as v:
                v.cam.azimuth = 90
                v.cam.elevation = -30
                v.cam.distance = 30
                v.cam.lookat = (25, 25, 0)

                while v.is_running():
                    bx, by = float(d.qpos[0]), float(d.qpos[1])
                    yaw = float(d.qpos[2])

                    if cnt % decision_every == 0:
                        lidar.update(bx, by, yaw)
                        obstacles = lidar.cluster()
                        dg = math.hypot(bx - goal[0], by - goal[1])

                        result = algo.choose_heading(
                            (bx, by), vc, goal, obstacles,
                        )
                        yaw = result.heading
                        vc = result.speed

                        if dg < 1.5 and cw < len(waypoints) - 1:
                            cw += 1
                            goal = waypoints[cw]
                            self.log(
                                f"[{cnt*m.opt.timestep:.0f}s] WP{cw} "
                                f"pts={lidar.hit_count} obs={len(obstacles)}"
                            )

                        if cw >= len(waypoints) - 1 and dg < 2.0:
                            elapsed = cnt * m.opt.timestep
                            self.log(
                                f"ARRIVED sim={elapsed:.0f}s "
                                f"avg={track_len/elapsed:.1f}m/s coll={coll}"
                            )
                            break

                    for i in range(d.ncon):
                        if d.contact[i].dist < -0.01:
                            coll += 1
                            break

                    d.qvel[0] = vc * math.cos(yaw)
                    d.qvel[1] = vc * math.sin(yaw)
                    d.qvel[2] = 0
                    mujoco.mj_step(m, d)
                    cnt += 1

                    if cnt % 500 == 0:
                        self.log(
                            f"[{cnt*m.opt.timestep:.0f}s] ({bx:.1f},{by:.1f}) "
                            f"wp{cw} v={vc:.1f} pts={lidar.hit_count} "
                            f"obs={len(obstacles)} coll={coll}"
                        )

        self.log(f"DONE sim={cnt*m.opt.timestep:.0f}s coll={coll}")
        self.log.close()
        return {"time": cnt * m.opt.timestep, "collisions": coll, "track_len": track_len}

    # ── 辅助 ──

    @staticmethod
    def _make_default_centerline():
        """生成与 trackgen 匹配的赛道中心线 (10段蛇形)"""
        cx, cy = [], []
        x, y = 5.0, 45.0
        turn_r = 5.0

        for i in range(10):
            if i % 2 == 0:
                for j in range(80):
                    cx.append(x + j * 0.5)
                    cy.append(y)
                x = cx[-1]
                n = int(math.pi * turn_r / 0.25)
                for j in range(1, n + 1):
                    a = math.pi / 2 * j / n
                    cx.append(x + turn_r * (1 - math.cos(a)))
                    cy.append(y - turn_r * math.sin(a))
                x, y = cx[-1], cy[-1] - turn_r * 2
            else:
                for j in range(80):
                    cx.append(x - j * 0.5)
                    cy.append(y)
                x = cx[-1]
                n = int(math.pi * turn_r / 0.25)
                for j in range(1, n + 1):
                    a = math.pi / 2 * j / n
                    cx.append(x - turn_r * (1 - math.cos(a)))
                    cy.append(y - turn_r * math.sin(a))
                x, y = cx[-1], cy[-1] - turn_r * 2

        return [(float(cx[i]), float(cy[i])) for i in range(len(cx))]

    @staticmethod
    def _make_default_waypoints():
        """生成 waypoints (每 8m 一个)"""
        cl = Simulation._make_default_centerline()
        cum = np.insert(
            np.cumsum([math.hypot(cl[i][0] - cl[i-1][0], cl[i][1] - cl[i-1][1])
                        for i in range(1, len(cl))]), 0, 0)
        wp = []
        nd = 8.0
        for i in range(len(cl)):
            if cum[i] >= nd:
                wp.append((cl[i][0], cl[i][1]))
                nd += 8.0
        if not wp or wp[-1] != (cl[-1][0], cl[-1][1]):
            wp.append((cl[-1][0], cl[-1][1]))
        return wp


# ── 自测 ──
if __name__ == "__main__":
    print("Simulation 模块导入正常")
    print(f"  Waypoints: {len(Simulation._make_default_waypoints())}")
    print(f"  Centerline: {len(Simulation._make_default_centerline())} 点")
