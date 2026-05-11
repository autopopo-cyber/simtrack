"""LidarSensor 单元测试 — 需要 MuJoCo"""

import pytest
import numpy as np

try:
    import mujoco
except ImportError:
    pytest.skip("mujoco not installed", allow_module_level=True)

from simtrack.lidar import LidarSensor


@pytest.fixture
def simple_scene():
    """简单场景: 平面 + 圆柱体 + 雷达 site"""
    xml = """<mujoco>
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
    import tempfile, os
    path = os.path.join(tempfile.gettempdir(), "lidar_test.xml")
    with open(path, "w") as f:
        f.write(xml)
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    return model, data


class TestLidarSensor:
    def test_init_defaults(self, simple_scene):
        model, data = simple_scene
        lidar = LidarSensor(model, data, site_name="lidar")
        assert lidar.rays == 120
        assert lidar.lines == 3
        assert lidar.range_m == 15.0
        assert lidar.hz == 10

    def test_scan_detects_obstacle(self, simple_scene):
        model, data = simple_scene
        lidar = LidarSensor(model, data, site_name="lidar", rays=36, lines=1, range_m=10)
        data.qpos[0:2] = [0, 0]

        # 跑几步让仿真稳定
        for _ in range(10):
            mujoco.mj_step(model, data)

        pts = lidar.update(0, 0, 0)
        assert len(pts) > 0, "应检测到障碍物"
        assert lidar.hit_count > 0

    def test_cluster_returns_centers(self, simple_scene):
        model, data = simple_scene
        lidar = LidarSensor(model, data, site_name="lidar", rays=36, lines=1, range_m=10)
        data.qpos[0:2] = [0, 0]

        for _ in range(10):
            mujoco.mj_step(model, data)

        lidar.update(0, 0, 0)
        clusters = lidar.cluster(grid_size=1.0, min_hits=2)
        assert len(clusters) > 0
        for c in clusters:
            assert len(c) == 3  # (cx, cy, r)

    def test_custom_params(self, simple_scene):
        model, data = simple_scene
        lidar = LidarSensor(
            model, data, rays=60, lines=1, range_m=5.0, hz=5,
            min_dist=0.5, min_z=0.2,
        )
        assert lidar.rays == 60
        assert lidar.lines == 1
        assert lidar.range_m == 5.0
        assert lidar.hz == 5
        assert lidar.step_interval == int(1.0 / 5 / model.opt.timestep)

    def test_empty_scene(self):
        """空场景无命中"""
        xml = """<mujoco>
        <option timestep="0.008"/>
        <worldbody>
        <geom type="plane" size="20 20 0.05"/>
        <body name="r" pos="0 0 0.5">
        <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
        <site name="lidar" pos="0 0 0.3" size="0.02"/>
        </body>
        </worldbody></mujoco>"""
        import tempfile, os
        path = os.path.join(tempfile.gettempdir(), "empty_test.xml")
        with open(path, "w") as f:
            f.write(xml)
        model = mujoco.MjModel.from_xml_path(path)
        data = mujoco.MjData(model)
        lidar = LidarSensor(model, data, site_name="lidar", rays=36, lines=1, range_m=10)
        pts = lidar.update(0, 0, 0)
        assert len(pts) == 0 or lidar.hit_count == 0
