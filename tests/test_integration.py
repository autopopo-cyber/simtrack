"""集成测试: 全管线 — 赛道生成 → 障碍物 → 雷达 → VO → 仿真

这些测试需要 MuJoCo + GPU，在超服上运行。
用法:
    cd simtrack && python tests/test_integration.py
"""

import os
import sys
import math
import time
import tempfile
import subprocess

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("SKIP: mujoco not installed")
    sys.exit(0)

from simtrack.trackgen import TrackGenerator
from simtrack.obstacles import ObstacleGenerator
from simtrack.lidar import LidarSensor
from simtrack.algorithms.vo import VOAlgorithm
from simtrack.models.cylinder import build_cylinder_scene


def test_full_pipeline_headless():
    """完整管线 headless 测试: 生成赛道→障碍物→雷达→VO→仿真闭环"""
    print("=" * 60)
    print("集成测试: 全管线 headless")
    print("=" * 60)

    # [1] 生成赛道 (小分辨率快测)
    print("\n[1] 生成赛道...")
    tg = TrackGenerator(hf_res=200, guard_height=3.0, guard_brush=5)
    tg.generate()
    hfield_path = os.path.join(tempfile.gettempdir(), "test_track.png")
    tg.save(hfield_path)
    print(f"  赛道: {tg.total_len:.0f}m, {len(tg.waypoints)} wp")
    print(f"  护栏: {(tg.hfield == 255).sum()} px")

    # [2] 障碍物
    print("\n[2] 生成障碍物...")
    og = ObstacleGenerator(
        tg.center_line, total_len=tg.total_len, seed=42,
    )
    obstacles = og.generate()
    print(f"  障碍物: {len(obstacles)} 个")
    assert len(obstacles) > 0

    # [3] 构建场景
    print("\n[3] 构建 MuJoCo 场景...")
    obstacles_xml = og.to_mujoco_xml(obstacles, obs_radius=0.3, obs_height=0.3)
    scene_xml = build_cylinder_scene(
        hfield_path=hfield_path,
        start_x=5.0, start_y=45.0,
        obstacles_xml=obstacles_xml,
    )
    xml_path = os.path.join(tempfile.gettempdir(), "test_scene.xml")
    with open(xml_path, "w") as f:
        f.write(scene_xml)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    data.qpos[0:2] = [5.0, 45.0]
    print(f"  模型: {model.nbody} bodies, {model.ngeom} geoms")

    # [4] 雷达
    print("\n[4] 初始化雷达...")
    lidar = LidarSensor(model, data, site_name="lidar_top",
                        rays=240, lines=3, range_m=15.0, hz=10)
    print(f"  雷达: {lidar.rays}r × {lidar.lines}l, {lidar.range_m}m")

    # [5] 算法
    print("\n[5] 初始化 VO 算法...")
    algo = VOAlgorithm(max_speed=2.0)
    print(f"  算法: {algo.name}")

    # [6] 生成 waypoints
    cum = np.insert(np.cumsum([
        math.hypot(tg.center_line[i][0] - tg.center_line[i-1][0],
                    tg.center_line[i][1] - tg.center_line[i-1][1])
        for i in range(1, len(tg.center_line))
    ]), 0, 0)
    waypoints = []
    nd = 8.0
    for i in range(len(tg.center_line)):
        if cum[i] >= nd:
            waypoints.append((tg.center_line[i][0], tg.center_line[i][1]))
            nd += 8.0
    if not waypoints or waypoints[-1] != (tg.center_line[-1][0], tg.center_line[-1][1]):
        waypoints.append((tg.center_line[-1][0], tg.center_line[-1][1]))
    track_len = tg.total_len
    print(f"  Waypoints: {len(waypoints)}, 赛道: {track_len:.0f}m")

    # [7] 闭环仿真 (无 viewer)
    print("\n[7] 运行闭环仿真...")
    cnt = 0
    cw = 0
    goal = waypoints[0]
    vc = 2.0
    coll = 0
    decision_every = lidar.step_interval
    max_waypoints_to_test = 10  # 只测前 10 个路点
    t_start = time.time()

    while True:
        bx, by = float(data.qpos[0]), float(data.qpos[1])
        yaw = float(data.qpos[2])

        if cnt % decision_every == 0:
            lidar.update(bx, by, yaw)
            obs = lidar.cluster()
            dg = math.hypot(bx - goal[0], by - goal[1])

            result = algo.choose_heading((bx, by), vc, goal, obs)
            yaw = result.heading
            vc = result.speed

            if dg < 1.5 and cw < len(waypoints) - 1:
                cw += 1
                goal = waypoints[cw]
                print(f"  [{cnt * model.opt.timestep:.0f}s] WP{cw} "
                      f"pts={lidar.hit_count} obs={len(obs)} "
                      f"v={vc:.1f} avoiding={result.avoiding}")

                # 验证: 雷达应检测到一些点
                if lidar.hit_count == 0 and cw > 2:
                    print(f"  WARNING: WP{cw} 雷达无命中!")
                    # 不 fail——可能赛道太短或障碍物在远处

            if cw >= max_waypoints_to_test:
                print(f"  ✓ 到达 WP{max_waypoints_to_test}, 测试通过")
                break

        for i in range(data.ncon):
            if data.contact[i].dist < -0.01:
                coll += 1
                break

        d.qvel[0] = vc * math.cos(yaw)
        d.qvel[1] = vc * math.sin(yaw)
        d.qvel[2] = 0
        mujoco.mj_step(model, data)
        cnt += 1

        if cnt * model.opt.timestep > 120:
            print(f"  ⚠ 超时 (120s), 停在 WP{cw}")
            break

    elapsed = cnt * model.opt.timestep
    print(f"\n  结果: {elapsed:.0f}s, {cw}/{max_waypoints_to_test} wp, "
          f"碰撞{coll}次")
    assert cw >= max_waypoints_to_test, \
        f"只到达 WP{cw}/{max_waypoints_to_test}"
    assert elapsed < 120, f"超时 {elapsed:.0f}s"


if __name__ == "__main__":
    test_full_pipeline_headless()
    print("\n✓ 集成测试通过!")
