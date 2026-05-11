"""
圆柱体仿真模型 — 2D 滑动物体，可替换为 G1 等复杂模型

提供 build_xml() 生成 MuJoCo 场景，包含:
- hfield 赛道碰撞
- 平面 (无摩擦)
- 圆柱体机器人 (slide x/y + hinge z)
- 顶部雷达 site
- 运行时障碍物
"""

import os
import math
import numpy as np

# hfield 编码: height = pixel/255 * scale - negative
# 3m 护栏: scale=6, negative=3, pixel=255 → 3m
HFIELD_HALF = 25.0
HFIELD_SCALE = 6.0
HFIELD_BASE = 3.0  # "negative" in trackgen encoding


def build_cylinder_scene(
    hfield_path: str,
    start_x: float = 5.0,
    start_y: float = 45.0,
    robot_radius: float = 0.25,
    robot_height: float = 0.3,
    obs_radius: float = 0.3,
    obs_height: float = 0.3,
    obstacles_xml: str = "",
    sim_dt: float = 0.008,
    site_height: float = 0.5,
) -> str:
    """生成圆柱体仿真 MuJoCo XML。

    Args:
        hfield_path: hfield PNG 路径
        start_x, start_y: 起点
        robot_radius: 机器人半径 (m)
        robot_height: 机器人半高 (m)
        obs_radius: 障碍物半径 (m)
        obs_height: 障碍物半高 (m)
        obstacles_xml: 障碍物 XML body 块
        sim_dt: 仿真步长 (s)
        site_height: 雷达 site 高度 (从机器人中心算)

    Returns:
        str: 完整 MuJoCo XML
    """
    return f"""<mujoco>
<compiler angle="radian"/>
<option timestep="{sim_dt}"/>
<visual><global offwidth="1280" offheight="720"/></visual>
<asset>
<hfield name="h" size="{HFIELD_HALF} {HFIELD_HALF} {HFIELD_SCALE} {HFIELD_BASE}" file="{hfield_path}"/>
<material name="mv" rgba="0.25 0.30 0.35 1"/>
<material name="mi" rgba="0.25 0.30 0.35 0"/>
</asset>
<worldbody>
<light pos="25 25 80" dir="0 0 -1" diffuse="1.5 1.5 1.5" specular="0.5 0.5 0.5"/>
<geom type="hfield" hfield="h" pos="25 25 0" material="mv"/>
<geom type="plane" size="0 0 0.05" material="mi"/>
<body name="r" pos="{start_x} {start_y} 0.5">
<inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
<joint name="x" type="slide" axis="1 0 0" damping="0"/>
<joint name="y" type="slide" axis="0 1 0" damping="0"/>
<joint name="z" type="hinge" axis="0 0 1" damping="0"/>
<geom type="cylinder" size="{robot_radius} {robot_height}" rgba="0.2 0.8 0.2 0.9"/>
<site name="lidar_top" pos="0 0 {site_height}" size="0.02"/>
</body>
{obstacles_xml}
</worldbody>
</mujoco>"""
