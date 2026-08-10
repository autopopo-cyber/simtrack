#!/usr/bin/env python3
"""验证 detect_finish：真实相机配置下，方位角符号与距离估计是否正确。"""
import math
import numpy as np
import mujoco
import glfw
glfw.init(); glfw.window_hint(glfw.VISIBLE, 0)
win = glfw.create_window(1280, 720, "off", None, None)
glfw.make_context_current(win)

HF_SURF = 2.0
FINISH = (10.0, 3.0)   # 测试终点
CAM_XML = '<camera name="bot_cam" pos="0.4 0 0.5" mode="fixed" euler="0 -1.5708 -1.5708"/>'
xml = f"""<mujoco>
  <compiler angle="radian"/><option timestep="0.005"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <worldbody>
    <light pos="25 25 80" dir="0 0 -1" ambient="0.5 0.5 0.55" diffuse="0.9 0.9 0.95"/>
    <geom type="plane" size="30 30 0.1" pos="25 25 {HF_SURF}" rgba="0.55 0.6 0.65 1.0"/>
    <body mocap="true" pos="{FINISH[0]} {FINISH[1]} {HF_SURF+1.5}">
      <geom type="sphere" size="1.5" rgba="0.2 1.0 0.2 0.8"/></body>
    <body name="bot" pos="0 0 {HF_SURF+0.5}">
      <joint type="slide" axis="1 0 0" damping="0"/>
      <joint type="slide" axis="0 1 0" damping="0"/>
      <joint name="yaw" type="hinge" axis="0 0 1" damping="0"/>
      <geom type="capsule" fromto="-0.4 0 0 0.4 0 0" size="0.2" rgba="1 0.9 0.1 1"/>
      {CAM_XML}
    </body>
  </worldbody>
</mujoco>"""
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
r = mujoco.Renderer(m, 720, 1280)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from test_scripts.finish_detect import detect_finish

# 狗在 (5,3)，球在 (10,3)：正前方 5m。yaw=0 → bearing 应 ≈0
# 再把球放到狗左侧 (+y)：world bearing=+45° 方向，检查符号
fov = float(m.cam_fovy[0])
print("cam fovy:", fov)
for (dx, dy, yaw_deg, note) in [
    (5, 0, 0, "正前方5m"),
    (5, 2, 0, "前方偏左(世界+y)"),
    (5, -2, 0, "前方偏右(世界-y)"),
    (0, 0, 90, "yaw=90 球在世界+x=狗右侧"),
    (20, 0, 0, "正前方20m"),
    (40, 0, 0, "正前方40m"),
]:
    fx_, fy_ = 5.0 + dx, 3.0 + dy
    d.qpos[0] = 5.0; d.qpos[1] = 3.0; d.qpos[2] = math.radians(yaw_deg)
    m.body("obot" if False else "bot").pos  # noop
    # 移动 mocap 球
    mid = m.body("mocap" if False else [b for b in range(m.nbody) if m.body(b).mocap][0] if False else 0)
    # 简化：直接用 mj 数据索引 —— mocap body 是第二个 body
    d.mocap_pos[0, 0] = fx_; d.mocap_pos[0, 1] = fy_; d.mocap_pos[0, 2] = HF_SURF + 1.5
    mujoco.mj_forward(m, d)
    r.update_scene(d, camera="bot_cam")
    img = r.render()
    obs = detect_finish(img, fovy_deg=fov)
    if obs is None:
        print(f"{note}: 未检测到")
        continue
    bearing, dist, area = obs
    # 真值方位（世界系，相对狗 yaw）
    wb = math.degrees(math.atan2(fy_ - 3.0, fx_ - 5.0)) - yaw_deg
    wd = math.hypot(fx_ - 5.0, fy_ - 3.0)
    print(f"{note}: bearing={math.degrees(bearing):+.1f}° (真值{wb:+.1f}°)  dist={dist:.1f}m (真值{wd:.1f}m)  area={area:.0f}px")
