#!/usr/bin/env python3
"""渲染真实场景 bot_cam 视角：狗在 ch9 右端看终点球（验证遮挡/检测）。"""
import math, sys, os
import numpy as np
import mujoco
import glfw
glfw.init(); glfw.window_hint(glfw.VISIBLE, 0)
win = glfw.create_window(1280, 720, "off", None, None)
glfw.make_context_current(win)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_scripts.landmarks import landmark_xml, wall_xml, HF_SURF, BOT_Z
from test_scripts.finish_detect import detect_finish

PROJ = os.path.expanduser("~/workspace/simtrack")
RENDER_MAP = os.path.join(PROJ, "confirmed/track_500_bin.png")
FINISH = (2.5, 47.5)
LM_ASSETS, LM_WORLD = landmark_xml()
WALL_XML = wall_xml()
CAM_XML = '<camera name="bot_cam" pos="0.4 0 0.5" mode="fixed" euler="0 -1.5708 -1.5708"/>'
FINISH_XML = f'<body mocap="true" pos="{FINISH[0]} {FINISH[1]} {HF_SURF + 1.5}"><geom type="sphere" size="1.5" rgba="0.2 1.0 0.2 0.8"/></body>'
xml = f"""<mujoco>
  <compiler angle="radian"/><option timestep="0.005"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset><hfield name="track" size="25.0 25.0 4.0 2.0" file="{RENDER_MAP}"/>
    {LM_ASSETS}
  </asset>
  <worldbody>
    <light pos="25 25 80" dir="0 0 -1" ambient="0.5 0.5 0.55" diffuse="0.9 0.9 0.95"/>
    <light pos="0 25 30" dir="0.3 0 -0.8" diffuse="0.3 0.3 0.35"/>
    <light pos="50 25 30" dir="-0.3 0 -0.8" diffuse="0.3 0.3 0.35"/>
    {FINISH_XML}
    <geom type="hfield" hfield="track" pos="25 25 0.0" rgba="0.55 0.6 0.65 1.0" friction="0 0 0" contype="0" conaffinity="0"/>
    {WALL_XML}
    {LM_WORLD}
    <body name="bot" pos="0 0 {BOT_Z}">
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
fov = float(m.cam_fovy[0])

from PIL import Image
for (bx, by, yaw_deg, note) in [
    (45.0, 47.5, 180, "ch9右端(45,47.5)朝-x，终点在42m外"),
    (25.0, 47.5, 180, "ch9中段(25,47.5)朝-x，终点22m"),
    (10.0, 47.5, 180, "ch9左段(10,47.5)朝-x，终点7.5m"),
    (45.0, 42.5, 90, "ch8右端(45,42.5)朝+y（转弯中，终点不可见预期）"),
]:
    d.qpos[0] = bx; d.qpos[1] = by; d.qpos[2] = math.radians(yaw_deg)
    mujoco.mj_forward(m, d)
    r.update_scene(d, camera="bot_cam")
    img = r.render()
    obs = detect_finish(img, fovy_deg=fov)
    true_d = math.hypot(FINISH[0]-bx, FINISH[1]-by)
    safe = note.encode("unicode_escape").decode()[:40]
    if obs:
        print(f"{safe}: 检测到 bearing={math.degrees(obs[0]):+.1f}° dist={obs[1]:.1f}m (真值{true_d:.1f}m) area={obs[2]:.0f}px")
    else:
        print(f"{safe}: 未检测到 (真值{true_d:.1f}m)")
    Image.fromarray(img).save(f"/tmp/botcam_{bx:.0f}_{by:.0f}.png")
print("帧已存 /tmp/botcam_*.png")
