#!/usr/bin/env python3
"""查看 track_clean.png 地图"""
import mujoco, mujoco.viewer, numpy as np, os

MAP = os.path.expanduser("~/workspace/simtrack/confirmed/track_clean.png")
xml = f"""<mujoco>
  <compiler angle="radian"/><option timestep="0.005"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset><hfield name="track" size="50.0 50.0 4.0 2.0" file="{MAP}"/></asset>
  <worldbody>
    <light pos="50 50 80" dir="0 0 -1"/>
    <geom type="hfield" hfield="track" pos="50 50 0.2" rgba="0.25 0.30 0.35 1.0"/>
    <geom type="box" size="10 10 0.2" pos="25 25 -0.1" rgba="0.3 0.3 0.4 0.5"/>
  </worldbody>
</mujoco>"""

m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
    v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.distance = 80; v.cam.elevation = -90; v.cam.azimuth = 0
    print("viewer ready", flush=True)
    while v.is_running():
        mujoco.mj_step(m, d)
        v.sync()
