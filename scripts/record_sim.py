#!/usr/bin/env python3
"""录制远程 MuJoCo 仿真画面为 mp4，下载到本地播放。

流程：
  1. 上传内嵌的远程 recorder 到 xiu2
  2. 远程跑：订阅 /odom 拿狗位姿 → EGL 离屏渲染 MuJoCo 俯视画面 → PNG 序列 → ffmpeg 合成 mp4
  3. 下载 mp4 → 自动用默认播放器打开

用法（或双击根目录 record_sim.bat）:
    .venv\\Scripts\\python scripts\\record_sim.py [duration_sec] [fps] [maze]
  默认 60 秒 / 10fps / MAZE 环境变量（无则 rooms5x5）
"""
import os
import platform
import subprocess
import sys

import paramiko

HOST, USER, PWD = "100.64.63.98", "qin", "1"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCAL_MP4 = os.path.join(ROOT, "sim_recording.mp4")
REMOTE_REC = "/tmp/record_sim_remote.py"
REMOTE_MP4 = "/tmp/sim_rec.mp4"

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
FPS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
MAZE = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("MAZE", "rooms5x5")

# ── 远程 recorder（顶格写，避免缩进）──
REMOTE_SRC = r'''import os, sys, math, time, subprocess
import numpy as np
import mujoco
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from PIL import Image

MAZE = os.environ.get("MAZE_R", "rooms5x5")
DURATION = float(os.environ.get("REC_DUR", "60"))
FPS = int(os.environ.get("REC_FPS", "10"))
CAM = os.environ.get("REC_CAM", "top")
OUT = "/tmp/sim_rec.mp4"
FRAMES = "/tmp/sim_frames"
PROJ = os.path.expanduser("~/simtrack")
MAZE_PNG = os.path.join(PROJ, "confirmed", "maze_%s.png" % MAZE)
PX_PER_M = 50


def yaw_from_q(q):
    siny = 2*(q.w*q.z + q.x*q.y)
    cosy = 1 - 2*(q.y*q.y + q.z*q.z)
    return math.atan2(siny, cosy)


def build():
    img = np.array(Image.open(MAZE_PNG))
    h, w = img.shape
    mw, mh = w / PX_PER_M, h / PX_PER_M
    hw, hh = mw / 2, mh / 2
    cam_h = max(mw, mh) * 1.3
    xml = "<mujoco><compiler angle='radian'/>" \
          "<visual><global offwidth='1280' offheight='1280'/></visual>" \
          "<asset>" \
          "<hfield name='maze' size='{hw} {hh} 4.0 2.0' file='{absp}'/></asset>" \
          "<worldbody>" \
          "<light pos='{hw} {hh} 30' dir='0 0 -1' diffuse='0.9 0.9 0.95' ambient='0.5 0.5 0.55'/>" \
          "<geom type='hfield' hfield='maze' pos='{hw} {hh} 0' rgba='0.55 0.6 0.65 1' contype='0' conaffinity='0'/>" \
          "<body name='bot' pos='0 0 0.5'>" \
          "<joint type='slide' axis='1 0 0'/><joint type='slide' axis='0 1 0'/>" \
          "<joint name='yaw' type='hinge' axis='0 0 1'/>" \
          "<geom type='capsule' fromto='-0.4 0 0 0.4 0 0' size='0.2' rgba='1 0.85 0.1 1' contype='0' conaffinity='0'/>" \
          "<site name='head' pos='0.4 0 0' size='0.12' rgba='1 0.3 0.1 1'/>" \
          "</body>" \
          "<camera name='top' pos='{hw} {hh} {ch}' mode='fixed'/>" \
          "<camera name='tilt' pos='{hw} {hy} {ch2}' euler='1.15 0 0' mode='fixed'/>" \
          "</worldbody></mujoco>"
    xml = xml.format(hw=hw, hh=hh, ch=cam_h, ch2=cam_h*0.55,
                     hy=hh - max(mh, 1)*0.35, absp=os.path.abspath(MAZE_PNG))
    m = mujoco.MjModel.from_xml_string(xml)
    return m, mujoco.MjData(m)


class Rec(Node):
    def __init__(self, m, d, rend):
        super().__init__("sim_recorder")
        self.m, self.d, self.rend = m, d, rend
        self.pose = (1.5, 1.5, 0.0)
        self.n = 0
        self.target = int(DURATION * FPS)
        os.makedirs(FRAMES, exist_ok=True)
        for f in os.listdir(FRAMES):
            os.remove(os.path.join(FRAMES, f))
        self.create_subscription(Odometry, "/odom", self._odom, 10)
        self.timer = self.create_timer(1.0 / FPS, self._snap)
        self.get_logger().info("REC %ss @%dfps cam=%s maze=%s -> %s (%d frames)"
                               % (DURATION, FPS, CAM, MAZE, OUT, self.target))

    def _odom(self, msg):
        p = msg.pose.pose.position
        self.pose = (p.x, p.y, yaw_from_q(msg.pose.pose.orientation))

    def _snap(self):
        x, y, yaw = self.pose
        self.d.qpos[0] = x; self.d.qpos[1] = y; self.d.qpos[2] = yaw
        mujoco.mj_forward(self.m, self.d)
        self.rend.update_scene(self.d, camera=CAM)
        pix = self.rend.render()
        Image.fromarray(pix).save(os.path.join(FRAMES, "f%05d.png" % self.n))
        self.n += 1
        if self.n % (FPS * 5) == 0:
            self.get_logger().info("  帧 %d/%d (%.0fs)" % (self.n, self.target, self.n/float(FPS)))
        if self.n >= self.target:
            self.timer.cancel()
            self._encode()

    def _encode(self):
        self.get_logger().info("编码 mp4 ...")
        cmd = ["ffmpeg", "-y", "-framerate", str(FPS), "-i", FRAMES + "/f%05d.png",
               "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", OUT]
        r = subprocess.run(cmd, capture_output=True)
        print("ENCODED", OUT, os.path.getsize(OUT) if os.path.exists(OUT) else "FAIL",
              r.returncode)
        raise SystemExit


def main():
    m, d = build()
    rend = mujoco.Renderer(m, 640, 640)
    rclpy.init()
    node = Rec(m, d, rend)
    rclpy.spin(node)
    node.destroy_node(); rclpy.shutdown()

main()
'''


def main():
    print("录制参数: %ss @%dfps  maze=%s" % (DURATION, FPS, MAZE))
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PWD, timeout=10)
    sftp = c.open_sftp()
    with sftp.open(REMOTE_REC, "w") as f:
        f.write(REMOTE_SRC)
    print("[1/3] 已上传 recorder，开始远程录制（会阻塞 %ss，机器狗必须在动）..." % int(DURATION))

    env = "source /opt/ros/jazzy/setup.bash; export MUJOCO_GL=egl; " \
          "export MAZE_R=%s; export REC_DUR=%s; export REC_FPS=%s; export REC_CAM=top; " \
          "/usr/bin/python3 %s" % (MAZE, DURATION, FPS, REMOTE_REC)
    # 阻塞运行，实时打印输出
    cmd = 'bash -c "%s"' % env.replace('"', '\\"')
    stdin, stdout, stderr = c.exec_command(cmd, timeout=DURATION + 60)
    for line in iter(stdout.readline, ""):
        print("  ", line.rstrip())
    err = stderr.read().decode()
    if err.strip():
        print("  [stderr]", err[:500])

    print("[2/3] 下载 mp4...")
    try:
        sftp.get(REMOTE_MP4, LOCAL_MP4)
    except Exception as e:
        print("下载失败：", e)
        sftp.close(); c.close(); return
    sftp.close(); c.close()
    print("[3/3] done ->", LOCAL_MP4, "(%d MB)" % (os.path.getsize(LOCAL_MP4) // 1024 // 1024))
    if platform.system() == "Windows":
        os.startfile(LOCAL_MP4)  # type: ignore[attr-defined]
    elif platform.system() == "Darwin":
        subprocess.run(["open", LOCAL_MP4])
    else:
        subprocess.run(["xdg-open", LOCAL_MP4])


if __name__ == "__main__":
    main()
