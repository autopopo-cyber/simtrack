#!/usr/bin/env python3
"""一键地图快照：远程抓 /map → 下载 → 本地渲染彩色 PNG → 自动打开。

输出：项目根的 map_snapshot.png
  白 = free（已探索且可通行）
  黑 = 墙/障碍
  灰 = 未知（还没探索到）
左下角对应地图 origin。

用法（或直接双击根目录 save_map.bat）:
    .venv\\Scripts\\python scripts\\snapshot_map.py
"""
import os
import platform
import subprocess

import numpy as np
import paramiko
from PIL import Image

HOST, USER, PWD = "100.64.63.98", "qin", "1"
REMOTE_DUMP = "/tmp/dump_map.py"
REMOTE_NPZ = "/tmp/map_snapshot.npz"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCAL_NPZ = os.path.join(ROOT, "_map_snapshot.npz")
LOCAL_PNG = os.path.join(ROOT, "map_snapshot.png")

# 远程执行：订阅一次 /map，dump 成 npz（顶格写，避免字符串缩进问题）
DUMP_SRC = '''import numpy as np, rclpy, time, sys
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
class D(Node):
    def __init__(s, p):
        super().__init__("map_dumper"); s.p=p; s.done=False
        s.create_subscription(OccupancyGrid, "/map", s.cb, QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST))
    def cb(s, m):
        if s.done: return
        np.savez(s.p, data=np.array(m.data, dtype=np.int16),
                 width=m.info.width, height=m.info.height,
                 resolution=m.info.resolution,
                 ox=m.info.origin.position.x, oy=m.info.origin.position.y)
        s.done=True
path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/map_snapshot.npz"
rclpy.init(); n = D(path); t0 = time.time()
while not n.done and time.time()-t0 < 10:
    rclpy.spin_once(n, timeout_sec=0.5)
n.destroy_node(); rclpy.shutdown()
print("dumped" if n.done else "FAIL: no map received in 10s")
'''


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PWD, timeout=10)

    def run(cmd, t=30):
        _, o, e = c.exec_command(cmd, timeout=t)
        return o.read().decode() + e.read().decode()

    print("[1/4] 上传 dump 脚本到远程...")
    sftp = c.open_sftp()
    with sftp.open(REMOTE_DUMP, "w") as f:
        f.write(DUMP_SRC)

    print("[2/4] 远程抓取 /map（最多等 10s）...")
    out = run(
        'bash -c "source /opt/ros/jazzy/setup.bash && /usr/bin/python3 %s %s"' % (REMOTE_DUMP, REMOTE_NPZ),
        30,
    )
    last = out.strip().splitlines()[-1] if out.strip() else "(无输出)"
    print("     ", last)
    if "dumped" not in last:
        print("抓取失败，退出。完整输出：\n", out)
        sftp.close(); c.close(); return

    print("[3/4] 下载 npz...")
    sftp.get(REMOTE_NPZ, LOCAL_NPZ)
    sftp.close(); c.close()

    print("[4/4] 本地渲染 PNG...")
    render()


def render():
    z = np.load(LOCAL_NPZ)
    data = z["data"]
    W, H = int(z["width"]), int(z["height"])
    res = float(z["resolution"])
    ox, oy = float(z["ox"]), float(z["oy"])
    grid = data.reshape((H, W))
    # OccupancyGrid: idx = y*width + x, y=0 在 origin（底部）。reshape((H,W)) → grid[y,x]
    img = np.full((H, W, 3), 205, dtype=np.uint8)  # 未知 = 中灰
    img[grid == 0] = [255, 255, 255]                # free = 白
    img[grid == 100] = [30, 30, 30]                 # 墙 = 近黑
    img = img[::-1]                                  # y=0 底部 → 翻转到图像坐标（顶在下）
    im = Image.fromarray(img).resize((W * 3, H * 3), Image.NEAREST)
    im.save(LOCAL_PNG)

    free = int((grid == 0).sum())
    wall = int((grid == 100).sum())
    unk = int((grid == -1).sum())
    known = free + wall
    print(f"     地图 {W}x{H} 像素 ({W*res:.1f}m x {H*res:.1f}m)  origin=({ox:.1f},{oy:.1f})")
    print(f"     free={free}  墙={wall}  未知={unk}  已知区探索率={100*free/max(free+unk,1):.1f}%")
    print("done ->", LOCAL_PNG)

    # 自动用默认图片查看器打开
    if platform.system() == "Windows":
        os.startfile(LOCAL_PNG)  # type: ignore[attr-defined]
    elif platform.system() == "Darwin":
        subprocess.run(["open", LOCAL_PNG])
    else:
        subprocess.run(["xdg-open", LOCAL_PNG])


if __name__ == "__main__":
    main()
