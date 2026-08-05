#!/usr/bin/env python3
"""机器狗模拟（水平圆柱）— 蛇形赛道全向运动测试

水平圆柱: 0.8m 长 × 0.4m 直径, 离地 0.5m, 全向 3-DOF (slide x/y + hinge yaw)
速度上限: 线 4 m/s, 角 1 rad/s
测试序列: 前进5m → 后退5m → 左移3m → 右移3m → 原地旋转360° → 斜向移动
"""
import os, sys, time, math, json, argparse
import numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco

# ── 常量 ──
MAX_V = 4.0   # 最大线速度 m/s
MAX_W = 1.0   # 最大角速度 rad/s
SIM_DT = 0.008
R_DOG = 0.2          # 圆柱半径 0.4m/2
L_DOG = 0.4          # 圆柱半长 0.8m/2
H_DOG = 0.5          # 离地高度
START = (5.0, 45.0)  # 赛道起点附近
TRACK_PNG = "/tmp/track_hd.png"
SCAN_DIR = os.path.expanduser("~/workspace/simtrack/scans")

# ── 运动学核心 ──
def clamp_cmd(vx, vy, w):
    """机体坐标指令 clamp：合速度≤MAX_V，角速度≤MAX_W"""
    v_mag = math.hypot(vx, vy)
    if v_mag > MAX_V:
        vx *= MAX_V / v_mag
        vy *= MAX_V / v_mag
    w = max(-MAX_W, min(MAX_W, w))
    return vx, vy, w

def body_to_world(vx, vy, yaw):
    """机体坐标 → 世界坐标（全向：侧移不依赖朝向）"""
    wx = vx * math.cos(yaw) - vy * math.sin(yaw)
    wy = vx * math.sin(yaw) + vy * math.cos(yaw)
    return wx, wy

# ── 场景 ──
def build_scene(track_png):
    return f"""<mujoco>
  <compiler angle="radian"/>
  <option timestep="{SIM_DT}"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset>
    <hfield name="track" size="25 25 6 3" file="{track_png}"/>
    <material name="v" rgba="0.25 0.30 0.35 1"/>
    <material name="i" rgba="0.25 0.30 0.35 0"/>
  </asset>
  <worldbody>
    <light pos="25 25 80" dir="0 0 -1" diffuse="1.5 1.5 1.5" specular="0.5 0.5 0.5"/>
    <geom type="hfield" hfield="track" pos="25 25 0" material="v"/>
    <geom type="plane" size="0 0 0.05" material="i"/>
    <body name="dog" pos="{START[0]} {START[1]} {H_DOG}">
      <inertial pos="0 0 0" mass="10" diaginertia="0.5 0.5 0.1"/>
      <joint name="x" type="slide" axis="1 0 0" damping="0"/>
      <joint name="y" type="slide" axis="0 1 0" damping="0"/>
      <joint name="yaw" type="hinge" axis="0 0 1" damping="0"/>
      <geom type="cylinder" size="{R_DOG} {L_DOG}" euler="0 1.5707963 0" rgba="0.2 0.8 0.2 0.9"/>
    </body>
  </worldbody>
</mujoco>"""

# ── 运动学测试序列 ──
def run_test_sequence(m, d, out_dir, render_every, renderer, max_steps):
    """6 段运动学测试: 前进5m → 后退5m → 左移3m → 右移3m → 旋转360° → 斜向"""
    segments = [
        ("forward",  (MAX_V, 0, 0),      5.0),
        ("backward", (-MAX_V, 0, 0),     5.0),
        ("left",     (0, MAX_V, 0),      3.0),
        ("right",    (0, -MAX_V, 0),     3.0),
        ("spin",     (0, 0, MAX_W),      2 * math.pi),
        ("diag",     (MAX_V, MAX_V, 0),  3.0),
    ]
    stats = []
    cnt = 0
    t0 = time.time()
    for name, cmd, target in segments:
        seg_start = time.time()
        start_pos = d.qpos[0:2].copy()
        start_yaw = d.qpos[2]
        max_speed = 0.0
        max_w = 0.0
        steps = 0
        reached = False
        while time.time() - seg_start < 60 and cnt < max_steps:  # 每段最多60s
            bx, by, yaw = d.qpos[0], d.qpos[1], d.qpos[2]
            vx_c, vy_c, w_c = clamp_cmd(*cmd)
            wx, wy = body_to_world(vx_c, vy_c, yaw)
            d.qvel[0] = wx; d.qvel[1] = wy; d.qvel[2] = w_c
            mujoco.mj_step(m, d); cnt += 1; steps += 1
            speed = math.hypot(d.qvel[0], d.qvel[1])
            max_speed = max(max_speed, speed)
            max_w = max(max_w, abs(d.qvel[2]))
            # 进度判定
            if name == "spin":
                progress = abs(((d.qpos[2] - start_yaw) % (2*math.pi)))
                if progress >= target: reached = True; break
            else:
                dx = d.qpos[0] - start_pos[0]; dy = d.qpos[1] - start_pos[1]
                if name in ("forward", "backward"):
                    progress = abs(dx * math.cos(start_yaw) + dy * math.sin(start_yaw))
                elif name in ("left", "right"):
                    progress = abs(-dx * math.sin(start_yaw) + dy * math.cos(start_yaw))
                else:  # diag
                    progress = math.hypot(dx, dy)
                if progress >= target: reached = True; break
            if render_every and cnt % render_every == 0 and renderer is not None:
                render_frame(renderer, d, out_dir, cnt)
        stats.append({
            "segment": name, "cmd": list(cmd), "reached": reached,
            "steps": steps, "max_speed": round(max_speed, 3),
            "max_w": round(max_w, 3),
            "pos": [round(d.qpos[0], 3), round(d.qpos[1], 3)],
            "yaw": round(d.qpos[2], 3),
            "time": round(time.time() - seg_start, 2),
        })
        print(f"[{name}] reached={reached} max_v={max_speed:.2f} "
              f"pos=({d.qpos[0]:.2f},{d.qpos[1]:.2f}) yaw={d.qpos[2]:.2f}", flush=True)
    return stats, cnt

# ── 渲染 ──
def render_frame(renderer, d, out_dir, cnt):
    try:
        renderer.update_scene(d, camera=-1)
        img = renderer.render()
        from PIL import Image
        Image.fromarray(img).save(os.path.join(out_dir, f"frame_{cnt:06d}.png"))
    except Exception:
        pass

def init_renderer(m):
    try:
        from mujoco import egl
        _ctx = egl.GLContext(1280, 720)
        _ctx.make_current()
        renderer = mujoco.Renderer(m, 720, 1280)
        print("  [RENDER] EGL 离屏渲染 OK", flush=True)
        return renderer
    except Exception as e:
        print(f"  [RENDER] 离屏渲染不可用: {e}", flush=True)
        return None

# ── 成绩单 ──
def save_stats(stats, total_steps, elapsed, out_json):
    data = {
        "robot": "horizontal_cylinder", "size": [0.8, 0.4], "height": 0.5,
        "max_v": MAX_V, "max_w": MAX_W,
        "segments": stats,
        "total_steps": total_steps, "time_sec": round(elapsed, 2),
        "all_reached": all(s["reached"] for s in stats),
        "max_speed_violated": any(s["max_speed"] > MAX_V + 0.01 for s in stats),
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {out_json}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=200000)
    ap.add_argument("--render-every", type=int, default=0, help="每N步渲染一帧，0=不渲染")
    ap.add_argument("--out-dir", type=str, default="/tmp/dog_frames")
    ap.add_argument("--save-name", type=str, default="", help="成绩单文件名（默认 dog_baseline_seed<N>.json）")
    args = ap.parse_args()

    print(f"=== 机器狗(水平圆柱) headless start: v_max={MAX_V} w_max={MAX_W} ===", flush=True)
    if not os.path.exists(TRACK_PNG):
        print(f"[ERROR] 赛道文件不存在: {TRACK_PNG}（先跑 trackgen_hd.py）", flush=True)
        sys.exit(1)

    xml = build_scene(TRACK_PNG)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    d.qpos[0] = START[0]; d.qpos[1] = START[1]
    mujoco.mj_forward(m, d)

    os.makedirs(args.out_dir, exist_ok=True)
    renderer = init_renderer(m)

    t0 = time.time()
    stats, total_steps = run_test_sequence(m, d, args.out_dir, args.render_every,
                                            renderer, args.max_steps)
    elapsed = time.time() - t0

    out_json = os.path.join(SCAN_DIR, args.save_name) if args.save_name else \
        os.path.join(SCAN_DIR, "dog_baseline.json")
    save_stats(stats, total_steps, elapsed, out_json)

    all_ok = all(s["reached"] for s in stats) and \
             not any(s["max_speed"] > MAX_V + 0.01 for s in stats)
    print(f"=== DONE sim={total_steps*SIM_DT:.1f}s wall={elapsed:.0f}s "
          f"all_reached={all_ok} ===", flush=True)
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
