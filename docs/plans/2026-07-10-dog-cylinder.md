# 机器狗模拟（水平圆柱）· 实现计划

> **For Hermes:** 按此计划逐任务实现，每任务完成后提交。

**Goal:** 用 MuJoCo 水平圆柱（0.8m×0.4m，离地 0.5m）模拟机器狗，在蛇形赛道（track_hd）上验证全向运动（前进/后退/平移/旋转），速度上限线 4 m/s、角 1 rad/s。

**Architecture:** 单文件 headless 脚本 `algo_dog_headless.py`，复用 simtrack 现有模式（trackgen_hd 赛道 + 帧渲染 + 成绩单 json）。先做运动学测试段（前进/后退/平移/旋转），验证速度 clamp 与绕中心旋转。

**Tech Stack:** Python 3.12（mujoco-venv）、MuJoCo 3.8.0、numpy 2.4.4、cv2 4.13.0

---

### Task 1: 创建运动学核心 — 速度 clamp + 全向换算

**Objective:** 实现全向运动学：机体坐标 (vx,vy,w) → 世界坐标，带速度上限 clamp。

**Files:**
- Create: `test_scripts/algo_dog_headless.py`

**Step 1: 写核心函数**

```python
#!/usr/bin/env python3
"""机器狗模拟（水平圆柱）— 蛇形赛道全向运动测试"""
import os, sys, time, math, json, argparse
import numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco

MAX_V = 4.0   # 最大线速度 m/s
MAX_W = 1.0   # 最大角速度 rad/s

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
```

**Step 2: 验证**

Run: `/home/qin/mujoco-venv/bin/python -c "import sys; sys.path.insert(0,'test_scripts'); import algo_dog_headless as a; print(a.clamp_cmd(5,0,2)); print(a.body_to_world(1,0,math.pi/2))"`
Expected: `(4.0, 0.0, 1.0)` 和 `(0.0, 1.0)`

**Step 3: Commit**
```bash
cd /home/qin/workspace/simtrack && git add test_scripts/algo_dog_headless.py && git commit -m "feat: 机器狗水平圆柱 — 运动学核心 (clamp + 全向换算)"
```

---

### Task 2: MuJoCo 场景 — 水平圆柱 + 蛇形赛道

**Objective:** 构建场景 XML：水平圆柱（0.8m×0.4m，z=0.5m，3-DOF）+ hfield 蛇形赛道。

**Files:**
- Modify: `test_scripts/algo_dog_headless.py`

**Step 1: 加场景构建**

```python
SIM_DT = 0.008
R_DOG = 0.2          # 圆柱半径 0.4m/2
L_DOG = 0.4          # 圆柱半长 0.8m/2
H_DOG = 0.5          # 离地高度

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
    <body name="dog" pos="5 45 {H_DOG}">
      <inertial pos="0 0 0" mass="10" diaginertia="0.5 0.5 0.1"/>
      <joint name="x" type="slide" axis="1 0 0" damping="0"/>
      <joint name="y" type="slide" axis="0 1 0" damping="0"/>
      <joint name="yaw" type="hinge" axis="0 0 1" damping="0"/>
      <geom type="cylinder" size="{R_DOG} {L_DOG}" euler="0 1.5707963 0" rgba="0.2 0.8 0.2 0.9"/>
    </body>
  </worldbody>
</mujoco>"""
```

注意：`euler="0 1.5707963 0"` 让圆柱从竖直转 90° 水平（绕 y 轴旋转），长度沿 x 轴。

**Step 2: 验证**

Run: `/home/qin/mujoco-venv/bin/python -c "import sys; sys.path.insert(0,'test_scripts'); import algo_dog_headless as a; xml=a.build_scene('/tmp/track_hd.png'); open('/tmp/dog_test.xml','w').write(xml); import mujoco; m=mujoco.MjModel.from_xml_path('/tmp/dog_test.xml'); print('model OK, nq=',m.nq)"`
Expected: `model OK, nq=3`

**Step 3: Commit**
```bash
git add test_scripts/algo_dog_headless.py && git commit -m "feat: 机器狗水平圆柱 — MuJoCo 场景 (水平圆柱 z=0.5 + 蛇形赛道)"
```

---

### Task 3: 运动学测试序列 — 前进/后退/平移/旋转

**Objective:** 顺序执行 6 段运动学测试，每段记录位移/朝向/速度，断言不超限。

**Files:**
- Modify: `test_scripts/algo_dog_headless.py`

**Step 1: 加测试序列**

```python
def run_test_sequence(m, d, out_dir, render_every):
    """6 段运动学测试: 前进5m → 后退5m → 左移3m → 右移3m → 旋转360° → 斜向"""
    segments = [
        ("forward",  (MAX_V, 0, 0),     5.0,  None),
        ("backward", (-MAX_V, 0, 0),    5.0,  None),
        ("left",     (0, MAX_V, 0),     3.0,  None),
        ("right",    (0, -MAX_V, 0),    3.0,  None),
        ("spin",     (0, 0, MAX_W),     2*math.pi, None),
        ("diag",     (MAX_V, MAX_V, 0), 3.0,  None),
    ]
    stats = []
    cnt = 0
    t0 = time.time()
    for name, cmd, target, _ in segments:
        seg_start = time.time()
        start_pos = d.qpos[0:2].copy()
        start_yaw = d.qpos[2]
        max_speed = 0.0
        steps = 0
        reached = False
        while time.time() - seg_start < 60:  # 每段最多60s
            bx, by, yaw = d.qpos[0], d.qpos[1], d.qpos[2]
            vx_c, vy_c, w_c = clamp_cmd(*cmd)
            wx, wy = body_to_world(vx_c, vy_c, yaw)
            d.qvel[0] = wx; d.qvel[1] = wy; d.qvel[2] = w_c
            mujoco.mj_step(m, d); cnt += 1; steps += 1
            speed = math.hypot(d.qvel[0], d.qvel[1])
            max_speed = max(max_speed, speed)
            # 进度判定
            if name == "spin":
                progress = abs(d.qpos[2] - start_yaw) % (2*math.pi)
                if progress >= target: reached = True; break
            else:
                dx = d.qpos[0]-start_pos[0]; dy = d.qpos[1]-start_pos[1]
                if name in ("forward","backward"):
                    progress = abs(dx*math.cos(start_yaw)+dy*math.sin(start_yaw))
                elif name in ("left","right"):
                    progress = abs(-dx*math.sin(start_yaw)+dy*math.cos(start_yaw))
                else:  # diag
                    progress = math.hypot(dx, dy)
                if progress >= target: reached = True; break
            if render_every and cnt % render_every == 0:
                render_frame(m, d, out_dir, cnt)
        stats.append({
            "segment": name, "cmd": list(cmd), "reached": reached,
            "steps": steps, "max_speed": round(max_speed, 3),
            "pos": d.qpos[0:3].tolist(), "yaw": round(d.qpos[2], 3),
            "time": round(time.time()-seg_start, 2),
        })
        print(f"[{name}] reached={reached} max_v={max_speed:.2f} pos=({d.qpos[0]:.2f},{d.qpos[1]:.2f})", flush=True)
    return stats, cnt
```

**Step 2: 加渲染函数 + main**

```python
def render_frame(m, d, out_dir, cnt):
    img = np.empty((720, 1280, 3), dtype=np.uint8)
    mujoco.mj_render(m, d, 0, 0, 1280, 720)  # offscreen
    # 实际用 mjr 渲染（见 algo3_headless 模式）
    ...
```

渲染复用 algo3_headless.py 的模式（读该文件 420-445 行的 offscreen 渲染）。

**Step 3: 验证**

Run: `/home/qin/mujoco-venv/bin/python test_scripts/algo_dog_headless.py --max-steps 60000 --render-every 0 --out-dir /tmp/dog_test1 --save-name dog_baseline.json`
Expected: 6 段全部 `reached=True`，`max_speed≤4.0`，json 写入 `scans/dog_baseline.json`

**Step 4: Commit**
```bash
git add test_scripts/algo_dog_headless.py && git commit -m "feat: 机器狗 — 6段运动学测试序列 (前进/后退/平移/旋转/斜向)"
```

---

### Task 4: 集成 tshell 跑测试 + 成绩单

**Objective:** 用 tshell 起 job 跑 headless 仿真，成绩单存 scans/，面板看进度。

**Files:**
- Modify: `test_scripts/algo_dog_headless.py`（加成绩单保存）

**Step 1: 成绩单保存**

```python
def save_stats(stats, total_steps, elapsed, out_json):
    data = {
        "robot": "horizontal_cylinder", "size": [0.8, 0.4], "height": 0.5,
        "max_v": MAX_V, "max_w": MAX_W,
        "segments": stats,
        "total_steps": total_steps, "time_sec": round(elapsed, 2),
        "all_reached": all(s["reached"] for s in stats),
        "max_speed_violated": any(s["max_speed"] > MAX_V + 0.01 for s in stats),
    }
    with open(out_json, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {out_json}", flush=True)
```

**Step 2: tshell 跑**

```bash
TSHELL_PORT=8765 python3 /home/qin/workspace/tshell/scripts/tshell.py start \
  "/home/qin/mujoco-venv/bin/python /home/qin/workspace/simtrack/test_scripts/algo_dog_headless.py --max-steps 100000 --render-every 1000 --out-dir /tmp/dog_run1 --save-name dog_baseline_seed7.json"
```

**Step 3: 验证**
- `jobs` 看状态，`tail` 拉输出，`panel` 开浏览器
- 完成后 `scans/dog_baseline_seed7.json` 存在且 `all_reached=true`

**Step 4: Commit**
```bash
git add test_scripts/algo_dog_headless.py && git commit -m "feat: 机器狗 — 成绩单 + tshell 集成"
```

---

## 验收标准

1. 6 段运动学测试全部 `reached=true`
2. 所有段 `max_speed ≤ 4.0`（速度 clamp 生效）
3. 旋转段质心几乎不动（绕中心旋转验证）
4. 成绩单 json 存 `scans/dog_baseline_seed7.json`
5. 用 tshell 起 job 能跑、面板能看到进度
