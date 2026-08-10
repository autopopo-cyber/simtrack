# algo3 萤火导航数值适配（机器狗 + 50m）实现计划

> **For Hermes:** 按本计划逐任务实现，每任务验证后提交。

**Goal:** 把 `test_scripts/algo3_headless.py` 的物理场景数值改成机器狗 + 50m 赛道，
算法逻辑零改动，跑到终点 (2.5, 47.5)。

**Architecture:** 纯常量/几何数值替换。改 6 个常量 + 4 处 XML 几何 + 1 处 qpos 起点。
算法函数（find_gates / astar_to / Mover / milestones）一行不动。

**Tech Stack:** Python 3 + MuJoCo 3.x（`~/mujoco-venv/bin/python`，EGL 离屏渲染）

---

## Task 1: 改世界常量（SCALE / SAFE_R / 速度）

**Objective:** 把世界比例和机器狗运动学常量改对。

**Files:**
- Modify: `test_scripts/algo3_headless.py` 第 28 行（SCALE）、第 30 行附近（SAFE_R）、速度常量

**Step 1: 确认当前行号**

Run: `grep -nE "SCALE =|SAFE_R|SPEED =|SPEED_MAX|YAW_RATE" test_scripts/algo3_headless.py`
Expected: 找到 5 个常量定义行

**Step 2: 修改常量**

```python
SCALE = 1.0                     # 原来是 2.0 (100m → 50m)
SAFE_R = 0.2                    # 原来是 0.5 (机器狗半径 0.4m/2)
SPEED = 4.0                     # 原来是 5.0 (MAX_V=4.0)
SPEED_MAX = 4.0                 # 原来是 8.0 (上限=线速度上限)
YAW_RATE = 1.0                  # 原来是 6.0 (MAX_W=1.0)
```

**Step 3: 验证常量已改**

Run: `grep -nE "SCALE =|SAFE_R =|SPEED =|SPEED_MAX|YAW_RATE" test_scripts/algo3_headless.py`
Expected: 显示新值

**Step 4: Commit**

```bash
git add test_scripts/algo3_headless.py
git commit -m "feat: algo3 headless 数值适配机器狗+50m — 世界/速度常量"
```

---

## Task 2: 改 FINISH 终点坐标

**Objective:** 终点从 100m 的 (3,95) 改为 50m 的 (2.5, 47.5)。

**Files:**
- Modify: `test_scripts/algo3_headless.py` FINISH 定义行（约第 61 行）

**Step 1: 修改**

```python
FINISH = (2.5, 47.5)   # 原来是 (3.0, 95.0) — 50m 终点（主人确认）
```

**Step 2: 验证**

Run: `grep -n "FINISH =" test_scripts/algo3_headless.py`
Expected: `FINISH = (2.5, 47.5)`

**Step 3: Commit**

```bash
git commit -am "feat: algo3 headless 终点适配 50m (2.5, 47.5)"
```

---

## Task 3: 改 XML 场景几何（hfield / light / 机器狗 body）

**Objective:** hfield 缩小到 25m 半宽、light 和 geom 移到 25 位置、机器人换成机器狗圆柱。

**Files:**
- Modify: `test_scripts/algo3_headless.py` build_xml 函数内（约第 296-313 行）

**Step 1: 确认当前 XML**

Run: `sed -n '296,315p' test_scripts/algo3_headless.py`

**Step 2: 修改 4 处**

```xml
<!-- hfield size: 50.0 50.0 → 25.0 25.0 -->
<asset><hfield name="track" size="25.0 25.0 4.0 2.0" file="{MAP}"/></asset>

<!-- light pos: 50 50 80 → 25 25 80 -->
<light pos="25 25 80" dir="0 0 -1"/>

<!-- hfield geom pos: 50 50 0.0 → 25 25 0.0 -->
<geom type="hfield" hfield="track" pos="25 25 0.0" rgba="..." friction="0 0 0"/>

<!-- bot geom cylinder: size="0.5 0.5" → size="0.2 0.4" (r=0.2 半长=0.4) -->
<geom type="cylinder" size="0.2 0.4" rgba="1 0.3 0 1" friction="0 0 0"/>
```

**Step 3: 验证**

Run: `grep -nE "hfield name|light pos|geom type=\"hfield\"|geom type=\"cylinder\"" test_scripts/algo3_headless.py`
Expected: hfield 25.0、light 25 25 80、hfield geom 25 25 0.0、cylinder 0.2 0.4

**Step 4: Commit**

```bash
git commit -am "feat: algo3 headless XML 场景适配 50m + 机器狗几何"
```

---

## Task 4: 改机器人起点 qpos

**Objective:** 起点从 (3,3) 改为 (2.5, 2.5)。

**Files:**
- Modify: `test_scripts/algo3_headless.py` 主循环 qpos 初始化行

**Step 1: 定位**

Run: `grep -n "qpos\[0\].*=.*3; d.qpos\[1\].*=.*3" test_scripts/algo3_headless.py`

**Step 2: 修改**

```python
d.qpos[0]=2.5; d.qpos[1]=2.5; mujoco.mj_forward(m,d)
```

**Step 3: Commit**

```bash
git commit -am "feat: algo3 headless 起点适配 (2.5, 2.5)"
```

---

## Task 5: 用 tshell 跑 3 遍验证

**Objective:** 无 DISPLAY 环境跑通完整导航到终点，3 个不同 seed 稳定。

**Files:**
- 无需改代码，用 tshell 启动

**Step 1: 确认 tshell 活着**

Run: `curl -s -m 5 -H "X-Token: orangepi" http://127.0.0.1:8765/ping`
Expected: `{"pong": true, ...}`

**Step 2: 启动 seed=7 跑**

```bash
curl -s -m 5 -H "X-Token: orangepi" -H "Content-Type: application/json" \
  -d '{"cmd":"cd /home/qin/workspace/simtrack && /home/qin/mujoco-venv/bin/python -u test_scripts/algo3_headless.py --seed 7 --max-steps 300000 --render-every 200 --save-name firefly_dog50_seed7.json"}' \
  http://127.0.0.1:8765/start
```

**Step 3: 轮询直到 done，检查输出**

Run: `curl -s -m 5 -H "X-Token: orangepi" "http://127.0.0.1:8765/jobs/<job_id>"`
Expected: status=done，stdout 显示导航轨迹、到达终点、all_reached=True（或 v3 终点判定）

**Step 4: 换 seed 99、seed 170456 重复**

**Step 5: 汇总 3 遍结果**

确认 3 遍都到达 FINISH=(2.5, 47.5)，无卡死/无超时。

---

## Task 6: 提交稳定版本 + 文档

**Objective:** 3 遍全绿后提交最终代码。

**Files:**
- `test_scripts/algo3_headless.py`（已提交）
- `scans/firefly_dog50_seed*.json`（成绩单）

**Step 1: 确认 git status**

Run: `cd /home/qin/workspace/simtrack && git status --short`

**Step 2: 提交成绩单**

```bash
git add scans/firefly_dog50_seed*.json
git commit -m "test: firefly 机器狗+50m 三遍验证成绩单"
```

**Step 3: 更新设计文档状态（可选）**

在 `docs/superpowers/specs/2026-07-11-algo3-firefly-dog50-design.md` 末尾加验证结果。
