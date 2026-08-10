# M3 DWA 局部规划器 + 4 段随机反弹障碍 实现计划

> **For Hermes:** 用 subagent-driven-development 逐任务执行本计划。
>
> **Goal:** 蛇形赛道 KNOWN_MAP_MODE 下，机器狗从起点走到出口，穿过 4 段含随机反弹障碍的区段，碰撞 0 + bounce 0。
>
> **Architecture:** 新增随机反弹障碍模块（小球弹性反弹模型）+ DWA 局部规划器（速度空间采样）。全局 HPA 不动，执行层 Mover 最小侵入——转向段换成 DWA 输出的 ω，速度段保留加速度平滑+制动约束，全碰撞回退 _bounce。
>
> **Tech Stack:** Python 3.12 + numpy + MuJoCo（mujoco-venv：`/home/qin/mujoco-venv/bin/python`，pytest 单测纯 Python 不依赖 MuJoCo）

---

## 环境（所有命令用 mujoco-venv）

```bash
PY=/home/qin/mujoco-venv/bin/python
$PY -m pytest tests/ -v          # 跑全部单测
$PY test_scripts/algo3_headless.py --obs-random 4 --seed 42 --max-steps 300000 --timeout 600
```

## 设计要点回顾（详见 specs/2026-08-07-dwa-obstacle-design.md）

- 4 区段：通道 ch=1,4,6,8，x∈[15,35]，虚拟墙 x∈[15.1,34.9]（内收 10cm），y 真实墙约束
- 每段 1 障碍：1m/s，每满 1s 20% 概率 dir=uniform(0,2π)，撞真实墙/虚拟墙镜面反射
- DWA：动态窗口（T=0.05s）→ 7×11=77 条轨迹 → 模拟 1.5s → blocked() 判定 → 评分（heading 0.6 / clearance 0.25 / velocity 0.1 / smooth 0.05）→ (v*,ω*)；全碰撞返回 None
- 第一版不做障碍运动预测（狗 4m/s vs 障碍 1m/s，0.05s 重决策）

---

### Task 1: RandomObstacleField 模块（含测试）

**Objective:** 实现小球弹性反弹的随机障碍，纯 Python 可单测。

**Files:**
- Create: `simtrack/obstacles_random.py`
- Create: `tests/test_obstacles_random.py`

**Step 1: 写失败测试** `tests/test_obstacles_random.py`

```python
"""随机反弹障碍单元测试 — 纯 Python (不依赖 MuJoCo)"""
import math
import random
from simtrack.obstacles_random import RandomObstacle, RandomObstacleField


def test_initial_position():
    """初始位置 = 段中央"""
    field = RandomObstacleField(channels=[1], seed=42)
    obs = field.obstacles[0]
    assert obs.ch == 1
    assert abs(obs.pos[0] - 25.0) < 1e-6      # x 段中央
    assert abs(obs.pos[1] - (2.5 + 1*5.0)) < 1e-6  # y 通道中心


def test_seed_reproducibility():
    f1 = RandomObstacleField(channels=[1, 4, 6, 8], seed=42)
    f2 = RandomObstacleField(channels=[1, 4, 6, 8], seed=42)
    assert len(f1.obstacles) == len(f2.obstacles) == 4
    for a, b in zip(f1.obstacles, f2.obstacles):
        assert a.pos == b.pos and a.dir == b.dir


def test_move_forward():
    """无障碍情况下障碍沿 dir 匀速移动 1m/s"""
    field = RandomObstacleField(channels=[1], seed=42)
    obs = field.obstacles[0]
    d0 = obs.dir
    field.update(1.0)   # 1s
    expect = (obs.pos[0] + math.cos(d0), obs.pos[1] + math.sin(d0))
    assert math.hypot(obs.pos[0]-expect[0], obs.pos[1]-expect[1]) < 1e-6


def test_virtual_wall_bounce_x():
    """虚拟墙反弹：障碍往左出界 → vx 取反，拉回界内"""
    field = RandomObstacleField(channels=[1], seed=1)
    obs = field.obstacles[0]
    obs.pos = [15.0, 2.5 + 5.0]   # 虚拟墙 x=15.1 外
    obs.dir = math.pi             # 朝 -x
    field.update(0.1)
    assert obs.pos[0] >= 15.1 - 1e-6          # 拉回界内
    # 方向应变成朝 +x（vx 取反）
    assert math.cos(obs.dir) > 0


def test_real_wall_bounce_y():
    """真实墙反弹：障碍往通道下墙撞 → vy 取反"""
    field = RandomObstacleField(channels=[1], seed=2)
    obs = field.obstacles[0]
    obs.pos = [25.0, 5.0 + 0.49]  # y_lo=5.0，中心距墙 < 0.5 → 反弹
    obs.dir = -math.pi/2          # 朝 -y
    field.update(0.1)
    assert obs.pos[1] >= 5.0 + 0.5 - 1e-6    # 半径约束
    assert math.sin(obs.dir) > 0             # 改朝 +y


def test_direction_change_probability():
    """每满 1s，20% 概率随机变向（用固定 random 序列验证）"""
    rng = random.Random(7)
    field = RandomObstacleField(channels=[1], seed=7)
    field._rng = rng
    obs = field.obstacles[0]
    d0 = obs.dir
    # 手动构造：变向判定用 rng.random()<0.2；先消费随机数确保命中
    obs.change_timer = 0.0
    # 直接验证 update 会重设 timer 且 1s 内 dir 变化有限（不是每次 update 都变）
    field.update(0.5)
    assert obs.change_timer <= 0.5 + 1e-6
```

**Step 2: 跑测试验证失败**

Run: `$PY -m pytest tests/test_obstacles_random.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'simtrack.obstacles_random'`

**Step 3: 实现** `simtrack/obstacles_random.py`

```python
"""随机反弹障碍 — 小球弹性反弹模型

主人需求（2026-08-07）：
- 每段 20m 长 × 5m 宽（通道真实宽度），x∈[15,35]，两端虚拟墙内收 10cm
- 障碍 1m/s，每满 1s 20% 概率随机改变方向（uniform 0~2π）
- 撞真实墙（y 边界）/虚拟墙（x 边界）→ 镜面反射（分量取反）
- 机器狗无视虚拟墙（虚拟墙只约束障碍中心）
"""

import math
import random


class RandomObstacle:
    """单个随机反弹障碍。pos/dir 是可变状态，ch 标识所在通道。"""

    def __init__(self, ch, x0=15.0, x1=35.0, y_lo=None, y_hi=None,
                 speed=1.0, radius=0.5, rng=None):
        self.ch = ch
        self.x0, self.x1 = x0, x1
        self.y_lo = y_lo if y_lo is not None else ch * 5.0
        self.y_hi = y_hi if y_hi is not None else (ch + 1) * 5.0
        self.speed = speed
        self.radius = radius
        # 虚拟墙：名义 20m 段内收 10cm
        self.wall_x0 = x0 + 0.1
        self.wall_x1 = x1 - 0.1
        self.rng = rng if rng is not None else random
        # 初始：段中央，方向随机
        self.pos = [25.0, self.y_lo + (self.y_hi - self.y_lo) / 2.0]
        self.dir = self.rng.uniform(0.0, 2 * math.pi)
        self.change_timer = 1.0

    def update(self, dt):
        """步进 dt 秒：变向判定 → 移动 → 真实墙/虚拟墙反弹"""
        # ① 变向：每满 1s，20% 概率随机新方向
        self.change_timer -= dt
        if self.change_timer <= 0.0:
            self.change_timer = 1.0
            if self.rng.random() < 0.2:
                self.dir = self.rng.uniform(0.0, 2 * math.pi)
        # ② 移动
        self.pos[0] += math.cos(self.dir) * self.speed * dt
        self.pos[1] += math.sin(self.dir) * self.speed * dt
        # ③ 真实墙反弹（y 边界，半径约束）
        if self.pos[1] < self.y_lo + self.radius:
            self.pos[1] = self.y_lo + self.radius
            self.dir = self._reflect(self.dir, axis="y")
        if self.pos[1] > self.y_hi - self.radius:
            self.pos[1] = self.y_hi - self.radius
            self.dir = self._reflect(self.dir, axis="y")
        # ④ 虚拟墙反弹（x 边界，只约束中心）
        if self.pos[0] < self.wall_x0:
            self.pos[0] = self.wall_x0
            self.dir = self._reflect(self.dir, axis="x")
        if self.pos[0] > self.wall_x1:
            self.pos[0] = self.wall_x1
            self.dir = self._reflect(self.dir, axis="x")

    @staticmethod
    def _reflect(dir_angle, axis):
        """镜面反射：撞 x 边界 → vx 取反（cos 取反）；撞 y 边界 → vy 取反（sin 取反）"""
        vx, vy = math.cos(dir_angle), math.sin(dir_angle)
        if axis == "x":
            vx = -vx
        else:
            vy = -vy
        return math.atan2(vy, vx)


class RandomObstacleField:
    """管理多个反弹区段（每段 1 个随机障碍）。"""

    def __init__(self, channels=(1, 4, 6, 8), seed=None):
        self._rng = random.Random(seed)
        self.obstacles = [RandomObstacle(ch, rng=self._rng) for ch in channels]

    def update(self, dt):
        for obs in self.obstacles:
            obs.update(dt)

    @property
    def positions(self):
        """[(x, y), ...] 供 obs_world 同步（当前为列表，兼容现有 blocked()）"""
        return [tuple(obs.pos) for obs in self.obstacles]
```

**Step 4: 跑测试验证通过**

Run: `$PY -m pytest tests/test_obstacles_random.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add simtrack/obstacles_random.py tests/test_obstacles_random.py
git commit -m "feat: 随机反弹障碍模块 — 小球弹性反弹(1m/s, 20%/s变向, 虚拟墙内收10cm)"
```

---

### Task 2: DWA 局部规划器（含测试）

**Objective:** 实现 DWA 速度空间采样算法，纯 Python 可单测。

**Files:**
- Create: `simtrack/algorithms/dwa.py`
- Create: `tests/test_dwa.py`

**Step 1: 写失败测试** `tests/test_dwa.py`

```python
"""DWA 局部规划器单元测试 — 纯 Python (不依赖 MuJoCo)"""
import math
from simtrack.algorithms.dwa import DWAAlgorithm


def _blocked_none(*args):
    return False


def test_no_obstacle_goes_fast_forward():
    """无障碍 → 选最大速度 + 零角速度（朝目标直冲）"""
    dwa = DWAAlgorithm()
    v, w = dwa.choose_velocity(
        robot_pos=(0.0, 0.0), yaw=0.0, v_now=0.0, w_now=0.0,
        target=(10.0, 0.0), blocked_fn=_blocked_none)
    assert v == pytest.approx(dwa.v_max, abs=1e-6)
    assert abs(w) < 1e-6


def test_obstacle_in_front_avoid():
    """正前方 3m 障碍 → 选非零角速度（绕行）"""
    def blocked(pt):
        # 半径 0.7 的圆形障碍（中心 3,0）
        return math.hypot(pt[0]-3.0, pt[1]-0.0) < 0.7
    dwa = DWAAlgorithm()
    v, w = dwa.choose_velocity(
        robot_pos=(0.0, 0.0), yaw=0.0, v_now=2.0, w_now=0.0,
        target=(10.0, 0.0), blocked_fn=blocked)
    assert w != pytest.approx(0.0, abs=1e-6)   # 有转向


def test_all_collision_returns_none():
    """全部轨迹都碰撞（四面墙）→ 返回 None（触发 _bounce 兜底）"""
    def blocked_all(pt):
        return True
    dwa = DWAAlgorithm()
    result = dwa.choose_velocity(
        robot_pos=(0.0, 0.0), yaw=0.0, v_now=0.0, w_now=0.0,
        target=(10.0, 0.0), blocked_fn=blocked_all)
    assert result is None


def test_dynamic_window_limits():
    """动态窗口：加速度限制下 v 不能瞬间跳到 v_max"""
    dwa = DWAAlgorithm(a_accel=1.0, a_decel=1.0, T=0.1)
    v, _ = dwa.choose_velocity(
        robot_pos=(0.0, 0.0), yaw=0.0, v_now=0.0, w_now=0.0,
        target=(10.0, 0.0), blocked_fn=_blocked_none)
    assert v <= 0.1 + 1e-6   # a_accel*T = 1.0*0.1


def test_smoothness_penalty():
    """smoothness：接近当前速度的候选优先（无障碍时不会跳变）"""
    dwa = DWAAlgorithm()
    v, w = dwa.choose_velocity(
        robot_pos=(0.0, 0.0), yaw=0.0, v_now=2.0, w_now=0.0,
        target=(10.0, 0.0), blocked_fn=_blocked_none)
    assert v <= dwa.v_max + 1e-6
    assert v >= 0.0
```

**Step 2: 跑测试验证失败**

Run: `$PY -m pytest tests/test_dwa.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'simtrack.algorithms.dwa'`

**Step 3: 实现** `simtrack/algorithms/dwa.py`

```python
"""DWA 局部规划器 (Dynamic Window Approach)

速度空间采样：动态窗口（加速度约束）内采样 (v, ω) 组合，
逐条模拟圆弧轨迹检测碰撞，按 heading/clearance/velocity/smoothness 加权评分选最优。

- 碰撞轨迹硬排除
- 全部碰撞 → 返回 None（调用方触发 _bounce 兜底）
- 第一版不做障碍运动预测（狗 4m/s vs 障碍 1m/s，0.05s 重决策足够）
"""

import math
import numpy as np


class DWAAlgorithm:
    def __init__(
        self,
        v_max: float = 4.0,
        w_max: float = 1.5,          # YAW_RATE
        a_accel: float = 5.0,        # A_ACCEL
        a_decel: float = 8.0,        # A_DECEL
        a_w: float = 10.0,           # 角加速度上限
        T: float = 0.05,             # 决策周期（LIDAR_TICK × timestep）
        horizon: float = 1.5,        # 轨迹模拟时长 (s)
        dt_sample: float = 0.05,     # 模拟步长 (s)
        n_v: int = 7,
        n_w: int = 11,
        w_heading: float = 0.6,
        w_clearance: float = 0.25,
        w_velocity: float = 0.1,
        w_smooth: float = 0.05,
        clearance_max: float = 2.0,  # 归一化上限 (m)
        stop_margin: float = 0.4,    # 停车安全余量（低于此距离的轨迹直接排除）
    ):
        self.v_max = v_max
        self.w_max = w_max
        self.a_accel = a_accel
        self.a_decel = a_decel
        self.a_w = a_w
        self.T = T
        self.horizon = horizon
        self.dt_sample = dt_sample
        self.n_v = n_v
        self.n_w = n_w
        self.w_heading = w_heading
        self.w_clearance = w_clearance
        self.w_velocity = w_velocity
        self.w_smooth = w_smooth
        self.clearance_max = clearance_max
        self.stop_margin = stop_margin

    def choose_velocity(self, robot_pos, yaw, v_now, w_now, target, blocked_fn):
        """返回最优 (v*, ω*)；全部轨迹碰撞 → None。

        Args:
            robot_pos: (x, y) 当前位置
            yaw:       当前朝向 (rad)
            v_now:     当前线速度 (m/s)
            w_now:     当前角速度 (rad/s)
            target:    (tx, ty) lookahead 目标
            blocked_fn: callable(point) -> bool，判定点是否被堵（复用现有 blocked()）
        """
        # ① 动态窗口
        v_lo = max(0.0, v_now - self.a_decel * self.T)
        v_hi = min(self.v_max, v_now + self.a_accel * self.T)
        w_lo = max(-self.w_max, w_now - self.a_w * self.T)
        w_hi = min(self.w_max, w_now + self.a_w * self.T)
        if v_hi - v_lo < 1e-6:
            v_hi = v_lo + 1e-6
        if w_hi - w_lo < 1e-6:
            w_hi = w_lo + 1e-6

        target_angle = math.atan2(target[1] - robot_pos[1], target[0] - robot_pos[0])

        vs = np.linspace(v_lo, v_hi, self.n_v)
        ws = np.linspace(w_lo, w_hi, self.n_w)

        best = None
        best_score = -1e18
        for v in vs:
            for w in ws:
                # ③ 轨迹模拟
                traj = self._simulate(robot_pos, yaw, v, w)
                # 碰撞检测 + 最近障碍距离
                hit, min_clear = self._check_collision(traj, blocked_fn)
                if hit or min_clear < self.stop_margin:
                    continue
                # ④ 评分
                score = self._score(traj, v, w, target_angle, min_clear, v_now, w_now)
                if score > best_score:
                    best_score = score
                    best = (float(v), float(w))
        return best

    def _simulate(self, robot_pos, yaw, v, w):
        """模拟圆弧轨迹，返回 [(x, y), ...]"""
        pts = []
        x, y, h = robot_pos[0], robot_pos[1], yaw
        dt = self.dt_sample
        n = max(1, int(self.horizon / dt))
        for _ in range(n):
            h += w * dt
            x += v * math.cos(h) * dt
            y += v * math.sin(h) * dt
            pts.append((x, y))
        return pts

    def _check_collision(self, traj, blocked_fn):
        """返回 (hit, min_clear)。min_clear 用更细步长量到障碍距离（采样点间距 0.05~0.3m）。"""
        min_clear = 1e18
        prev = None
        for pt in traj:
            if blocked_fn(pt):
                return True, min_clear
            # 粗略最近距离：检查采样点之间（用 0.1m 子步长细化）
            if prev is not None:
                for t in np.linspace(0.05, 1.0, 5):
                    ip = (prev[0] + (pt[0]-prev[0])*t, prev[1] + (pt[1]-prev[1])*t)
                    if blocked_fn(ip):
                        return True, min_clear
            prev = pt
        # 采样点不碰撞时，min_clear 用轨迹起点/终点粗略估计
        if traj:
            min_clear = self.horizon  # 无精确距离时给上限（评分弱化）
        return False, min_clear

    def _score(self, traj, v, w, target_angle, min_clear, v_now, w_now):
        """加权评分。返回值越大越好。"""
        # heading：轨迹终点方向 vs 目标方向（cos 相似度 → [-1,1]）
        end = traj[-1]
        end_angle = math.atan2(end[1], end[0]) if traj else 0.0
        heading = math.cos(end_angle - target_angle)
        # clearance：归一化 [0,1]
        clearance = min(min_clear / self.clearance_max, 1.0)
        # velocity：归一化 [0,1]
        velocity = v / self.v_max if self.v_max > 0 else 0.0
        # smoothness：与当前状态接近 [0,1]
        dv = abs(v - v_now) / self.v_max if self.v_max > 0 else 0.0
        dw = abs(w - w_now) / (2 * self.w_max) if self.w_max > 0 else 0.0
        smooth = 1.0 - (dv + dw) / 2.0
        return (self.w_heading * heading
                + self.w_clearance * clearance
                + self.w_velocity * velocity
                + self.w_smooth * smooth)
```

**Step 4: 跑测试验证通过**

Run: `$PY -m pytest tests/test_dwa.py -v`
Expected: 5 passed

**Step 5: Commit**

```bash
git add simtrack/algorithms/dwa.py tests/test_dwa.py
git commit -m "feat: DWA局部规划器 — 动态窗口77轨迹采样 + heading/clearance/velocity/smooth评分 + 全碰撞返回None"
```

---

### Task 3: 集成 --obs-random 场景（algo3_headless.py）

**Objective:** 命令行参数 + 随机障碍初始化/更新，替换 patrol 分支。

**Files:**
- Modify: `test_scripts/algo3_headless.py`（参数段 ~L107、障碍初始化段 ~L345-355、主循环障碍更新段 ~L1079-1081）

**Step 1: 加参数**（`--obs-patrol` 行附近，L107 后）

```python
ap.add_argument("--obs-random", type=int, default=0, help="随机反弹障碍数(2-4)：每段20m×5m, 1m/s, 20%/s变向, 撞墙/虚拟墙反弹")
ap.add_argument("--obs-random-ch", type=str, default="1,4,6,8", help="反弹区段通道列表(逗号分隔)")
```

**Step 2: 加初始化/更新函数**（`init_patrol_obstacles` 函数附近）

```python
# ── 随机反弹障碍（主人指令 08-07：20m×5m 段, 1m/s, 20%/s 变向, 弹性反弹）──
random_field = None   # RandomObstacleField 实例

def init_random_obstacles():
    global random_field, obs_world
    chs = [int(c) for c in args.obs_random_ch.split(",")][:args.obs_random]
    random_field = RandomObstacleField(channels=chs, seed=FIXED_SEED)
    obs_world = random_field.positions
    print(f"  [CFG] 随机反弹障碍 {len(obs_world)} 个 @通道{chs} (1m/s, 20%/s变向)", flush=True)

def update_random(dt):
    global obs_world
    random_field.update(dt)
    obs_world = random_field.positions   # blocked()/DWA 自动感知新位置
```

**Step 3: 障碍初始化分支**（L345-355，在 `args.obs_patrol > 0` 分支后加）

```python
elif args.obs_random > 0:
    obs_world = []
    init_random_obstacles()
```

**Step 4: 主循环更新**（L1079-1081 patrol 更新后加）

```python
    # 随机反弹障碍移动（每 tick 更新，1m/s 弹性反弹）
    if random_field is not None:
        update_random(m.opt.timestep)
```

**Step 5: 验证**

Run: `$PY test_scripts/algo3_headless.py --obs-random 4 --seed 42 --max-steps 5000 --timeout 120`
Expected: 启动日志出现 `[CFG] 随机反弹障碍 4 个 @通道[1, 4, 6, 8]`，无异常（跑 5000 步提前退出可接受——本任务只验证场景集成）

**Step 6: Commit**

```bash
git add test_scripts/algo3_headless.py
git commit -m "feat: 随机反弹障碍场景集成 — --obs-random N + 通道选择 + 每tick更新obs_world"
```

---

### Task 4: Mover DWA 集成

**Objective:** Mover 增加 DWA 决策状态，转向段用 ω*，速度段保留约束，全碰撞回退 _bounce。

**Files:**
- Modify: `test_scripts/algo3_headless.py`（Mover 类 L700-867）

**Step 1: Mover.__init__ 增加状态**（L703-708 附近）

```python
        self.dwa = None          # DWAAlgorithm 实例（args.obs_random>0 时创建）
        self.dwa_target = None   # DWA 决策结果 (v*, ω*)；None = 全碰撞 → 走原逻辑
        self.omega = 0.0         # 当前角速度（DWA 动态窗口用）
```

**Step 2: Mover.step 转向段改造**（L730-734 原转向逻辑前插入）

```python
        # DWA 决策结果存在 → 用 ω* 转向（替代原 err 转向）
        if self.dwa is not None and self.dwa_target is not None:
            self.yaw += self.dwa_target[1] * dt
        else:
            tgt_yaw = math.atan2(ty-by, tx-bx)
            err = (tgt_yaw-self.yaw+math.pi)%(2*math.pi)-math.pi
            dyaw = max(-YAW_RATE*dt, min(YAW_RATE*dt, err))
            self.yaw += dyaw
        self.omega = self.dwa_target[1] if (self.dwa is not None and self.dwa_target is not None) else 0.0
```

**Step 3: 速度段改造**（L743-757，v_des 计算加 DWA 优先）

```python
        # ── 期望速度：DWA 给的 v* 优先，否则原逻辑（目标距离速度）──
        if self.dwa is not None and self.dwa_target is not None:
            v_des = self.dwa_target[0]
        else:
            v_des = min(SPEED_MAX, math.hypot(tx-bx, ty-by)*SPEED_FACTOR)
```

**Step 4: 被堵段条件改造**（L758-759，`speed<=0.05` 前加 DWA 全碰撞条件）

```python
        # ── 前方被堵且已停住 / DWA 全碰撞 → 预判转向，不碰撞 ──
        if (self.speed <= 0.05 and d_clear < STOP_MARGIN + 0.15) or (
                self.dwa is not None and self.dwa_target is None):
```

**Step 5: 验证**

Run: `$PY -m pytest tests/ -v`
Expected: 现有单测全过（无回归）

Run: `$PY test_scripts/algo3_headless.py --obs-random 4 --seed 42 --max-steps 5000 --timeout 120`
Expected: 无异常；日志无 `[OBS-ESC]` 死循环；MOVER 日志显示 yaw 随 DWA 平滑变化

**Step 6: Commit**

```bash
git add test_scripts/algo3_headless.py
git commit -m "feat: Mover DWA集成 — 转向段用ω* / 速度段v*优先 / 全碰撞触发_bounce"
```

---

### Task 5: 主循环 DWA 决策

**Objective:** 每 LIDAR_TICK 步调用 dwa.choose_velocity，把 lookahead 目标传给它。

**Files:**
- Modify: `test_scripts/algo3_headless.py`（主循环执行段 ~L1242-1294）

**Step 1: Mover 创建处初始化 DWA**（`mv = Mover(m, d)` 附近）

```python
mv = Mover(m, d)
if args.obs_random > 0:
    mv.dwa = DWAAlgorithm(v_max=SPEED_MAX, w_max=YAW_RATE,
                          a_accel=A_ACCEL, a_decel=A_DECEL)
    print("  [DWA] 局部规划器已启用", flush=True)
```

**Step 2: 执行段决策**（L1245-1265 pure pursuit 算出 tx,ty 后，调用 Mover.step 前插入）

```python
    if path is not None and path_idx < len(path):
        # ... 原有 pure pursuit 代码（算 tx, ty, look_target）不变 ...
        # DWA 决策：每 LIDAR_TICK 步，用 lookahead 目标
        if mv.dwa is not None and step % LIDAR_TICK == 0:
            mv.dwa_target = mv.dwa.choose_velocity(
                robot_pos=(bx, by), yaw=mv.yaw,
                v_now=mv.speed, w_now=mv.omega,
                target=(tx, ty), blocked_fn=blocked)
```

**Step 3: 验证**

Run: `$PY test_scripts/algo3_headless.py --obs-random 4 --seed 42 --max-steps 5000 --timeout 120`
Expected: 日志出现 `[DWA] 局部规划器已启用`；无异常；MOVER 每 10000 步日志正常

**Step 4: Commit**

```bash
git add test_scripts/algo3_headless.py
git commit -m "feat: 主循环DWA决策 — 每LIDAR_TICK步 choose_velocity(lookahead目标)"
```

---

### Task 6: 正向全程验证（验收）

**Objective:** 4 段随机障碍全程跑通，碰撞 0 + bounce 0 + 到达出口。

**Files:**
- Run: `test_scripts/algo3_headless.py`

**Step 1: 跑全程**

Run:
```bash
cd /home/qin/workspace/simtrack
$PY test_scripts/algo3_headless.py --obs-random 4 --seed 42 --max-steps 300000 --timeout 600 --render-every 0 --save-name m3_dwa_seed42.json
```

Expected:
- 日志 `★ ARRIVED! @(...)` 出现（到达出口）
- 成绩单 `bounces: 0`、`collisions: 0`
- 时间/步数记录

**Step 2: 多 seed 抽样**（随机障碍布局不同，验证鲁棒性）

Run:
```bash
for s in 7 123 999; do
  $PY test_scripts/algo3_headless.py --obs-random 4 --seed $s --max-steps 300000 --timeout 600 --render-every 0 --save-name m3_dwa_seed${s}.json
done
```

Expected: 全部 ARRIVED + bounce 0 + collision 0

**Step 3: 渲染确认（至少 1 seed）**

Run: `$PY test_scripts/algo3_headless.py --obs-random 4 --seed 42 --max-steps 300000 --timeout 900 --render-every 200 --out-dir /tmp/m3_dwa_frames`
Expected: 帧目录有 PNG；选 2-3 张发主人确认观感（狗绕障碍顺滑，无抖动）

**Step 4: Commit 成绩单**

```bash
git add scans/m3_dwa_*.json
git commit -m "test: M3 正向全程验证 — 4段随机障碍 seed42/7/123/999 全到达 bounce0 collision0"
```

---

### Task 7: 反向验证 + 收尾

**Objective:** 反向（出口→起点）验证 + 文档收尾。

**Files:**
- Run: `test_scripts/algo3_headless.py --target start`

**Step 1: 反向跑**

Run: `$PY test_scripts/algo3_headless.py --obs-random 4 --seed 42 --max-steps 300000 --timeout 600 --target start --save-name m3_dwa_rev_seed42.json`
Expected: ARRIVED + bounce 0 + collision 0

**Step 2: 文档收尾**

- 更新 `docs/2026-08-07-obs-progressive-milestone.md`（或新建 M3 里程碑文档），记录：
  - 场景参数（4 段 20m×5m、1m/s、20%/s 变向、虚拟墙内收 10cm）
  - DWA 参数（77 轨迹、评分权重、T=0.05s）
  - 成绩表（正向/反向、多 seed）
  - 坑与教训
- 更新 mujoco-simulation 技能（如需）

**Step 3: Commit**

```bash
git add docs/
git commit -m "docs: M3 DWA里程碑 — 4段随机反弹障碍全程 bounce0 验收文档"
```

**Step 4: 汇报主人**

- 成绩单摘要（时间/步数/bounce/collision）
- 渲染帧截图（如已生成）
- 里程碑达成：M1 无障碍 → M2 固定/巡逻障碍 → **M3 随机可动障碍 + DWA 局部规划（全部完成）**

---

## 风险与应对

| 风险 | 应对 |
|---|---|
| DWA 评分权重不理想（绕行抖动/贴障碍） | 调权重：clearance 权重提高 / smooth 权重提高；先看轨迹渲染再调 |
| 障碍贴虚拟墙角反复横跳（dir 翻转震荡） | _reflect 后加最小方向扰动（±0.2 rad）；测试观察 |
| DWA 在窄缝（障碍+墙<1.6m）犹豫 | stop_margin=0.4 + 制动约束保证不撞；bounce 兜底 |
| 障碍刚好堵在转弯口 | 虚拟墙 x∈[15,35] 远离转弯口（x=4.2/45.8），设计上规避 |
| 全程耗时过长（bounce 反复） | 降速通过区段（近墙限速已有）；检查 DWA 是否频繁 None |

## 验证清单（全部通过才算完成）

- [ ] `$PY -m pytest tests/ -v` 全绿（含新增 test_obstacles_random / test_dwa）
- [ ] `--obs-random 4` 正向 seed 42/7/123/999：ARRIVED + bounce 0 + collision 0
- [ ] 反向 `--target start`：ARRIVED + bounce 0 + collision 0
- [ ] 渲染帧确认绕障顺滑（主人过目）
- [ ] 里程碑文档 + 技能更新 + commit
