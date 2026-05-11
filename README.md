# SimTrack — 模块化赛道避障仿真系统

[![tests](https://img.shields.io/badge/tests-15/15-brightgreen)](tests/)
[![python](https://img.shields.io/badge/python-3.8+-blue)]()

**地图/障碍物/雷达/算法/模型五组件可替换的 MuJoCo 赛道避障仿真。**

```
simtrack/
├── simtrack/
│   ├── trackgen.py       # 蛇形赛道 hfield 生成 (2000×2000, 3m 护栏)
│   ├── obstacles.py      # 沿赛道中轴线随机障碍物生成
│   ├── lidar.py          # 多线激光雷达 (10Hz, 点数/线数可调)
│   ├── simulation.py     # 主仿真运行器 (组件化)
│   ├── algorithms/
│   │   ├── base.py       # 避障算法抽象基类
│   │   └── vo.py         # VO 避障 (朗毅, 切线搜索+滞后防抖)
│   └── models/
│       └── cylinder.py   # 圆柱体仿真模型
├── tests/
│   ├── test_trackgen.py
│   ├── test_obstacles.py
│   └── test_vo.py
└── examples/
    └── run_cylinder_vo.py
```

## 五组件可替换

| 组件 | 默认 | 替换方式 |
|------|------|---------|
| **地图** | `TrackGenerator` (10段蛇形, 3m护栏) | `generate()` + `save()` 到 PNG |
| **障碍物** | `ObstacleGenerator` (沿赛道中轴线, 间距4-8m) | 传入自定义 `ObstacleGenerator` |
| **雷达** | `LidarSensor` (240射线/3线/15m/10Hz) | 传入自定义 `LidarSensor` |
| **算法** | `VOAlgorithm` (切线搜索+滞后防抖) | 实现 `AvoidanceAlgorithm` 基类 |
| **模型** | `build_cylinder_scene()` (圆柱体) | 实现自定义 `build_*_scene()` |

## 快速开始

```python
from simtrack import Simulation, TrackGenerator
from simtrack.algorithms import VOAlgorithm

# 1. 生成赛道
tg = TrackGenerator(hf_res=2000, guard_height=3.0)
tg.generate()
tg.save("/tmp/track_hd.png")

# 2. 运行仿真
sim = Simulation(
    track_hfield="/tmp/track_hd.png",
    algorithm=VOAlgorithm(max_speed=3.0),
    lidar_rays=240,
    seed=None,  # 每次不同障碍物布局
)
sim.setup()
result = sim.run(headless=True)
print(f"完成: {result['time']:.0f}s, 碰撞{result['collisions']}次")
```

## 架构: 三层导航

```
┌─────────────────────────────────────────┐
│ L1 全局导航 (Waypoints)                  │
│ 提供方向。绝对坐标，不可靠但足够用。        │
│ → waypoints 序列，每 8m 一个              │
├─────────────────────────────────────────┤
│ L2 通道识别 (SectorNav)                  │
│ 雷达点云→扇区距离→通道判断。               │
│ 墙壁=通道边界，障碍物=要绕开的威胁。        │
│ 不聚类、不识别、只问"这个方向通不通"        │
│ → 扇区距离图 + 通/堵判定                  │
├─────────────────────────────────────────┤
│ L3 动态避障 (TBD)                        │
│ 活动障碍物快速反应，尽量少减速。            │
│ → 暂留空                                  │
└─────────────────────────────────────────┘
```

**L1+L2已实现**。扇形导航内部: waypoints给目标方向→扇区距离判断通堵→选最近通扇区。导航和避障是同一套逻辑。

### 已验证事实链

```
hfield墙壁射线命中 ✅ (mj_forward后正常)
纯墙壁→17个误检聚类 ❌
VO输入混入墙壁误检 → 永久避障 ❌
扇形导航不看聚类只看扇区距离 → v=2.0全速 ✅
```

**核心认知**: 墙壁≠障碍物。通道边界和要绕开的威胁本质不同。扫地机器人逻辑——扫一圈，距离近=堵，距离远=通。

## 安装

```bash
pip install -e .
# 需要: numpy, mujoco>=3.0, opencv-python-headless
```

## 测试

```bash
pytest tests/ -v
# 15 个纯 Python 测试，无需 MuJoCo 或 GPU
```

## 扩展

### 新避障算法

```python
from simtrack.algorithms.base import AvoidanceAlgorithm, AvoidanceResult

class MyAlgorithm(AvoidanceAlgorithm):
    def choose_heading(self, robot_pos, robot_speed, target_pos, obstacles):
        # 你的逻辑
        return AvoidanceResult(heading=0.0, speed=self.max_speed, avoiding=False)

sim = Simulation(algorithm=MyAlgorithm(max_speed=5.0))
```

### 新模型 (G1 等)

```python
def build_g1_scene(hfield_path, ...):
    # 返回 MuJoCo XML 字符串
    ...

sim = Simulation(model_builder=build_g1_scene)
```

## 许可

MIT
