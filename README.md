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

## 关键经验

### 护栏必须足够厚

hfield 墙壁是高度图上的像素——在 2000px 分辨率下每像素 2.5cm。

```
120 射线 × 15m 外弧间距 = 78cm
3px 墙壁宽 = 7.5cm → 仅 10% 射线命中

240 射线 × 15m 外弧间距 = 39cm
5px 墙壁宽 = 12.5cm → ~32% 射线命中 ✓
```

**推荐配置**: `lidar_rays=240` + `guard_brush=5`

### 障碍物密度控制

564m 赛道 + 4-8m 间距 ≈ 90-140 个障碍物。即使墙壁完全遮挡，单赛道 15m 前视也有 2-4 个。配合 VO 算法的 `avoid_speed_ratio=0.6` 可顺畅通过。

降低密度: `spacing_range=(10, 15)` → 约 40-55 个

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
