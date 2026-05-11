"""
SimTrack — 模块化赛道仿真系统

组件:
    trackgen   — 生成蛇形赛道 hfield PNG (2000×2000)
    obstacles  — 沿赛道中轴线随机生成障碍物
    lidar      — 多线激光雷达 (10Hz, 点数/线数可调)
    algorithms — 可替换避障算法 (VO / Tangent Arc)
    models     — 可替换仿真模型 (圆柱体 / G1)
    policies   — 可替换控制策略 (Protrain / 自定义)
    simulation — 主仿真运行器
"""

__version__ = "1.0.0"
