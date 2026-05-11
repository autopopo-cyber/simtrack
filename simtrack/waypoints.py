"""
simtrack.waypoints — 导航点 + 分段障碍物生成

沿中心线生成导航点（间距可调），在每段随机放置障碍物。
障碍物避开中心线，随机出现在路边缘。
"""
import math, random
import numpy as np
from . import map as simmap

# ── 导航点 ──

def from_centerline(centerline_world, spacing_m=8.0):
    """从中心线按间距均匀采样导航点（世界坐标）。
    
    Args:
        centerline_world: 中心线点列表 [(wx,wy), ...]
        spacing_m: 导航点间距(米)
    Returns:
        list[(wx,wy)]: 均匀分布导航点
    """
    if len(centerline_world) < 2:
        return list(centerline_world)
    
    # 累积距离
    cum = [0.0]
    for i in range(1, len(centerline_world)):
        dx = centerline_world[i][0] - centerline_world[i-1][0]
        dy = centerline_world[i][1] - centerline_world[i-1][1]
        cum.append(cum[-1] + math.hypot(dx, dy))
    
    total = cum[-1]
    waypoints = [centerline_world[0]]
    d = spacing_m
    idx = 0
    while d < total:
        # 找到 cum 中刚好 >= d 的位置
        while idx < len(cum) - 1 and cum[idx+1] < d:
            idx += 1
        if idx >= len(centerline_world) - 1:
            break
        # 线性插值
        seg_len = cum[idx+1] - cum[idx]
        if seg_len < 0.001:
            d += spacing_m
            continue
        t = (d - cum[idx]) / seg_len
        wx = centerline_world[idx][0] + t * (centerline_world[idx+1][0] - centerline_world[idx][0])
        wy = centerline_world[idx][1] + t * (centerline_world[idx+1][1] - centerline_world[idx][1])
        waypoints.append((wx, wy))
        d += spacing_m
    
    # 终点
    if waypoints[-1] != centerline_world[-1]:
        waypoints.append(centerline_world[-1])
    
    return waypoints

# ── 分段障碍物 ──

def generate_segment_obstacles(centerline_world, spacing_m=5.0, 
                                 road_half=4.0, obs_radius=0.3, 
                                 density=0.3, seed=42):
    """沿中心线分段生成障碍物。
    
    每 spacing_m 米为一段，在每段路边缘随机放置 0-N 个障碍物。
    障碍物偏离中心线 road_half*0.6~road_half*1.2 米到左右两侧。
    
    Args:
        centerline_world: 中心线点列表 [(wx,wy), ...]
        spacing_m: 分段间距(米)
        road_half: 路面半宽(米)
        obs_radius: 障碍物半径(米)
        density: 障碍物密度 (0-1, 每段期望障碍数)
        seed: 随机种子
    
    Returns:
        list[(wx,wy)]: 障碍物世界坐标列表
    """
    rng = random.Random(seed)
    obstacles = []
    
    # 累积距离
    cum = [0.0]
    for i in range(1, len(centerline_world)):
        dx = centerline_world[i][0] - centerline_world[i-1][0]
        dy = centerline_world[i][1] - centerline_world[i-1][1]
        cum.append(cum[-1] + math.hypot(dx, dy))
    total = cum[-1]
    
    d = spacing_m / 2  # 从半段开始
    idx = 0
    while d < total - spacing_m:
        # 定位当前段中心
        while idx < len(cum) - 1 and cum[idx+1] < d:
            idx += 1
        if idx >= len(centerline_world) - 1:
            break
        
        seg_len = cum[idx+1] - cum[idx]
        t = (d - cum[idx]) / seg_len if seg_len > 0.001 else 0
        cx = centerline_world[idx][0] + t * (centerline_world[idx+1][0] - centerline_world[idx][0])
        cy = centerline_world[idx][1] + t * (centerline_world[idx+1][1] - centerline_world[idx][1])
        
        # 切线方向（法线用于偏移）
        tx = centerline_world[idx+1][0] - centerline_world[idx][0]
        ty = centerline_world[idx+1][1] - centerline_world[idx][1]
        tlen = math.hypot(tx, ty)
        if tlen < 0.001:
            d += spacing_m
            continue
        nx = -ty / tlen
        ny = tx / tlen
        
        # 本段放 0-N 个障碍物
        n_obs = 1 if rng.random() < density else 0
        if rng.random() < density * 0.5:
            n_obs += 1
        
        for _ in range(n_obs):
            side = rng.choice([-1, 1])
            offset = road_half * rng.uniform(0.5, 1.3)  # 路边缘附近
            # 沿段内随机微调
            local_offset = rng.uniform(-spacing_m * 0.3, spacing_m * 0.3)
            
            ox = cx + side * offset * nx + local_offset * tx / tlen
            oy = cy + side * offset * ny + local_offset * ty / tlen
            
            obstacles.append((ox, oy))
        
        d += spacing_m
    
    return obstacles

# ── MuJoCo XML 生成 ──

def obstacles_to_xml(obstacles, radius=0.3, height=1.0, rgba="0.9 0.3 0.3 0.8"):
    """将障碍物列表转为 MuJoCo body XML"""
    xml = ""
    for i, (ox, oy) in enumerate(obstacles):
        xml += (f'<body name="obs{i}" pos="{ox:.1f} {oy:.1f} {height/2}">'
                f'<geom type="cylinder" size="{radius} {height/2}" '
                f'rgba="{rgba}"/></body>\n')
    return xml

def checkpoints_to_xml(checkpoints_world, radius=1.0, rgba="0.2 0.5 1 0.8"):
    """将检查点转为可视化球体 XML (mocap)"""
    xml = ""
    for i, (x, y) in enumerate(checkpoints_world):
        xml += (f'<body mocap="true" pos="{x} {y} 2">'
                f'<geom type="sphere" size="{radius}" rgba="{rgba}"/></body>\n')
    return xml
