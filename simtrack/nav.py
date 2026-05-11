"""
simtrack.nav — 扇区导航算法

使用激光雷达点云，将 360° 分为 N 个扇区，
选目标方向最近的通畅扇区作为前进方向。
碰撞检测兜底。
"""
import math, random
from . import map as simmap

class SectorNav:
    """扇区导航: 雷达扇区扫描 + 向目标点导航"""
    
    def __init__(self, n_sectors=36, safe_dist=2.0, speed=2.0, cp_radius=3.0):
        self.n_sectors = n_sectors
        self.safe_dist = safe_dist  # 扇区内无障碍距离阈值(m)
        self.speed = speed
        self.cp_radius = cp_radius
        self.sector_deg = 360.0 / n_sectors
        self._bounce_count = 0
    
    def steer(self, bx, by, tx, ty, lidar_pts, hf):
        """计算下一步速度和方向。
        
        Args:
            bx, by: 机器人当前位置(世界坐标)
            tx, ty: 目标点(世界坐标)
            lidar_pts: 雷达点云 [(x,y), ...] 或 []
            hf: hfield 图像数组
        
        Returns:
            (vx, vy, reached): 速度向量 + 是否到达目标
        """
        dist_to_target = math.hypot(tx - bx, ty - by)
        if dist_to_target < self.cp_radius:
            return (0.0, 0.0, True)
        
        target_angle = math.atan2(ty - by, tx - bx)
        
        # ── 扇区扫描 ──
        sector_min = [float('inf')] * self.n_sectors
        
        for px, py in lidar_pts:
            dx, dy = px - bx, py - by
            dist = math.hypot(dx, dy)
            ang = math.atan2(dy, dx)
            rel = (ang - target_angle) % (2 * math.pi)
            si = int(rel / (2 * math.pi / self.n_sectors)) % self.n_sectors
            if dist < sector_min[si]:
                sector_min[si] = dist
        
        # ── 选最优扇区: 目标方向最近的通畅扇区 ──
        best_sector = 0
        for offset in range(self.n_sectors // 2):
            for sign in [1, -1]:
                si = (offset * sign) % self.n_sectors
                if sector_min[si] >= self.safe_dist:
                    best_sector = si
                    break
            else:
                continue
            break
        
        yaw = target_angle + best_sector * 2 * math.pi / self.n_sectors
        
        # ── 速度 ──
        # 前方通畅度调整速度
        front_dist = sector_min[0]
        if front_dist < self.safe_dist * 0.5:
            spd = self.speed * 0.3  # 减速
        elif front_dist < self.safe_dist:
            spd = self.speed * 0.7
        else:
            spd = self.speed
        
        vx = math.cos(yaw) * spd
        vy = math.sin(yaw) * spd
        
        # ── 碰撞检测兜底 ──
        nx = bx + vx * 0.005  # 预测 1 timestep
        ny = by + vy * 0.005
        if simmap.detect_collision(hf, nx, ny, 0.55):
            yaw = random.uniform(0, 2 * math.pi)
            vx = math.cos(yaw) * self.speed * 0.5
            vy = math.sin(yaw) * self.speed * 0.5
            self._bounce_count += 1
        
        return (vx, vy, False)
    
    @property
    def bounces(self):
        return self._bounce_count
