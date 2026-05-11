"""
避障算法基类 — 可替换接口

所有避障算法实现此接口，仿真运行器通过此接口调用。
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class AvoidanceResult:
    """避障算法返回值"""
    heading: float          # 推荐航向 (弧度)
    speed: float            # 推荐速度 (m/s)
    avoiding: bool          # 是否处于避障状态
    status: str = "ok"      # "ok" | "stuck" | "blocked" | "map_error"


class AvoidanceAlgorithm(ABC):
    """避障算法抽象基类。

    子类实现 choose_heading() 方法。
    仿真运行器每 decision_interval 调用一次。
    """

    def __init__(self, max_speed: float = 2.0, robot_radius: float = 0.25):
        self.max_speed = max_speed
        self.robot_radius = robot_radius

    @abstractmethod
    def choose_heading(
        self,
        robot_pos: Tuple[float, float],
        robot_speed: float,
        target_pos: Tuple[float, float],
        obstacles: List[Tuple[float, float, float]],
    ) -> AvoidanceResult:
        """选择最优航向。

        Args:
            robot_pos:  (x, y) 机器人当前位置
            robot_speed: 当前速度 (m/s)
            target_pos:  (x, y) 目标路点
            obstacles:   [(cx, cy, radius), ...] 障碍物列表

        Returns:
            AvoidanceResult
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
