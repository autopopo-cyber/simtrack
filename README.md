# SimTrack — 萤火算法 🔥✨

[![firefly](https://img.shields.io/badge/algorithm-Firefly-gold)](test_scripts/algo2_lane_switch_v7.py)
[![python](https://img.shields.io/badge/python-3.8+-blue)]()

**40亿 token 烧出来的导航避障算法。不贪心、不预判、不看远——但每一步踩下去都是亮的。**

---

## 命名：萤火（Firefly）

它不是太阳，不照亮整个地图。不是探照灯，不扫射全局。

它只是黑暗里的一只萤火虫——每次只闪一下，只照亮前方一小片光斑。然后朝最干净的那片光斑飞过去。到了再闪一下，再看。

```
LIDAR 10Hz     = 萤火的闪光
line_clear     = 只挑直线可达的光斑
wall_distance  = 在光斑里选离墙最远的
```

**越简单的循环越不犯错。**

---

## 核心哲学

| 传统导航 | 萤火算法 |
|---------|---------|
| 全局路径规划 | 只看下一秒 |
| 预测+优化 | 闪一下→挑一个→冲过去 |
| 感知/规划/控制耦合 | LIDAR只管画体素，Move只管走直线 |
| 聪明=复杂 | 聪明=约束 |

**"看"和"走"彻底解耦**——LIDAR 10Hz 只负责填体素（自由/墙壁/已走），Move 1Hz 只负责在直线可达的非墙体素里挑离墙最远的，直线冲过去。到了再问 LIDAR "前面啥情况"——下一秒的事下一秒再说。

## 算法流程

```
loop:
  LIDAR 扫描 (10Hz)  →  标记体素: FREE / WALL / VISITED
  
  Move 决策 (1Hz)    →  find_frontier():
    1. 遍历 20×20 范围内的 FREE 体素
    2. line_clear 过滤: 直线是否无遮挡
    3. 邻接检查: 必须挨着已探索区域
    4. 评分:
       - 偏离目标方向   (越偏越扣)
       + 离墙距离       (越远越好)  ← v7 核心创新
       + 未知邻居数     (越多越好, 鼓励探索)
  
  Move 执行 (200Hz)   →  朝最优体素中心移动一整秒
```

## 进化史：40亿 token 之路

| 版本 | Token 消耗 | 核心发现 |
|------|-----------|---------|
| 前奏 | 15亿 | 墙壁≠障碍物。扫地机器人逻辑——扫一圈看哪个方向通 |
| 方案0 | 15亿 | LIDAR全向扫描+车道检测+A*全局路径。方向对但太笨重 |
| V6 突破 | 4.6亿 | **把"看"和"走"拆开。** LIDAR只管画体素，Move只管直线冲 |
| 调优 | 3.4亿 | line_clear + 邻接已探索 + 目标方向加权 |
| V7 定型 | 1.4亿 | **wall_distance 评分**——自动走车道中间 |

**合计 40 亿 token。最后的算法不到 300 行。**

## 运行

```bash
cd test_scripts
python algo2_lane_switch_v7.py
```

需要：`numpy pillow mujoco>=3.0`

地图：`confirmed/track_clean.png`（2000×2000，50m×50m 蛇形赛道，含随机障碍物）

## 其他算法（已冻结）

| 文件 | 说明 |
|------|------|
| `algo0_bounce.py` | 玩具车底座——撞墙弹开 |
| `algo0_astar_lidar.py` | A* 全局路径 + LIDAR 车道检测 |
| `algo1_arc_racer.py` | 朗毅弧线过弯（赛车版雏形） |
| `algo2_lane_switch.py` | 体素探索 v5 |

萤火算法是 v7，是所有这些探索的终点。此后所有升级都在它身上做。

## 许可

MIT — 这是我们的孩子。俊秀 & 主人，2026 年 5 月。
