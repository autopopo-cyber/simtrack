# algo3 萤火导航数值适配 — 机器狗 + 50m 赛道 设计文档

> 日期: 2026-07-11
> 项目: `~/workspace/simtrack/`
> 目标: 把萤火导航的物理场景数值改成机器狗 + 50m 赛道，算法逻辑零改动，
> 达到之前 v3 的效果（走到终点 FINISH=(2.5, 47.5)）。

## 背景

- v3 已锁定（commit 0e4097c, `v3-locked`），在 100m 世界（SCALE=2.0）跑通全程。
- **headless 版 `test_scripts/algo3_headless.py` 与 v3 算法完全一致**（diff 验证：
  find_gates / astar_to / Mover / milestones / 黄球路点全部相同），仅渲染层
  viewer.launch_passive → EGL 离屏渲染。tshell 环境无 DISPLAY，必须用 headless 版。
- 主人 2026-07-10 指令：赛道改为 50×50m（SCALE=1.0），路宽 1.5m、每段 5m。
- 机器人从旧圆柱（r=0.5）换成机器狗（0.8m 长 × 0.4m 径水平圆柱，MAX_V=4.0, MAX_W=1.0）。
- 起点主人指定 (2.5, 2.5)（第 4 段赛道，周围 2m 零护栏），终点主人确认 (2.5, 47.5)。

## 改动范围

**只改 `test_scripts/algo3_headless.py` 的物理场景常量与几何，算法逻辑
（find_gates / 三级跳 A* / Mover / 路标 / 迷失恢复 / 黄球路点）零改动。**

### 常量表

| # | 位置 | 现值 | 新值 | 说明 |
|---|------|------|------|------|
| 1 | `SCALE` | 2.0 | 1.0 | 世界坐标比例（50m 世界） |
| 2 | `SAFE_R` | 0.5 | 0.2 | 机器狗圆柱半径 0.4m/2 |
| 3 | `SPEED` | 5.0 | 4.0 | MAX_V=4.0 |
| 4 | `SPEED_MAX` | 8.0 | 4.0 | 上限=线速度上限 |
| 5 | `YAW_RATE` | 6.0 | 1.0 | MAX_W=1.0 |
| 6 | `FINISH` | (3.0, 95.0) | (2.5, 47.5) | 50m 终点（主人确认） |

### XML 场景表（build_xml 内）

| # | 元素 | 现值 | 新值 |
|---|------|------|------|
| 7 | `<hfield size>` | `50.0 50.0 4.0 2.0` | `25.0 25.0 4.0 2.0` |
| 8 | `<light pos>` | `50 50 80` | `25 25 80` |
| 9 | hfield geom `pos` | `50 50 0.0` | `25 25 0.0` |
| 10 | bot geom cylinder `size` | `0.5 0.5` | `0.2 0.4`（r=0.2, 半长=0.4） |

### 起点

`d.qpos[0]=3; d.qpos[1]=3` → `d.qpos[0]=2.5; d.qpos[1]=2.5`

## 不动的部分

- `VOXEL=0.1`（格子分辨率）
- `LIDAR_RANGE=15.0 / LIDAR_RAYS=120 / LIDAR_TICK`（激光）
- `ROBOT_R = int(SAFE_R/VOXEL)` → 自动变为 2（0.2m），CLEARANCE 跟随
- `MIN_SPEED / SPEED_FACTOR`（速度自适应）
- find_gates / merge_gates / pick_gate / astar_to / gen_yellow_waypoints / Mover.step / 迷失恢复
- 障碍物生成（OBS_R=1.0, OBS_CLEAR）
- EGL 渲染与成绩单 JSON 输出

## 验证标准

1. 脚本无 DISPLAY 可运行（EGL 离屏）
2. 机器人从 (2.5, 2.5) 出发
3. 完整导航跑到 FINISH=(2.5, 47.5)（距终点 < 3m 判定到达）
4. 成绩单 JSON 输出终点命中
5. 用 tshell 跑 ≥3 遍确认稳定（不同 seed）
