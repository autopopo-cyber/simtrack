# 机器狗模拟（水平圆柱）· 设计规格

> 日期: 2026-07-10 | 项目: simtrack | 状态: 已确认（主人拍板）

## 目标

用 MuJoCo 水平圆柱模拟机器狗，直接放蛇形赛道（track_hd）测试：
- 验证全向运动能力：前进/后退、左右平移、绕中心旋转
- 速度上限：线速度 4 m/s、角速度 1 rad/s
- 机器人离地 0.5m（模拟有腿机器狗，身体悬空）

## 机器人模型

- **形状**: MuJoCo `cylinder` geom，**水平放置**（euler 旋转 90°）
  - 尺寸: 长 0.8m × 直径 0.4m（半径 0.2m）
  - 位置: z=0.5m（离地高度）
- **自由度**: 全向 3-DOF
  - `slide x` + `slide y`（平面平移）
  - `hinge yaw`（绕中心旋转）
- **速度上限**: 线 4 m/s，角 1 rad/s（在控制层 clamp）

## 运动学控制

- 输入: 机体坐标系指令 `(vx_cmd, vy_cmd, w_cmd)`
- 世界系换算: `vx = vx_cmd·cosθ − vy_cmd·sinθ`, `vy = vx_cmd·sinθ + vy_cmd·cosθ`
- 写入 `d.qvel[0:2]`, `d.qvel[2] = w_cmd`
- 纯旋转指令时质心不动（绕中心旋转）

## 场景

- 蛇形赛道: `/tmp/track_hd.png`（746.9m，10 段，路宽 5m，护栏 3m）
- 起点: waypoints[0] 附近
- 障碍物: 运行时生成（复用 cyl_independent 模式）

## 测试脚本

`test_scripts/algo_dog_headless.py`（headless，无 viewer）:
1. 加载蛇形赛道 + 水平圆柱机器人
2. 顺序执行运动学测试: 前进 5m → 后退 5m → 左移 3m → 右移 3m → 原地旋转 360° → 斜向移动
3. 每段记录: 位置/朝向/速度，断言速度不超限
4. 输出成绩单 json 到 `scans/dog_baseline_seed<N>.json`
5. 渲染帧到 out-dir（复用 algo3_headless 的 render-every 模式）

## 测试基建

- tshell 起 job 跑 headless 仿真（HTTP 轮询，不怕截断）
- 浏览器面板看进度
- mujoco-venv: `/home/qin/mujoco-venv/bin/python`（numpy 2.4.4 + cv2 4.13.0 + mujoco 3.8.0）

## 非目标（Phase 2）

- 激光雷达避障导航（复用 MultiLineLidar）
- 赛道全程导航成绩（先验证运动学）
