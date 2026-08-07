# M3 里程碑：DWA 局部规划器 + 4 段随机反弹障碍

> 日期：2026-08-07
> 状态：✅ 完成（正向 4 seed + 反向验证全通）
> tag：m3-dwa-obstacle-clear

## 场景（主人指定 08-07）

- 蛇形赛道 KNOWN_MAP_MODE（地图已知，HPA* 全局规划）
- 4 个反弹区段：通道 ch=1,4,6,8，x∈[15,35]（20m 长 × 5m 宽通道）
- 每段 1 个随机障碍：1m/s，每满 1s 20% 概率随机变向（uniform 0~2π），撞真实墙/虚拟墙镜面反弹
- 虚拟墙 x∈[15.1,34.9]（内收 10cm），只约束障碍中心，机器狗无视虚拟墙
- 狗任务：起点 (2.5,2.5) → 出口 (2.5,47.5)，可正向/反向

## 架构：两层规划（ROS 标准）

```
全局层（已有）：HPA* 分层寻路 → 路径 → pure pursuit → lookahead 目标
局部层（新增）：DWA 每 LIDAR_TICK=10 步决策速度空间 → (v*, ω*) → Mover 执行
```

## DWA 实现要点（simtrack/algorithms/dwa.py）

- 动态窗口：v∈[0, v_now+a_accel·T]（v_lo 强制 0——全速接近障碍可选低速/停车），ω∈[ω_now±a_w·T]
- 采样：7 档 v × 11 档 ω = 77 条轨迹，每条模拟 1.5s 圆弧逐点 blocked() 判定
- 评分：heading(0.6) + clearance(0.25) + velocity(0.1) + smoothness(0.05)，碰撞轨迹硬排除
- **障碍运动预测**：obstacles_motion=[(ox,oy,vx,vy,r)] 传入，模拟时障碍按速度外推（pred_t=0.5s 截断）判定未来位置——障碍朝狗移动则提前避让
- 全碰撞 → 返回 None → Mover 触发 _bounce 兜底

## 成绩（正向）

| seed | arrived | bounce | collision | time |
|---|---|---|---|---|
| 42 | ✅ | 0 | 0 | 413s |
| 7 | ✅ | 0 | 0 | 412s |
| 999 | ✅ | 0 | 0 | 423s |
| 123 | ✅ | 2 | 0 | 419s |

反向（--target start, 起点=出口）：seed 42 ✅ bounce 0 collision 0（392s）

## 途中踩坑（三个 DWA 致命 bug）

### 坑 1：heading 参照系——end_angle 用世界原点而非机器人位置
- 现象：狗在 ch1 右端 (45,8) 原地打转 20000 步（yaw 累加 11906°），lookahead 目标在身后
- 根因：`_score` 里 `end_angle = atan2(end[1], end[0])`（世界原点），target_angle 相对机器人——参照系不一致。机器人 (0,0) 时恰好重合，单测全在原点抓不到
- 修复：`atan2(end[1]-robot_pos[1], end[0]-robot_pos[0])`
- 教训：单测必须覆盖非原点位置

### 坑 2：动态窗口 v_lo 未含 0——全速接近障碍全碰撞
- 现象：seed 123 在 ch8 (23.4,42.9) 距障碍 1m 时 bounce 循环 → collision 272 次
- 根因：v_now=4.0 时窗口 v∈[3.6,4.0] 全高速，77 条轨迹全穿过障碍 → 全碰撞 → bounce；bounce 后障碍贴脸 → escape 方向受限 → 死循环
- 修复：v_lo 强制 0（经典 DWA 允许停车候选）

### 坑 3：障碍主动撞向狗——运动预测 vs 盲目膨胀
- 现象：seed 123 ch8 障碍恰好朝狗移动，静止判定"乐观"→ 选到会撞的轨迹
- 演进：膨胀 0.5（碰撞 0 但 bounce 激增——过度保守堵路）→ 膨胀 0.3 → **最终障碍运动预测**（速度向量外推，pred_t=0.5s 截断）
- 为什么截断：障碍每 1s 20% 变向 + 撞墙反弹，1.5s 长期外推不可靠（seed 7 出现"预测障碍会来但实际没来"的误判 bounce）
- 结果：4 seed 全部 collision 0，seed 42/7/999 bounce 0

## 相关文件

- `simtrack/algorithms/dwa.py` — DWA 局部规划器（含障碍运动预测）
- `simtrack/obstacles_random.py` — 随机反弹障碍（小球弹性反弹 + 虚拟墙）
- `test_scripts/algo3_headless.py` — 集成（--obs-random / --obs-random-ch / --target start 反向）
- `tests/test_dwa.py` / `tests/test_obstacles_random.py` — 14 个单测

## 待办

- [ ] test_integration.py / test_lidar.py 预存损坏（import 已删除的 LidarSensor，非本次改动）
- [ ] DWA 性能优化（单次决策 37ms 纯 Python，占步速 ~30%；可降采样密度或 numpy 矢量化）
