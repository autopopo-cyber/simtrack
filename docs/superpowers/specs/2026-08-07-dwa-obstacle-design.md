# M3 设计：DWA 局部规划器 + 4 段随机反弹障碍

> 日期：2026-08-07
> 状态：已获主人批准（"试试吧"）
> 里程碑：M3（最后里程碑）——建好图的地图里加入随机可动障碍，稳定顺滑回避

## 背景

M2 障碍渐进完成（巡逻 2 可动障碍 141.94s bounce 0）。核心矛盾：执行层是**反应式**（撞了才 bounce），对随机不可预测障碍，"等撞上再 escape"就谈不上顺滑。全局 HPA 假设静态，重建 ~1.7s 跟不上。

主人拍板 B 方案：**引入 DWA 局部规划器**（速度空间采样实时避障），ROS 标准两层架构——全局 HPA 给参考路径，局部 DWA 实时修正。

## 场景（主人指定）

- 蛇形赛道（KNOWN_MAP_MODE，地图已探索，机器狗知道地图）
- 4 个反弹区段，每段在一条通道内：
  - 20m 长（x∈[15,35]）居中，两端留 10m 不挡转弯口/标牌
  - 5m 宽（通道天然宽度，上下是真实 hfield 墙）
  - **虚拟墙**：x∈[15.1, 34.9]（内收 10cm），只约束障碍中心，**机器狗无视虚拟墙**（虚拟墙不进地图/碰撞检测）
  - y 方向真实墙天然约束
- 每段 1 个随机障碍，初始位置段中央 (25, yc)，方向随机
- 4 区段默认通道 ch = 1, 4, 6, 8（均匀分布）
- 无其他障碍
- 机器狗任务：起点 (2.5,2.5) → 出口 (2.5,47.5)，可正向也可反向（先正向验证）

## 随机障碍运动模型（小球弹性反弹）

```
update(dt):
  ① 变向倒计时：每满 1s，20% 概率 dir = uniform(0, 2π)  ← 完全随机新方向
  ② 移动：pos += dir · 1.0 m/s · dt
  ③ 真实墙反弹（y 边界）：中心距墙 < 0.5m → 镜面反射（vy 取反），拉回界内
  ④ 虚拟墙反弹（x 边界）：中心出 [15.1, 34.9] → vx 取反，拉回界内
```

- 速度恒 1.0 m/s（主人指定）
- 镜面反射 = 弹性碰撞（分量取反），类似小球撞墙

## DWA 局部规划器（新文件 simtrack/algorithms/dwa.py）

```
choose_velocity(robot_pos, yaw, v_now, ω_now, target, blocked_fn):
  ① 动态窗口（受加速度约束）：
     T = 决策周期 = LIDAR_TICK × timestep = 10 × 0.005 = 0.05s
     v ∈ [max(0, v_now - A_DECEL·T), min(v_max, v_now + A_ACCEL·T)]
     ω ∈ [max(-ω_max, ω_now - A_W·T), min(ω_max, ω_now + A_W·T)]
     A_W = 角加速度上限（默认 10 rad/s²，可调）；ω_max = YAW_RATE = 1.5 rad/s
  ② 采样：7 档 v × 11 档 ω = 77 条轨迹
  ③ 每条模拟 1.5s 圆弧（dt_sample），逐点 blocked() 判定（复用现有碰撞检测）
  ④ 评分（加权和）：
     heading    weight 0.6 — 轨迹终点方向 vs 目标方向（cos 相似度）
     clearance  weight 0.25 — 轨迹上最近障碍距离（归一化 min(d,D_max)/D_max）
     velocity   weight 0.1  — v / v_max
     smoothness weight 0.05 — 与当前 (v,ω) 差距惩罚（防抖）
     碰撞轨迹硬排除
  ⑤ 返回最优 (v*, ω*)；全碰撞 → 返回 None（触发 _bounce 兜底）
```

### 决策频率

每 LIDAR_TICK=10 步（0.05s）重决策一次——障碍 1m/s 只动 5cm，足够顺滑。

### 第一版不做障碍运动预测（YAGNI）

用障碍当前位置静态判定。理由：狗 4m/s vs 障碍 1m/s，速度差 4 倍；每 0.05s 重决策；1.5m 预测误差对 5m 宽通道可接受。测试不顺再加"障碍速度估计+预测"。

## 集成（Mover 最小侵入）

```
Mover 增加 dwa 成员 + (v_target, ω_target) 状态
每 LIDAR_TICK=10 步：dwa.choose_velocity() → 存 (v_target, ω_target)
每步执行：
  转向段   ← yaw += clamp(ω_target·dt)（替代原 err 转向）
  速度段   ← 保留现有加速度平滑 + 制动约束 + 近墙限速
  被堵段   ← DWA 全碰撞时触发（原 _bounce/escape 兜底不变）
```

- 新增 `--obs-random N`（N=4，取代 patrol 分支；patrol 代码保留不删）
- `obs_world` 每 tick 更新为随机障碍位置，blocked()/DWA 自动感知
- KNOWN_MAP_MODE 下 HPA 规划不变，pure pursuit 提供 lookahead 目标给 DWA

## 验收标准

- 正向全程：起点 (2.5,2.5) → 出口 (2.5,47.5)，穿过 4 个反弹区段
- **碰撞 0 + bounce 0 + 到达出口**，记录时间/步数
- 若正向跑通，再反向验证

## 对照实验

- DWA off（纯 M2 执行层）vs DWA on，同 seed 同布局
- 预期：DWA on 顺滑通过（bounce 0），DWA off 在随机障碍前可能 bounce/碰撞

## 参考

- 主人原话："机器狗速度远大于障碍，其实保持距离就可以了。难度应该没那么大。"
- M2 巡逻障碍实现：algo3_headless.py `gen_patrol_path` / `update_patrol`（保留，可参考随机障碍结构）
- 现有执行层：`Mover.step()`（转向/限速/制动/被堵）
- 碰撞检测：`blocked()` / `is_obstacle_world()`（精确圆判定，OBS_CLEAR=0.7）
