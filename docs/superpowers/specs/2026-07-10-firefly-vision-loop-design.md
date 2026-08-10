# Firefly algo3 视觉闭环测试 + 物理模型升级 设计文档

> 日期: 2026-07-10
> 项目: `~/workspace/simtrack/`
> Baseline: `test_scripts/algo3_firefly.py` (v3-locked)

## 1. 背景与目标

algo3 是萤火虫导航算法的锁定版（0.1m 字典 SLAM + find_gates + 三级跳 A*）。
受限于开发时 LLM 能力，有四个已知缺陷：

1. **作弊定位**：直接读 MuJoCo `d.qpos` 真值，无里程计模拟
2. **车体形状错误**：当前是圆形机器人（radius 0.5m），实际是 80cm×40cm 长条
3. **转弯模型错误**：当前绕点旋转，实际机器狗绕自身中心旋转
4. **激光雷达位置错误**：当前在原点，实际在头部、离地 50cm

**目标**：先在 baseline 上跑通完整闭环测试（数值成绩单 + GLM-5V 视觉巡检），
再升级 MuJoCo 物理模型，再实现真实定位/运动学。

## 2. 执行顺序（主人明确铁律）

```
Phase 1: 先跑完 baseline 测试 ← 现在开始
Phase 2: 再修改 mujoco 物理模型
Phase 3: 再实现功能（里程计/长条运动学）
```

**绝不跳过 Phase 1**。baseline 成绩单是后续所有改动的对照基准。

## 3. Phase 1: headless baseline 闭环测试

### 3.1 改动（最小侵入）
- 新建 `test_scripts/algo3_headless.py`（复制 algo3_firefly.py 主体）
- 替换渲染层：`mujoco.viewer.launch_passive` → EGL 离屏渲染
- **算法逻辑零改动**：find_gates / astar_to / Mover / milestones 原样保留

### 3.2 指标采集
| 指标 | 来源 |
|------|------|
| 探索覆盖率 | grid FREE+WALL / track_clean.png 真值可通行格 |
| 到达终点 | 距 FINISH(3,95) < 3m |
| 总步数/耗时 | step 计数 + wall time |
| 路标数 | len(milestones) |
| bounce 次数 | mv.bounce |
| 门探索数 | [GATE] 日志计数 |
| 回溯次数 | [BACK] 日志计数 |
| 迷失恢复次数 | [LOST] 日志计数 |

### 3.3 GLM-5V 视觉巡检（智谱直连）
- 每 200 步离屏渲染 PNG
- 随机抽帧 → 调智谱 `glm-5v-turbo` 判断：
  - "前方是通道还是死路？"
  - "机器人是否在贴墙转圈？"
- 输出 JSON：`{"step": N, "question": "...", "answer": "..."}`

### 3.4 输出产物
- `scans/baseline_<seed>.json` — 成绩单
- `/tmp/firefly_frames/*.png` — 渲染帧
- GIF 时间轴（可选）

## 4. Phase 2: MuJoCo 物理模型升级

### 4.1 长条车体（80cm × 40cm）
```xml
<geom type="box" size="0.4 0.2 0.25" .../>  <!-- 80cm长 40cm宽 高50cm -->
```
- 替换当前 `cylinder size="0.5 0.5"`

### 4.2 旋转中心
- 当前 `d.qpos[0..1]` 是圆柱中心
- 改为：qpos 表示**车体中心**，前端/后端通过车体朝向偏移
- 转弯：绕中心旋转（yaw 变化），前后端点位置随 yaw 更新

### 4.3 激光雷达位置
- 当前 LIDAR 在机器人原点
- 改为：雷达在头部（距中心 +0.4m 沿朝向方向），离地 0.5m
- `scan(bx, by)` 的起点改为头部位置，而不是车体中心

## 5. Phase 3: 功能实现

### 5.1 里程计定位（替代作弊 qpos）
- 从运动学模型积分：速度 × dt → 位置增量
- 加高斯噪声模拟轮子打滑
- EKF 融合（复用 `~/workspace/slam-ekf/` 的现成实现）

### 5.2 车体运动学
- 差速驱动模型：`v = (vl + vr)/2`, `ω = (vr - vl)/W`
- blocked() 碰撞检测用长条 bounding box（非圆形）
- 贴墙安全距离按长条宽度重新算

## 6. 风险与对策

| 风险 | 对策 |
|------|------|
| EGL 软渲染慢 | 降渲染频率（200步一帧），物理步进不受影响 |
| GLM-5V 小图回答空 | max_tokens≥500，图≥480px |
| 长条碰撞检测复杂 | bounding box + 分段线段检测 |
| 里程计漂移 | EKF + 定期重定位（回门时校正） |

## 7. 验收标准

- Phase 1: baseline 成绩单生成，GLM-5V 至少 3 帧巡检回答有效
- Phase 2: 长条车体在 MuJoCo 中正确运动，LIDAR 从头部扫描
- Phase 3: 里程计定位误差 < 1m（无真值辅助），对比作弊版
