# 2026-08-07 M2 里程碑：障碍渐进避让 — 动态安全距离 + 撞障碍学习闭环

> 成果：**每段直道 1 障碍 / 每段弯道 1 障碍 / 混合 20 障碍全部跑通**，碰撞全 0。
> 关键突破：① 撞障碍学习闭环（STOP→escape→写安全圈→HPA重建→绕行）② 动态安全距离的正确实现 ③ blocked 精确圆判定。

---

## 一、为什么需要"渐进式障碍"

主人渐进方案（08-05 原话）：**先去掉所有障碍只有墙 → 看轨迹 → 每段直道加入一个随机障碍 → 跑通优化 → 每段弯道加障碍 → 逐步增加**。

渐进的原因：一次加满障碍无法区分"算法缺陷"和"场景过难"。每段 1 个是上限（后面发现 2-3 个会物理堵死）。

### M2 全场景数据（seed7，KNOWN_MAP 全程 476m）

| 场景 | 障碍数 | 完成时间 | bounce | collisions |
|---|---|---|---|---|
| 无障碍（v1.0） | 0 | 125.13s | 0 | 0 |
| 直道 1/段（v1.1） | 10 | 280s | 6 | 0 |
| 弯道 1/段（v1.1） | 10 | 266.17s | 0 | 0 |
| 直道+弯道 混合（v1.2） | 20 | 325.76s | **1** | 0 |

**每段最多 1 个是物理上限**：通道仅 5m 宽，1 障碍+安全圈占 ~1.6m，2-3 个横排安全圈叠加把通道堵死（直道 2/段 bounce 277 卡通道1、3/段 bounce 343 卡通道0——不是 bug，是通道装不下）。

---

## 二、撞障碍学习闭环（M2 核心设计）

```
狗撞到障碍（speed≈0 且 前方 d_clear < 0.55m）
  ├─① STOP 识别：附近 1.5m 内有障碍？（_near_obs）
  ├─② escape：全向扫描选【可走距离最大】方向，走 150 步(~2m)
  ├─③ 写安全圈：障碍周围 0.8m 半径格 → static_grid WALL（永久层）
  ├─④ 重建 HPA：门网络/距离场含障碍圈（1.7s）
  └─⑤ 重规划：HPA 新路径绕开安全圈
```

**效果**：每个障碍撞 1 次 → 学习写圈 → 重规划绕行，之后不再撞。混合 20 障碍全程 bounce 仅 1（唯一 1 次是学习第一个障碍）。

---

## 三、动态安全距离（主人指令 08-07）

主人原话："安全距离应该随速度变化。速度低的时候，安全距离相应降低。这样就可以挤过一些狭窄的地方。"

### 3.1 错误实现（踩坑）：动态 blocked 判定 → 死锁

```python
# ❌ 错误：把刹车距离塞进碰撞判定
dyn_clear = OBS_CLEAR + (speed*speed) / (2.0 * A_DECEL)
if math.hypot(wx-ox, wy-oy) < dyn_clear: return True
```

**死锁**：狗离障碍 1.2m，加速到 3m/s 时安全距离 = 0.7+9/16 = 1.26m > 1.2m → **当前位置第一点就 blocked → d_clear=0 → v_brake=0 → 永远无法加速**。高速靠近障碍时反而被自己"困死"。

### 3.2 正确实现：物理边界固定 + 制动约束自适应

```python
# ✓ 物理碰撞边界固定 0.7m（障碍0.5+狗半径0.2），speed 不参与判定
def is_obstacle_world(wx, wy):
    if sample_hf(wx, wy) != ROAD_PIX: return True
    for ox, oy in obs_world:
        if math.hypot(wx-ox, wy-oy) < OBS_CLEAR: return True
    return False

# 动态安全距离由 v_brake 制动约束自动实现：
# 接近障碍 → d_clear 短 → v_brake = √(2·A_DECEL·(d_clear-STOP_MARGIN)) 小 → 限速
# 低速 → 能贴近物理边界 0.7m → 挤过窄缝
```

**原理**：碰撞判定永远用物理边界（确定"能不能走"），速度自适应由制动约束完成（确定"走多快"）。接近障碍自动减速，低速自然能挤窄缝——不需要显式"安全距离随速度变化"。

---

## 四、blocked 精确圆判定（窄缝挤过的关键）

### 旧版（过保守 +0.283m）

```python
def blocked(wx, wy):
    vx, vy = int(wx/VOXEL), int(wy/VOXEL)
    for dy in range(-ROBOT_R, ROBOT_R+1):   # ±2 格 = 0.2m
        for dx in range(-ROBOT_R, ROBOT_R+1):
            if dx*dx+dy*dy <= ROBOT_R*ROBOT_R:
                nx, ny = vx+dx, vy+dy
                if is_obstacle_world((nx+0.5)*VOXEL, (ny+0.5)*VOXEL):
                    return True
```

5×5 邻域格判定：邻域格中心距狗中心最远 √(2²+2²)×0.1 = **0.283m**，加上物理边界 0.7 → 狗实际被挡在障碍 **0.98m** 外。**窄缝（障碍中心距 1.4m）永远挤不过**。

### 新版（精确圆判定）

```python
def blocked(wx, wy):
    if not (0.0 <= wx <= 50.0 and 0.0 <= wy <= 50.0):
        return True
    return is_obstacle_world(wx, wy)   # 狗中心距障碍 <0.7m 直接判定，无格离散冗余
```

狗中心距障碍 ≥0.7m（物理边界）即安全——**能贴 0.7m 边界，窄缝（中心距 ≥1.4m）可挤过**。

---

## 五、写安全圈 0.8m（不是 1.0m）

| 半径 | 通道缝 | 结果 |
|---|---|---|
| 1.0m | (5 - 1.0×2) - 墙膨胀 0.8 = 1.1m < 执行层需求 1.6m | ❌ 挤不过死循环（bounce 269） |
| **0.8m** | (5 - 0.8×2) - 0.8 = 1.7m > 1.6m | ✅ 通过 |

- 0.8m = 物理边界 0.7 + 格偏移 0.1（round 后格中心可能偏 0.05）
- 配合精确圆判定后无 0.283 冗余，0.8m 足够

---

## 六、HPA 门网络静态问题（必须重建）

**现象**：写圈后 HPA plan 返回 **None（0 格）**——狗卡死，bounce 299。

**根因**：`HPAStar.__init__` 构建门网络 + 距离场时用当时的 static 快照。**写圈后门可能被障碍圈堵住**（cell 边界开口被 WALL 覆盖），但粗层门网络不知道 → nearest_gate 返回被堵的门 → 细层 A* 失败 → None。

**修复**：写圈后重建 HPA：
```python
hpa = HPAStar(_hpa_wall, verbose=False)   # 重建 1.7s（含障碍圈），门网络/距离场更新
```

**注意**：v1.0 时重建导致 bounce 1741——那是 **1.0m 圈 + 5×5 判定**的组合把缝堵死；**0.8m 圈 + 精确圆判定后重建可行**。三个修复必须配套，单独一个都不够。

---

## 七、弯道障碍放置（U 型弯内部）

### 错误：放直道末端
```python
ox = rng.uniform(44.0, 49.0)   # ❌ 堵在转弯路径上，狗转弯正撞上，bounce 233
```

### 正确：放 U 型弯内部
```python
ox = rng.uniform(46.5, 49.0)   # ✓ 偶通道右端转弯段内部
ox = rng.uniform(1.0, 3.5)     # ✓ 奇通道左端转弯段内部
```

放直道末端（x=44-46）会堵死转弯入口；放转弯段内部狗绕行空间大。

---

## 八、escape 方向演进（三版）

| 版本 | 逻辑 | 结果 |
|---|---|---|
| v1 朝目标 | `dot*4 + min(d,2)` 评分 | ❌ 选 90° 撞 y=20 墙（score 5.5 > 路径方向 4.7）→ 死循环 |
| v2 远离障碍 | `atan2(by-oy, bx-ox)` | ❌ 远离方向可能 1m 就撞分界墙（如 328° → y=5 墙）→ 死循环 |
| **v3 可走最远** | 全向扫描 `_forward_clear` 最大 | ✅ 选 0° 通畅 4m+，逃出障碍区 |

**核心**：escape 不是"朝哪"的问题，是"哪能走远"的问题——撞墙方向的 d_clear 短，天然被淘汰。

---

## 九、文件变更

| 文件 | 变更 |
|---|---|
| `test_scripts/algo3_headless.py` | 障碍渐进框架 + 撞障碍闭环 + 精确圆判定 + HPA 重建 + ch<N> 目标 |
| `scripts/hpa_star.py` | _astar 实时 wall 检查（含 live grid 障碍）|

## 十、命令速查

```bash
# 直道 1/段（10 障碍）
python test_scripts/algo3_headless.py --seed 7 --known-raw 1 --obs-straight 1 --timeout 380

# 弯道 1/段（10 障碍）
python test_scripts/algo3_headless.py --seed 7 --known-raw 1 --obs-turn 1 --timeout 380

# 混合 20 障碍（最终形态）
python test_scripts/algo3_headless.py --seed 7 --known-raw 1 --obs-straight 1 --obs-turn 1 --timeout 380
```

---

## 十一、仍待办

- [ ] **M2-4 可动障碍**：少量固定范围内慢速移动的障碍（主人路线图下一步）
- [ ] **M3 建图+随机可动障碍**：最后里程碑，稳定顺滑回避
- [ ] 其他形状迷宫（远期）
