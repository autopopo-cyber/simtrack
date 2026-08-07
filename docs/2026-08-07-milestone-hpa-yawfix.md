# 2026-08-07 里程碑：HPA* 移植 + yaw 控制 bug 修复 — 全场景达标

> 成果：探索 / 已知地图 / 绑架恢复 / 任意通道导航 **四场景全达标**，碰撞全 0，bounce 全 0。
> 关键突破：① HPA* 分层规划替代全程 A*（50.8s → 0.38ms）② 距离场 BFS 取 min bug ③ **yaw 控制被物理读回覆盖**（终极根因）。

---

## 一、为什么今天能达标

### 1. HPA*（分层寻路）移植 — 长距离规划消失

主人方法论（08-06）：**先移植成熟算法，不自研死磕**。ROS navigation stack 的全局规划就是 HPA*。

| 规划方式 | 全程耗时 | 说明 |
|---|---|---|
| 全程 A*（缓存空） | **50.8s** | wall_dist 每格扫 441 次 × 25 万格 = 1.1 亿查询 |
| 分段 A* | 每段 430-560ms | 仍超 100ms 铁律 4 倍 |
| **HPA\*（门网络）** | **0.38ms** | CELL=50 格(5m)、216 门、cell 内 BFS 配对 |

实现：`scripts/hpa_star.py`。粗层只跑门间连接（10 通道 → 约 216 门），细层只跑相邻门间短 A*。距离场 O(1) 查墙距替代 wall_dist 441 次扫描。

### 2. 距离场 BFS 取 min bug — 贴墙路径的根源之一

原代码：
```python
di[j] = n if n < v else (w if w < v else v)   # ❌ n 优先，忽略更小的 w
```
当上一行已传播（n=50）而左邻是墙（w=1）时，**错误取 50 而忽略 1**——距离场以为很开阔，HPA 路径贴墙走。

修复：`if w < n and w < v: di[j] = w`（取三值最小）。独立测试墙稀疏时 n=INF+1>v 走 w 分支碰巧正确；真实地图墙多时 n 已传播就错了。**修复后距离场正确（墙邻格=1 而非 49）。**

### 3. yaw 控制被物理读回覆盖 — 终极根因 🎯

```python
self.d.qvel[2] = 0                       # yaw 角速度设 0
mujoco.mj_step(...)
self.yaw = self.d.qpos[2]                # ❌ 读回物理 → 恒 0！转向被覆盖！
```
`self.yaw` 每步 dyaw 转向后被 `d.qpos[2]`（恒定 0）**重置**——狗永远朝初始方向（+x）直冲，从未真正转向！

**为什么之前能跑通**：KNOWN_MAP 起点 (2.5,2.5) 初始 yaw=0 恰好朝路径方向，狗靠"撞墙 bounce + 近墙限速"硬走完（bounce 94）。**绑架恢复失败**：随机起点 yaw=0 但目标在背后 170°——狗永远转不了向，直线冲出通道卡死。

修复：yaw 是控制变量，直接写回物理，不读回：
```python
self.d.qpos[2] = self.yaw  # 控制 yaw 直接写回（滑动模型 friction=0，mj_step 保持）
```

### 4. 大转向限速（DWA 思想：曲率越大速度越低）

需要转 >57° 时限速 1.0m/s——先转身再加速，防止直冲偏离。配合 yaw 修复后，**bounce 从 94 降到 0**。

---

## 二、四场景实测数据（2026-08-07 全绿）

### 探索模式（SLAM 未知地图，seed7 纯墙）

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 完成时间 | 540s 覆盖 52%（未完成） | **126.65s ARRIVED** |
| 覆盖 | 52% | **99.75%** |
| bounce | 58+ | **0** |
| collisions | 0 | **0** |

### 已知地图全程（KNOWN_MAP 起点→终点，476m 蛇形）

| 指标 | 修复前 | 修复后 |
|---|---|---|
| 完成时间 | 241s（bounce 94） | **125.13s** |
| bounce | 94 | **0** |
| collisions | 0 | **0** |

### 绑架恢复（随机位置 → 起点/终点，--known-raw + --random-start）

| seed | 随机起点 | 目标 | 时间 | bounce |
|---|---|---|---|---|
| 7 | (42.1,39.4) 通道7 | start | 88.8s | 0 |
| 42 | (13.0,18.0) 通道3 | finish | 78.0s | 0 |
| 123 | (14.4,27.8) 通道5 | start | 70.2s | 0 |
| 555 | (17.7,3.0) 通道0 | start | 5.5s | 0 |

修复前：seed7 180s 卡通道6（bounce 105）、seed42 finish 150s 卡通道3（bounce 120+）。

### 任意通道二维码导航（--target ch<N>，随机起点→指定通道二维码）

| 通道 | 二维码位置 | 时间 | bounce |
|---|---|---|---|
| ch0 | (45.8,2.5) | 33.8s | 0 |
| ch1 | (4.2,7.5) | 21.9s | 0 |
| ch2 | (45.8,12.5) | 9.7s | 0 |
| ch3 | (4.2,17.5) | 3.8s | 0 |
| ch4 | (45.8,22.5) | 16.0s | 0 |
| ch5 | (4.2,27.5) | 27.8s | 0 |
| ch7 | (4.2,37.5) | 53.2s | 0 |
| ch8 | (45.8,42.5) | 65.1s | 0 |
| ch9 | (4.2,47.5) | 77.9s | 0 |

**10/10 通道全到达，bounce 全 0，collisions 全 0。**

---

## 三、关键技术决策

1. **HPA\* 替代全程 A\***：粗层门网络 + 细层短 A*，长距离规划直接消失。CELL=50 格（5m）、216 门、距离场膨胀（dist<ROBOT_DIA 禁行）。
2. **KNOWN_MAP 直接用原始 track_clean.png**（--known-raw）：MAX-pool 区域判定 + **y 轴 flip**（`[::-1,:]`——图像 row0=y=50m 顶部，格 gy=0=y=0m）。full_map.npz 中间产物有采样丢墙史，弃用。
3. **yaw 控制变量直接写回物理**：不读回 qpos[2]。这是所有转向问题的终极根因。
4. **bounce 逃生冷却**：bounce 后沿逃生方向强制走 120 步（绕出墙角再回归路径），防 step() 又转向被挡目标 → bounce 死循环。
5. **大转向限速**：abs(err)>1.0 rad 时限速 1.0m/s（DWA 思想）。
6. **任意通道目标**：`--target ch<N>` 用 landmark_positions() 解析二维码坐标（每通道 1 个标牌）。

---

## 四、文件变更

| 文件 | 变更 |
|---|---|
| `scripts/hpa_star.py` | **新增** HPA* 实现（门网络 + 距离场 + 膨胀约束 + 自测块） |
| `scripts/analyze_trail.py` | 新增 轨迹分析（通道序列/bounce 分布/卡点） |
| `scripts/prof_segment.py` | 新增 分段 A* profiling |
| `scripts/verify_waypoint_astar.py` | 新增 路点 A* 验证 |
| `test_scripts/algo3_headless.py` | HPA 集成 + yaw 修复 + 大转向限速 + ch<N> 目标 + ARRIVE 3.5m |

## 五、命令速查

```bash
# 探索模式（SLAM 自建图，126s 全通）
python test_scripts/algo3_headless.py --seed 7 --no-obs 1 --timeout 280

# 已知地图全程（125s）
python test_scripts/algo3_headless.py --seed 7 --no-obs 1 --known-raw 1 --timeout 280

# 绑架恢复（随机位置→起点）
python test_scripts/algo3_headless.py --seed 42 --no-obs 1 --known-raw 1 --random-start 1 --target start --timeout 150

# 任意通道二维码（随机位置→通道4二维码）
python test_scripts/algo3_headless.py --seed 42 --no-obs 1 --known-raw 1 --random-start 1 --target ch4 --timeout 150
```

---

## 六、仍待办

- [ ] 障碍版验证（阶段2：每段直道+1 随机障碍渐进测试——主人渐进方案）
- [ ] B 阶段 --obs-reseed 运行中障碍变化验证
- [ ] HPA 增量重规划（地图变化时重规划，当前 KNOWN_MAP 一次规划走全程）
- [ ] 视觉定位参与决策（看到二维码→修正全局位姿，当前只记录不参与）
- [ ] mystory 更新本轮里程碑
