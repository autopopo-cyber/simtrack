# simtrack — 机器狗 50m 蛇形迷宫导航

> **核心索引** · 2026-08-07 M2 里程碑更新
> 项目：`~/workspace/simtrack/` · 分支：`exp/dog-navigation`（master 干净）
> 主战场：`test_scripts/algo3_headless.py`（萤火 Firefly v3 SLAM 导航）

---

## 一、项目是什么

MuJoCo 仿真中，机器狗在 **50×50m 蛇形迷宫**（10 条水平通道 + U 型弯，路径约 **476m**）里自主导航。核心约束（主人铁律）：**现实中不允许碰撞**——运动学约束避障（前瞻测距 + 制动约束 v≤√(2·A_DECEL·d)），碰撞=0。**不作弊**——SLAM 建图，不用世界真值规划。

## 二、目标（主人 10 小时开发指令，2026-08-05 晚）

| # | 目标 | 状态 |
|---|---|---|
| ① | **探索效率**：未知地图从头扫描建图更优 | ✅ **126.65s 全程 + 99.75% 覆盖**（08-07） |
| ② | **已知地图快速寻路**：障碍变化后复用旧地图 | ✅ **125.13s 全程 + bounce 0**（08-07，HPA*） |
| ③ | **绑架恢复**：随机位置快速到达起点/终点 | ✅ **4/4 全到达 5.5~89s bounce 0**（08-07） |
| 主线 | **任意二维码导航**：每通道 1 标牌，导航到任意通道 | ✅ **ch0-9 全通道 100% 到达 bounce 0**（08-07） |
| 主线 | **墙面二维码视觉定位**：ArUco 码+数字，狗看码人看数字 | ✅ 完整落地（10 标牌识别 + 级联粗筛 + 相机修正） |

### M2 障碍渐进（主人 08-07 追加路线图）
| # | 里程碑 | 状态 |
|---|---|---|
| M2-1 | **每段直道 1 障碍** | ✅ **280s 全程 bounce 6**（v1.1） |
| M2-3 | **每段弯道 1 障碍** | ✅ **266s 全程 bounce 0**（v1.1） |
| M2-2/3 | **混合 20 障碍**（直道1+弯道1，最终形态） | ✅ **325.76s 全程 bounce 1**（v1.2） |
| M2-4 | **少量慢速可动障碍** | ⏳ 下一步 |
| M3 | **建图 + 随机可动障碍**（最后里程碑） | ⏳ |

## 三、当前系统能力（2026-08-07 实测全绿）

### 导航（四场景全达标）
- **探索模式**（SLAM 自建图）：**126.65s ARRIVED，覆盖 99.75%**，bounce 0
- **已知地图模式**（--known-raw + HPA*）：**125.13s 全程 476m**，bounce 0
- **绑架恢复**（随机位置→起点/终点）：**4/4 全到达**（5.5~88.8s），bounce 0
- **任意通道导航**（--target ch<N>）：**ch0-9 全到达**（3.8~77.9s），bounce 0
- **碰撞全 0** ✅（所有测试）
- 步速 ~200 步/s；HPA 全程规划 **0.38ms**（原全程 A* 50.8s）

### 障碍避让（M2，撞障碍学习闭环）
- **每段直道 1 / 弯道 1 / 混合 20 全部跑通**，bounce 1-6，碰撞 0
- **动态安全距离**：物理边界固定 0.7m + v_brake 制动约束自适应（低速挤窄缝）
- **blocked 精确圆判定**：去 5×5 邻域 +0.283m 过保守冗余
- **写圈 0.8m + HPA 重建**：写圈后门网络含障碍，重规划绕行
- 详见 → [docs/2026-08-07-obs-progressive-milestone.md](docs/2026-08-07-obs-progressive-milestone.md)

### 核心技术（08-07 里程碑）
- **HPA\* 分层寻路**（`scripts/hpa_star.py`）：门网络替代全程 A*，长距离规划消失
- **yaw 控制修复**：控制变量直接写回物理（不再被 qpos[2] 读回覆盖）——终极根因
- **距离场 BFS min bug 修复**：墙邻格距离正确（1 而非 49），路径不贴墙
- **大转向限速**（DWA 思想）：>57° 限速 1.0m/s，bounce 94→0
- 详见 → [docs/2026-08-07-milestone-hpa-yawfix.md](docs/2026-08-07-milestone-hpa-yawfix.md)

### 视觉二维码标牌
- **10 个标牌**（每通道 1 个）立在通道中心线终点端，2m×2m，DICT_7X7，中心离地 1m = 相机高度
- **图像金字塔多尺度**（0.25x~3x 六档）+ **级联粗筛**（0.25x Laplacian 能量 <200 跳过 → 0.3ms vs 25ms）
- 相机 roll 修正：`euler="0 -1.5708 -1.5708"`（天空在上、地面在下）
- 识别率：10/10 标牌双方向全绿；导航中持续识别

### 迷宫与障碍
- 476m 蛇形迷宫**完全连通**（BFS 23.5 万格全通，起点→终点无墙阻断）
- 障碍：**固定位置**（seed 决定，seed7=10 个，半径 0.5m 圆柱）；`--obs-reseed` 可运行中换位置
- 另有 **45° 薄斜墙**（hfield 地形自带，0.1m 厚）——早期 bounce 主要来源，yaw 修复后 bounce=0

## 四、文档索引

### 里程碑 / 复盘（最重要！）
| 文档 | 内容 |
|---|---|
| [**2026-08-07-obs-progressive-milestone.md**](docs/2026-08-07-obs-progressive-milestone.md) | **M2 障碍渐进：动态安全距离 + 撞障碍学习闭环 + 混合 20 跑通** |
| [2026-08-07-milestone-hpa-yawfix.md](docs/2026-08-07-milestone-hpa-yawfix.md) | HPA\* 移植 + yaw bug + 距离场 min bug — 四场景全达标 |
| [2026-08-06-retrospective.md](docs/2026-08-06-retrospective.md) | 48 小时开发复盘（视觉花 70% 时间但核心目标未验收完的审视） |
| [2026-08-05-dog50-maze-pitfalls.md](docs/2026-08-05-dog50-maze-pitfalls.md) | 全部踩坑「现象→根因→修复」（卡死三连环/斜墙漏扫/相机roll/box墙/M2六坑） |

### 架构/设计（docs/superpowers/specs/）
| 文档 | 内容 |
|---|---|
| `2026-08-05-nav-efficiency-design.md` | 10 小时目标设计：双层地图/射线清除/Voronoi 走中间/两阶段/绑架/标牌贴墙 |
| `2026-08-06-vision-cascade-design.md` | 二维码标牌系统：金字塔级联粗筛 + 2m 贴墙标牌 |
| `2026-07-11-algo3-firefly-dog50-design.md` | 萤火 v3 导航核心设计（早期）|
| `2026-07-10-firefly-vision-loop-design.md` | 视觉闭环设计（早期）|
| `2026-07-10-dog-cylinder-design.md` | 机器狗圆柱建模（早期）|

### 实现计划（docs/plans/）
| 文档 | 内容 |
|---|---|
| `2026-08-05-nav-efficiency.md` | 10 小时目标实现计划（289 行，Task1-9）|
| `2026-08-06-vision-cascade.md` | 视觉级联实现计划（4 Task）|
| `2026-07-11-algo3-firefly-dog50.md` / `2026-07-10-*.md` | 早期计划 |

### 记忆系统（mystory）
- 话题「机器人仿真」·story.md — 长期记忆（待补 08-07 里程碑）

## 五、关键文件

| 文件 | 作用 |
|---|---|
| `test_scripts/algo3_headless.py` | 主战场：SLAM/门探索/HPA*/执行层/视觉全部 |
| `scripts/hpa_star.py` | **HPA\* 分层寻路**（门网络 + 距离场 + 膨胀约束）|
| `scripts/analyze_trail.py` | 轨迹分析（通道序列/bounce 分布/卡点）|
| `scripts/prof_segment.py` / `scripts/verify_waypoint_astar.py` | A* 分段验证/profiling |
| `test_scripts/landmarks.py` | 标牌系统：位置/墙/XML（10 标牌 + 分界墙留转弯口 + 背景板）|
| `test_scripts/vision_landmark.py` | 视觉识别：ArUco + 金字塔多尺度 + 级联粗筛 |
| `scripts/gen_aruco_landmarks.py` | 生成 30 个 ArUco 标牌 PNG（DICT_7X7）|
| `assets/landmarks/aruco_XX.png` | 标牌纹理（前 10 个在用）|
| `confirmed/track_clean.png` | 碰撞原图（2000×2000，128=路 191=墙）**KNOWN_MAP 唯一地图源** |

## 六、常用命令

```bash
# 探索模式（SLAM 自建图，126s 全通）
python test_scripts/algo3_headless.py --seed 7 --no-obs 1 --timeout 280

# 已知地图全程（125s，HPA*）
python test_scripts/algo3_headless.py --seed 7 --no-obs 1 --known-raw 1 --timeout 280

# 绑架恢复（随机位置→起点）
python test_scripts/algo3_headless.py --seed 42 --no-obs 1 --known-raw 1 --random-start 1 --target start --timeout 150

# 任意通道二维码（随机位置→通道4二维码）
python test_scripts/algo3_headless.py --seed 42 --no-obs 1 --known-raw 1 --random-start 1 --target ch4 --timeout 150

# 视觉标牌测试
python test_scripts/algo3_headless.py --landmarks 1 --vision 1 --timeout 60

# 混合 20 障碍（直道1+弯道1，M2 最终形态）
python test_scripts/algo3_headless.py --seed 7 --known-raw 1 --obs-straight 1 --obs-turn 1 --timeout 380
```

## 七、本轮收获（08-05 ~ 08-07）

1. **撞障碍学习闭环**（08-07 M2）：STOP识别→escape(可走最远)→写安全圈(static_grid)→HPA重建→重规划。每障碍撞 1 次即学会绕行，混合 20 全程 bounce 1
2. **动态安全距离**（08-07 主人指令）：物理边界固定 0.7m + v_brake 制动约束自适应——接近障碍自动减速，低速挤窄缝。**错误做法**（刹车距离塞进碰撞判定）会死锁：狗离障碍 1.2m 加速到 3m/s 安全距离 1.26m > 1.2m → 永远无法加速
3. **blocked 精确圆判定**（08-07）：去 5×5 邻域 +0.283m 过保守冗余——窄缝（中心距 ≥1.4m）可挤过
4. **HPA\* 分层寻路**（08-07）：全程 A* 50.8s → HPA* 0.38ms（13 万倍）。主人方法论"先移植成熟算法"的胜利
5. **yaw 控制变量直接写回物理**（08-07）：`self.yaw = d.qpos[2]` 读回导致每步转向被重置——所有 bounce/卡死的终极根因，修复后 bounce 94→0
6. **距离场 BFS 取 min**（08-07）：`n if n<v else (w if w<v else v)` 忽略更小 w——墙邻格被误判开阔 49 倍
7. **MAX-pool y 轴 flip**（08-06）：图像 row0=y=50m 顶部，格 gy=0=y=0m——不翻转则墙位置上下颠倒（早期所有诡异卡点的根源）
8. **薄斜墙漏扫**（08-06）：LIDAR_RAYS=120 在 15m 处射线间距 0.78m > 斜墙 0.1m 厚 → 漏扫。360 射线修复
9. **级联检测**（08-06）：低分辨率 Laplacian 能量判别"有无特征"——无标牌帧 25ms→0.3ms

## 八、待办 / 未完成

- [ ] **M2-4 可动障碍**：少量固定范围内慢速移动（主人路线图下一步）
- [ ] **M3 建图+随机可动障碍**：最后里程碑，稳定顺滑回避
- [ ] **B 阶段**：--obs-reseed 运行中障碍变化后的重规划
- [ ] **HPA 增量重规划**：地图变化时只重算受影响区域（当前写圈后全量重建 1.7s）
- [ ] **标牌与导航联动**：看到二维码 → 定位修正参与决策（当前只记录不参与导航）
- [ ] **mystory 更新**：机器人仿真 story.md 补 08-07 里程碑
- [ ] 早期遗留：seed 99/170456 到达终点（差 8m/3m，加时即可）
