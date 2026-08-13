# PRD — 机器狗室内迷宫/房间自主导航（SLAM + Frontier 探索）

> **本文是项目的稳定入口（living document，非日期稿）。** 会话压缩后从这里恢复上下文。
> 最后更新：2026-08-12 · 维护者：俊秀 · 状态：复审中（待主人确认改进方向）
>
> 阅读顺序：[一、目标](#一目标与成功标准) → [二、当前真实状态](#二当前真实状态诚实) → [四、架构](#四系统架构与数据流) → [六、已知问题根因](#六已知问题与根因诊断) → [七、改进方向](#七改进方向按优先级算力感知)。
> 如只看一节，看 [六、已知问题根因](#六已知问题与根因诊断)（卡慢的真相）。

---

## 一、目标与成功标准

### 1.1 项目目标
MuJoCo 仿真里，四足机器狗在**未知室内环境**（当前：50×50m 蛇形迷宫，可扩展到普通房间）自主完成：
1. **SLAM 建图**——只用激光雷达 + 相机，不用世界真值（铁律："不作弊"）
2. **Frontier 探索**——发现门（已知/未知边界）、确认门后是什么、找到终点
3. **避障导航**——运动学约束保证 **collision=0**（铁律："现实中不允许碰撞"）

最终要迁移到**真实机器狗**（宇树 A2 级，RK3588 算力，2D 激光 + 相机 + IMU + 腿式里程计）。

### 1.2 成功标准（可量化）
| 指标 | 目标 | 当前（2026-08-12 实测） | 判定 |
|---|---|---|---|
| 50m 迷宫到达 | 物理 150–200s 内到达终点（500m / ~3m/s 均速） | 纯墙 490s / 81240 步（**慢 2.5×**） | ❌ 不达标 |
| collision=0 | 全程零碰撞（算法禁行保证） | 0（实测复现） | ✅ |
| 探索+建图 | 无地图先验、纯感知栅格 | 真实（95.7% 覆盖，墙 recall 高） | ✅ |
| 决策层零真值 | 定位/DWA 速度/经验墙不用真值 | 真实（2026-08-12 整改，源码+实跑核实） | ✅ |
| 运行稳定性 | 多次跑都到达，方差小 | **bounce 从 0 到 159617（灾难性方差）** | ❌ 严重不达标 |
| 换房间可迁移 | 门宽/尺寸/地图源可配置 | 硬编码 50×50、PASS_CLEAR 按 5m 走廊调 | ❌ 待做 |

> **关键诚实声明**：能力是真的（探索/建图/零碰撞/无真值定位都切实做到），但**性能数字和稳定性不达标**。详见第二节和第六节。

### 1.3 非目标（明确不做）
- ❌ 不做 3D / 多层 / 台阶（纯 2D 平面假设）
- ❌ 不做动态环境的人群语义识别（移动障碍按刚体圆盘 1m/s 处理）
- ❌ 不刻意制造机器人无法识别的场景（玻璃墙、纯白无特征墙）——主人原则

---

## 二、当前真实状态（诚实）

### 2.1 能做到的（已验证，附证据）
| 能力 | 证据 |
|---|---|
| SLAM 建图（无先验） | 实跑：FREE 纯度高、墙 recall 高、95.7% 覆盖；scan() 物理投射(真值)/写图(估计)分离 |
| Frontier 探索（门发现/选择） | find_gates + pick_gate advance 评分；幽灵门/伪前沿门多重过滤 |
| collision=0 | 制动约束 v≤√(2·A·d) + 滚动局部层兜底；实跑 0（注：MuJoCo contype=0 物理上也不接触，"0"是算法禁出来的） |
| 决策层零真值定位 | 默认 `--odom 1`；主循环 L1966 把真值覆盖为估计位姿；scan-matching + 二维码修正 |
| 感知跟踪障碍速度 | ObstacleTracker（聚类+关联+EMA），DWA 只吃 `tracker.moving()`（默认） |
| 经验墙感知确认 | HIT_CONFIRMED（激光直命中）替代 sample_hf 真值 |
| scan-matching 激光里程计 | scan_matching.py 网格搜索，实跑 380 次/8124 次尝试（4.7%） |

### 2.2 做不到 / 不达标的（同样诚实）
| 问题 | 实测 |
|---|---|
| **速度太慢** | 纯墙 490s（应 150-200s）；22% 步数原地不动；移动均速仅 1.47m/s（上限 4） |
| **运行方差灾难性** | 同 seed7，bounce 从 0 到 159617 都有；取决于里程计 RNG |
| **特征障碍/混合场超时** | README 已标 ⚠️，长直道/弯口磨蹭吃掉预算 |
| **数字不可复现** | README 称 322s/278bounce，实测 490s/430bounce（+52%） |
| **换房间不可迁移** | 50×50 硬编码、PASS_CLEAR=0.6 按 5m 走廊调、地图源绑死 track_clean.png |
| **无回环检测** | scan-matching 只压局部漂移，大环形建筑会双墙 |

### 2.3 已废弃 / 死代码（待清理）
- **`simtrack/scan_match.py`**：cv2 距离场版 ScanMatcher，**未被 import**（主脚本用 `scan_matching.py`）。`docs/2026-08-11-scan-match-odom.md` 描述的就是它，底部 `RESULTS-PLACEHOLDER` 至今未填。→ 应删除或合并。
- **风暴隔离 `_STORM`**：恒 False（L1972 注释，实测切断射线清除自救→死地）。
- bounce_launcher.sh 引用的 algo0/1/2 早期脚本（已被 algo3 取代）。

---

## 三、仿真 vs 真实环境差距（迁移风险）

> 主人要求：确认机器狗在"普通室内、可能有长走廊、不刻意制造无法识别场景"的屋子里能跑完。

### 3.1 当前仿真特有的"不真实"（会让仿真偏乐观，迁移会变难）
| 仿真 | 真实 | 影响 |
|---|---|---|
| 激光投射用**真值位姿**（狗物理在哪射线从哪出） | 真实激光装在机身上随机抖动，有 **机械标定误差/振动** | 仿真点云相对机身"太干净"，真机有额外噪声 |
| 移动障碍 = 0.5m 圆柱、1m/s、mocap 渲染 | 行人形状不规则、速度 0.5-1.5m/s、会突然变向 | ObstacleTracker 质心法对行人不够鲁棒 |
| 墙 = hfield 地形，**完美刚性、无门窗洞口** | 真实墙有门、窗、家具、踢脚线 | 仿真墙是理想几何 |
| 地面**绝对平整**（friction=0 滑动） | 地毯/瓷砖/门槛，腿式里程计误差更大 | 真机 5%/s 漂移模型可能仍偏乐观 |
| 相机 hfield 渲染，二维码**绝对平贴墙** | 真实二维码反光/遮挡/脏污 | 视觉识别率会降 |
| 滑动模型，**原地转向无足迹代价** | 真实四足转身有最小角速度+净空需求 | 窄通道转身行为不可迁移 |
| **contype=0**：物理上根本不接触墙 | 真机撞了就是撞了 | 仿真的"collision=0"没有真机保险 |

### 3.2 真实房间会遇到、仿真没覆盖的
| 问题 | 根源 | 严重度 |
|---|---|---|
| **房门 0.8-1.0m** 被 PASS_CLEAR=0.6 当栅栏陷阱封闭 | 门宽阈值按 5m 走廊调 | 致命 |
| **长走廊无特征** → scan-matching 沿走廊方向退化 | ROS 已知对称性退化；对策=特征锚点 | 高（主人已用固定障碍缓解） |
| **大空间/开阔大厅** → 前沿呈斑点簇、朝向保持失效 | 评分按走廊调 | 中 |
| **薄墙/玻璃墙** | 墙厚先验 2 格；激光看不见玻璃 | 中 |
| **多房间拓扑**（门后是另一房间而非走廊） | 当前"门=前沿"，无房间分割 | 中 |
| **回环路径**（绕一圈回原地） | 无位姿图回环，漂移→双墙 | 高（大建筑） |

→ 详见 [七、改进方向 §7.4 迁移适配](#74-换房间迁移)。

---

## 四、系统架构与数据流

### 4.1 概念链（门-墙-前线 = frontier + 拓扑）
```
探索哲学（主人定义）：
  探索不是把地图画完再走，而是不断【发现门 → 走向门 → 确认门后是什么】的过程
  每个【门】都可能是疑似终点——答案隐藏在未知后

门(gate)   = 已知FREE 与 未知UNKNOWN 的边界格（= ROS frontier）
            + 过滤：开阔前沿(5×5无墙) + 门宽度(PASS_CLEAR≥0.6m) + 黑名单
墙(wall)   = blocked() 判定不可通行（全局地图 G/SG 的 WALL 格 + keepout）
前线       = 门聚类后的 region（pick_gate 用 advance 评分选最优）
终点       = 相机看到绿球 → 多帧方位角三角定位 → finish_est → 视觉伺服直奔
```

### 4.2 数据流（写图 → 判定 → 决策 → 执行）
```
┌─ scan() [L570] ──────────────────────────────────────────────────┐
│ 物理投射(真值位姿) → 命中点 → scan-matching修正odom →             │
│ 写图(估计位姿): G=FREE(射线清除)/WALL(命中+墙厚先验+掠射填充),    │
│                LOCAL_STAMP(命中时间戳), HIT_CONFIRMED, OBS_SEEN  │
└──────────────────────────────────────────────────────────────────┘
        │ G, SG
        ├─→ blocked() [L848] 读 G+SG（⚠️ 不读 LOCAL_STAMP）
        │     → _forward_clear [L1362] (blocked OR LOCAL_STAMP)
        │     → STOP/bounce/escape [L1446] → 硬防穿墙 [L1512]
        │     → blocked_batch [L862] (DWA, 叠加 LOCAL_STAMP)
        ├─→ PG=SG⊕G → DIST/PASS净空场 → find_gates门宽度过滤
        └─→ 主循环: path耗尽/bounce>8/need_replan → replan
              → find_gates → pick_gate(advance评分) → fine_path/A*/HPA*
              → pure-pursuit(path_idx推进) → DWA choose_velocity → mv.step
```

### 4.3 关键常量（调参锚点）
| 常量 | 值 | 含义 | 调参风险 |
|---|---|---|---|
| `VOXEL` | 0.1m | 感知格 | 改小→算力爆炸 |
| `SPEED_MAX` | 4 m/s | 最大速度 | 窄处过冲 |
| `YAW_RATE` | 1.5 rad/s | 转向速率 | 提高→窄缺口过冲卡死 |
| `STOP_MARGIN` | 0.4m | 停车安全余量 | |
| `LOOKAHEAD` | 4.0m | 前瞻测距上限 | |
| `OBS_CLEAR` | 0.7m | 障碍膨胀碰撞圈 | 窄走廊封死 |
| `PASS_CLEAR_M` | 0.6m | 门净宽阈值 | **真房间门会被封** |
| `WALL_MAP_RANGE` | 10m | odom模式墙标记最远距离 | |
| `LOCAL_WIN` | 60扫(~3s) | 滚动局部层有效期 | |
| `MATCH_RANGE` | 8m | scan-matching 命中点最远距离 | |

---

## 五、核心算法（逐模块）

> 每个算法的详细实现见 `test_scripts/algo3_headless.py` 对应行号；踩坑见 [第九节](#九踩坑总账索引)。

### 5.1 定位栈（无真值）
分层修正，对标 ROS `robot_localization + Cartographer + AMCL`：
1. **Odometry**（`simtrack/odometry.py`）：腿式航位推算模型。线速 5%/s 量级（慢变偏差 ±6% + 1% 白噪），陀螺偏置 ±0.3°/s 封顶。→ `odom.update()` 积分。
2. **ScanMatcher**（`simtrack/scan_matching.py`）：scan-to-map 激光里程计。当前帧墙命中点 vs 已建地图墙掩码，(dx,dy,δyaw) 粗-细网格搜索，得分=落墙比例；零偏移优先（防退化方向噪声拖动）；限幅+增益<1。→ 写图前修正 `odom`。
3. **二维码绝对修正**：看到标牌N→反解狗位姿指数拉回（`odom.correct()`）。
- **已知退化**：长直走廊沿墙方向平移不可观测（ROS 已知对称性）→ 对策=二维码锚点 + 直道特征障碍（`--obs-feature 1`）。

### 5.2 scan() 写图（`algo3_headless.py:570`）
- **物理/估计分离**（2026-08-12）：投射用真值位姿（激光从真实位置出），写图用估计位姿（狗以为自己在哪）。这是真实 SLAM 形态。
- **像素级步进** SCAN_STEP=0.025m（1px）：跨不过 <0.1m 真缝（修"幽灵门"）。
- **掠射填充**：相邻射线都命中→沿命中点连线补WALL（+自由空间反证，>25%已FREE则拒填）。
- **墙厚先验**：命中点往里 +0.1/+0.2m 也标WALL（防墙脸伪前沿门）。
- **射线清除**：射线穿过的格标FREE（唯一解堵通道）。
- **WALL_MAP_RANGE=10m**（odom模式）：远处命中不标WALL，走近再标。

### 5.3 门发现与选择（find_gates L1037 / pick_gate L1183）
- **门** = 未知前沿格，过滤：开阔前沿(5×5无墙) + clear_ok(PASS净空) + 黑名单(bad_gates/dead_gates)。
- **聚类** cluster_gates：4-连通BFS，质心+门格集合（黑名单按集合拉黑）。
- **advance评分**：`0.55·advance + 0.25·(1/dist) + 0.20·(size/50)`。看到终点→advance=朝终点投影；没看到→朝向保持（沿走廊，**无特权**，不读FINISH真值）。

### 5.4 路径规划
- 探索：find_gates→pick_gate→fine_path。
- 已知地图：**HPA\*** 分层（`scripts/hpa_star.py`，全程 A\* 50.8s→0.38ms）。
- fallback：三级跳 A\*（jump_steps）。

### 5.5 执行层（Mover L1383）
- 前瞻测距 `_forward_clear` → 制动约束 `v≤√(2·A_DECEL·(d-STOP_MARGIN))`。
- 近墙限速：d_clear<2m 时 v≤1.5。
- 大转向限速：>57° 限速 1.0（先转身再加速）。
- DWA（`simtrack/algorithms/dwa.py`）：速度空间采样，障碍运动预测=ObstacleTracker估计。

### 5.6 滚动局部层（LOCAL_STAMP，2026-08-12）
ROS rolling obstacle layer 思想：每次扫描的直接命中格打时间戳（3s窗口）。**设计意图**：全局地图漂移时执行层仍看到新鲜真墙。**实现断层见 [六.3](#63-概念链缺口门-墙-前线缺什么)。**

---

## 六、已知问题与根因诊断

> 这是当前最关键的章节。性能不达标的根因，都有实跑证据。

### 6.1 症状：慢 + 卡 + 方差大
- **慢**：纯墙 490s（应 150-200s）；22% 步数原地不动；移动均速 1.47m/s（上限4）。
- **卡**：狗在 **U 型弯区域**（如 4.x, 24.8 / 4.9, 29.9）反复 bounce，y 几分钟只涨几米；最坏卡在边界墙 (30.2, 0.8) 自旋 105 圈。
- **方差**：同 seed7，bounce 从 0 到 159617；README 的 322s 无法稳定复现。

### 6.2 根因（三个叠加）

**根因 A：里程计 RNG 决定成败（元凶）**
Odometry 的 bias random-walk（线速 ±6%、陀螺 ±0.3°/s）每次跑落点不同：
- 温和时（bias ~2%）：漂移 ≤1m，scan-matching 压得住，能跑通（76 bounce）。
- 恶劣时（bias 触顶 -5~6%）：漂移 2-3m，**全局地图被写成幻影墙**，后续一切崩（159617 bounce）。
- 证据：b76 跑 drift=0.41m；490s 跑 drift=1.95m（bias=-5.11%）；trail 历史方差巨大。
- **本质**：5%/s 纯推算 + 稀疏地标是真实四足级难题，scan-matching 的修正率（~5%）在某些漂移轨迹下压不住。

**根因 B：U 型弯楔入 + 逃逸不可靠**
- 狗追门/路径冲进走廊端头墙角 → 原地自旋（yaw 实测转 100+ 圈，位置 std 0.035m）。
- escape 逻辑（全向扫最远净空方向）原理对，但：①`_forward_clear` 先调 `blocked()`，真实角落多数方向本就堵；②escape_steps 过后 path-following 又拉回同一个门；③replan 拉黑当前门，但新门可能绕回同一区域。
- 叠加根因 A 的幻影墙 → 连"退回原路"方向也被堵 → 永冻。

**根因 C：地图损坏无法自愈**
- 全局地图 G 是**每帧按估计位姿写的活栅格**，漂移→墙错位→幻影墙。这与 ROS 的 static layer（来自位姿优化过的 SLAM，几何一致）本质不同。
- `blocked()`（执行层主判定）读的就是这个可能损坏的 G。
- 无"我迷路了"检测：漂移损坏地图时，没有信号触发重定位（额外匹配/减速/自旋重扫）。狗只能磨到超时或靠运气。

### 6.3 概念链缺口（门-墙-前线缺什么）
对比 ROS costmap_2d 标准（见 `docs/research/ros-costmap-and-rk3588.md` 调研）：

| 维度 | ROS 标准 | 本项目 | 缺口 |
|---|---|---|---|
| **全局/局部 costmap 分离** | global(static-ish,规划用) vs local(rolling+raytrace清除,控制用) | `blocked()` 一个函数同时服务规划和执行 | **执行层应读 rolling local costmap** |
| **static 被局部free观测清除** | 默认 `combination_method=Max`，static 不被覆盖（安全） | `blocked()` 不读 LOCAL_STAMP | 与 ROS 默认一致，**非主要问题**（见下） |
| **恢复行为** | nav2 标准：ClearCostmap / Spin / BackUp / Wait | 只有 bounce+escape（原地转） | **缺 BackUp 后退 + 主动 Spin 重扫** |
| **丢失检测** | 局部极小→recovery→重触发 frontier | 无显式"lost"状态 | **缺漂移/卡死检测→重定位** |
| **房间拓扑** | frontier + 房间/门分割 | 纯 frontier（门=前沿） | 多房间场景缺拓扑层 |

**关键澄清（修正初判）**："`blocked()` 不被局部层清除"本身**不算违背 ROS**——ROS 默认 Max 融合也是 static 不被覆盖。真正问题是 **G 不是干净的 static（它是漂移污染的活栅格）**，把它当 static 用就错了。**修法不是让 blocked 可被局部清除（不安全），而是：(1)提升 SLAM 一致性（位姿图/回环）；(2)执行层读独立的 rolling local costmap（raytrace 清除过的干净副本）。**

### 6.4 性能预算分解（为何 490s 而非 200s）
500m 路程，理论 500/4=125s（全速）。实际 490s 的去向：
| 项 | 估算 | 占比 |
|---|---|---|
| 有效移动（78% 时间 × 1.47m/s） | ~380s | 78% |
| 原地自旋/卡死（22% 时间） | ~110s | 22% |
| 路径冗余（绕路、回头、replan） | 未量化 | 显著 |

移动均速 1.47m/s（非 4）的原因：近墙限速（d_clear<2m→1.5）+ 大转向限速 + U弯密集减速。**提速度的关键不是提高 SPEED_MAX，而是减少卡死 + 让狗敢在走廊中段跑满速。**

---

## 七、改进方向（按优先级，算力感知）

> 目标硬件：**RK3588**（4×A76 2.4G + 4×A55 + Mali-G610 + 6TOPS NPU）。调研结论：**2D SLAM 全栈对 RK3588 是富余算力**（slam_toolbox 在 Pi4 上都能跑，RK3588 强 1.5-2×）。瓶颈不在算力，在算法鲁棒性。算力预算见 [第八节](#八目标硬件与算力预算)。
>
> **2026-08-12 实验后重排**：最小版 P1（执行层清幻影墙）已实测否决——见踩坑§17。
> 它能让狗满速跑（4m/s、bounce 降 5×）但**破坏 collision=0**（coll 31）：drift 2-4m 时
> 局部层继承漂移误差，"估计系 free"≠"真实系 free"，清墙=撞真墙。**结论：必须先治漂移，
> 执行层解冻 trick 才安全。** 故 P0-4（漂移）升为最高优先。

### 7.1 P0：止血——降方差 + 修卡死（不改架构）
| # | 改进 | 预期 | 算力 | 风险 |
|---|---|---|---|---|
| **P0-4 ★最高** | **漂移检测 + 二维码主动寻址重定位**：连续低匹配分 / 楔入频发 / drift 估计 >1.5m → 标记 lost → **主动转向最近可见二维码方向**做绝对修正（odom.correct 拉回）。不破坏 collision=0 的唯一安全提速路径 | 压漂移→幻影墙自消→楔入自解 | 低 | 中 |
| **P0-1** | **U弯前瞻减速 + 切弯路径**：approach U-bend 提前减速、路径贴内角（Voronoi/中线偏置），别冲进角 | 减少几何楔入 | 零 | 低 |
| **P0-2** | **nav2 式 recovery 行为**：连续 N 次 bounce/卡死 → ①Spin 360°（重扫+给 scan-matching 多观测降漂移）②BackUp 后退 0.5m ③ClearCostmap 局部 | 自愈卡死 | 零 | 低（注：Spin 治不了已 corrupted 地图，但能给匹配更多观测） |
| ~~P0-3~~ | ~~固定 odom RNG seed~~ | **已完成（验证）**：代码本就可复现（同 seed 逐位一致），方差来自不同代码版本非运行间 | — | — |

### 7.2 P1：架构补全——执行层读 rolling local costmap（**门槛：先完成 P0-4 把漂移压到 ≤0.5m**）
> ⚠️ 2026-08-12 实测：当前漂移水平（2-4m）下做执行层清墙**破坏 collision=0**。此项必须
> 等漂移可控后启用，否则局部层继承的漂移误差会让狗撞真墙。

| # | 改进 | 说明 |
|---|---|---|
| **P1-1** | 新增 `local_blocked()`：rolling 窗口的 raytrace 清除副本（3s 有效），执行层读它而非全局 blocked() | 对标 ROS local_costmap。**前提：漂移 ≤0.5m**（否则局部层不准，见踩坑§17） |
| **P1-2** | 经验墙/安全圈只写 SG(static 规划层)，不污染执行层的 local 副本 | 防假墙堵死执行 |

### 7.3 P2：定位升级（真机必需，仿真可选）
| # | 改进 | 说明 |
|---|---|---|
| **P2-1** | **scan-matching 换 PLICP**（point-to-line ICP，Censi 2008）：比当前"落墙比例"快且准，纯 CPU 毫秒级，有里程计初值 | RK3588 友好；直接提匹配率(当前~5%) |
| **P2-2** | **位姿图 + 回环**（轻量版 g2o/Ceres）：大环形建筑必需；或直接用 slam_toolbox（nav2 官方） | 真机大场景 |
| **P2-3** | EKF 融合 odom+IMU+scan-matching（robot_localization 思想） | 平滑+消漂 |

### 7.4 换房间迁移
| # | 改进 | 说明 |
|---|---|---|
| **M-1** | **参数化环境尺寸**：GRID_N/边界/PASS_CLEAR/OBS_CLEAR 全部 CLI 可配 | 解 50×50 硬编码 |
| **M-2** | **地图源通用化**：新房间用 SLAM 在线建图替代 track_clean.png 真值；物理碰撞用障碍层而非 hfield 真值图 | 解地图源绑死 |
| **M-3** | **PASS_CLEAR 自适应**：按检测到的走廊宽度自动设（中位数走廊宽 × 0.4），而非固定 0.6 | 解房门被封 |
| **M-4** | **门检测/房间分割**：frontier + 拓扑（门后是房间还是走廊），多房间更高效 | 多房间 |

### 7.5 已否决方向（别再试，踩坑文档有记录）
- 风暴隔离（冻结建图自救→死地）
- scan-to-scan 细网格帧间匹配（量化噪声放大器）
- 匹配参照剔除固定障碍锚（放弃纵向可观测性）
- 局部层墙戳加膨胀（走廊变窄→刹车风暴）
- PASS_CLEAR 下探到 0.3m（实测否决）

---

## 八、目标硬件与算力预算

### 8.1 RK3588 能力（调研结论）
- CPU 4×A76(2.4G)+4×A55，GPU Mali-G610 MP4 ~1.4TFLOPS，NPU 6TOPS(INT8)，内存可达 32GB。
- **2D SLAM 全栈（slam_toolbox/cartographer/hector）单 A76 核就够**，RK3588 是富余算力。
- 瓶颈在**四足步态控制 + 感知融合**，不在 2D SLAM。

### 8.2 各模块算力预算（当前仿真实测 + RK3588 估算）
| 模块 | 仿真实测(本机) | RK3588 预估 | 备注 |
|---|---|---|---|
| scan() 写图（360射线×1200步 numpy） | ~5ms | ~8-12ms | 向量化，A76 够 |
| ScanMatcher（120点×3000候选） | <15ms | <20ms | 可换 PLICP 更快 |
| find_gates + pick_gate | ~1ms | ~2ms | dict→numpy 已优化 |
| HPA* 全程规划 | 0.38ms | <1ms | |
| DWA choose_velocity | ~1ms | ~2ms | 向量化 |
| 视觉二维码（金字塔级联） | 0.3-25ms/帧 | 1-30ms/帧 | 降频跑 |
| **决策频率** | 174 步/s | ~100-150 步/s | 够 10Hz 实时 |

**结论**：当前算法在 RK3588 上能跑 50-100Hz 决策，远超 10Hz 激光频率。**算力不是障碍，算法鲁棒性才是。**

### 8.3 省算力方向（如未来加视觉/3D/gait 算力紧张）
1. **scan() 降射线数**：当前 360 射线，可降到 180（薄墙靠掠射填充补），省一半 scan 时间。
2. **find_gates 增量更新**：当前每帧全图扫，可只扫机器人周围 + frontier 缓存。
3. **scan-matching 用 PLICP**：比当前网格搜索快 3-5×。
4. **NPU 跑视觉**：二维码/语义识别量化后放 6TOPS NPU，省 CPU 给步态。
5. **规划层降频**：HPA*/find_gates 不必每 tick 跑，5-10Hz 够。

---

## 九、踩坑总账（索引）

> 详细"现象→根因→修复"见 [`docs/2026-08-05-dog50-maze-pitfalls.md`](2026-08-05-dog50-maze-pitfalls.md)（30KB，15+ 节）。这里只列高价值索引，新增踩坑追加到那份文档并在此登记。

| # | 坑 | 根因一句话 | 文档定位 |
|---|---|---|---|
| 1 | 卡死三连环 | scan墙前膨胀→find_gates空门→path=None不step | pitfalls §2 |
| 2 | 幽灵门 | 0.1m步进跨过<0.1m真缝→墙后标FREE→不可达门 | pitfalls §8.1 |
| 3 | bad_gates黑名单失效 | 存质心vs过滤原始门格key错位 | pitfalls §8.2 |
| 4 | 墙脸伪前沿门 | 掠射把墙面标FREE、墙背UNKNOWN→门后是墙 | pitfalls §8.3 |
| 5 | yaw读回覆盖 | `self.yaw=qpos[2]`每步转向被重置→bounce 94 | pitfalls §七.5 |
| 6 | 距离场BFS取min | `w if w<v`忽略更小值→墙邻格误判开阔49× | pitfalls §七.6 |
| 7 | MAX-pool y轴flip | row0=y=50m顶部不翻转→墙上下颠倒 | pitfalls §七.7 |
| 8 | 薄斜墙漏扫 | LIDAR_RAYS=120在15m间距0.78m>墙厚0.1m | pitfalls §七.8 |
| 9 | 里程计2%失败 | README旧称收敛0.1-0.5m，实测12万步碰撞出界 | noprivilege-fixes §3 |
| 10 | 假终点死锁 | 远距针孔噪声+三角锁到脚边→A*空转9万步 | no-truth §三 |
| 11 | 嵌墙10万撞 | 只信地图→狗开进真墙（contype=0无物理兜底） | no-truth §三.五 |
| 12 | DWA自旋锁 | v≈0时heading无信号，smoothness锁死大ω | pitfalls §8.7 |
| 13 | 接触区豁免误擦固定障碍 | 豁免对obs_world全生效→固定障碍被擦→卡7700步 | square-maze §9 |

---

## 十、文档地图（wiki 结构）

> 目标：关键概念互相链接，压缩上下文后从 PRD.md 能找到一切。

```
docs/PRD.md                          ← 本文件，稳定入口（目标/状态/架构/问题/改进）
docs/2026-08-05-dog50-maze-pitfalls.md  ← 踩坑总账（现象→根因→修复），持续追加
docs/2026-08-12-no-truth-localization.md ← 无真值化整改（定位/障碍/经验墙去特权）
docs/2026-08-11-scan-match-odom.md     ← ⚠️ 描述已废弃的 scan_match.py，待清理/合并
docs/2026-08-10-gate-width.md          ← 门宽度判断（PASS净空场）
docs/2026-08-09-square-maze.md         ← 方角地图 + 障碍mocap + 接触区豁免根修
docs/2026-08-09-noprivilege-fixes.md   ← 15项去特权根因清单
docs/research/
  ros-costmap-and-rk3588.md            ← ROS costmap分层/RK3588算力/scan-matching调研（本轮新增）
README.md                              ← 里程碑成绩单（已诚实标注⚠️超时项）
代码程序审核-2026-08-11.txt            ← 上一次审核报告（已闭环）
DESIGN.md / DESIGN_V4.md               ← 早期架构设计
docs/superpowers/specs/                ← 各阶段设计稿
```

**待办（文档清理）**：
- [ ] 删除/合并 `simtrack/scan_match.py`（死代码）+ 同步 08-11 文档
- [ ] 把本轮 ROS 调研落盘到 `docs/research/ros-costmap-and-rk3588.md`
- [ ] PRD.md 作为唯一稳定入口，日期稿只记"变更日志"指向 PRD

---

## 十一、如何运行/验证

```bash
# 单测（应 35/35）
.venv/Scripts/python -m pytest tests/ -q

# 纯墙无真值（默认 odom+scan-matching+二维码）—— 带 trail+map 存盘便于诊断
PYTHONIOENCODING=utf-8 MUJOCO_GL=glfw .venv/Scripts/python \
  test_scripts/algo3_headless.py --no-obs 1 --render-every 0 --seed 7 \
  --timeout 600 --trail-every 10 --save-map scans/wall_map.npz

# 复审"决策层零真值"（应零命中决策路径）：
grep -n "qpos" test_scripts/algo3_headless.py        # 只在物理/统计/轨迹/--odom 0
grep -n "velocities" test_scripts/algo3_headless.py  # 只在 --dwa-truth-vel（默认0）
grep -n "sample_hf" test_scripts/algo3_headless.py   # 只在物理投射/碰撞统计/场景生成

# 诊断卡死（轨迹分析）
.venv/Scripts/python scripts/analyze_trail.py scans/trail_*.npz
```

**关键指标看输出**：`loc_mode=odom`（无真值生效）、`collisions=0`、`match_corrections`（匹配率）、`odom_drift_final`（漂移）、`bounces`（卡死程度）。

---

## 十二、术语表

| 术语 | 含义 |
|---|---|
| **门 gate / 前线 frontier** | 已知FREE与未知UNKNOWN的边界格（=Yamauchi frontier） |
| **blocked()** | 执行层通行性判定，读全局G+SG（注：不读LOCAL_STAMP） |
| **LOCAL_STAMP** | 滚动局部层，最近3s激光直接命中的格时间戳 |
| **HIT_CONFIRMED** | 激光直命中过的格（经验墙感知确认凭据） |
| **PASS_CLEAR** | 门净宽阈值（0.6m），窄于此当前沿被当栅栏陷阱封闭 |
| **scan-matching** | 激光里程计，当前帧墙命中点对已建地图求位姿修正 |
| **odom 模式** | `--odom 1`，决策/建图用估计位姿（默认，无特权） |
| **幻影墙** | 里程计漂移导致墙被写到错误位置，在地图里和真墙无法区分 |
| **contype=0** | MuJoCo 碰撞组屏蔽，机器人物理上不接触墙（collision=0 部分源于此） |
| **rolling local costmap** | ROS 概念，跟随机器人的局部代价图，raytrace清除（本项目缺口） |

---

## 变更日志
- 2026-08-12：初版。复审实跑（纯墙490s到达，b76/b430），诊断三大根因（odom方差/U弯楔入/地图损坏无自愈），ROS调研，PRD 落地。
- 2026-08-12（晚）：P1 实验轮。实测最小版执行层清幻影墙（LOCAL_CLEAR+exec_blocked 两变体）→ **决定性否决**（变体A 卡死step10600；变体B coll 31 破坏铁律）。根因：drift 2-4m 时局部层继承漂移误差。**重排优先级：P0-4 漂移治理升为最高**。代码已干净回退到安全 baseline（0 collision）。详见踩坑§17。
- 2026-08-13：P0-4 漂移根治调查轮（自主）。贴墙=漂移驱动铁证（CLIP：每次贴墙 drift 1.4-6m）。
  三路实测：①宽窗重定位**无效**（identity=1.0@2m漂移，自洽漂移盲）；②lidar体坐标刹车**贴墙减半但bounce风暴**（弯道错工具）；③**密集二维码 `--qr-spacing` 落地**（spacing15：drift 2.68→2.02m，clip17→14%，0 collision，真实有效）。**架构结论：toy定位栈漂移天花板≈2m，根治需真SLAM后端（位姿图，RK3588可跑slam_toolbox）**。详见踩坑§17.7-17.8。
