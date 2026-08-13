# ROS costmap 分层 / RK3588 算力 / scan-matching 调研

> 2026-08-12 为 PRD §六（概念链缺口）和 §八（算力预算）做的调研。
> 目的：对比 ROS 标准做法，定位本项目"门-墙-前线"概念链缺什么；评估 RK3588 能否承载。

## 一、ROS costmap_2d 分层融合（对比本项目 blocked()/LOCAL_STAMP）

### 1.1 层种类与融合
- 层：`static_layer`（SLAM地图）→ `obstacle_layer`/`voxel_layer`（实时观测）→ `inflation_layer`（衰减cost，放最后）。
- 每层维护**自己的栅格**，按 plugins 顺序写入共享 master costmap。
- **执行层(控制器)读 local_costmap（rolling window，几米，5-20Hz）；规划层读 global_costmap（整图，~1Hz）。两者都是融合后的总 costmap。** ← 本项目 blocked() 不分全局/局部，是关键差距。

### 1.2 rolling window + raytrace 清除
- local costmap 窗口跟随机器人，窗外 cell 丢弃。
- mark/clear = **raytrace**：从传感器原点向回波画射线，路径 cell 置 free，终点置 lethal。"障碍要被清除，必须有射线物理穿过它"。
- 标准 obstacle_layer **无时间衰减**；要衰减用 spatio_temporal_voxel_layer(stvl)。

### 1.3 ★关键：static 障碍 vs 最新 free 观测，ROS 怎么处理
由 obstacle_layer 的 `combination_method` 决定：
- **`1=Max`（默认）**：static lethal(254) 与 obstacle free(0) 取 max → **static 不被动态清除**（安全，宁可绕远不让穿墙）。
- **`0=Overwrite`**：obstacle 值直接覆盖 → static 可被擦掉（动态环境用）。
- static_layer 内部状态**永远**不被清除，只在新地图到达时刷新。

**对本项目的结论**：
- 本项目 `blocked()` 不被 LOCAL_STAMP 清除——**与 ROS 默认 Max 一致，本身不算错**。
- 真正问题：本项目的 G **不是干净的 static**（它是每帧按估计位姿写的活栅格，漂移→幻影墙），把它当 static 用才错。
- **正解**：执行层读独立的 rolling local costmap（raytrace 清除过的干净副本），全局 G 只给规划/门发现。对标 ROS local vs global 分离。

来源：[Nav2 costmap 配置](https://docs.nav2.org/configuration/packages/configuring-costmaps.html)、[obstacle layer combination_method](https://docs.nav2.org/configuration/packages/costmap-plugins/obstacle.html)、[static vs obstacle 覆盖案例](https://robotics.stackexchange.com/questions/90591)

### 1.4 inflation 分档
5 档 cost：LETHAL(254,障碍) / INSCRIBED(253,内切圆必碰) / POSSIBLY-CIRCUMSCRIBED(252,外接圆) / 指数衰减区 / FREE(0)。`inflation_radius` 是衰减上界（非机器人半径），`cost_scaling_factor` 控衰减快慢。本项目 OBS_CLEAR=0.7 单档膨胀，无衰减cost。

## 二、探索/前沿算法（对比"门-前线"）

- **Yamauchi 1997 frontier**（祖宗）：frontier = free 邻接 unknown 的 cell；聚类成 region；选目标（原版=最近）。
- **explore_lite/m-explore**：最近 frontier（NavFn 代价）；进阶加信息增益/朝向。
- **室内多房间**：纯 frontier 易卡（门口 frontier 瞬时遮挡漏判"探完"）；推荐 **frontier + 门/房间拓扑分割**（本项目"门"概念方向正确，缺拓扑层）。
- **nav2 recovery**：ClearCostmap→Spin(原地转一圈重扫)→Wait→BackUp(后退)。本项目只有 bounce+escape，**缺 Spin/BackUp**。

来源：[Yamauchi 1997](https://www.cs.cmu.edu/~motionplanning/papers/integrated1/yamauchi_frontiers.pdf)、[nav2 recovery](https://docs.nav2.org/configuration/packages/configuring-behavior-server.html)

## 三、★RK3588 算力（目标硬件）

RK3588：4×A76(2.4G)+4×A55，Mali-G610 MP4 ~1.4TFLOPS，NPU 6TOPS(INT8)，内存≤32GB。
**2D SLAM 全栈对 RK3588 是富余算力**（slam_toolbox 在 Pi4 都能跑，RK3588 强 1.5-2×）。瓶颈在步态控制+感知融合，不在 2D SLAM。

### 3.1 2D SLAM 方案对比（ARM 边缘）
| 方案 | 回环 | 大场景 | 轻量度 | 本项目选型 |
|---|---|---|---|---|
| **slam_toolbox** | 强(lifelong) | 好 | 中（nav2官方推荐） | **真机首选** |
| **cartographer** | 强 | 优秀(子图) | 中(重依赖) | 大场景备选 |
| **hector_slam** | 无 | 中(无回环漂) | **最轻**(无需里程计) | 兜底/无里程计 |
| gmapping | 弱(粒子退化) | 差 | 中 | 已过时不选 |

### 3.2 scan matching（ARM 最快）
- **PLICP**(Censi2008, point-to-line)：最快之一，毫秒级，需好初值（腿式里程计有）→ **本项目首选替换当前网格搜索**。
- **Hector 多分辨率 Gauss-Newton**：轻量，~40Hz，**无需里程计** → 无里程计时用。
- CSM(Olson)：鲁棒但单次~1.55s，留作全局重定位。
- GICP：2D 性价比低。

### 3.3 规划/控制
- 全局：**Smac Hybrid-A\***（带heading，窄通道鲁棒）或 A\*。
- 局部：**MPPI**（nav2新主推，原生支持原地转身，四足可零转弯半径）或 TEB。
- 四足澄清：可原地转身，"最小转弯半径"不是硬约束；真实约束是步态最小角速度+窄通道净空。

来源：[slam_toolbox](https://github.com/SteveMacenski/slam_toolbox)、[Nav2 MPPI](https://docs.nav2.org/configuration/packages/configuring-mppic.html)、[Nav2算法选择](https://docs.nav2.org/setup_guides/algorithm/select_algorithm.html)、[leggedrobotics/se2_navigation(四足)](https://github.com/leggedrobotics/se2_navigation)

## 四、对本项目的落点建议
1. **地图分层**：新增 rolling local costmap 给执行层；全局 G 只给规划。→ 解根因C（PRD §7.2 P1-1）。
2. **scan-matching**：换 PLICP（PRD §7.3 P2-1）。
3. **recovery**：补 Spin/BackUp（PRD §7.1 P0-2）。
4. **真机 SLAM**：slam_toolbox 起点，RK3588 富余。
5. **规划/控制**：全局 Smac/A\*，局部 MPPI——四足原生支持原地转身。
