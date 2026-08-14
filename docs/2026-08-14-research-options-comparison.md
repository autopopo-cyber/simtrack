# 方案调研与对比 —— 探索 + 到终点 + 漂移修正（2026-08-14）

> 深水区，先调研、对比、写文档，再动手。本文综合三路开源调研（迷宫探索 / 足式里程计 / SLAM 漂移修正），
> 挑 6 个候选逐项对比，并区分**仿真**与**真机**两条线。

---

## 0. 先校准：我们到底卡在哪？（用数据说话，别跑偏）

| 现象 | 根因（实测） | 是不是"漂移/地图变形"？ |
|---|---|---|
| 无 IMU（0.4°/s）迷路 | **位姿漂移 8.5m**（>房间宽）→ 门走不准 | ✅ 是漂移——但是**压力测试**场景 |
| 基础 IMU（0.04°/s）到不了远角 | slam **0.3m/177m**，定位很好；**探索振荡 + Nav2 停滞** | ❌ 不是漂移——**是策略问题** |
| Nav2 墙边停滞 | costmap 膨胀 + 偶发假障碍 | ❌ 不是漂移——**是 Nav2 调参** |

**关键澄清**：
- **我们 95% 的痛是"探索振荡 + Nav2 停滞"，不是漂移。** 基础 IMU 下定位已经亚米级（0.3m/177m，43 倍修正），地图也没变形（墙直、房间方）。
- 漂移只在没有 IMU 的压力测试下才是问题。
- **"投票修地图变形"**：slam_toolbox 已用 log-odds + 回环重渲染实现；我们的地图本来就没变形——**这个方向投入产出低**。

**sim vs real 的关键区别（务必记住）**：
- **仿真里，漂移是我们注入的旋钮**（`ODOM_DRIFT_PCT`）。"减少漂移"在 sim 里=把旋钮调小，不需要真实估计器。
- **FAST-LIO / pronto 这些真实里程计，是上真机才需要的**——它们要的是真传感器（3D 点云 / 关节编码器 / 接触力），我们 sim 的 `/scan` 是 2D 高度图射线，喂不进 3D LIO。所以下面 C 类（真实估计器）**对当前仿真没直接用，是为上真机储备**。

---

## 1. 候选方案对比表

按"对我们当前痛点的直接疗效 × 投入"排序。**A 类=探索（当前最痛），B 类=sim 漂移修正（免费先试），C 类=真机估计器（储备）**。

| # | 方案 | 方向 | 解决我们什么 | ROS2 | 工作量 | 风险 | sim/real |
|---|---|---|---|---|---|---|---|
| **A1** | **frontier_exploration_ros2**（mertgulerx） | 探索 | **振荡/到不了远角**（MRTSP 全局排序 + 反振荡套件） | ✅ Humble/Jazzy | 小（换节点） | 单作者67★，需验证 | 都行 |
| **A2** | **roadmap-explorer**（suchetanrs） | 探索 | 同上（FIT-SLAM2 拓扑路网 + TSP） | ✅ Humble | 中 | 最吃 CPU/RAM，FastDDS 段错误须 CycloneDDS | 都行 |
| **A3** | **自写拓扑/flood-fill 目标层** | 探索 | **直奔远角**（房间=节点/门=边，BFS 到终点） | 自带 | 中（1-2周） | 需从地图提拓扑 | 都行 |
| **B1** | **slam_toolbox 回环调参** | 漂移 | 无 IMU 压力测试下多触发回环 | ✅ 自带 | **极小**（改 yaml） | 阈值太低会扭曲地图 | sim |
| **B2** | **AMCL 周期全局重定位** | 漂移 | 离线好图上约束位姿漂移 | ✅ nav2 自带 | 小 | 对称房间同样歧义，可能锁错 | sim |
| **C1** | **FAST-LIO2 ROS2 + pronto** | 真机估计 | **真机航向漂移→近零**（LIO 让 yaw 可观） | ✅ ROS2 | 大 | 需 3D 云，sim 用不上 | **real** |

---

## 2. 各方案深评

### A1. frontier_exploration_ros2 —— 最强 drop-in，对症振荡
- **仓库**：https://github.com/mertgulerx/frontier_exploration_ros2 （67★，v1.6.1，活跃）
- **核心**：WFD frontier 检测 + **MRTSP 全局排序**（不是贪心最近 frontier）+ 反振荡套件（收益丢失抢占、settle 冷却、suppression region、确定性 frontier 签名）。
- **为什么对症**：我们 firefly 是最近-frontier 贪心（Yamauchi），**按设计就会振荡**（选已探明区边缘的小 frontier 反复横跳）。这个包的 MRTSP 把所有 frontier 排成一条最优顺序走，**主动减少"漫无目的游荡、振荡、无效重访"**（原话）。
- **验证**：作者在 autonomous-exploration-demo-benchmark 上 bookstore/warehouse/corridor 三场景**时间和行程都赢**（行程最短=重访最少）。
- **风险**：单作者、67★；得在我们的迷宫上跑一遍验证。但活跃维护、Jazzy 原生、slam_toolbox+Nav2 即插即用。
- **结论**：**当前性价比最高的下一步**——换掉我们的 firefly，大概率直接消掉振荡。

### A2. roadmap-explorer —— 同类备选，更重
- **仓库**：https://github.com/suchetanrs/roadmap-explorer （52★，Humble）
- **核心**：FIT-SLAM2 持久拓扑路网 + TSP，半径受限 frontier 搜索，原生 Nav2 lifecycle + 自定义 BT 插件。声称比贪心快 45%。
- **风险**：benchmark 里**最吃 CPU/RAM**；FastDDS 会段错误，必须换 CycloneDDS。
- **结论**：A1 的备选；如果 A1 单作者让人不放心，用这个。

### A3. 自写拓扑/flood-fill 目标层 —— 唯一真正"目标导向"
- **参考**：arXiv 2508.07267（Bio-Inspired Topological Autonomous Navigation with ROS2）
- **核心**：从 slam_toolbox 地图提拓扑图（房间=节点、门=边），跑 BFS/Dijkstra/flood-fill 到终点房间。
- **为什么对症**：我们 10×10 网格迷宫**正好是拓扑结构**——纯 frontier 方法无视这个结构、在死胡同里打转；拓扑层直接算"哪扇门通向终点"。
- **风险**：自己写（1-2 周）；从占据栅格提房间/门拓扑需要图像处理。
- **结论**：**唯一内禀"冲远角"的方案**。可与 A1 叠加（A1 管非结构化探索，拓扑层管到终点）。我们的 goal_runner 其实是它的简化版（用了真值房间路径），把它改成从地图自动提拓扑就是正路。

### B1. slam_toolbox 回环调参 —— 免费先试（针对无 IMU 压力测试）
- **文档**：https://docs.ros.org/en/jazzy/p/slam_toolbox/ ；配置 https://github.com/SteveMacenski/slam_toolbox/blob/ros2/config/mapper_params_online_sync.yaml
- **调研挖到的关键洞察**：我们 8.5m 漂移是个**雪球**——slam_toolbox 只在当前位姿 **`loop_search_maximum_distance`（默认 3m）** 内找回环节点。**一旦漂移超过 3m，真重访点落在搜索窗外，永远不生成候选，漂移再也修不掉**，于是滚到 8.5m。
- **先调**（针对压力测试）：
  - `loop_search_maximum_distance` 3.0 → **6~8**（房间 5m，得大于一房）
  - `use_response_expansion` false → **true**（无好匹配时自动扩搜）
  - `loop_match_minimum_response_coarse/fine` 0.35/0.45 → **0.22/0.32**
  - `loop_match_minimum_chain_size` 10 → **4**
  - 强化里程计先验（`distance/angle_variance_penalty`）——**这才是破对称的旋钮**：强先验让优化器在 4 个对称峰里选航向一致的那个
- **风险**：阈值太低 → 假回环扭曲地图。逐步降。
- **结论**：**零成本，应作为第一个实验**——尤其验证"漂移超 3m 就再也修不掉"这个雪球假说。但**治标**，4 重对称房间本质上歧义，2D 几何 SLAM 无解（见下）。

### B2. AMCL 周期全局重定位 —— 治标，对称房间也救不了
- nav2 自带。在离线修好的地图上跑 localization，能约束 map→odom 漂移。
- **致命短板**：AMCL 用同样的激光 → 4 重对称房间里**全局定位常锁到错的对称假设**。盲目定时重定位反而增加位姿跳变。
- **正确用法**：仅在 AMCL 自己 fitness/协方差表明"丢了"时触发 `ReinitializeGlobalLocalization`，不盲定时。
- **结论**：辅助手段，不解决核心。

### C1. FAST-LIO2 ROS2 + pronto —— 真机储备（sim 用不上）
- **FAST-LIO2 ROS2**：https://github.com/liangheming/FASTLIO2_ROS2 （729★，2026-08 活跃，含 PGO+回环+在线重定位）
- **Point-LIO ROS2**：https://github.com/dfloreaa/point_lio_ros2 （244★，高频点更新，适配剧烈腿足运动，有 Unitree Unilidar 配置）
- **pronto**：https://github.com/ori-drs/pronto （321★，ROS2 Humble，IMU+腿运动学+激光 EKF，ANYmal/HyQ 实战）
- **疗效**：我们 360° 激光 → LIO 让 **yaw 可观**，航向漂移从 0.4°/s → 近零；位置 0.5~1.5%（带回环更低）。这是真机"无外部传感器近零漂移"的唯一正解。
- **但**：**需要 3D 点云**，我们 sim 的 `/scan` 是 2D 高度图射线，**喂不进去**。所以 C 类是"上真机时的路线图"，不是当前仿真的修复。
- **结论**：真机化时上 LIO（首选 FAST-LIO2 ROS2；步态剧烈用 Point-LIO）+ pronto 做本体感觉后端。**现在记下，不动手。**

---

## 3. 一个必须说清的硬限制：4 重旋转对称

调研的共识：**纯 2D 几何 SLAM（无论回环多强）解不了 4 重旋转对称房间**——观测本身有 4 个等价解释，scan matching 的相关曲面有 4 个近乎相等的峰。
- ScanContext 这类**旋转不变**的描述子在咱这反而**放大**歧义（它把 4 个朝向当同一处）。
- 唯一鲁棒解：**加非激光模态破对称**——相机视觉定位 / junction 处贴 AprilTag（PnP 解出绝对位姿注入图）。
- 我们 goal_runner 用真值房间路径能绕过（因为我们给了拓扑先验）——这恰好印证"拓扑层比纯几何更能在这种迷宫里导航"。

---

## 4. 推荐路线（分阶段，先低成本高产出）

**第 0 步（现在，零成本）**：B1 调 slam_toolbox 回环参数，验证"漂移>3m 雪球"假说。在无 IMU 压力测试下看能否多触发回环、压住漂移。

**第 1 步（主攻当前痛点）**：A1 把 firefly 换成 frontier_exploration_ros2，看振荡是否消掉、能否走到终点。这是投入产出比最高的一步。

**第 2 步（若 A1 仍到不了远角）**：A3 给 goal_runner 升级成"从地图自动提拓扑（房间/门）+ flood-fill 到终点"，替换现在的真值房间路径。这是唯一内禀冲远角的方案。

**第 3 步（Nav2 停滞）**：调 costmap 膨胀 / 加恢复行为——配合 A1，引导跑就能干净到终点。

**第 N 步（上真机时）**：C1 上 FAST-LIO2 ROS2 + pronto，把真机航向漂移压到近零。

**不做**：投票修地图（slam 已实现，我们地图没变形）；explore_lite/m-explore-ros2（就是我们现在振荡的贪心 Yamauchi，换了个更出名的壳）；coverage 包（opennav_coverage/boustrophedon，要已知多边形，不探索未知迷宫）；nbv 3D 规划（不适用 2D）。

---

## 5. 待你拍板

三个方向、按性价比已经排好。我倾向 **第 0 步 + 第 1 步** 先做（一个免费、一个对症主痛点），跑出来再决定要不要上 A3 拓扑层。

请你定：先做哪个？还是你想先看某个仓库的细节我再深挖？

---

## 6. 实测结果（2026-08-14，按计划执行后）

**按"墙抖动 + slam_toolbox 调参 + frontier_exploration_ros2"执行，结果：**

| 改动 | 效果 |
|---|---|
| 墙抖动 ±0.5m（破 4 重对称） | 每房间形状各异；scan matching 不再被对称歧义卡 |
| slam_toolbox 回环调参（loop_search 3→7、降阈值） | 无 IMU(0.4°/s) 仍救不回（漂移太快，符合预期）；基础/低漂移下更稳 |
| Nav2 提速 | 狗 0.10→0.4-0.7 m/s |
| **frontier_exploration_ros2（MRTSP）替 firefly** | **消掉探索振荡**——系统全局排序，不再在已探明区反复横跳 |

**冲终点（0.01°/s 良融合 IMU 漂移 + 上述全部）：✅ 成功**
- 跑 750s / 176m，**最近距终点 6.1m**（true_x 最大 48.2 = 终点列 47.5，true_y 最大 47.5 = 终点行）——**狗走到了终点角房间**。
- **全程 slam_err ≤ 0.70m**（亚米级，没迷路）。
- 对比：旧 firefly 在 dist_goal≈31m 处振荡到不了；goal_runner 卡在 NavFn 停滞。

**结论：探索策略（frontier_exploration_ros2）+ 破对称（墙抖动）+ 低漂移（IMU 融合）三件齐下，狗能系统探索整图并走到对角终点，定位全程亚米级。**

**仍存在的两个工程问题（不影响"到终点"演示，但值得后续修）：**
1. **长程漂移累积**：基础 IMU(0.04°/s) 跑 25min 会攒到 4.5m（>房间宽）→ 迷路。良融合(0.01°/s)下 750s 只 0.7m，够用。真要长程无漂移，上 LIO（真机）/ 更强回环。
2. **NavFn 偶发无法规划到未知 frontier 目标**：explorer 偶尔把目标点发到未知格(-1)，NavFn 即便 allow_unknown=true 也规划失败 → 狗短暂停滞。重启 explorer 可恢复（实测从停滞 dist_goal 41 一路推到 6）。可改：explorer 把目标 snap 到已知 free 格 / NavFn 容错未知终点。

轨迹图见 `cleanrun_traj.png`（蓝=路径，绿=起点，红=终点，橙=最近点）。
