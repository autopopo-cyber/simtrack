# 调研：步进失败/探索卡滞——Nav2 社区的根因与解法

> 2026-08-15。背景：批量窄门统计（§十一）显示真正耗时大头是普通房"步进失败 4 次跳过"
> （2% 房次吃 10% 时间）+ error_code=208 循环。本文档调研 ROS 社区对同类问题的根因和解法，
> 结论给出改造方案。**先文档后代码**（主人原则）。

## 一、我们的症状 → 社区根因（全部对上号）

| 我们的症状 | 社区根因 | 证据 |
|---|---|---|
| "Failed to create a plan from potential when a legal potential was found. This shouldn't happen." | **NavFn 梯度回溯有硬编码步数上限**（max(地图长,宽)×4）：远目标穿大片平坦 unknown，回溯路径超限即报错。**不是 unknown 的锅**（allow_unknown 时 NO_INFORMATION 被置中性代价，Dijkstra 能穿），死在回溯提取 | [navigation2#2016](https://github.com/ros-planning/navigation2/issues/2016) [#4655](https://github.com/ros-navigation/navigation2/issues/4655)（#4655 里 R0xy42 描述的现象与我们逐字一致：远目标→旋转→清图→死循环） |
| error_code=208 | **208 = ComputePathToPose 的 NO_VALID_PATH**（2xx=规划器段；1xx 才是控制器段）——即批量实验的"步进失败"全是**规划器层失败**，不是 MPPI 的锅 | [ComputePathToPose.action](https://github.com/ros-navigation/navigation2/blob/main/nav2_msgs/action/ComputePathToPose.action) |
| 修复状态 | **2026-08-06 已合并 PR #6308**（`max_cycles_factor` 参数，可设 6+），但 backport 只到 lyrical——**Jazzy 二进制包没有**，要么 cherry-pick 要么换规划器 | [PR #6308](https://github.com/ros-navigation/navigation2/pull/6308) |
| 清 costmap 恢复越清越糟 | 清全局图抹掉已观测障碍再缓慢重累积，算法性上限未变，重规划必再失败。#4655 建议恢复加重试上限或去清图 | #4655 + [官方 BT 树](https://docs.nav2.org/behavior_trees/trees/nav_to_pose_with_consistent_replanning_and_if_path_becomes_invalid.html) |

## 二、解法一：换 SmacPlanner2D（社区证据最足，一行配置）

官方 README（Macenski）原话："2D A* 没有 NavFn 梯度波前实现的怪异伪影，稍慢但路径质量值得"。

| 特性 | 对我们的意义 |
|---|---|
| 纯 A* 搜索，**无梯度回溯=无路径长度上限** | 直接消灭 "legal potential" 报错 |
| 精确路径找不到时**返回 tolerance 内最接近可行路径** | 目标落未知/非法格不再硬失败（208 的主要来源） |
| Jazzy 自带 | 无需 cherry-pick |
| `downsample_costmap: true` | 大地图长路径"极其有益"（官方 README）——50×50m 场景正合适 |

参数（[官方文档](https://docs.nav2.org/configuration/packages/smac/configuring-smac-2d.html)）：
`tolerance: 0.5`（探索可放宽），`allow_unknown: true`，`max_iterations: 1000000`（或 -1），
`downsample_factor: 1→2` 试。

## 三、解法二：explore_lite 的三层失败处理（代码级核实，最可迁移）

读了 [m-explore-ros2 explore.cpp 全文](https://github.com/robo-friends/m-explore-ros2/blob/main/explore/src/explore.cpp)：

1. **ABORTED 即拉黑**：result 回调里 ABORTED → 该 frontier 入黑名单（5×分辨率半径命中判定），
   **立即**从排序列表取下一个未拉黑目标。error_code==0（被抢占）不拉黑。
2. **进度超时兜底**（关键！）：周期检查"到目标的 min_distance 是否在变小"，超过
   `progress_timeout`（默认 30s）无进展 → 拉黑换目标。**覆盖 Nav2 挂起不报错的场景**——
   我们批跑里的"门舞呆 180-222s"正属此类。
3. **同目标活跃期绝不重发**：`if (same_goal && goal_active_) return;`——直接治"重发同一 goal 死循环"。
4. 已知缺陷：黑名单无 TTL（永久误杀），社区靠重启节点绕过——**我们实现时加 60s TTL**。

注：我们在用的 frontier_exploration_ros2（作者实为 mertgulerx，[仓库](https://github.com/mertgulerx/frontier-exploration-ros-2)）
只有**事前过滤**（双 costmap 占据校验/blocked 即跳/够近即完成+临时排斥半径），**没有 ABORTED 重试逻辑**。

## 四、解法三：Nav2 调参（防误杀合法慢行）

| 参数 | 现默认 | 建议 | 理由 |
|---|---|---|---|
| `movement_time_allowance`（SimpleProgressChecker） | 10s | **25s** | 窄门对准的原地旋转被误判"卡死"→假 ABORTED。或换 PoseProgressChecker（旋转也算进步） |
| `required_movement_radius` | 0.5m | **0.25m** | 同上，探索慢行是常态 |
| `failure_tolerance`（controller_server） | 0.0 | **3.0** | 首个控制异常不再立即中止 |
| 恢复 BT | 清 local+global | **只清 local**，global 清图移除 | §一第四行；用 Wait/BackUp 替代 Spin |
| goal checker | xy 0.25? | 检查确认；探索用 PositionGoalChecker（忽略朝向） | 目标在膨胀区贴不近的假失败 |

## 五、改造方案（按性价比排序）

| # | 改动 | 成本 | 预期收益 | 风险 |
|---|---|---|---|---|
| 1 | **NavFn → SmacPlanner2D**（configs/nav2_fast_params.yaml） | 一处配置 | 消灭 208 主源（远/未知目标规划失败） | A* 稍慢（50×50m 降采样可抵） |
| 2 | **progress_checker + failure_tolerance 调参** | 三行配置 | 消灭门舞旋转被误杀的假 ABORTED | 无（只是变宽容） |
| 3 | **BT 去掉全局清图恢复** | 自定义 BT xml | 消灭"清图→更糟→死循环" | 恢复能力略降（保留 local 清+Wait/BackUp） |
| 4 | goal_runner 失败处理升级：步进失败 2 次 → **拉黑该子目标格(0.5m 半径, TTL 60s) → A\* 重规划给备选子目标**；再加 explore_lite 式进度超时（30s 无进展换路） | ~60 行 | 消灭"4 次失败跳过整航点"的浪费（直接换路不弃站） | 需防拉黑振荡（TTL 兜底） |
| 5 | （备选）cherry-pick PR #6308 给 NavFn 加 max_cycles_factor | 中 | 修 NavFn 本体 | 维护成本 > 换 Smac，不推荐 |

**验证协议**：改完 1+2+3（纯配置）先跑 3 个种子快速对比；若 208 消失再做 4；
最终用 §十一同协议 12 种子批跑对比基线（mean 30s / >120s 占 10% 时间 / 到达 3/11）。
成功标准：**>120s 时间占比 <3%，"步进失败跳过"事件 <2 次/12 种子，到达数 ≥4**。

## 六、方法论沉淀（记入踩坑思维）

1. 社区调研要读到**代码级/issue 级**根因，别停在 README——本次 "legal potential" 报错
   的真根因（回溯步数硬上限）只在 issue 分析里，文档只字未提。
2. error_code 分段速记：**2xx=规划器，1xx=控制器**——先看码再猜层。
3. 别人的成熟模式先抄（explore_lite 三层），缺陷（无 TTL）顺手修。

## 七、落地结果（当日完成，12 种子×600s 同协议批跑）

### 7.1 版本迭代史（每版都被实测数据否决或确认）

| 版本 | 机制 | 实测 | 判决 |
|---|---|---|---|
| 1-3 号改造 | Smac2D + 调参 + BT（纯配置） | quick1: seed3 26/41(基线17)、慢房清零、窄门过；seed1/3 **零失败行** | ✅ **208 彻底消失**（v4.x 全程 0 次 Nav2 ABORT） |
| v4.0 | 超时拉黑+直线度量 | seed3 连打 6 次误杀：狗在走 A* 给的**绕行**，直线距离却在变大 | ❌ 度量错：迷宫里直线不是进度 |
| v4.1 | 路线长度二级确认 | seed3 30；但 seed6 只到 8——拉黑封了树状迷宫**唯一门**→A*无路→雪球跳站 | ❌ 拉黑在无替代路时是毒药 |
| v4.2 | 超时不拉黑+换向脱离 | seed1 掉到 15：脱离把狗**扔出 15m 外**（迷宫里后退=改道） | ❌ 脱离机动方向性错误 |
| v4.3 | +墙格代打+基线持久 | seed1 14：A* nudge 分支**无限递归**炸节点（seed2 只到 2） | ❌ bug：allow_nudge 传了没检查 |
| **v4.4** | 条件拉黑（拉黑后 A* 验证仍有路才保留，否则回滚） | 见 7.2 | ✅ **当前最优** |

### 7.2 v4.4 vs 基线（11 共同种子）

| 指标 | 基线 | v4.4 | 判定 |
|---|---|---|---|
| 完全到达种子数 | 3 | **5**（1/4/7/9/11） | ✓ |
| 到达房间总和 | 198 | **234**（+18%） | ✓ |
| 窄门通过种子数 | 6 | **8** | ✓ |
| 房均通过 mean / 中位 / p90 | 30.1 / 23 / 41s | **22.4 / 19 / 26s** | ✓✓✓ |
| <30s 房次占比 | ~76% | **93%** | ✓ |
| >120s 慢房房次 | 3 | 1 | ✓ |
| >120s 时间占比 | 10.5% | 8.3% | ✓ 改善未达标(<3%) |
| 最差房 | 222s | 414s（seed10 预算跳站雪崩） | ✗ |
| 跳站事件 | （旧日志无此统计） | 4 | ✗ 未达标(<2) |

**成功标准 3 条达成 1 条**（到达 5≥4 ✓；慢房占比 8.3%≥3% ✗；跳站 4≥2 ✗）——但结构变了：
基线是**普遍性慢**（每房都 30s+、门舞遍地），v4.4 是**快而双峰**（93% 房 <30s，剩余风险
集中在 1-2 个"地图变形封区"种子：seed8/10，其 A* 连代打格都无路——§九误差下限的
极端形态，goal_runner 层无解，属 SLAM 层问题）。9/11 项指标改善。

### 7.3 残余问题与下一步

- **硬尾巴=地图变形封区**（seed8 类："路线 ?" 无路可规划）：治本在 SLAM（回环密度/
  全局重定位），不在规划层。PRD backlog 的"floor-obstacle+垂直FOV"之前先攻这个。
- seed10 的 414s：240s 预算跳站后重进近又卡——考虑预算降到 180s 或跳站后先拉黑整房。
- 单种子方差 ±10 房（变形抽签），对比结论要看聚合，别看单种子（seed3 三轮 26/30/16）。
