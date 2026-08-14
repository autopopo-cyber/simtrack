# simtrack — 机器狗自主导航（ROS2 + MuJoCo 仿真 → Unitree A2 实机）

> **本文是纯索引。** 项目一句话：机器狗自主进入未知屋子（<300㎡，随机房间，地面障碍物，
> 60–80cm 窄通道），SLAM+探索+避障全程自主，最终上 A2 实机（L2 激光）。
> 里程碑：`v2.0-ros2-slam-nav`（2026-08-14）。旧 README 全文在 git 历史 `2bcda70`。

## 📖 文档地图（会话恢复从这开始）

### 第一部分：PRD（要做什么）

| 文档 | 内容 |
|---|---|
| **[docs/PRD.md](docs/PRD.md)** | 目标/验收标准与当前状态/硬件 spec（L2 全参数）/主人铁律/里程碑/下一步/术语表 |

### 第二部分：实现（怎么做的）

| 文档 | 内容 | 什么时候看 |
|---|---|---|
| **[docs/pitfalls-ros2.md](docs/pitfalls-ros2.md)** | ⭐ **踩坑总账（血泪重点）**：20 条 症状→根因→修法，两条定律，旧时期索引 | **接手第一读** |
| [docs/architecture.md](docs/architecture.md) | 架构：数据流全景/关键设计决策及理由/远程部署拓扑/坐标系约定/真机目标架构 | 改架构前 |
| [docs/api.md](docs/api.md) | 怎么跑：环境变量表/话题QoS/SimBackend接口/maze_gen CLI/标准实验流程 | **每次跑实验** |
| [docs/algorithms.md](docs/algorithms.md) | 关键算法：漂移模型/扫描匹配/周期重定位/A*推进/墙抖动/slam调参/窄门标定/两条定律 | 改算法前 |
| [docs/code-structure.md](docs/code-structure.md) | 仓库地图：活跃文件 vs 历史文件/git 模型/接手清单 | 找文件时 |

### 原始记录（按日期，实验的完整叙事）

| 文档 | 内容 |
|---|---|
| [2026-08-14-maze-drift-goalreaching.md](docs/2026-08-14-maze-drift-goalreaching.md) | §一到§十：迷宫/漂移实验/到终点/周期重定位/自建图修正/L2真实参数/窄门分级（最新最全） |
| [2026-08-14-research-options-comparison.md](docs/2026-08-14-research-options-comparison.md) | 6 候选方案对比（frontier_exploration_ros2 等）+"不做"清单 |
| [2026-08-13-ros2-mujoco-pipeline.md](docs/2026-08-13-ros2-mujoco-pipeline.md) | ROS2+MuJoCo 迁移全程 + 12 坑 + 被超越方案清单 |
| [2026-08-05-dog50-maze-pitfalls.md](docs/2026-08-05-dog50-maze-pitfalls.md) | 纯 Python 时期踩坑 17 章（§17 = 迁移触发点） |
| 其余 2026-08-0*.md | 旧时期里程碑/优化/整改过程记录 |

## 🚀 快速上手（3 行版，完整版在 api.md §五）

```bash
# 远程 qin@100.64.63.98（密码"1"）tmux session "sim"：0:bridge 1:slam 2:nav2 3:drive 4:mon
MAZE=rooms10x10n80 ODOM_DRIFT_PCT=5 ODOM_DRIFT_YAW_BIAS_DEG=0.4 CORRECT_PERIOD_S=30 \
CORRECT_REF=map LIDAR_RANGE=10 LIDAR_NOISE_M=0.03 /usr/bin/python3 -m simtrack.sim_bridge
# ↑ 必须 /usr/bin/python3（hermes-venv 劫持坑）；启动顺序 bridge→slam→nav2→goal_runner
```

## ★ 当前状态一览（2026-08-14，详见 PRD §三）

| 验收项 | 状态 |
|---|---|
| 探索+建图+导航（2500㎡，比目标难 8 倍） | ✅ 到终点区 |
| 定位漂移（L2 真实参数+30s 自建图重定位） | ✅ mean 0.08m，**结题** |
| 0.8m 窄通道 | ✅ 任务级 |
| 0.6m 窄通道 | ⚠️ 需"对门"行为（±14° 对准容差） |
| 地面障碍物 / 300㎡ 目标尺度验收 / 真机集成 | ❌ 待做（PRD §七） |
