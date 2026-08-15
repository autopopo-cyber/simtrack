# 踩坑总账（ROS2 栈时期）— 血泪重点文档

> **本文是接手项目必读第一文档。** 每条都是实打实耗掉的时间。格式：症状→根因→修法→预防。
> 纯 Python 原型时期的踩坑（17 章，同样有价值）见 [2026-08-05-dog50-maze-pitfalls.md](2026-08-05-dog50-maze-pitfalls.md)。
> 最后更新 2026-08-14。

## 定律区（先背这两条，能避一半坑）

**定律一：误差下限 = 退化几何暴露时长，不是传感器。**
实测：雷达从 15m 无噪降到 10m±3cm，定位反而好 20 倍（1.81→0.08m），因为健康的探索模式让狗
一直暴露在二维富几何环境。走廊/贴墙 = 不可观测方向积累漂移。对策排序：探索模式 > 修正频率 > 传感器。

**定律二：门舞 = 退化几何暴露 = 地图变形。同一行为三副面孔。**
狗贴墙蹭（找门/找路）→ 该方向几何退化 → 地图变形 1.9m → 定位伤。治探索行为同时治定位。

---

## 运维/环境类

### 1. tmux kill-session 杀不死 launch 子进程树 ★★★
- **症状**：重启栈后诡异行为（地图不更新/goal 被抢/新节点起不来）。
- **根因**：`ros2 launch` 的子进程不在 tmux pane 进程组里，kill-session 只杀了 shell。
  旧 slam/nav2 残留 → **双 /map（都 latched）双 goal 竞争**。
- **修法**：`pkill -9 -f <模式们>` 后必须 `pgrep -af ...` **确认为空**，必要时按 PID 补刀，再重启。
- **预防**：重启流程写死在 api.md §五，照抄。

### 2. 远端 python3 被 hermes-venv 劫持 ★★★
- **症状**：tmux 里 `python3 -m simtrack.sim_bridge` 报 `No module named 'numpy'`，非交互 SSH 执行却正常。
- **根因**：`~/.bashrc` 把 `~/hermes-venv/bin` 塞进 PATH 首位（hermes-gateway 装的），交互 shell 的
  python3 = hermes venv（无 numpy/rclpy）。
- **修法**：所有 tmux 命令显式 **`/usr/bin/python3`**。
- **预防**：api.md 命令模板里已写死。

### 3. nav2 在无 /clock 时 activation 超时弃疗 ★★
- **症状**：`Failed to bring up all requested nodes. Aborting bringup`，controller_server inactive。
- **根因**：use_sim_time 节点激活时等 /clock；bridge（时钟主人）还没起。
- **修法**：**启动顺序铁律：bridge → slam → nav2(等16s) → drive**。nav2 挂了单独重启即可（此时钟已在）。

### 4. tmux 80 列折行截断日志数字 ★★
- **症状**：抓取的修正日志只剩"@t=123s：修正 pos"，数值全被切掉。
- **修法**：`capture-pane -S -4000 | grep -A3 "关键字"`，然后重组行（`c.replace(chr(10),'')`）再正则。

### 5. TUN 代理 push 不稳 ★
- **症状**：`schannel: SEC_E_MESSAGE_ALTERED`，push 连续失败。
- **修法**：重试循环（sleep 10-15s，一般 <12 次内过）。**本地提交永远先做**，推送可以等。

### 6. QoS 不匹配静默丢消息 ★
- /scan 是 BEST_EFFORT（订阅端用默认 RELIABLE → 收不到 + 一行 WARN）
- /map 是 TRANSIENT_LOCAL latched（订阅端必须同 QoS 才能收到迟加入前的消息）

## 规划/驱动类

### 7. NavFn "Failed to create a plan from potential when a legal potential was found" ★★★
- **症状**：explorer 反复发 goal，NavFn 报此错 + recovery 清 costmap 循环，狗定住。
- **根因**：**远目标穿大片平坦 unknown**——Dijkstra 潜在场在 unknown 上梯度太平，回溯提取路径
  超时/失败。allow_unknown=true 也没用。
- **修法**：**永远发近距离 free 子目标**（A* 推进的连续 free 段末尾），让 NavFn 在已知 free 上规划。
- **教训**：这是 frontier 类探索器（对未知边界发远目标）与 NavFn 的结构性冲突。

### 8. "等地图"死锁：静止狗的地图永远不长 ★★★
- **症状**：goal_runner 打"前方待探明，等地图…"，永远等下去。
- **根因**：slam_toolbox 关键帧式建图——**不移动不出关键帧，不出关键帧地图不长**。360° 激光也没用
  （观测有，但图不更新）。
- **修法**：必须让狗动——A* 推进的子目标永远在已知 free 内（狗一定动得起来），动起来图就长。

### 9. 扇形扫描贴墙振荡 ★★
- **症状**：狗在墙前 ±1m 来回蹭（y 或 x 坐标锯齿），不前进。
- **根因**：扇形兜底（朝航点 ±15..±90° 找 0.8m 步）每步重锚航向，3m 探针短视；墙前两侧对称。
- **修法**：A* 推进（第 4 节算法）——它能表达"绕远穿过已知门"，扇形只留兜底。

### 10. closest-to-waypoint 目标函数的墙前鞍点 ★★
- **症状**：BFS 选"连通区内离航点最近的格子"，狗钉在离航点最近的墙点上来回蹭。
- **根因**：直线距离评不了绕行收益——绕行两侧到航点几乎等距（鞍点）；不知道门在哪就不知道哪边绕近。
- **修法**：换成 A*（unknown=8 可穿），由代价自然表达"穿门比绕已知远路便宜"。

### 11. 地图数组边界外 ≠ 非法 ★★
- **症状**：A* 推进永远返回 None（回退扇形振荡）。
- **根因**：航点常在已建图 bbox **外**（目标房间没探过），按"数组越界=错误"处理直接放弃。
- **修法**：数组外 = unknown（从未观测），照常计费穿越。

### 12. 狗贴墙时自身格读作"占据" ★
- **症状**：A* 路径回溯的"连续 free 段"长度为 0。
- **根因**：狗贴墙，它站的格子在图里是墙像素。
- **修法**：跳过起点格（A* 不穿墙，脏格最多只有起点这一格）。

## 定位/SLAM 类

### 13. 走廊沿轴不可观测 → 1.9m 形变 ★★★
- **症状**：slam_err 稳定在 ~1.87m，且**几乎全是单一轴向偏移**（y），x 完全准。
- **根因**：狗沿南北长走廊推进时，走廊几何对 y 平移不可观测（slam 和 scan-match 都看不见），
  漂移静默积累进地图；进二维富几何房间后**自愈**（0.57m），再进新走廊又犯。
- **教训**：这不是 bug 是几何本质。对策=环境破对称（墙抖动）+ 避免长走廊暴露 + 回环重访。

### 14. 自建图修正的误差下限 = 地图局部变形 ★★
- **症状**：对自建图周期重定位后 odom-真值 flat 在 ~1.8m（15m 基线轮），比对真图修正（0.87m）差一倍。
- **根因**：修正把 odom **钉到地图系**——地图哪里变形 odom 就继承哪里。参考图自身就是伤员。
- **定位**：这不是失败——**有界性成立**（8.5m 失控 → ≤2m 有界），只是下限由地图质量决定。

### 15. 低分匹配必须拒绝修正 ★
- **症状/风险**：狗在未探明区域，图上没几面墙，匹配器乱给一个"最优"→ 把 odom 锚到错误位置。
- **修法**：score<40 拒绝（"该区域建图不足"日志）。类似 AMCL 低似然不更新。

### 16. OccupancyGrid 行约定：本系统是 no-flip ★★
- **症状风险**：若按标准 ROS 翻转约定（row0=顶）转坐标，整张图上下镜像，匹配得分 ~0。
- **实测**：本系统 slam_toolbox 发的 /map 用 **row=(y-origin_y)/res 随 y 增**（firefly/goal_runner
  一直这么用且导航正确）。已由 `scripts/probe_map_convention.py` 独立验证（no-flip 命中 65/360
  且匹配位姿距真值 0.36m；镜像 1%）。
- **方法论**：**把匹配器本体当探针**——从 odom 初始位姿做相关搜索，对的约定高分且落在真值附近，
  错的 ~0 分。比投影真值端点查图靠谱（后者被地图变形干扰，曾 0% 全空）。

### 17. 雪球效应：回环只在搜索窗内找候选 ★★
- 漂移 > loop_search_maximum_distance（默认 3m）→ 真回访点落窗外 → 永不回环 → 漂移再也修不掉。
- 已调 7m + 关键帧更密 + 阈值降低（configs/slam_tuned_params.yaml）。

## 实验方法类

### 18. 录制窗口必须对准"狗在动" ★★
- **症状**：870s 录制全是静止狗（我自己的流程失误，还覆盖了前一份有运动数据的文件）。
- **修法**：开录前 `tail _progress.log` 确认位姿在变；中途再查一次。

### 19. 修正量对账 = 同 run 反事实证据 ★★
- 不用另跑对照组：修正日志每次的 pos/yaw 修正量累计 ≈ 不修正本会漂掉的量
  （yaw mean 1.5°/次 = 0.05°/s×30s 精确对账）。省一半实验。

### 20. 同 seed 只保证同抽签，不保证同实验条件 ★
- 对比实验要检查 drift 抽签值（bridge 启动日志"scale_f=… yaw_bias=…"）——不同轮若 env 组合
  不同，抽签可能不同（如 yaw_bias -0.31 vs -0.05），横向误差压力差 6 倍。

### 21. pkill -f 自杀陷阱：模式匹配执行者自身的命令行 ★★★
- **症状**：清理链 `pkill -9 -f simtrack.sim_bridge; pkill …` 看似执行了，实际**第一条就把自己杀了**
  ——SSH exec_command 的 bash cmdline 含全部模式串，`-f` 全命令行匹配时 bash 自己就是命中目标
  （pkill 只排除 pkill 进程本身，不排除父 shell）。链中后续 pkill 全部没跑。
- **为什么一直没炸**：tmux kill-session 触发 `ros2 launch` 的 SIGTERM 级联清了大部分子树，
  pkill 只是"补刀"——补刀刀断了没人发现，直到 #22 累积爆雷。
- **修法**：方括号正则 `pkill -9 -f 'monitor_[p]rogress'`——正则不匹配含方括号的自身 cmdline。
  所有清理脚本已改（batch_run.py / restart_stack.py）。

### 22. 脱离终端的 `(进程 &)` = 永生孤儿，耗尽 CycloneDDS participant 池 ★★★
- **症状**：新进程起不来，报 `Failed to find a free participant index for domain 0`；
  录制/监控悄悄死掉（批跑里表现为 traj 文件缺失、progress.log 为空）。
- **根因链**：`(monitor_progress.py &)` 从 pane shell 脱离 → kill-session 后 PPID=1 永生 →
  又因 #21 pkill 链自杀没补刀 → 跨 20+ 次重启累积 17 个 → DDS participant 池耗尽。
- **修法**：长跑进程一律放**独立 tmux 窗口前台**（随 session 死）；批跑干脆去掉非必需 monitor。
  排查命令：`pgrep -af 'monitor_[p]rogress'`、看 `/dev/shm`。

### 23. 日志抓取不能只 grep 关键词——Traceback 被滤掉，崩溃查无此案 ★★
- **症状**：seed 只到 2 个航点，runner 日志"恰好停在一条正常 WARN 后面"，无任何错误痕迹。
- **根因**：抓取命令 `capture-pane | grep -E "✅|→|超时"` 把 Traceback 行滤掉了——除了
  恰好含"进度超时"的**源码行**混进结果（那行反而成了破案线索）。
- **修法**：抓取永远多存一份**未过滤的 pane 尾部**（batch_run 已加 seed<k>_pane.log）。

### 24. 疑难崩溃上随机性质测试，几分钟顶一小时盲猜 ★★★
- **案例**：goal_runner 在超时→条件拉黑→路线验证链上神秘暴毙。本地桩+固定场景复现不出；
  换**随机地图/随机位姿/随机黑名单轰炸 3000 次**，第 8 次就炸出 `RecursionError`——
  `_astar` nudge 分支传了 `allow_nudge=False` 但分支根本没检查它，代打格也被墙围死时
  无限自递归 1000 层。
- **教训**：① 传防御标志必须检查（写完自查一遍调用链）；② "我只改了一小处"的 bug
  最适合性质测试兜底；③ 崩溃类问题优先让机器找，别人肉枚举路径。

### 25. Windows 后台任务的 TaskStop 杀不死 python 子进程 = 幽灵实例 ★★★
- **症状**：两个批跑实例"同时"跑，远程栈被互相 pkill/restart，数据全废。
- **根因**：Windows 上杀外层 bash 不会级联杀 python.exe 子进程；TaskStop 后它继续跑完。
- **修法**：批跑加**心跳锁**（`_lock` 文件每 30s touch，启动前查 mtime<90s 即退出）；
  人工清理用 PowerShell 按 CommandLine 过滤后 Stop-Process -Force。

## 已知未修（TODO）

- `record_traj.py` 的 slam 列与 monitor 不一致（疑似 TF 时间戳问题）——用 monitor + CSV odom 列替代。
- CORRECT_PERIOD_S=15 参数扫描未做（预期收益有限，下限卡在地图变形）。

---

## 旧时期（纯 Python）血泪索引 → [2026-08-05-dog50-maze-pitfalls.md](2026-08-05-dog50-maze-pitfalls.md)

仍然高频引用的几条：
- §13 yaw 读回恒 0 bug（狗从来不转弯的终极根因）
- §15.7 PASS_CLEAR 0.3m 否决（0.6m 是误差余量不是碰撞余量）
- §17.7 RELOC 触发 0 次（scan-matching 对自洽漂移是瞎的——**ROS2 迁移的直接原因**）
- §16.7 scan-to-scan 帧间匹配否决（参照物比算法重要）
