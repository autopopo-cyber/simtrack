# 提速攻坚：15 分钟 → 2.3 分钟（2026-08-09 晚）

> 主人目标：跑完 500m 无特权探索从 ~15 分钟墙钟压到 2-5 分钟（2 分钟 = 物理极限 120s）。
> 方法论：文档祖训"先 profile 再优化"（pitfalls #11）+"瓶颈在 bounce 不在直线速度"（pitfalls 六）。

## 一、成绩

| 场景 | 物理时间 | 墙钟 | 步数 | bounce | collision | 对比优化前 |
|---|---|---|---|---|---|---|
| 纯墙无特权探索 | **135s**（极限 120s） | **140s = 2.3min** | 27040 | **10** | 0 | 物理 720s / 墙钟 ~16min / bounce 185 |
| 混合 20 固定障碍 | **160s** | **175s = 2.9min** | 31960 | **11** | 0 | 物理 826s / bounce 296 |
| 随机移动障碍×4+DWA | **142s** | **161s = 2.7min** | 28444 | **0** | 0 | 物理 583s / bounce 183 |

三个场景全部无特权到达（距真值 2.4-3.5m）、collision 0、二维码 10/10。

纯墙场：里程 455m（<476m 中心线——过弯自然切角），**78.5% 时间满速 >3m/s**，每通道均匀 ~12s，
10/10 二维码识别，终点估计误差 2.94m，到达距真值 2.41m。**无特权探索跑出了原作弊版的物理极限成绩。**

## 二、计算优化（150 → 360 步/s 冒烟段，全程均速 ~192）

cProfile（4000 步）定位的三大头 + 处置：

1. **栅格 dict → numpy 数组**（gget 430 万次/调用 6s/4000 步）：
   `G`/`SG` 两个 500×500 int8 数组替代 dict；`gget/gset/gget_plan` 语义不变（越界读=UNKNOWN）。
   scan 标记改 fancy indexing 批量写（先 FREE 后 WALL = 同格 WALL 优先），省掉逐格 Python 循环
   和 np.unique。`_cnt` 计数器 → 按需 count_nonzero。
2. **PG 物化规划视图**：gget_plan 230 万次/call 是 find_gates 主成本 → scan/gset 置脏 +
   一次性重建（`PG = where(SG==WALL, WALL, where(G!=UNKNOWN, G, SG))`），find_gates/wall_dist/
   _open_frontier 全程 PG 直读。find_gates 115ms → ~20ms/call。
3. **视觉降本**：渲染 1280×720 → 640×360（mjr_render 30→8ms/帧）；ArUco 金字塔 6 尺度 → 3 尺度
   （0.5/1.0/2.0，2m 标牌 25m 处在 1.0 尺度仍有 ~35px，3x 放大 4K 白跑）+ 每 3 帧才检一次
   （0.6s 物理一检，10/10 识别率不降）。终点球检测每帧不受影响。
4. **前瞻采样 0.05→0.1m**（blocked 调用减半；墙厚≥0.2m+keepout 邻格覆盖，不漏墙）。
5. **DWA 向量化**（移动障碍场景）：77 轨迹×30 点逐点 Python 回调是全场最大热点
   （62s/3000 步）→ `choose_velocity(..., blocked_batch=)` 快路径：全部轨迹一次性 numpy 模拟 +
   栅格批量判定（38 → 246 步/s，6.4x）。标量路径保留（tests/test_dwa.py 8/8 过）。

### DWA 陈旧 None 死锁（移动障碍回归暴露）

- **现象**：移动障碍场狗锁死在 (29.1,8.8)，bounce 660 次刷频，DWA 报全碰撞但前瞻 4m 畅通。
- **根因**：`dwa_target=None` 被当作"DWA 全碰撞"进 STOP 分支，但 DWA 只在路径跟随分支
  每 10 步咨询一次——path=None 的窗口里 None 是"没咨询"不是"全碰撞"：STOP 每步触发 →
  bounce 刷频 → need_replan → escape 被 _bounce 清零速度 → 永不移动。旧版被僵尸路径 bug
  掩盖（zombie 踏步用陈旧但非 None 的 dwa_target），僵尸修复后暴露。
- **修复**：`mv.dwa_t` 记录咨询时刻，None 只在 ≤LIDAR_TICK 内新鲜才判全碰撞；
  无路径分支（朝估计/门/前方直走）同样每 10 步咨询 DWA。

## 三、行为优化（物理 720s → 135s 的决定性一刀）

### 僵尸路径 bug（bounce 主头，占旧版 ~500s/720s）

- **现象**：狗到达门后原地踏步 20s+，攒 ~8 次 bounce 才换下一个门。
- **根因**：主循环重规划分支全在 `if path is None` 之下，但"门 3m 即达消耗"只把
  `path_idx = len(path)` 而不置 `path = None` → 路径耗尽后任何重规划都不触发，
  执行层 else 分支朝旧门格僵尸踏步 → 贴墙 STOP/bounce，直到攒够 `bounce - base > 8`
  才强制重规划。每个门白打 ~8 次 bounce ≈ 16-25s，全程 27 门 ≈ 500s。
- **修复**：重规划块入口 `if path_idx >= len(path): path = None`——一行。
- **为何旧版能跑通**：bounce 风暴本身就是触发器（>8 才重规划），成绩被 bounce 数掩盖。
  08-07 yaw 修复后 bounce≈0 的 KNOWN_MAP 模式不受影响（直奔路径 path_is_goal 不提前消耗）。

### 掠射填充（幽灵前沿门 2.0）

- **现象**：狗被吸到长直墙脸前（如底墙 (32.5,0.7)）反复 STOP/bounce。
- **根因**：0.5° 射线间隔 + 30m 量程边界 → 掠射墙面命中点稀疏（间距 4-5m），
  墙皮前留数米宽 UNKNOWN 条 → find_gates 长出"墙脸门"（已有过滤全失效：
  wall_dist/_open_frontier 都依赖附近有墙标记，稀疏命中 5×5 窗口罩不住）。
- **修复**：相邻射线命中对连线补 WALL + **自由空间反证**（弦内部格 >25% 已 FREE →
  横穿开阔空间，拒填——开口/拐角/不同墙天然排除；同墙掠射弦贴墙皮，无射线能穿过）。
  障碍命中对不填（防假墙封死障碍↔墙 1.5m 缝）。

## 四、本轮踩坑

- **y 翻转镜像图**：`cy = py >> 2` 忘了 py 是图像行（row0=y=50m 顶部）→ 整场标记写成
  y 镜像地图，狗在镜像世界里漫游 22 万步不到终点。**教训：感知管线改动后，4000 步冒烟
  看不出行为对错——必须跑全程比对 step 数/bounce 基线**（正确基线：ms≈13/4000 步、全程 ~14 万步）。
- 全程速率 192 vs 冒烟 360：后期 find_gates 搜索范围大 + 视觉帧恒定开销。可接受。

## 五、复现命令

```bash
# 纯墙全程（墙钟 140s）
python test_scripts/algo3_headless.py --seed 7 --no-obs 1 --timeout 1200 \
  --vision 1 --landmarks 1 --render-every 0 --trail-every 20 --max-steps 300000
# 混合 20 固定障碍（墙钟 175s）：--obs-straight 1 --obs-turn 1
# 随机移动障碍×4（墙钟 161s）：--obs-random 4
# Windows 本机需 MUJOCO_GL=glfw + USERPROFILE="D:\\"（PROJ 写死 ~/workspace/simtrack）
```

日志/轨迹：`scans/opt_full_v3.log`（纯墙）、`opt_fixed_v3.log`（固定）、`opt_random_v5.log`（移动）；
速度分析：`scripts/analyze_speed.py`。
