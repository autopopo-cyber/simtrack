# 架构设计 — ROS2 + MuJoCo 导航栈

> 会话恢复速查。最后更新 2026-08-14（v2.0-ros2-slam-nav）。

## 一、全景数据流

```
【离线】maze_gen.py ──→ confirmed/maze_<name>.png (高度图, 墙255/路128, 50px/m)
                        + maze_<name>.meta.json (start/goal sidecar——sim_bridge 读它定起点)

【仿真核心，无ROS】sim_server.SimBackend
   MuJoCo 运动学狗(slide x/y + hinge yaw, contype=0 不物理碰撞)
   解析式高度图射线扫描 get_scan()  ← 噪声 LIDAR_NOISE_M 加在这（/scan和修正吃同一份）
   相关扫描匹配 scan_match()        ← 对真迷宫图（仿真特权版匹配器）
   碰撞判定 _check_collision()      ← 0.8×0.4m 胶囊 5 点采样，撞墙阻塞平移/旋转
        ▲ import
【ROS2 桥】sim_bridge.SimBridge (rclpy Node "sim_bridge")
   发: /scan(10Hz,BEST_EFFORT) /odom(漂移位姿!) /clock /true_pose(诊断,frame="true")
       TF: odom→base_footprint(漂移) + 静态 base_footprint→base_link / →laser(z=0.5)
   订: /cmd_vel → set_cmd_vel
   OdometryDrift: 5%尺度+陀螺零偏+噪声（env 控制，模拟腿式里程计）
   _correction_cb: 每 CORRECT_PERIOD_S(30s) 对参考图做相关匹配重置漂移里程计
       CORRECT_REF=true→真迷宫图(特权) | map→slam自建的/map(诚实,土法AMCL)
        │ /scan /odom /clock /tf
        ▼
【定位建图】slam_toolbox online_sync (configs/slam_tuned_params.yaml)
   关键调参: loop_search_maximum_distance 3→7(破雪球) 关键帧更密 阈值更低
   发: /map (latched TRANSIENT_LOCAL) + TF map→odom
        │ /map + TF
        ▼
【规划控制】Nav2 (configs/nav2_fast_params.yaml)
   NavFn(allow_unknown=true,robot_radius 0.22) + MPPI(iteration 3, vx_max 0.7)
   + velocity_smoother + 恢复行为(清costmap/旋转)
        │ navigate_to_pose action
        ▼
【驱动层·二选一】
   goal_runner  — 21房间航点逐个发 NavigateToPose + A*推进(冷启动/找门) ← 当前主力
   firefly_explorer — frontier质心→NavigateToPose(自研, 被frontier_exploration_ros2替代)
   frontier_exploration_ros2 — 第三方MRTSP全局排序探索(~/exploration_ws, 反振荡)
        │ /cmd_vel
        ▼
   (回到 sim_bridge → SimBackend)
```

## 二、关键设计决策（为什么是现在这样）

| 决策 | 理由 | 出处 |
|---|---|---|
| MuJoCo 而非 Gazebo | 轻量headless+高度图解析射线快；Gazebo太重 | 2026-08-13-ros2-mujoco-pipeline.md |
| slam_toolbox 而非自研 | 纯Python原型的决定性教训：scan-matching对**自洽漂移**是瞎的（地图本身就是漂移位姿写的，恒等分=1.0）；位姿图+回环才有<1m | 踩坑§17.8（迁移触发点） |
| 狗=运动学滑块(contype=0) | 步态不是研究对象；碰撞用胶囊采样判定（阻塞而非物理弹开） | sim_server |
| /odom 发**漂移**位姿，/scan 永远真值 | 让slam拿"错的里程计+对的激光"——复现真机处境；真值只从 /true_pose 出（测量用） | sim_bridge |
| 周期重定位在 bridge 内而非独立节点 | 修正的是 bridge 自己的 OdometryDrift 状态；真机上等价物=AMCL(复访)或LIO本身 | §七-九 |
| 驱动用 goal_runner(A*推进) 而非 frontier | NavFn远目标穿大片unknown会梯度回溯失败；短步进free子目标免疫。A*给短目标 | pitfalls-ros2 #3 |
| 两阶段定位架构 | 未知屋首访=slam_toolbox建图；复访=AMCL/slam localization模式(存图后)。真机L2自带POINT-LIO当odom | §八/用户讨论 |
| 墙抖动±0.5m破4重旋转对称 | 纯2D几何SLAM解不了对称歧义；抖动让每房形状不同 | maze_gen |

## 三、远程部署拓扑（所有实验在这跑）

```
Windows (D:\workspace\simtrack, 本地仓库+文档)
   │ SSH/SFTP (paramiko, 密码"1"——主人提供; 无sshpass)
   ▼
Linux qin@100.64.63.98 (Ubuntu, ROS2 Jazzy /opt/ros/jazzy)
   ~/simtrack/            ← 平铺部署(注意: 不是git clone, 是散文件! 用sftp单传)
       simtrack/*.py      run_{slam,nav2,frontier}.sh
       configs/*.yaml     confirmed/maze_*.png(+meta)
       record_traj.py monitor_progress.py ...
   ~/exploration_ws/      ← frontier_exploration_ros2 (colcon build)
   tmux session "sim": 5 windows = 0:bridge 1:slam 2:nav2 3:drive(goal_runner) 4:mon
```

**启动顺序有讲究**：bridge 先起（它是 /clock 主人）→ slam → nav2（等 lifecycle active，无/clock会activation超时弃疗）→ drive → mon。

**重启铁律**：`tmux kill-session` **杀不死** launch 子进程树！必须 pkill 全部 + `pgrep -af` 确认为空再重启，否则双 slam 双 nav2 双 /map 双 goal 竞争，症状诡异。

## 四、帧与坐标系约定（易错！）

| 约定 | 内容 | 坑位 |
|---|---|---|
| 迷宫世界系 | 原点=左下, x右y上, 单位米 | maze_gen.py 头注 |
| 高度图像素 | col=x*50, row=(H-1)-y*50（row0=顶部=max y） | sim_server._match_score |
| **/map OccupancyGrid 行约定** | **no-flip**: row=(y-oy)/res 随 y 增（实测验证！标准ROS翻转约定在这套系统是错的） | probe_map_convention.py; firefly/goal_runner 同款 |
| 机身系 | angle 0=前方(+x), CCW 正 | sim_server.get_scan |
| TF 链 | map→odom(slam发)→base_footprint(bridge发,漂移)→base_link→laser | base_footprint 给 slam, base_link 给 Nav2 |

## 五、真机目标架构（未实现，规划）

```
A2 实机: L2(前后) ──UDP──→ l2_bridge节点(新ament包)
   ├→ POINT-LIO odom → /odom (常规层, 漂移0.1-1%)
   └→ PointCloud2 → z带切片(10-40cm) → /scan (slam_toolbox+Nav2照用)
 腿式里程计 → EKF(robot_localization, 融合LIO+腿odom+IMU) → 容灾
 仿真栈的周期重定位(CORRECT_REF=map) = LIO故障时的降级保险
```
