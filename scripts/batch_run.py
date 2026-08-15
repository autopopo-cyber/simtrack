#!/usr/bin/env python3
"""批量窄门实验编排器（本地跑，后台长任务）。

对每个 seed：本地生成 rooms10x10b 迷宫（BFS 路径中段一扇 0.8m 窄门）→ 部署远程 →
干净重启全栈（bridge/slam/nav2/goal_runner）→ 600s 轨迹录制 → 拉回数据。
结果落 results/batch1/：seed<k>_traj.csv / _meta.json / _runner.log / _progress.log
状态实时写 results/batch1/_status.log。
"""
import os
import subprocess
import sys
import time

import paramiko

SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
RUN_S = 600
HOST, USER, PW = "100.64.63.98", "qin", "1"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "batch1")
VENV_PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")

# 用法: batch_run.py [seeds逗号表] [结果目录名]
#   例: batch_run.py 1,3,8 quick1   → results/quick1/
if len(sys.argv) > 1:
    SEEDS = [int(x) for x in sys.argv[1].split(",")]
if len(sys.argv) > 2:
    OUT = os.path.join(REPO, "results", sys.argv[2])

BRIDGE_ENV = ("MAZE=rooms10x10b ODOM_DRIFT_PCT=5 ODOM_DRIFT_YAW_BIAS_DEG=0.4 "
              "ODOM_DRIFT_SEED=42 CORRECT_PERIOD_S=30 CORRECT_REF=map "
              "LIDAR_RANGE=10 LIDAR_NOISE_M=0.03")


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(os.path.join(OUT, "_status.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sh(ssh, cmd, timeout=30):
    _, out, err = ssh.exec_command(cmd, timeout=timeout)
    return (out.read().decode() + err.read().decode()).strip()


def run_seed(seed):
    # 1) 本地生成迷宫
    r = subprocess.run([VENV_PY, "-m", "simtrack.maze_gen", "rooms10x10b", str(seed)],
                       cwd=REPO, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        log("seed%d 生成失败: %s" % (seed, r.stderr[-200:]))
        return
    # 2) 部署
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PW, timeout=15)
    sftp = ssh.open_sftp()
    for f in ("maze_rooms10x10b.png", "maze_rooms10x10b.meta.json"):
        sftp.put(os.path.join(REPO, "confirmed", f),
                 "/home/qin/simtrack/confirmed/" + f)
    # 3) 干净重启（铁律：kill-session 杀不死 launch 子树，pkill+pgrep 确认）
    sh(ssh, "pkill -9 -f simtrack.sim_bridge; pkill -9 -f simtrack.goal_runner; "
            "pkill -9 -f slam_toolbox; pkill -9 -f nav2_bringup; pkill -9 -f record_traj; "
            "pkill -9 -f monitor_progress; pkill -9 -f component_container; sleep 2")
    sh(ssh, "tmux kill-session -t sim 2>/dev/null; sleep 1; "
            "tmux new-session -d -s sim -n bridge; tmux new-window -t sim -n slam; "
            "tmux new-window -t sim -n nav2; tmux new-window -t sim -n drive; "
            "tmux new-window -t sim -n mon; rm -f /home/qin/simtrack/_progress.log")
    left = sh(ssh, 'pgrep -af "slam_toolbox|nav2_bringup|sim_bridge|goal_runner" | grep -v pgrep')
    if left:
        log("seed%d 残留进程未清，跳过: %s" % (seed, left[:120]))
        ssh.close()
        return
    # 4) 启动（顺序铁律：bridge 先起=/clock 主人）
    sh(ssh, 'tmux send-keys -t sim:0 "cd ~/simtrack && source /opt/ros/jazzy/setup.bash && '
            '%s /usr/bin/python3 -m simtrack.sim_bridge" Enter' % BRIDGE_ENV)
    time.sleep(7)
    sh(ssh, 'tmux send-keys -t sim:1 "bash ~/simtrack/run_slam.sh" Enter')
    time.sleep(6)
    sh(ssh, 'tmux send-keys -t sim:2 "bash ~/simtrack/run_nav2.sh" Enter')
    time.sleep(16)
    sh(ssh, 'tmux send-keys -t sim:3 "cd ~/simtrack && source /opt/ros/jazzy/setup.bash && '
            'MAZE=rooms10x10b /usr/bin/python3 -m simtrack.goal_runner" Enter')
    time.sleep(4)
    sh(ssh, 'tmux send-keys -t sim:4 "cd ~/simtrack && source /opt/ros/jazzy/setup.bash && '
            '(/usr/bin/python3 monitor_progress.py > _mon_stdout.log 2>&1 &) && '
            '/usr/bin/python3 record_traj.py %d _b%d_traj.csv" Enter' % (RUN_S + 30, seed))
    log("seed%d 栈已起，录制 %ds…" % (seed, RUN_S))
    # 5) 等待
    time.sleep(RUN_S + 15)
    # 6) 收数据
    for remote, local in (
            ("/home/qin/simtrack/_b%d_traj.csv" % seed, "seed%d_traj.csv" % seed),
            ("/home/qin/simtrack/confirmed/maze_rooms10x10b.meta.json", "seed%d_meta.json" % seed)):
        try:
            sftp.get(remote, os.path.join(OUT, local))
        except Exception as e:
            log("seed%d 拉 %s 失败: %s" % (seed, local, e))
    for name, cap in (("seed%d_runner.log" % seed,
                       'tmux capture-pane -p -t sim:3 -S -3000 | grep -E "✅|→|跳过|被拒|拉黑|超时" | tail -80'),
                      ("seed%d_progress.log" % seed, "tail -200 ~/simtrack/_progress.log")):
        try:
            _, out, _ = ssh.exec_command(cap, timeout=15)
            open(os.path.join(OUT, name), "w", encoding="utf-8").write(out.read().decode())
        except Exception as e:
            log("seed%d 抓 %s 失败: %s" % (seed, name, e))
    wp = sh(ssh, 'tmux capture-pane -p -t sim:3 -S -3000 | grep -c "✅"')
    last = sh(ssh, "tail -1 ~/simtrack/_progress.log")
    log("seed%d 完成: 航点达成=%s | %s" % (seed, wp, last))
    ssh.close()


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    for seed in SEEDS:
        try:
            run_seed(seed)
        except Exception as e:
            log("seed%d 异常(继续): %r" % (seed, e))
    log("全部完成，总耗时 %.0f 分钟" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
