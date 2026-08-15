#!/usr/bin/env python3
"""远程栈部署/重启/健康检查工具（本地跑）。

用法:
  restart_stack.py deploy [文件...]     # sftp 推文件到远程同路径（相对 repo 根）
  restart_stack.py restart [seed]       # 生成 rooms10x10b 种子迷宫→干净重启→起全栈
  restart_stack.py health [秒]          # 等 N 秒后抓 nav2/runner/progress 三窗健康快照
"""
import os
import subprocess
import sys
import time

import paramiko

HOST, USER, PW = "100.64.63.98", "qin", "1"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")

BRIDGE_ENV = ("MAZE=rooms10x10b ODOM_DRIFT_PCT=5 ODOM_DRIFT_YAW_BIAS_DEG=0.4 "
              "ODOM_DRIFT_SEED=42 CORRECT_PERIOD_S=30 CORRECT_REF=map "
              "LIDAR_RANGE=10 LIDAR_NOISE_M=0.03")


def sh(ssh, cmd, timeout=30):
    _, out, err = ssh.exec_command(cmd, timeout=timeout)
    return (out.read().decode() + err.read().decode()).strip()


def connect():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PW, timeout=15)
    return ssh


def deploy(ssh, files):
    sftp = ssh.open_sftp()
    for f in files:
        remote = "/home/qin/simtrack/" + f.replace("\\", "/")
        sftp.put(os.path.join(REPO, f), remote)
        print("  推送 %s → %s" % (f, remote))
    sftp.close()


def restart(ssh, seed):
    r = subprocess.run([VENV_PY, "-m", "simtrack.maze_gen", "rooms10x10b", str(seed)],
                       cwd=REPO, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        print("迷宫生成失败:", r.stderr[-300:])
        sys.exit(1)
    sftp = ssh.open_sftp()
    for f in ("maze_rooms10x10b.png", "maze_rooms10x10b.meta.json"):
        sftp.put(os.path.join(REPO, "confirmed", f),
                 "/home/qin/simtrack/confirmed/" + f)
    sftp.close()
    print("迷宫 rooms10x10b seed=%d 已部署" % seed)
    # 干净重启（铁律：kill-session 杀不死 launch 子树，pkill+pgrep 确认）
    sh(ssh, "pkill -9 -f simtrack.sim_bridge; pkill -9 -f simtrack.goal_runner; "
            "pkill -9 -f slam_toolbox; pkill -9 -f nav2_bringup; pkill -9 -f record_traj; "
            "pkill -9 -f monitor_progress; pkill -9 -f component_container; sleep 2")
    sh(ssh, "tmux kill-session -t sim 2>/dev/null; sleep 1; "
            "tmux new-session -d -s sim -n bridge; tmux new-window -t sim -n slam; "
            "tmux new-window -t sim -n nav2; tmux new-window -t sim -n drive; "
            "tmux new-window -t sim -n mon; rm -f /home/qin/simtrack/_progress.log")
    left = sh(ssh, 'pgrep -af "slam_toolbox|nav2_bringup|sim_bridge|goal_runner" | grep -v pgrep')
    if left:
        print("❌ 残留进程未清:", left[:200])
        sys.exit(1)
    # 启动（顺序铁律：bridge 先起=/clock 主人）
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
            '(/usr/bin/python3 monitor_progress.py > _mon_stdout.log 2>&1 &) " Enter')
    print("全栈已起（bridge/slam/nav2/goal_runner/monitor）")


def health(ssh, wait_s):
    time.sleep(wait_s)
    alive = sh(ssh, 'pgrep -af "nav2_bringup|sim_bridge|goal_runner|slam_toolbox" '
                    '| grep -v pgrep | awk \'{print $1, $3, $4}\'')
    print("=== 存活进程\n%s" % alive)
    nav2 = sh(ssh, 'tmux capture-pane -p -t sim:2 -S -120 | grep -vi "elapsed" | tail -25')
    print("=== nav2 窗口尾部\n%s" % nav2)
    runner = sh(ssh, 'tmux capture-pane -p -t sim:3 -S -200 | grep -E "✅|→|跳过|被拒|拉黑|超时|失败" | tail -12')
    print("=== runner 关键行\n%s" % runner)
    prog = sh(ssh, "tail -3 ~/simtrack/_progress.log 2>/dev/null")
    print("=== progress 尾部\n%s" % prog)
    ec = sh(ssh, 'tmux capture-pane -p -t sim:2 -S -2000 | grep -ci "error_code\\|legal potential"')
    print("=== nav2 错误行计数(含legal potential): %s" % ec)


def main():
    ssh = connect()
    try:
        cmd = sys.argv[1] if len(sys.argv) > 1 else ""
        if cmd == "deploy":
            deploy(ssh, sys.argv[2:])
        elif cmd == "restart":
            restart(ssh, int(sys.argv[2]) if len(sys.argv) > 2 else 3)
        elif cmd == "health":
            health(ssh, int(sys.argv[2]) if len(sys.argv) > 2 else 0)
        else:
            print(__doc__)
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
