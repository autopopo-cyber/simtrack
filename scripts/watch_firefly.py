#!/usr/bin/env python3
"""实时监控远程 firefly_explorer 自主探索进度。

每 3 秒刷新本地窗口：firefly 日志尾部 + 累计到达/拉黑次数。
Ctrl-C 退出。依赖项目 .venv 里的 paramiko。

用法（或直接双击根目录 watch_firefly.bat）:
    .venv\\Scripts\\python scripts\\watch_firefly.py
"""
import os
import re
import time
import paramiko

HOST, USER, PWD = "100.64.63.98", "qin", "1"


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PWD, timeout=10)

    def run(cmd, t=12):
        _, o, e = c.exec_command(cmd, timeout=t)
        return o.read().decode() + e.read().decode()

    print("已连接远程，开始监控（Ctrl-C 退出）...")
    time.sleep(0.5)
    try:
        while True:
            os.system("cls" if os.name == "nt" else "clear")
            log = run("tmux capture-pane -J -t sim:firefly -p -S -22 2>/dev/null", 8)
            # 从较长历史里解析权威统计：firefly 每次完成/到达都会打 sent=ok=fail= 行，
            # 它不受回溯窗口滚动影响（最近的统计行一定在窗口里）。比数关键字可靠得多。
            # -J 合并被终端 80 列折行的行（中文长行不折，关键字才不会断）
            hist = run("tmux capture-pane -J -t sim:firefly -p -S -600 2>/dev/null", 8)
            sent = ok = fail = None
            for line in reversed(hist.splitlines()):
                m = re.search(r"sent=(\d+)\s+ok=(\d+)\s+fail=(\d+)", line)
                if m:
                    sent, ok, fail = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    break
            # fallback：解析不到统计行（还在跑、没打过）就数确认行
            if ok is None:
                ok = hist.count("✅ 到达")
                fail = hist.count("拉黑区域")
                sent = ok + fail
            # 状态判定：倒序找最近一个事件行（-J 已合并折行，关键字不会断）
            status = "…"
            for line in reversed(hist.splitlines()):
                if "选 frontier" in line:
                    status = "🔄 探索中"; break
                if "探索完成" in line or "探索结束" in line:
                    status = "✅ 探索已完成"; break
                if "✅ 到达" in line:
                    status = "🔄 到达一点，选下一个中"; break
            last = ""
            for line in hist.splitlines():
                if "剩余" in line:
                    last = line.strip()

            print("=" * 64)
            print("   firefly_explorer 自主探索监控    " + time.strftime("%H:%M:%S"))
            print("=" * 64)
            print(f"  状态: {status}")
            print(f"  累计: 发送 {sent}  到达 ✅ {ok}  失败/拉黑 ❌ {fail}")
            tail = last.split("剩余=")[-1].strip() if "剩余=" in last else "—"
            print(f"  最近剩余 frontier 数: {tail}")
            print("-" * 64)
            print(log.strip() or "[firefly 窗口无输出 —— 检查 tmux 的 sim:firefly 是否在跑]")
            print("-" * 64)
            print("（3 秒后刷新，Ctrl-C 退出）", flush=True)
            time.sleep(3)
    except KeyboardInterrupt:
        print("\n退出监控。")
    finally:
        c.close()


if __name__ == "__main__":
    main()
