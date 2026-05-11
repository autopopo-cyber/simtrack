#!/bin/bash
# bounce_obs launcher — 一步到位，玩具车避障仿真
# 用法: bash bounce_launcher.sh [秒数默认30]

set -e
DURATION=${1:-30}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 确认赛道文件存在
[ -f confirmed/track_clean.png ] || { echo "!!! confirmed/track_clean.png 缺失"; exit 1; }
echo "赛道: confirmed/track_clean.png"

# 确保 Xvfb 在跑
if ! pgrep -x Xvfb >/dev/null; then
    Xvfb :99 -screen 0 1280x720x24 -ac &
    sleep 1
fi

# 跑仿真
echo "=== 启动仿真 ${DURATION}s ==="
DISPLAY=:99 timeout "$DURATION" python3 test_scripts/bounce_obs.py 2>&1 || true
echo "=== 完毕 ==="
