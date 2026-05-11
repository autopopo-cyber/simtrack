#!/bin/bash
# bounce launcher — 玩具车避障 / 导航仿真
# 用法: bash bounce_launcher.sh [bounce_obs|bounce_nav] [秒数默认30]

set -e
SCRIPT=${1:-bounce_obs}
DURATION=${2:-30}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

[ -f confirmed/track_clean.png ] || { echo "!!! confirmed/track_clean.png 缺失"; exit 1; }
echo "赛道: confirmed/track_clean.png | 脚本: $SCRIPT | 时长: ${DURATION}s"

[ -f "test_scripts/${SCRIPT}.py" ] || { echo "!!! test_scripts/${SCRIPT}.py 缺失"; exit 1; }

# Xvfb
if [ -z "$DISPLAY" ]; then
    if ! pgrep -x Xvfb >/dev/null; then
        Xvfb :99 -screen 0 1280x720x24 -ac &
        sleep 1
    fi
    export DISPLAY=:99
fi
echo "DISPLAY=$DISPLAY"

echo "=== 启动仿真 ==="
timeout "$DURATION" python3 "test_scripts/${SCRIPT}.py" 2>&1 || true
echo "=== 完毕 ==="
