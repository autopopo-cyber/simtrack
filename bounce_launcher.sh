#!/bin/bash
# 仿真启动器 — 多算法支持
# 用法: bash bounce_launcher.sh <algo> [秒数]
# algo: algo0_bounce | algo1_arc_racer | algo2_lane_nav

set -e
ALGO=${1:-algo1_arc_racer}
DURATION=${2:-30}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

[ -f confirmed/track_clean.png ] || { echo "!!! confirmed/track_clean.png 缺失"; exit 1; }
[ -f "test_scripts/${ALGO}.py" ] || { echo "!!! test_scripts/${ALGO}.py 缺失"; exit 1; }

echo "=== ${ALGO} ${DURATION}s ==="

if [ -z "$DISPLAY" ]; then
    if ! pgrep -x Xvfb >/dev/null; then
        Xvfb :99 -screen 0 1280x720x24 -ac & sleep 1
    fi
    export DISPLAY=:99
fi

timeout "$DURATION" python3 "test_scripts/${ALGO}.py" 2>&1 || true
echo "=== 完毕 ==="
