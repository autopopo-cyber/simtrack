#!/bin/bash
# bounce_obs launcher — 一步到位，玩具车避障仿真
# 用法: bash bounce_launcher.sh [秒数默认30]

set -e
DURATION=${1:-30}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 生成赛道 (4px围墙 + 路面)
echo "=== 生成赛道 ==="
python3 -c "
import numpy as np, cv2, os, math
PPM=40; RR=int(2.5*PPM)
hf=np.full((2000,2000),191,dtype=np.uint8)
pts=[]
y0=2.5
for s in range(10):
    y=y0+s*5.0; l2r=(s%2==0)
    xs=np.arange(5,45.01,0.25) if l2r else np.arange(45,4.99,-0.25)
    for x in xs: pts.append((x,y))
    if s<9:
        ny=y0+(s+1)*5.0; cx=45 if l2r else 5; cy=(y+ny)/2
        sa,ea=(math.pi/2,3*math.pi/2) if l2r else (3*math.pi/2,5*math.pi/2)
        for j in range(1,17):
            a=sa+(ea-sa)*j/17; pts.append((cx+5*math.cos(a),cy+5*math.sin(a)))
for cx,cy in pts:
    px,py=int(cx*PPM),1999-int(cy*PPM)
    cv2.circle(hf,(px,py),RR,128,-1)
hf[0:4,:]=191; hf[-4:,:]=191; hf[:,0:4]=191; hf[:,-4:]=191
os.makedirs('confirmed',exist_ok=True)
cv2.imwrite('confirmed/track_clean.png',hf)
print(f'赛道生成完毕 Road:{np.sum(hf==128)} Wall:{np.sum(hf==191)}')
"

# 确保 Xvfb 在跑
if ! pgrep -x Xvfb >/dev/null; then
    Xvfb :99 -screen 0 1280x720x24 -ac &
    sleep 1
fi

# 跑仿真
echo "=== 启动仿真 ${DURATION}s ==="
DISPLAY=:99 timeout "$DURATION" python3 test_scripts/bounce_obs.py 2>&1 || true
echo "=== 完毕 ==="
