"""地标标牌系统：20 个 ArUco 二维码标牌（DICT_7X7，唯一 ID）。

主人设计（2026-08-06）：
- 标准黑白二维码（DICT_7X7，四角定位标志明显，机器狗能看更复杂的码）
- 每个标牌唯一 ID 0-19：idx = ch*2+side（ch=通道 0-9, side=0起点/1终点）
- **标牌 2m×2m**（放大 2 倍，主人指令），中心离地 1m = 相机高度 = 视觉中心
- **贴横墙内表面**（通道两端），狗沿通道走正对看到
- **三面 box 墙**（通道两侧纵墙 + 两端横墙）提供闭合背景——实测单面墙时
  ArUco 边界分离失败，三面墙闭合后全部距离识别成功
- emissive 纹理（hfield 地形暗，非 emissive 对比不足）

关键发现（2026-08-06）：
- **MuJoCo plane 绕 y 旋转（横墙）纹理可靠，绕 x 旋转（侧墙）纹理异常**
  （绕 x 后 ArUco 所有距离检测失败）→ 标牌全部贴横墙
- hfield 是实体（0~4.996m），标牌不能嵌进去，需贴 box 墙表面凸出 0.3m
- 机器狗必须站 hfield 表面上方（body z=HF_SURF+0.5），否则相机在地底下全黑

用法（在 algo3_headless.py 主循环中）：
    vis = VisionLandmark(m, d, renderer, cam_name="bot_cam")
    detected = vis.scan_once(step)   # 返回 [(step, idx, ch, side)]
"""
import os

PROJ = os.path.expanduser("~/workspace/simtrack")
LM_DIR = os.path.join(PROJ, "assets/landmarks")

# ArUco 字典
ARUCO_DICT = "DICT_7X7_1000"

# hfield 表面高度（通道 128 灰度）：关键世界修正
# 2026-08-08 定稿：渲染图换 track_500_bin.png（从 clean 块判定生成的无抗锯齿二值图，
# 纯 128/191）→ hfield_data 路=0 墙=1 → 路表面 = zoffset 2.0 + zscale 4.0*0 = 2.0m，
# 与碰撞图 track_clean.png 完全一致。此前 track_500.png 抗锯齿（121-199）路表面 2.36m，
# 且墙边界 0.3m 斜坡导致狗贴墙渲染上墙。
HF_SURF = 2.0

# 标牌几何：2m×2m plane（size 是半尺寸 1.0），中心离地 1m = 相机高度
LM_HALF = 1.0
LM_CENTER_Z = HF_SURF + 1.0

# 机器狗 body 高度（站 hfield 表面上方 0.5m），相机在 body 局部 +0.5 → 世界 HF_SURF+1.0
BOT_Z = HF_SURF + 0.5

# 贴墙参数：凸出横墙面 0.3m（路牌式，避免嵌进 box 墙）
WALL_X_OFF = 0.3

def landmark_positions():
    """返回 10 个标牌的 (idx, ch, side, wx, wy, wz, quat)。
    每通道 1 个标牌，立在**通道中心线**（狗正前方，2m 大码最清晰），
    贴终点端墙头 x 位置（转弯口前 4m，狗走到尽头转弯前看到）。
    关键：标牌中心必须在通道中心（y=2.5+ch*5），不能贴分界墙边缘——
    否则 2m 标牌会嵌进分界墙 box（墙 y=5±0.175），码只显示一半识别失败。
    """
    out = []
    for ch in range(10):
        y_center = 2.5 + ch * 5.0    # 通道中心线
        if ch % 2 == 0:
            # 偶通道（朝 +x）：终点 x=50，标牌在 x=45.8，法线 -x
            out.append((ch, ch, 1, 45.8, y_center, LM_CENTER_Z, "0.7071 0 -0.7071 0"))
        else:
            # 奇通道（朝 -x）：终点 x=0，标牌在 x=4.2，法线 +x
            out.append((ch, ch, 0, 4.2, y_center, LM_CENTER_Z, "0.7071 0 0.7071 0"))
    return out

def wall_xml():
    """分界墙 box（contype=0 纯可视化），必须**留出 U 型转弯口**！
    迷宫结构（track_clean.png 实测）：蛇形通道，分界墙在 y=5k（k=1..9）。
    转弯口交替开口：
      - y=5,15,25,35,45（k 奇数）：x=50 端开口（通道 0→1, 2→3... 在 x=50 转弯）
      - y=10,20,30,40（k 偶数）：x=0 端开口（通道 1→2, 3→4... 在 x=0 转弯）
    边界 y=0 / y=50：全封闭。
    教训：之前侧墙 x∈[0,50] 全程 + 横墙 x=0.25/49.75 堵死了所有 U 型弯（主人指出）。
    """
    walls = []
    for k in range(11):
        y = k * 5.0
        if k == 0 or k == 10:
            # 地图边界：全墙
            walls.append(f'<geom type="box" size="25.0 0.175 2.0" pos="25 {y} {HF_SURF+1.0}" '
                         f'rgba="0.5 0.5 0.55 1" contype="0" conaffinity="0"/>')
        elif k % 2 == 1:
            # 奇数分界：x=50 端开口（转弯口），墙只到 x=45.5
            cx = 22.75   # (0 + 45.5)/2
            walls.append(f'<geom type="box" size="22.75 0.175 2.0" pos="{cx} {y} {HF_SURF+1.0}" '
                         f'rgba="0.5 0.5 0.55 1" contype="0" conaffinity="0"/>')
        else:
            # 偶数分界：x=0 端开口（转弯口），墙从 x=4.5 开始
            cx = 27.25   # (4.5 + 50)/2
            walls.append(f'<geom type="box" size="22.75 0.175 2.0" pos="{cx} {y} {HF_SURF+1.0}" '
                         f'rgba="0.5 0.5 0.55 1" contype="0" conaffinity="0"/>')
    return "\n".join(walls)

def landmark_xml():
    """生成标牌 XML：立放 plane + emissive 纹理 + **背景板**。
    背景板：标牌在转弯口开口区（无闭合墙背景），ArUco 边界分离失败；
    在码后面 0.6m 加深色 box（2.6m 比标牌大 0.3m/边）提供均匀背景——实测必须。
    """
    assets = []
    world = []
    for idx, ch, side, wx, wy, wz, quat in landmark_positions():
        tex_name = f"lm{idx}"
        mat_name = f"lm_mat{idx}"
        bg_name = f"lm_bg{idx}"
        assets.append(f'<texture name="{tex_name}" type="2d" file="{LM_DIR}/aruco_{idx:02d}.png"/>')
        assets.append(f'<material name="{mat_name}" texture="{tex_name}" texrepeat="1 1" '
                      f'emission="1.0" specular="0"/>')
        # 背景板：标牌后面 0.6m（法线 -x 的标牌背景在 +x；法线 +x 的背景在 -x）
        bg_dir = 0.6 if side == 1 else -0.6
        world.append(
            f'<geom name="lm{idx}" type="plane" size="{LM_HALF} {LM_HALF} 0.01" '
            f'pos="{wx:.2f} {wy:.2f} {wz:.2f}" quat="{quat}" '
            f'material="{mat_name}" contype="0" conaffinity="0"/>'
        )
        world.append(
            f'<geom name="{bg_name}" type="box" size="0.3 {LM_HALF+0.3} {LM_HALF+0.3}" '
            f'pos="{wx+bg_dir:.2f} {wy:.2f} {wz:.2f}" rgba="0.15 0.15 0.18 1" '
            f'contype="0" conaffinity="0"/>'
        )
    return "\n".join(assets), "\n".join(world)

if __name__ == "__main__":
    a, w = landmark_xml()
    print(f"标牌数: {len(landmark_positions())}")
    print(f"asset 行: {len(a.splitlines())}, world 行: {len(w.splitlines())}")
    print(f"墙行数: {len(wall_xml().splitlines())}")
    print(f"HF_SURF={HF_SURF}, LM_CENTER_Z={LM_CENTER_Z}, BOT_Z={BOT_Z}")
