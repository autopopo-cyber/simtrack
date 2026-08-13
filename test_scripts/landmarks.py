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

# 中间锚点二维码间距（米，0=关）：>0 时每通道沿中心线按此间距加浮空二维码(idx 10-29)，
# 作为长直道的密集绝对锚点（破纯墙沿走廊方向漂移不可观测）。contype=0 不挡路、lidar 不检测。
# algo3 由 --qr-spacing 设置。真实环境等效物=自然视觉特征/SLAM。
QR_SPACING = [0.0]

def landmark_positions():
    """返回标牌列表 (idx, ch, side, wx, wy, wz, quat)。
    2026-08-09 贴墙版（主人指令）：标牌贴**端头边界墙**（x≈0/50），法线沿通道，
    狗从通道另一端 40m 外正对可见；不再立在路中间（旧版 x=45.8/4.2 悬在通道中心，
    虽然 contype=0 不挡路，但碍眼且 ch9 背景板曾挡终点球视线）。
    配合方角弯道地图（scripts/gen_square_maze.py）：转弯区无斜墙遮挡，
    贴墙标牌全程视线通畅。y 保持通道中心线（2m 码不能贴 y 向分界墙——会嵌墙只显示一半）。
    2026-08-13 QR_SPACING>0：每通道加中间浮空二维码（密集锚点，降沿走廊漂移）。
    """
    out = []
    _next_idx = [10]   # 中间二维码 idx 从 10 起（aruco PNG 仅 00-29；idx10-29=中间锚点）
    for ch in range(10):
        y_center = 2.5 + ch * 5.0    # 通道中心线
        if ch == 9:
            out.append((ch, ch, 1, 49.85, y_center, LM_CENTER_Z, "0.7071 0 -0.7071 0"))
            _face_side, _x_end, _xdir = 1, 49.85, -1   # ch9 也朝 -x（同偶通道）
        elif ch % 2 == 0:
            # 偶通道（朝 +x）：终点 x=50，标牌贴右端墙 x=49.85，法线 -x
            out.append((ch, ch, 1, 49.85, y_center, LM_CENTER_Z, "0.7071 0 -0.7071 0"))
            _face_side, _x_end, _xdir = 1, 49.85, -1
        else:
            # 奇通道（朝 -x）：终点 x=0，标牌贴左端墙 x=0.15，法线 +x
            out.append((ch, ch, 0, 0.15, y_center, LM_CENTER_Z, "0.7071 0 0.7071 0"))
            _face_side, _x_end, _xdir = 0, 0.15, +1
        # 中间锚点（沿通道每隔 QR_SPACING 米加一个，idx 10-29，与端墙标牌同朝向）
        sp = QR_SPACING[0]
        if sp > 0:
            _quat = "0.7071 0 -0.7071 0" if _face_side == 1 else "0.7071 0 0.7071 0"
            k = 1
            while _next_idx[0] < 30:
                _mx = _x_end + _xdir * sp * k
                if _mx < 5.0 or _mx > 45.0:
                    break    # 离转弯口 ≥5m，不堵转弯、不嵌端墙
                out.append((_next_idx[0], ch, _face_side, _mx, y_center, LM_CENTER_Z, _quat))
                _next_idx[0] += 1
                k += 1
    return out

def wall_xml():
    """分界墙 box（contype=0 纯可视化），与方角地图（scripts/gen_square_maze.py）一致：
    分隔墙 y=5k（k=1..9）：k 奇数右端开口（墙 x∈[0,45]，x=50 转弯）；
    k 偶数左端开口（墙 x∈[5,50]，x=0 转弯）。转弯区方角 `[ ---` 形，无斜墙。
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
            # 奇数分界：x=50 端开口（转弯口），墙 x∈[0,45]
            cx = 22.5   # (0 + 45)/2
            walls.append(f'<geom type="box" size="22.5 0.175 2.0" pos="{cx} {y} {HF_SURF+1.0}" '
                         f'rgba="0.5 0.5 0.55 1" contype="0" conaffinity="0"/>')
        else:
            # 偶数分界：x=0 端开口（转弯口），墙 x∈[5,50]
            cx = 27.5   # (5 + 50)/2
            walls.append(f'<geom type="box" size="22.5 0.175 2.0" pos="{cx} {y} {HF_SURF+1.0}" '
                         f'rgba="0.5 0.5 0.55 1" contype="0" conaffinity="0"/>')
    return "\n".join(walls)

def landmark_xml():
    """生成标牌 XML：立放 plane + emissive 纹理 + **背景板**。
    背景板：深色 box 提供均匀背景（ArUco 边界分离必须）。
    2026-08-09 贴墙版：标牌贴端头边界墙（x≈0/50），背景板改薄（0.06m）贴标牌身后
    与墙面齐平——旧版在路中间凸出 0.6m，贴墙后不再凸进通道。
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
        # 背景板：标牌后面 0.04m、厚 0.06m（法线 -x 的标牌背景在 +x；法线 +x 的背景在 -x）
        bg_dir = 0.04 if side == 1 else -0.04
        world.append(
            f'<geom name="lm{idx}" type="plane" size="{LM_HALF} {LM_HALF} 0.01" '
            f'pos="{wx:.2f} {wy:.2f} {wz:.2f}" quat="{quat}" '
            f'material="{mat_name}" contype="0" conaffinity="0"/>'
        )
        world.append(
            f'<geom name="{bg_name}" type="box" size="0.03 {LM_HALF+0.3} {LM_HALF+0.3}" '
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
