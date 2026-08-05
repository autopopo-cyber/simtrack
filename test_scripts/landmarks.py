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
HF_SURF = 4.008

# 标牌几何：2m×2m plane（size 是半尺寸 1.0），中心离地 1m = 相机高度
LM_HALF = 1.0
LM_CENTER_Z = HF_SURF + 1.0

# 机器狗 body 高度（站 hfield 表面上方 0.5m），相机在 body 局部 +0.5 → 世界 HF_SURF+1.0
BOT_Z = HF_SURF + 0.5

# 贴墙参数：凸出横墙面 0.3m（路牌式，避免嵌进 box 墙）
WALL_X_OFF = 0.3

def landmark_positions():
    """返回 20 个标牌的 (idx, ch, side, wx, wy, wz, quat)。
    布局：10 通道（通道 ch 的 y 范围 [5ch, 5ch+5]，中心 y=5ch+2.5）。
    每通道 2 个标牌贴两端横墙：
      side 0 = 起点横墙（x=0.4，法线 +x 面向从 x=50 来的狗）quat 绕 y +90°
      side 1 = 终点横墙（x=49.6，法线 -x 面向从 x=0 来的狗）quat 绕 y -90°
    """
    out = []
    for ch in range(10):
        y_center = 2.5 + ch * 5.0
        out.append((ch*2+0, ch, 0, 0.4, y_center, LM_CENTER_Z, "0.7071 0 0.7071 0"))
        out.append((ch*2+1, ch, 1, 49.6, y_center, LM_CENTER_Z, "0.7071 0 -0.7071 0"))
    return out

def wall_xml():
    """三面 box 墙（contype=0 纯可视化）：每通道两侧纵墙 + 两端横墙。
    闭合背景是 ArUco 识别的关键（实测单面墙失败）。
    """
    walls = []
    for ch in range(10):
        y0 = ch * 5.0
        y1 = y0 + 5.0
        yc = y0 + 2.5
        # 两侧纵墙（x 全程 0~50）
        walls.append(f'<geom type="box" size="25.0 0.1 2.0" pos="25 {y0} {HF_SURF+1.0}" '
                     f'rgba="0.5 0.5 0.55 1" contype="0" conaffinity="0"/>')
        walls.append(f'<geom type="box" size="25.0 0.1 2.0" pos="25 {y1} {HF_SURF+1.0}" '
                     f'rgba="0.5 0.5 0.55 1" contype="0" conaffinity="0"/>')
        # 两端横墙（x=0 和 x=50）
        walls.append(f'<geom type="box" size="0.1 2.5 2.0" pos="0 {yc} {HF_SURF+1.0}" '
                     f'rgba="0.5 0.5 0.55 1" contype="0" conaffinity="0"/>')
        walls.append(f'<geom type="box" size="0.1 2.5 2.0" pos="50 {yc} {HF_SURF+1.0}" '
                     f'rgba="0.5 0.5 0.55 1" contype="0" conaffinity="0"/>')
    return "\n".join(walls)

def landmark_xml():
    """生成标牌 XML：贴横墙 plane + emissive 纹理（20 个，256px ArUco，2m×2m）。
    标牌贴横墙内表面，法线朝通道（绕 y 旋转——MuJoCo 唯一可靠的立放旋转）。
    """
    assets = []
    world = []
    for idx, ch, side, wx, wy, wz, quat in landmark_positions():
        tex_name = f"lm{idx}"
        mat_name = f"lm_mat{idx}"
        assets.append(f'<texture name="{tex_name}" type="2d" file="{LM_DIR}/aruco_{idx:02d}.png"/>')
        assets.append(f'<material name="{mat_name}" texture="{tex_name}" texrepeat="1 1" '
                      f'emission="1.0" specular="0"/>')
        world.append(
            f'<geom name="lm{idx}" type="plane" size="{LM_HALF} {LM_HALF} 0.01" '
            f'pos="{wx:.2f} {wy:.2f} {wz:.2f}" quat="{quat}" '
            f'material="{mat_name}" contype="0" conaffinity="0"/>'
        )
    return "\n".join(assets), "\n".join(world)

if __name__ == "__main__":
    a, w = landmark_xml()
    print(f"标牌数: {len(landmark_positions())}")
    print(f"asset 行: {len(a.splitlines())}, world 行: {len(w.splitlines())}")
    print(f"墙行数: {len(wall_xml().splitlines())}")
    print(f"HF_SURF={HF_SURF}, LM_CENTER_Z={LM_CENTER_Z}, BOT_Z={BOT_Z}")
