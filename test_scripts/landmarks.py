"""地标标牌系统：30 个 ArUco 二维码标牌（DICT_7X7，唯一 ID）。

主人设计（2026-08-06）：
- 标准黑白二维码（DICT_7X7，四角定位标志明显，机器狗能看更复杂的码）
- 每个标牌唯一 ID 0-29：idx = ch*3+slot（ch=通道 0-9, slot=开头/中间/结尾）
- 数字（id 两位数）给人看，码给机器狗识别
- 标牌 = 地面平铺 plane + emissive 纹理（MuJoCo 已验证）

世界模型（2026-08-06 关键发现）：
- hfield `size="25 25 4 2"` → 表面高度 = 2 + 4*(灰度/255)。通道(128)≈4.008m，墙(191)≈4.996m
- **机器狗必须站在 hfield 表面上方**（body z = HF_SURF+0.5），否则相机在地底下全黑
- **标牌必须贴 hfield 表面**（z = HF_SURF+0.02），地面 z=0.02 会被 hfield 埋住
- 相机朝下 14°（euler=(0,-1.326,0)）看前方 2m 地面标牌
- 材质用 emissive（自发光），hfield 地形暗，非 emissive 标牌对比度不足
- **图像金字塔多尺度检测**：码偏大时 scale=0.5 识别（码太大 ArUco 检测器失败）

用法（在 algo3_headless.py 主循环中）：
    vis = VisionLandmark(m, d, renderer, cam_name="bot_cam")
    detected = vis.scan_once(step)   # 返回 [(step, idx, ch, slot)]
"""
import os

PROJ = os.path.expanduser("~/workspace/simtrack")
LM_DIR = os.path.join(PROJ, "assets/landmarks")

# ArUco 字典
ARUCO_DICT = "DICT_7X7_1000"

# hfield 表面高度（通道 128 灰度）：关键世界修正
HF_SURF = 4.008

# 标牌几何：0.75m × 0.75m plane（size 是半尺寸 0.375），贴 hfield 表面
# （1.5m 标牌在狗 2m 距离看太大占满视野，ArUco 检测失败；0.75m 配合金字塔 scale=0.5 稳定识别）
LM_HALF = 0.375
LM_Z = HF_SURF + 0.02

# 机器狗 body 高度（站 hfield 表面上方 0.5m）
BOT_Z = HF_SURF + 0.5

def landmark_positions():
    """返回 30 个标牌的 (idx, ch, slot, wx, wy, wz) —— 贴 hfield 表面。
    布局：10 通道（y=2.5+5k），每通道 3 个（开头 x=8 / 中间 x=25 / 结尾 x=42）。
    """
    out = []
    for ch in range(10):
        y_center = 2.5 + ch * 5.0
        slots = [8.0, 25.0, 42.0]
        for slot, x in enumerate(slots):
            idx = ch * 3 + slot
            out.append((idx, ch, slot, x, y_center, LM_Z))
    return out

def landmark_xml():
    """生成标牌 XML：贴 hfield 表面 plane + emissive 纹理（30 个，256px ArUco）"""
    assets = []
    world = []
    for idx, ch, slot, wx, wy, wz in landmark_positions():
        tex_name = f"lm{idx}"
        mat_name = f"lm_mat{idx}"
        assets.append(f'<texture name="{tex_name}" type="2d" file="{LM_DIR}/aruco_{idx:02d}.png"/>')
        assets.append(f'<material name="{mat_name}" texture="{tex_name}" texrepeat="1 1" '
                      f'emission="1.0" specular="0"/>')
        world.append(
            f'<geom name="lm{idx}" type="plane" size="{LM_HALF} {LM_HALF} 0.01" '
            f'pos="{wx:.2f} {wy:.2f} {wz:.2f}" material="{mat_name}" '
            f'contype="0" conaffinity="0"/>'
        )
    return "\n".join(assets), "\n".join(world)

if __name__ == "__main__":
    a, w = landmark_xml()
    print(f"标牌数: {len(landmark_positions())}")
    print(f"asset 行: {len(a.splitlines())}, world 行: {len(w.splitlines())}")
    print(f"HF_SURF={HF_SURF}, LM_Z={LM_Z}, BOT_Z={BOT_Z}")
