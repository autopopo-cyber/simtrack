"""地标标牌系统：30 个 ArUco 二维码标牌（DICT_7X7，唯一 ID）。

主人设计（2026-08-06）：
- 标准黑白二维码（DICT_7X7，四角定位标志明显，机器狗能看更复杂的码）
- 每个标牌唯一 ID 0-29：idx = ch*3+slot（ch=通道 0-9, slot=开头/中间/结尾）
- 数字（id 两位数）给人看，码给机器狗识别
- 标牌 = 地面平铺 plane + 纹理（MuJoCo 已验证：plane 纹理渲染 + ArUco 识别可靠；
  box 纹理有 UV bug 不可用）

技术要点：
- plane 纹理 UV 正常（box 不行）
- 相机取景要完整（标牌入画，否则"缺 1/4"）
- 检测用图像金字塔多尺度（主人指令：避免只拍到局部）
"""
import os

PROJ = os.path.expanduser("~/workspace/simtrack")
LM_DIR = os.path.join(PROJ, "assets/landmarks")

# ArUco 字典
ARUCO_DICT = "DICT_7X7_1000"

# 标牌几何：1.5m × 1.5m plane，悬空 z=2.0（hfield 深度会遮挡地面 plane——实测 z=0.02 不渲染，
# 抬到 2.0m 悬空才可见）。contype=0 不挡路，机器狗从下方经过。
LM_HALF = 0.75
LM_Z = 2.0

def landmark_positions():
    """返回 30 个标牌的 (idx, ch, slot, wx, wy, wz) —— 悬空 plane 位置。
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
    """生成标牌 XML：悬空 plane + 纹理（30 个，256px ArUco）"""
    assets = []
    world = []
    for idx, ch, slot, wx, wy, wz in landmark_positions():
        tex_name = f"lm{idx}"
        mat_name = f"lm_mat{idx}"
        assets.append(f'<texture name="{tex_name}" type="2d" file="{LM_DIR}/aruco_{idx:02d}.png"/>')
        assets.append(f'<material name="{mat_name}" texture="{tex_name}" texrepeat="1 1"/>')
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
    print(w.splitlines()[0][:120])
