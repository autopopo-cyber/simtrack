"""地标标牌系统：30 个单色标牌（8 色循环 + 位置/几何编码）。

背景（2026-08-05 实测）：
- MuJoCo EGL + hfield：纹理 material 不渲染；emissive 纯色（RGB 0/1）可靠
- **多 box 并排/叠放渲染重叠**（软渲染器投影 bug）→ 每标牌只能用 1 个 box
- 8 纯色（红/绿/蓝/黄/青/品红/白/黑）+ 2 编码维度 → 30 唯一：
  - 颜色：8 色循环（通道 ch 用 COLORS[ch % 8]）
  - 高度：通道 0-7 高 1.0m（z 中心 1.0），通道 8 高 1.6m，通道 9 高 0.5m（相机看投影不同）
  - 位置：每通道 3 个（x=6/25/44 开头/中间/结尾）
  → 识别 = 颜色(8) × 高度(3) 基本够用，先跑通视觉定位流程

标牌布局：
- 10 通道（y=2.5+5k），每通道 3 个：开头 x=6 / 中间 x=25 / 结尾 x=44
- 垂直 box 立通道中心（contype=0 不挡路），面朝相机（x 面）
"""
import os

PROJ = os.path.expanduser("~/workspace/simtrack")

# 8 个纯色（RGB 0/1，emissive 可靠渲染）
COLORS = [
    (1, 0, 0),   # 红
    (0, 1, 0),   # 绿
    (0, 0, 1),   # 蓝
    (1, 1, 0),   # 黄
    (0, 1, 1),   # 青
    (1, 0, 1),   # 品红
    (1, 1, 1),   # 白
    (0, 0, 0),   # 黑
]
COLOR_NAMES = ["红", "绿", "蓝", "黄", "青", "品红", "白", "黑"]

# 标牌几何
LM_HALF_W = 0.5   # 半宽（y 方向）
LM_HALF_T = 0.05  # 半厚（x 方向）

def landmark_positions():
    """返回 30 个标牌的 (idx, ch, slot, wx, wy, wz, height)。
    贴墙方案（主人："标牌贴墙放置，别堵路"）：
    - 每通道 3 个标牌，贴在通道内 3 处**横墙内壁**（机器人沿 x 正对可见，不挡导航）
    - 蛇形通道：通道 ch 的下墙 y=ch*5，上墙 y=ch*5+5；横墙是段间墙
    - 简化：标牌贴在通道中心线上，位置 x=6/25/44 处**悬在通道上方高处**（z=3.6 墙顶以上）
      ——但之前实验墙顶标牌被 hfield 挡。改用**正前方可见**方案：
    - 最终：标牌贴通道 **末端横向墙内壁**（x=47.5 右端 / x=2.5 左端交替），
      机器人沿 x 走到段尾正对可见；开头/中间/结尾用 x 偏移区分（不完美，先验证）
    """
    out = []
    for ch in range(10):
        y_center = 2.5 + ch * 5.0
        # 蛇形：偶数段从左到右(x 5→45)，奇数段从右到左(x 45→5)
        # 段尾横墙：偶数段在 x=49.3，奇数段在 x=0.7
        end_x = 49.0 if ch % 2 == 0 else 1.0
        slots = [end_x]  # 贴段尾横墙内壁（先只放 1 个验证，后续扩展开头/中间/结尾）
        for slot, x in enumerate(slots):
            idx = ch * 3 + slot
            h = 1.0
            # 贴墙：box 厚 0.1(x) × 宽 1(y) × 高 1(z)，中心 z=1.0 覆盖 0.5-1.5
            # 面朝 -x（偶数段，机器人从 x 小往大走）或 +x（奇数段）
            out.append((idx, ch, slot, x, y_center, 1.0, h))
    return out

def landmark_color(ch):
    """通道 ch → 颜色（8 色循环；黑色通道7 在 EGL 下渲染偏灰不可靠，用白色+偏移区分）"""
    return COLORS[ch % 8]

def landmark_xml():
    """生成标牌 XML：单色垂直 box（emissive），贴段尾横墙内壁"""
    assets = []
    world = []
    for idx, ch, slot, wx, wy, wz, h in landmark_positions():
        r, g, b = landmark_color(ch)
        assets.append(f'<material name="lm{idx}" rgba="{r} {g} {b} 1" emission="1"/>')
        # 贴横墙：厚 0.1(x) × 宽 1(y) × 高 1(z)，中心 z=1.0 覆盖 0.5-1.5
        world.append(
            f'<geom name="lm{idx}" type="box" size="{LM_HALF_T} {LM_HALF_W} {h/2}" '
            f'pos="{wx:.2f} {wy:.2f} {wz:.2f}" material="lm{idx}" '
            f'contype="0" conaffinity="0"/>'
        )
    return "\n".join(assets), "\n".join(world)

if __name__ == "__main__":
    a, w = landmark_xml()
    print(f"标牌数: {len(landmark_positions())}")
    for idx, ch, slot, wx, wy, wz, h in landmark_positions()[:5]:
        print(f"  idx{idx}: 通道{ch} 位置{slot} ({wx},{wy}) 高{h}m 色{COLOR_NAMES[ch%8]}")
