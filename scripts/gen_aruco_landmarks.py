"""生成 30 个 ArUco 标牌纹理：DICT_7X7 唯一码（机器狗识别）+ 大数字（人看）。

主人设计（2026-08-06）：
- 标准黑白二维码（DICT_7X7，四角定位标志最明显）
- 每个标牌唯一 ID（0-29），机器狗看码识 ID，人看数字
- 纹理用于地面平铺 plane（MuJoCo 已验证：plane 纹理渲染 + ArUco 识别可靠）

生成：assets/landmarks/aruco_{idx:02d}.png（纯码）+ 组合大图（码+数字）
"""
import os, cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.expanduser("~/workspace/simtrack/assets/landmarks")
os.makedirs(OUT, exist_ok=True)

DICT_ID = cv2.aruco.DICT_7X7_1000
ARUCO_PX = 500       # 码的像素
TEXT_PX = 150        # 数字区像素
PAD = 10             # 边距

def gen_all():
    d = cv2.aruco.getPredefinedDictionary(DICT_ID)
    for idx in range(30):
        # 1. 纯 ArUco 码
        aruco = cv2.aruco.generateImageMarker(d, idx, ARUCO_PX)
        aruco_img = Image.fromarray(aruco).convert("RGB")

        # 2. 组合：码（上）+ 数字（下）
        W = ARUCO_PX + 2 * PAD
        H = ARUCO_PX + TEXT_PX + 3 * PAD
        canvas = Image.new("RGB", (W, H), "white")
        canvas.paste(aruco_img, (PAD, PAD))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", TEXT_PX - 30)
        except Exception:
            font = ImageFont.load_default()
        text = f"{idx:02d}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((W - tw) // 2 - bbox[0], ARUCO_PX + 2 * PAD - bbox[1]), text, fill=(0, 0, 0), font=font)

        # 3. 保存：纯码（纹理用）+ 组合（人看预览）
        aruco_img.save(os.path.join(OUT, f"aruco_{idx:02d}.png"))
        canvas.save(os.path.join(OUT, f"aruco_{idx:02d}_full.png"))
    print(f"生成 30 个标牌 → {OUT}")
    print(f"字典: DICT_7X7_1000, 码 {ARUCO_PX}px, 组合图 {W}x{H}")

if __name__ == "__main__":
    gen_all()
