"""生成 30 个墙标牌纹理：ArUco 二维码（机器人识别唯一ID）+ 大数字（人看）。
每通道 3 个：开头/中间/结尾。标牌 1024×1024 PNG，白底黑码黑数字。
"""
import cv2, os, numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = "/home/qin/workspace/simtrack/assets/landmarks"
os.makedirs(OUT, exist_ok=True)

SIZE = 1024
ARUCO_DICT = cv2.aruco.DICT_4X4_50

def gen_landmark(idx, channel, slot):
    """idx: 全局唯一ID 0-29, channel: 0-9, slot: 0开头/1中间/2结尾"""
    # 1. ArUco 码（占上方 2/3）
    d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    aruco_img = cv2.aruco.generateImageMarker(d, idx, 640)
    aruco_img = cv2.cvtColor(aruco_img, cv2.COLOR_GRAY2RGB)

    # 2. 合成画布
    canvas = np.ones((SIZE, SIZE, 3), dtype=np.uint8) * 255
    # ArUco 居中偏上
    y0 = (SIZE - 640) // 2 - 40
    x0 = (SIZE - 640) // 2
    canvas[y0:y0+640, x0:x0+640] = aruco_img

    # 3. 大数字（下方 1/3，人看）
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    # 找一个大字体（ttf 优先，fallback 默认）
    font = None
    for fp in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
               "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, 220)
            break
    if font is None:
        font = ImageFont.load_default()
    text = f"{idx:02d}"
    # 居中
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    draw.text(((SIZE-tw)//2 - bbox[0], 700 - bbox[1]), text, fill=(0, 0, 0), font=font)
    # 通道号小字（角落，人看）
    small = ImageFont.truetype(font.path, 60) if hasattr(font, 'path') else font
    draw.text((30, 30), f"CH{channel} S{slot} #{idx}", fill=(60, 60, 60), font=small)

    pil.save(os.path.join(OUT, f"landmark_{idx:02d}_ch{channel}_s{slot}.png"))
    return idx, channel, slot

# 生成 30 个：channel 0-9, 每通道 slot 0/1/2
manifest = []
for ch in range(10):
    for slot in range(3):
        idx = ch * 3 + slot
        manifest.append(gen_landmark(idx, ch, slot))

# 写 manifest（坐标由算法侧计算，这里只存 ID 映射）
with open(os.path.join(OUT, "manifest.txt"), "w") as f:
    f.write("# idx channel slot\n")
    for idx, ch, slot in manifest:
        f.write(f"{idx} {ch} {slot}\n")

print(f"生成 {len(manifest)} 个标牌 → {OUT}")
print("manifest:", manifest[:5], "...")
