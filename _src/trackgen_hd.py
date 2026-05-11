#!/usr/bin/env python3
"""
TrackGen v2: 50×50m 蛇形赛道hfield生成器
==========================================
规格:
  地图: 50×50m, hfield 500×500
  赛道: 蛇形3m宽, 总长约260m (5段+R5m弯头)
  护栏: 1m高连续护栏沿赛道两侧
  障碍: 直径1m×高2m圆柱, 每3-8m随机一个, 路宽内0.5-1.0m随机偏移
  起终点5m内无障碍
  输出: hfield PNG + MuJoCo XML + 预览PNG
"""

import os
os.environ["MUJOCO_GL"] = "egl"
import math, numpy as np, cv2, sys, random

# ═══ 规格常量 ═══
MAP_SZ = 50.0           # 50×50m
HF_RES = 2000            # 500×500像素
ROAD_W = 5.0            # 路宽5m
ROAD_H_ABS = 0.0        # 路面绝对高度0m
GUARD_H = 3.0           # 护栏1m高
OBS_R = 0.5             # 障碍半径0.5m (直径1m)
OBS_H = 2.0             # 障碍高2m
OBS_INTERVAL_MIN = 6.0  # 6m (减少障碍密度)
OBS_INTERVAL_MAX = 15.0 # 15m
START_CLEAR = 5.0
END_CLEAR = 5.0
N_SEG = 10              # 10段蛇形
TURN_R = 5.0             # 弯头半径5m
START_END_LEN = 5.0      # 起终点直道各5m

# hfield映射
HEIGHT_SCALE = 6.0
NEGATIVE = 3.0
ROAD_PIX = 128                     # 0m
GUARD_PIX = 255                    # 3m
OBS_PIX = 212                     # 2m (hfield baked, runtime overrides)

# 输出
OUT_DIR = "/tmp"
PNG_PATH = f"{OUT_DIR}/track_hd.png"
XML_PATH = f"{OUT_DIR}/track_hd.xml"
PREVIEW_PATH = f"{OUT_DIR}/track_hd_preview.png"

print(f"ROAD_PIX={ROAD_PIX} GUARD_PIX={GUARD_PIX} OBS_PIX={OBS_PIX}")

def gen_snake_center():
    """
    蛇形赛道: 10段水平直道 + R=5m U型弯头
    直线x:5→45(40m), 弯头圆心x=5/L x=45/R
    偶数段左→右, 奇数段右→左
    起点段嵌入赛道顶部内侧(y_start+2.5m)
    返回: (中心线点列表, 总长, waypoints列表)
    """
    y_start, y_end = 45.0, 5.0
    dy = (y_start - y_end) / (N_SEG - 1)
    pts = []
    waypoints = []

    # 起点直道 (嵌入内侧, 避免贴边)
    y0 = y_start - dy / 2  # 段0上方半间距处
    waypoints.append((5.0, y0))
    waypoints.append((45.0, y0))
    for x in np.arange(5.0, 45.01, 0.25):
        pts.append((float(x), float(y0)))

    for i in range(N_SEG):
        y = y_start - i * dy
        left_to_right = (i % 2 == 0)
        if left_to_right:
            waypoints.append((5.0, float(y)))   # 段起点
            waypoints.append((45.0, float(y)))  # 段终点
        else:
            waypoints.append((45.0, float(y)))  # 段起点
            waypoints.append((5.0, float(y)))   # 段终点
        xs = np.arange(5.0, 45.01, 0.25) if left_to_right else np.arange(45.0, 4.99, -0.25)
        for x in xs:
            pts.append((float(x), float(y)))

        # 段间U型弯头 (半圆弧, R=TURN_R)
        if i < N_SEG - 1:
            ny = y_start - (i + 1) * dy
            cx = 45.0 if left_to_right else 5.0
            cy = (y + ny) / 2.0
            sa = math.pi/2 if left_to_right else 3*math.pi/2
            ea = 3*math.pi/2 if left_to_right else 5*math.pi/2
            n_arc = max(10, int(math.pi * TURN_R / 0.25))
            for j in range(1, n_arc + 1):
                a = sa + (ea - sa) * j / n_arc
                pts.append((cx + TURN_R * math.cos(a), cy + TURN_R * math.sin(a)))

    # 终点直道 (嵌入内侧)
    last_left_to_right = ((N_SEG - 1) % 2 == 0)
    ye = y_end + dy / 2  # 最后一段下方半间距处
    waypoints.append((5.0 if last_left_to_right else 45.0, float(y_end)))  # 终点段起点
    waypoints.append((45.0 if last_left_to_right else 5.0, float(ye)))     # 终点段终点
    for x in np.arange(5.0, 45.01, 0.25):
        pts.append((float(x), float(ye)))

    # 计算总长
    total = 0.0
    for i in range(1, len(pts)):
        total += math.hypot(pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1])
    print(f"  中心线点数: {len(pts)}, 赛道总长: {total:.1f}m, waypoints: {len(waypoints)}个")

    return pts, total, waypoints


def w2p(wx, wy):
    """世界坐标→像素"""
    px = int(wx / MAP_SZ * HF_RES)
    py = int(wy / MAP_SZ * HF_RES)
    return min(max(px, 0), HF_RES - 1), min(max(py, 0), HF_RES - 1)


def build_hfield(center_line):
    """构建hfield数组 + 障碍物位置列表 (全地图铺平, 无深沟)"""
    road_half_px = int(ROAD_W / MAP_SZ * HF_RES / 2)
    obs_r_px = int(OBS_R / MAP_SZ * HF_RES) + 1

    # 全地图初始化为路面高度 (无深沟!)
    hf = np.full((HF_RES, HF_RES), ROAD_PIX, dtype=np.uint8)

    # === 护栏: 沿赛道中心线两侧画连续护栏 ===
    # v2.1: 法线防翻转 + 1px宽护栏
    n_guard = 0
    prev_nx, prev_ny = 0.0, 0.0
    for i in range(len(center_line)):
        cx, cy = center_line[i]
        # 计算法线方向
        if i < len(center_line) - 1:
            tx = center_line[i+1][0] - cx
            ty = center_line[i+1][1] - cy
        elif i > 0:
            tx = cx - center_line[i-1][0]
            ty = cy - center_line[i-1][1]
        else:
            continue
        tlen = math.hypot(tx, ty)
        if tlen < 0.001:
            continue
        nx_dir = -ty / tlen
        ny_dir = tx / tlen
        # 防翻转 + 几何锚定: 弯头区强制法线指向赛道外侧
        if prev_nx * nx_dir + prev_ny * ny_dir < 0:
            nx_dir, ny_dir = -nx_dir, -ny_dir
        # 左弯(cx<15)法线指向左, 右弯(cx>35)法线指向右
        if cx < 15.0 and nx_dir > 0:
            nx_dir, ny_dir = -nx_dir, -ny_dir
        elif cx > 35.0 and nx_dir < 0:
            nx_dir, ny_dir = -nx_dir, -ny_dir
        prev_nx, prev_ny = nx_dir, ny_dir
        # 两侧护栏 (1像素宽, 无3×3 brush)
        for side in [-1, 1]:
            gx = cx + side * (ROAD_W / 2) * nx_dir
            gy = cy + side * (ROAD_W / 2) * ny_dir
            gpx, gpy = w2p(gx, gy)
            if 0 <= gpx < HF_RES and 0 <= gpy < HF_RES:
                if hf[gpy, gpx] < GUARD_PIX:
                    hf[gpy, gpx] = GUARD_PIX
                    n_guard += 1

    print(f"  护栏像素: {n_guard}")

    # === 两端弧形护栏 (半圆帽) ===
    # 起点: 左半圆连接顶部→底部
    sx, sy = center_line[0]
    for a in np.linspace(np.pi/2, -np.pi/2, 80):
        gx = sx + (ROAD_W/2) * np.cos(a)
        gy = sy + (ROAD_W/2) * np.sin(a)
        gpx, gpy = w2p(gx, gy)
        if 0 <= gpx < HF_RES and 0 <= gpy < HF_RES:
            if hf[gpy, gpx] < GUARD_PIX:
                hf[gpy, gpx] = GUARD_PIX
                n_guard += 1

    # 终点: 右半圆连接底部→顶部
    ex, ey = center_line[-1]
    for a in np.linspace(-np.pi/2, np.pi/2, 80):
        gx = ex + (ROAD_W/2) * np.cos(a)
        gy = ey + (ROAD_W/2) * np.sin(a)
        gpx, gpy = w2p(gx, gy)
        if 0 <= gpx < HF_RES and 0 <= gpy < HF_RES:
            if hf[gpy, gpx] < GUARD_PIX:
                hf[gpy, gpx] = GUARD_PIX
                n_guard += 1

    print(f"  护栏+弧形帽: {n_guard}")

    # === 计算累计距离 ===
    cum_dists = [0.0]
    for i in range(1, len(center_line)):
        cum_dists.append(cum_dists[-1] + math.hypot(
            center_line[i][0] - center_line[i-1][0],
            center_line[i][1] - center_line[i-1][1]))
    total_len = cum_dists[-1]

    # 障碍物由运行时脚本生成 (cyl_independent.py)，不在hfield中烘焙
    obs_list = []
    print(f"  障碍物: 0 个 (由运行时生成)")

    return hf, n_guard, len(obs_list), total_len, obs_list


def build_xml(png_path):
    """生成MuJoCo XML"""
    return f"""<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.008"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset>
    <hfield name="track" size="{MAP_SZ/2} {MAP_SZ/2} {HEIGHT_SCALE} {NEGATIVE}" file="{png_path}"/>
    <material name="ground_mat" rgba="0.25 0.30 0.35 1.0"/>
  </asset>
  <worldbody>
    <light pos="25 25 80" dir="0 0 -1" diffuse="1.5 1.5 1.5" specular="0.5 0.5 0.5"/>
    <geom type="hfield" hfield="track" pos="25 25 {ROAD_H_ABS}" material="ground_mat"/>
  </worldbody>
</mujoco>"""


def render_preview(hf_img):
    """生成彩色预览PNG"""
    h, w = hf_img.shape
    preview = np.zeros((h, w, 3), dtype=np.uint8)

    # 路面(128) → 浅灰
    preview[hf_img == ROAD_PIX] = [180, 180, 185]
    # 护栏(191) → 红色
    preview[hf_img == GUARD_PIX] = [220, 40, 40]
    # 障碍(255) → 黄色
    preview[hf_img == OBS_PIX] = [240, 220, 30]

    cv2.imwrite(PREVIEW_PATH, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    return preview


def main():
    print("=" * 60)
    print("TrackGen v2: 50x50m 蛇形赛道 hfield 生成器")
    print(f"路宽={ROAD_W}m 护栏={GUARD_H}m 障碍=Ø1m×2m 间距{OBS_INTERVAL_MIN}-{OBS_INTERVAL_MAX}m (v2.1)")
    print("=" * 60)

    # [1] 中心线
    print("\n[1/3] 生成蛇形中心线...")
    center_line, total_len, waypoints = gen_snake_center()

    # [2] hfield
    print("\n[2/3] 构建hfield...")
    hf_img, n_guard, n_obs, total_len2, obs_list = build_hfield(center_line)
    cv2.imwrite(PNG_PATH, hf_img)

    unique = np.unique(hf_img)
    print(f"  hfield唯一值: {unique.tolist()}")
    print(f"  PNG: {PNG_PATH}")

    # waypoints json
    import json
    wp_json_path = f"{OUT_DIR}/track_hd_waypoints.json"
    with open(wp_json_path, "w") as f:
        json.dump([{"x": w[0], "y": w[1]} for w in waypoints], f, indent=2)
    print(f"  Waypoints: {wp_json_path} ({len(waypoints)}个)")

    # [3] XML + 预览
    print("\n[3/3] 生成XML + 预览图...")
    xml = build_xml(PNG_PATH)
    with open(XML_PATH, "w") as f:
        f.write(xml)

    preview = render_preview(hf_img)

    print(f"  XML: {XML_PATH}")
    print(f"  预览: {PREVIEW_PATH}")
    print(f"\n{'=' * 60}")
    print(f"TrackGen v2 完成!")
    print(f"  地图: {MAP_SZ}×{MAP_SZ}m, hfield {HF_RES}×{HF_RES}")
    print(f"  赛道: 路宽{ROAD_W}m, 总长{total_len:.1f}m, {N_SEG}段蛇形")
    print(f"  护栏: {GUARD_H}m高, {n_guard}边缘像素")
    print(f"  障碍: {n_obs}个, 直径1m×高2m, 间距{OBS_INTERVAL_MIN}-{OBS_INTERVAL_MAX}m")
    print(f"  映射: pixel/255 × {HEIGHT_SCALE} - {NEGATIVE}")
    print(f"    road={ROAD_PIX} → 0m | guard={GUARD_PIX} → 1m | obs={OBS_PIX} → 2m")
    print("=" * 60)


if __name__ == "__main__":
    main()
