#!/usr/bin/env python3
"""
maze_gen.py — 为 MuJoCo + ROS2 SLAM/Nav2 生成干净的迷宫高度图。

坐标系约定（消除旧版 y 翻转混乱）：
  世界系：x 向右, y 向上, 原点 (0,0) = 迷宫左下角
  图像系：col = x * PX_PER_M,  row = (MAZE_H - y) * PX_PER_M
          （row 0 = 图像顶部 = 世界 y 最大处，标准图像约定）

输出（每种迷宫一组）：
  confirmed/maze_<name>.png   — 高度图（road=128, wall=255），MuJoCo hfield 引用
  confirmed/maze_<name>_annot.png — 彩色标注图（仅供人眼核对，不给 MuJoCo）

用法：
  python -m simtrack.maze_gen loop20            # 旧版绕方块回环（默认）
  python -m simtrack.maze_gen rooms5x5          # 5x5 房间 3m×3m（seed=42）
  python -m simtrack.maze_gen rooms10x10        # 10x10 房间 5m×5m
  python -m simtrack.maze_gen rooms10x10 7      # 指定 seed

输出还会写 maze_<name>.meta.json（start/goal/尺寸）——sim_bridge 读它决定机器狗起点，
所以不同房间尺寸的迷宫起点会自动正确（3m房→(1.5,1.5)，5m房→(2.5,2.5)）。
"""
import os
import sys
import json
import random
from collections import deque

import numpy as np
from PIL import Image, ImageDraw

# ── 全局渲染参数 ──
PX_PER_M = 50          # 分辨率：2cm/像素
WALL_T = 0.3           # 墙厚 (m)
ROAD_VAL = 128         # 地面像素值
WALL_VAL = 255         # 墙像素值


# ══════════════════════════════════════════════
# 迷宫定义：每个 gen_*() 返回 dict
#   walls: [((x1,y1),(x2,y2)), ...] 世界坐标墙段
#   w, h: 迷宫宽高 (m)
#   start: (x, y) 起点世界坐标
#   start_yaw: 起点朝向 (rad)
#   goal: (x, y) 终点世界坐标 或 None
#   info: dict 额外信息（房间图、门等，可选）
# ══════════════════════════════════════════════
def gen_loop20():
    """旧版：20×20m，绕中央方块走一圈的回环迷宫。"""
    W = H = 20.0
    return {
        "walls": [
            ((0, 0), (W, 0)), ((W, 0), (W, H)),
            ((W, H), (0, H)), ((0, H), (0, 0)),
            ((6, 6), (14, 6)), ((14, 6), (14, 14)),
            ((14, 14), (6, 14)), ((6, 14), (6, 6)),
            ((3, 3), (3, 10)), ((17, 10), (17, 17)),
        ],
        "w": W, "h": H,
        "start": (1.5, 1.5), "start_yaw": 0.0, "goal": None,
        "info": {},
    }


def gen_rooms_grid(rows=5, cols=5, room=3.0, door_w=1.5, extra_prob=0.18,
                   wall_jitter=0.0, seed=42):
    """网格房间迷宫：rows×cols 个 room×room 的房间，相邻房间间的墙上有门(宽 door_w)。

    生成保证：
      - 随机 DFS 生成树 → 所有房间全连通（每个房间至少 1 扇门）。
      - 额外按 extra_prob 给非树相邻对开门 → 形成环路/死胡同，
        所以不是所有房间都在 起点→终点 的直路上（但仍可达，因为全连通）。
      - 起点=(0,0) 左下房间中心，终点=(rows-1,cols-1) 右上房间中心，生成树保证连通。
      - wall_jitter>0：9 排横墙/9 排纵墙**每排整体**±wall_jitter 随机抖动（边界固定），
        → 每个房间长宽/位置都不同，**破掉 4 重旋转对称**，让 scan matching/回环不再
        被对称歧义卡死。抖动是"整排线"抖（不是逐段），保证网格仍连贯、墙角无缝。

    房间 (r,c) 占据四边形 (X[c],Y[r])-(X[c+1],Y[r+1])，X/Y 为（可能抖动的）网格线。
    """
    rnd = random.Random(seed)
    R, C = rows, cols

    # 网格线位置（边界固定 0 和 C*room/R*room；内部线 ±wall_jitter 随机抖动）
    X = [0.0] + [c * room + rnd.uniform(-wall_jitter, wall_jitter) for c in range(1, C)] + [C * room]
    Y = [0.0] + [r * room + rnd.uniform(-wall_jitter, wall_jitter) for r in range(1, R)] + [R * room]

    def nbrs(r, c):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C:
                yield (nr, nc)

    # ── 随机 DFS 生成树（迭代，避免深递归）──
    start_room = (0, 0)
    visited = {start_room}
    tree = set()        # frozenset({a,b})
    stack = [start_room]
    while stack:
        r, c = stack[-1]
        cand = [n for n in nbrs(r, c) if n not in visited]
        if not cand:
            stack.pop()
            continue
        nxt = rnd.choice(cand)
        tree.add(frozenset({(r, c), nxt}))
        visited.add(nxt)
        stack.append(nxt)
    assert len(visited) == R * C, "生成树未覆盖所有房间"

    # ── 额外门（环路）──
    all_adj = set()
    for r in range(R):
        for c in range(C):
            for n in nbrs(r, c):
                all_adj.add(frozenset({(r, c), n}))
    doors = set(tree)
    for e in all_adj:
        if e not in tree and rnd.random() < extra_prob:
            doors.add(e)

    # ── 房间邻接图（用于校验 + info）──
    adj = {(r, c): set() for r in range(R) for c in range(C)}
    for e in doors:
        a, b = tuple(e)
        adj[a].add(b)
        adj[b].add(a)
    door_count = {rc: len(adj[rc]) for rc in adj}

    # ── 渲染墙段：每条内部墙，有门则中间留 door_w 缺口，否则整墙（用抖动网格线 X/Y）──
    walls = _rooms_to_walls(R, C, X, Y, door_w, doors)
    W, H = C * room, R * room
    goal_room = (R - 1, C - 1)
    # 房间中心用抖动后的网格线（保证落在 free 空间、不在墙上）
    cx0, cy0 = (X[0] + X[1]) / 2.0, (Y[0] + Y[1]) / 2.0              # 房间(0,0)中心
    cgx, cgy = (X[C - 1] + X[C]) / 2.0, (Y[R - 1] + Y[R]) / 2.0      # 右上房间中心

    return {
        "walls": walls,
        "w": W, "h": H,
        "start": (cx0, cy0),
        "start_yaw": 0.0,
        "goal": (cgx, cgy),
        "info": {
            "rows": R, "cols": C, "room": room, "door_w": door_w, "wall_jitter": wall_jitter,
            "doors": doors, "adj": adj, "door_count": door_count,
            "start_room": start_room, "goal_room": goal_room,
        },
    }


def _rooms_to_walls(R, C, X, Y, door_w, doors):
    """把房间-门图转成世界坐标墙段列表（用抖动网格线 X[0..C], Y[0..R]）。

    门居中于墙段，缺口宽=door_w。每排墙整排在一条网格线上（X[c] 或 Y[r]），
    所以抖动是"整排"抖——网格仍连贯、墙角无缝、每房间形状各异（破对称）。
    """
    walls = []
    half = door_w / 2.0
    # 水平内墙在 Y[r]（隔开 row r-1 与 r），r∈[1,R-1]，每段 X[c]→X[c+1]
    for r in range(1, R):
        for c in range(C):
            y = Y[r]
            x0, x1 = X[c], X[c + 1]
            if frozenset({(r - 1, c), (r, c)}) in doors:
                cx = (x0 + x1) / 2.0
                walls.append(((x0, y), (cx - half, y)))
                walls.append(((cx + half, y), (x1, y)))
            else:
                walls.append(((x0, y), (x1, y)))
    # 竖直内墙在 X[c]（隔开 col c-1 与 c），c∈[1,C-1]，每段 Y[r]→Y[r+1]
    for c in range(1, C):
        for r in range(R):
            x = X[c]
            y0, y1 = Y[r], Y[r + 1]
            if frozenset({(r, c - 1), (r, c)}) in doors:
                cy = (y0 + y1) / 2.0
                walls.append(((x, y0), (x, cy - half)))
                walls.append(((x, cy + half), (x, y1)))
            else:
                walls.append(((x, y0), (x, y1)))
    # 外边界（整墙无门，边界网格线固定）
    walls.append(((X[0], Y[0]), (X[C], Y[0])))
    walls.append(((X[C], Y[0]), (X[C], Y[R])))
    walls.append(((X[C], Y[R]), (X[0], Y[R])))
    walls.append(((X[0], Y[R]), (X[0], Y[0])))
    return walls


# ══════════════════════════════════════════════
# 渲染
# ══════════════════════════════════════════════
def _wp(x, y, maze_h):
    """世界 → 像素 (col, row)。"""
    col = int(x * PX_PER_M)
    row = int(maze_h * PX_PER_M) - 1 - int(y * PX_PER_M)
    return col, row


def render_heightfield(maze):
    """生成高度图 numpy 数组，road=128, wall=255。"""
    W, H = maze["w"], maze["h"]
    img_w, img_h = int(W * PX_PER_M), int(H * PX_PER_M)
    img = Image.new("L", (img_w, img_h), ROAD_VAL)
    draw = ImageDraw.Draw(img)
    wt = max(1, int(WALL_T * PX_PER_M))
    for (x1, y1), (x2, y2) in maze["walls"]:
        c1, r1 = _wp(x1, y1, H)
        c2, r2 = _wp(x2, y2, H)
        if x1 == x2:      # 竖墙
            draw.rectangle([c1 - wt // 2, min(r1, r2), c1 + wt // 2, max(r1, r2)], fill=WALL_VAL)
        elif y1 == y2:    # 横墙
            draw.rectangle([min(c1, c2), r1 - wt // 2, max(c1, c2), r1 + wt // 2], fill=WALL_VAL)
        else:
            draw.line([c1, r1, c2, r2], fill=WALL_VAL, width=wt)
    return np.array(img)


def render_annotated(maze):
    """彩色标注图（人眼核对用）：墙=深灰，门缺口高亮，起点绿/终点红，房间标门数。"""
    W, H = maze["w"], maze["h"]
    img_w, img_h = int(W * PX_PER_M), int(H * PX_PER_M)
    img = Image.new("RGB", (img_w, img_h), (235, 235, 235))
    draw = ImageDraw.Draw(img)
    wt = max(2, int(WALL_T * PX_PER_M))
    for (x1, y1), (x2, y2) in maze["walls"]:
        c1, r1 = _wp(x1, y1, H)
        c2, r2 = _wp(x2, y2, H)
        if x1 == x2:
            draw.rectangle([c1 - wt // 2, min(r1, r2), c1 + wt // 2, max(r1, r2)], fill=(50, 50, 50))
        elif y1 == y2:
            draw.rectangle([min(c1, c2), r1 - wt // 2, max(c1, c2), r1 + wt // 2], fill=(50, 50, 50))
        else:
            draw.line([c1, r1, c2, r2], fill=(50, 50, 50), width=wt)
    info = maze.get("info") or {}
    if "door_count" in info:
        room = info["room"]
        for (r, c), n in info["door_count"].items():
            cx, cy = c * room + room / 2, r * room + room / 2
            pc, pr = _wp(cx, cy, H)
            tag = "S" if (r, c) == info["start_room"] else ("G" if (r, c) == info["goal_room"] else str(n))
            color = (40, 160, 40) if tag == "S" else ((200, 40, 40) if tag == "G" else (20, 20, 20))
            draw.ellipse([pc - 14, pr - 14, pc + 14, pr + 14], outline=color, width=2)
            draw.text((pc - 5, pr - 8), tag, fill=color)
    return np.array(img)


# ══════════════════════════════════════════════
# 校验：BFS 确认起点→终点连通 + 打印门数分布
# ══════════════════════════════════════════════
def verify(maze):
    info = maze.get("info") or {}
    if "adj" not in info:
        print("  (无房间图，跳过校验)")
        return
    adj = info["adj"]
    s, g = info["start_room"], info["goal_room"]
    # BFS 全图（不提前 break，否则连通性统计会少算房间）
    prev = {s: None}
    q = deque([s])
    while q:
        cur = q.popleft()
        for n in adj[cur]:
            if n not in prev:
                prev[n] = cur
                q.append(n)
    if g not in prev:
        print("  ❌ 起点{} 到不了 终点{}！".format(s, g))
        return
    path = []
    cur = g
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    # 全连通校验
    reach = set(prev) | {g}
    print("  房间数 {} | 起点{}→终点{} 路径长 {} 步: {}".format(
        len(adj), s, g, len(path) - 1, "→".join(str(p) for p in path)))
    print("  全连通: {} (可达 {} 个房间)".format(len(reach) == len(adj), len(reach)))
    dc = info["door_count"]
    dist = {}
    for n in dc.values():
        dist[n] = dist.get(n, 0) + 1
    print("  每房间门数分布(门数:房间数): {}".format(
        " ".join("{}:{}".format(k, dist[k]) for k in sorted(dist))))


def save_heightfield(arr, path, maze):
    Image.fromarray(arr).save(path)
    wall_pct = 100.0 * (arr == WALL_VAL).sum() / arr.size
    print("  高度图: {}  {}×{}px  墙{:.1f}%  {}px/m".format(
        path, arr.shape[1], arr.shape[0], wall_pct, PX_PER_M))
    print("  迷宫: {}×{}m  起点:{} 朝{}rad  终点:{}".format(
        maze["w"], maze["h"], tuple(round(v, 1) for v in maze["start"]),
        maze["start_yaw"], tuple(round(v, 1) for v in maze["goal"]) if maze["goal"] else None))


MAZES = {
    "loop20": lambda seed=42: gen_loop20(),
    "rooms5x5": lambda seed=42: gen_rooms_grid(5, 5, 3.0, 1.5, 0.08, 0.0, seed),
    "rooms10x10": lambda seed=42: gen_rooms_grid(10, 10, 5.0, 1.5, 0.08, 0.5, seed),
    # 窄门变体：同 seed=42 → 同 DFS 拓扑/同门位 → goal_runner 航点表仍有效，只缩门宽。
    # 狗足迹 0.8×0.4 胶囊：0.8m 门两侧各 0.2m 余量（舒服），0.6m 各 0.1m（极限）。
    "rooms10x10n80": lambda seed=42: gen_rooms_grid(10, 10, 5.0, 0.8, 0.08, 0.5, seed),
    "rooms10x10n60": lambda seed=42: gen_rooms_grid(10, 10, 5.0, 0.6, 0.08, 0.5, seed),
}


def save_meta(maze, path):
    """写迷宫元数据 sidecar：sim_bridge 读它决定机器狗起点（不同房间尺寸起点不同）。"""
    meta = {
        "start": [float(maze["start"][0]), float(maze["start"][1])],
        "start_yaw": float(maze["start_yaw"]),
        "goal": [float(maze["goal"][0]), float(maze["goal"][1])] if maze["goal"] else None,
        "w": float(maze["w"]), "h": float(maze["h"]),
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print("  元数据: {}  start={} goal={}".format(
        path, tuple(round(v, 1) for v in meta["start"]),
        tuple(round(v, 1) for v in meta["goal"]) if meta["goal"] else None))


def main():
    args = sys.argv[1:]
    name = args[0] if args else "loop20"
    if name not in MAZES:
        print("未知迷宫:", name, "可选:", ", ".join(sorted(MAZES)))
        sys.exit(1)
    seed = int(args[1]) if len(args) > 1 else 42
    maze = MAZES[name](seed)

    out_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "confirmed"))
    os.makedirs(out_dir, exist_ok=True)
    hf_path = os.path.join(out_dir, "maze_%s.png" % name)
    ann_path = os.path.join(out_dir, "maze_%s_annot.png" % name)
    meta_path = os.path.join(out_dir, "maze_%s.meta.json" % name)

    print("== 迷宫 %s ==" % name)
    verify(maze)
    arr = render_heightfield(maze)
    save_heightfield(arr, hf_path, maze)
    Image.fromarray(render_annotated(maze)).save(ann_path)
    print("  标注图:", ann_path)
    save_meta(maze, meta_path)


if __name__ == "__main__":
    main()
