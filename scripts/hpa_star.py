#!/usr/bin/env python3
"""HPA* (Hierarchical Pathfinding A*) — 标准分层 A* 实现

成熟算法（游戏业/ROS 通用）：粗层 cell 图 + 门网络，细层短 A*。
- 构建一次（加载地图时 ~2.5s）：距离场 + 门 + cell 内配对
- 规划每次毫秒级（目标 <100ms，满足主人 10fps 铁律）
- 长距离规划消失：永远只跑相邻门间的短 A*

用法：
    hpa = HPAStar(wall_fn)   # wall_fn(vx,vy)->bool，构建图
    path = hpa.plan(sx, sy, gx, gy)   # 返回 [(x,y),...] 世界格坐标
"""
import time, math, heapq
import numpy as np

VOXEL = 0.1
CELL_M = 5.0
CELL = int(CELL_M / VOXEL)   # 50 格 = 5m
ROBOT_DIA = 4                # 0.4m 机器人直径（格）
INF = 10**6

class HPAStar:
    def __init__(self, wall_fn, map_size=500, cell_m=CELL_M, verbose=False, wall_mask=None):
        """wall_fn(vx,vy)->bool：某格是否墙（运行中可用 live 地图）
        wall_mask: 可选 numpy bool 数组 (S,S)，提供则距离场构建向量化（快 100 倍）"""
        self.wall = wall_fn
        self.S = map_size
        self.cell = int(cell_m / VOXEL)
        self.N = map_size // self.cell
        self.verbose = verbose
        t0 = time.time()
        self._build_dist_field(wall_mask)
        self._build_gates_and_edges()
        if verbose:
            print(f"[HPA] 构建完成 {time.time()-t0:.1f}s: 门={len(self.id_pos)} 配对={sum(len(v) for v in self.cell_edges.values())}")

    # ── 距离场（O(1) 查墙距，替代 wall_dist 441 次扫描） ──
    def _build_dist_field(self, wall_mask=None):
        if wall_mask is None:
            wall_mask = np.zeros((self.S, self.S), dtype=bool)
            for vy in range(self.S):
                for vx in range(self.S):
                    if self.wall(vx, vy):
                        wall_mask[vy, vx] = True
        d = np.where(wall_mask, 0, INF).astype(np.int32)
        for i in range(1, self.S):
            di, d1 = d[i], d[i-1]
            for j in range(1, self.S):
                # 必须取 min(v,n,w) 三值最小！原代码 n 优先会忽略更小 w（墙邻格被误判开阔）
                v = di[j]
                n = d1[j] + 1
                w = di[j-1] + 1
                di[j] = n if n < v else (w if w < v else v)
                if w < n and w < v:
                    di[j] = w
        for i in range(self.S-2, -1, -1):
            di, d1 = d[i], d[i+1]
            for j in range(self.S-2, -1, -1):
                v = di[j]
                n = d1[j] + 1
                w = di[j+1] + 1
                di[j] = n if n < v else (w if w < v else v)
                if w < n and w < v:
                    di[j] = w
        self.dist = d

    def _cell_bounds(self, cx, cy):
        return cx*self.cell, (cx+1)*self.cell, cy*self.cell, (cy+1)*self.cell

    def _boundary_gates(self, cx, cy, side):
        """cell 一条边界的门（连续 FREE 段中点，宽度≥机器人直径）"""
        x0, x1, y0, y1 = self._cell_bounds(cx, cy)
        if side == 'E':   coords = [(x1, y) for y in range(y0, y1)]
        elif side == 'W': coords = [(x0, y) for y in range(y0, y1)]
        elif side == 'N': coords = [(x, y1) for x in range(x0, x1)]
        else:             coords = [(x, y0) for x in range(x0, x1)]
        free = []
        for gx, gy in coords:
            ok = True
            for dx in (-2,-1,0,1,2):
                for dy in (-2,-1,0,1,2):
                    if self.wall(gx+dx, gy+dy):
                        ok = False; break
                if not ok: break
            if ok: free.append((gx, gy))
        if not free: return []
        segs = [[free[0]]]
        for i in range(1, len(free)):
            prev, cur = free[i-1], free[i]
            if abs(cur[0]-prev[0]) + abs(cur[1]-prev[1]) == 1:
                segs[-1].append(cur)
            else:
                segs.append([cur])
        return [s[len(s)//2] for s in segs if len(s) >= ROBOT_DIA]

    def _build_gates_and_edges(self):
        self.gates = {}    # (cx,cy) -> [(gx,gy),...]
        self.gate_id = {}  # (cx,cy,gx,gy) -> id
        self.id_pos = {}   # id -> (cx,cy,gx,gy)
        gid = 0
        for cy in range(self.N):
            for cx in range(self.N):
                gs = []
                for side in ('E','W','N','S'):
                    gs.extend(self._boundary_gates(cx, cy, side))
                self.gates[(cx, cy)] = gs
                for g in gs:
                    self.gate_id[(cx, cy, g[0], g[1])] = gid
                    self.id_pos[gid] = (cx, cy, g[0], g[1])
                    gid += 1
        # cell 内配对：BFS 门连通性
        self.cell_edges = {}
        for cy in range(self.N):
            for cx in range(self.N):
                gs = self.gates[(cx, cy)]
                if len(gs) < 2: continue
                ids = [self.gate_id[(cx, cy, g[0], g[1])] for g in gs]
                x0, x1, y0, y1 = self._cell_bounds(cx, cy)
                for gi, g in enumerate(gs):
                    q = [(g[0], g[1])]
                    seen = {(g[0], g[1])}
                    reach = set()
                    while q:
                        vx, vy = q.pop(0)
                        if (vx, vy) != (g[0], g[1]):
                            for gi2, g2 in enumerate(gs):
                                if g2 == (vx, vy): reach.add(ids[gi2])
                        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                            nx, ny = vx+dx, vy+dy
                            if not (x0 <= nx <= x1 and y0 <= ny <= y1): continue
                            if self.wall(nx, ny) or (nx, ny) in seen: continue
                            seen.add((nx, ny)); q.append((nx, ny))
                    for rid in reach:
                        if rid != ids[gi]:
                            self.cell_edges.setdefault((cx, cy), set()).add(tuple(sorted((ids[gi], rid))))

    def _neighbors(self, cid):
        cx, cy, gx0, gy0 = self.id_pos[cid]
        nbrs = []
        for pair in self.cell_edges.get((cx, cy), set()):
            if cid in pair:
                nbrs.append(pair[0] if pair[1] == cid else pair[1])
        for ddx, ddy in ((1,0),(-1,0),(0,1),(0,-1)):
            ncx, ncy = cx+ddx, cy+ddy
            if not (0 <= ncx < self.N and 0 <= ncy < self.N): continue
            for g in self.gates.get((ncx, ncy), []):
                nid = self.gate_id.get((ncx, ncy, g[0], g[1]))
                if nid is not None and g[0] == gx0 and g[1] == gy0:
                    nbrs.append(nid)
        return nbrs

    def _astar(self, sx, sy, gx, gy, max_expand=30000, voronoi=True):
        """细层 A*：短距离（相邻门间）。路径必须离墙 ≥ROBOT_DIA(0.2m)——用距离场判定
        起点/终点允许贴墙（机器人实际位置可能在墙边），只约束扩展路径"""
        if not (0 <= sx < self.S and 0 <= sy < self.S): return None
        if not (0 <= gx < self.S and 0 <= gy < self.S): return None
        if self.dist[sy, sx] < 1 or self.dist[gy, gx] < 1: return None
        open_set = [(0.0, sx, sy)]
        came = {}; gs = {(sx, sy): 0.0}
        while open_set and len(gs) < max_expand:
            f, cx, cy = heapq.heappop(open_set)
            if (cx, cy) == (gx, gy):
                path = []
                while (cx, cy) in came:
                    path.append((cx, cy)); cx, cy = came[(cx, cy)]
                path.append((sx, sy)); path.reverse()
                return path
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx, ny = cx+dx, cy+dy
                if not (0 <= nx < self.S and 0 <= ny < self.S): continue
                # 实时障碍检查：wall_fn（含 live grid 障碍，SLAM/bounce 写回后可见）
                # —— 障碍不在 static 距离场里，必须查 wall_fn 才能绕过！
                if self.wall(nx, ny): continue
                # 膨胀约束：距墙 <ROBOT_DIA 格禁行（机器人半径 0.2m，ROS costmap inflation）
                if self.dist[ny, nx] < ROBOT_DIA: continue
                ng = gs.get((cx, cy), 9999) + 1.0
                if voronoi:
                    wd = self.dist[ny, nx]
                    ng += 12.0 / (max(1, wd) * max(1, wd))  # C=12：走中间代价增强
                    if wd < 2: ng += 30.0
                if ng < gs.get((nx, ny), 9999):
                    gs[(nx, ny)] = ng
                    came[(nx, ny)] = (cx, cy)
                    heapq.heappush(open_set, (ng + math.hypot(gx-nx, gy-ny), nx, ny))
        return None

    def plan(self, sx, sy, gx, gy, max_expand=30000, voronoi=True):
        """HPA* 主入口：返回完整格路径 [(vx,vy),...] 或 None"""
        scx, scy = sx // self.cell, sy // self.cell
        gcx, gcy = gx // self.cell, gy // self.cell
        if (scx, scy) == (gcx, gcy):
            return self._astar(sx, sy, gx, gy, max_expand, voronoi)
        def nearest_gate(cx, cy, px, py):
            gs = self.gates.get((cx, cy), [])
            if not gs: return None
            return min(gs, key=lambda g: math.hypot(g[0]-px, g[1]-py))
        start_g = nearest_gate(scx, scy, sx, sy)
        goal_g = nearest_gate(gcx, gcy, gx, gy)
        if start_g is None or goal_g is None: return None
        sid = self.gate_id[(scx, scy, start_g[0], start_g[1])]
        gid = self.gate_id[(gcx, gcy, goal_g[0], goal_g[1])]
        if sid == gid:
            return self._astar(sx, sy, gx, gy, max_expand, voronoi)
        # 粗层：门网络 A*
        open_set = [(0.0, sid)]
        came = {}; gs = {sid: 0.0}
        found = None
        while open_set:
            f, cid = heapq.heappop(open_set)
            if cid == gid:
                found = []
                while cid in came:
                    found.append(cid); cid = came[cid]
                found.append(sid); found.reverse()
                break
            for nid in self._neighbors(cid):
                ng = gs.get(cid, 9999) + 1.0
                if ng < gs.get(nid, 9999):
                    gs[nid] = ng
                    came[nid] = cid
                    heapq.heappush(open_set, (ng + 1.0, nid))
        if found is None: return None
        # 细层：起→门1→门2→...→终
        full = []
        cur = (sx, sy)
        gate_seq = [start_g] + [self.id_pos[i][2:] for i in found[1:]] + [goal_g]
        uniq = []
        for g in gate_seq:
            if not uniq or g != uniq[-1]: uniq.append(g)
        for g in uniq:
            seg = self._astar(cur[0], cur[1], g[0], g[1], max_expand, voronoi)
            if seg is None: return None
            full.extend(seg[:-1])
            cur = g
        seg = self._astar(cur[0], cur[1], gx, gy, max_expand, voronoi)
        if seg is None: return None
        full.extend(seg)
        return full

if __name__ == "__main__":
    # 自测
    data = np.load('/home/qin/workspace/simtrack/scans/full_map.npz')
    grid = data['grid']
    wm = (grid == 2)
    def wf(vx, vy):
        if not (0 <= vx < 500 and 0 <= vy < 500): return True
        return wm[vy, vx]
    h = HPAStar(wf, verbose=True, wall_mask=wm)
    t0 = time.time()
    p = h.plan(25, 25, 25, 475)
    print(f"HPA* 全程: {(time.time()-t0)*1000:.1f}ms path={len(p) if p else 0}格")
    if p:
        times = []
        for i in range(0, len(p), 100):
            sx, sy = p[i]
            t0 = time.time()
            p2 = h.plan(sx, sy, 25, 475)
            times.append((time.time()-t0)*1000)
        print(f"增量重规划: 平均 {np.mean(times):.1f}ms 最大 {np.max(times):.1f}ms")
