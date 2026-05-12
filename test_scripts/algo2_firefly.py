#!/usr/bin/env python3
"""萤火算法 Firefly v17 — 127中心线导航点 + 动态选点 + 跳过不可达

体素:
  UNKNOWN=0 — 未扫描(目标背后的新区)
  FREE=1    — 扫描过但没走过(门! A*的目标)
  WALL=2    — 墙/障碍
  VISITED=3 — 实际到达过(OLD, 只当通路)

导航: 127个中心线点全做路标。机器人找最近可达点, 朝终点走。堵了跳过。
"""
import sys, os, math, time, random, heapq
from collections import deque
import numpy as np
from PIL import Image
import mujoco, mujoco.viewer

MAP = os.path.expanduser("~/workspace/simtrack/confirmed/track_clean.png")
hf = np.array(Image.open(MAP))

SCALE = 2.0; HF_RES = 2000; PIX_PER_M = 40; ROAD_PIX = 128
SAFE_R = 1.0; SPEED = 5.0; SPEED_MAX = 8.0; YAW_RATE = 6.0
CP_RADIUS = 2.0; LIDAR_RANGE = 15.0
VOXEL = 0.5; W = 200

UNKNOWN, FREE, WALL, VISITED = 0, 1, 2, 3
vox = np.zeros((W, W), dtype=np.int8)

def gen_centerline():
    pts = []; y0 = 2.5
    for seg in range(10):
        y = y0+seg*5.0; x0, x1 = (5.0,45.0) if seg%2==0 else (45.0,5.0)
        for j in range(10): pts.append((x0+(j/9.0)*(x1-x0), y))
    for mx, my in [(46.5,3.75),(47.5,5.0),(46.5,6.25)]:
        for gy in range(5): pts.append((mx, my+gy*10.0))
    for mx, my in [(3.5,8.75),(2.5,10.0),(3.5,11.25)]:
        for gy in range(4): pts.append((mx, my+gy*10.0))
    return pts

rng = random.Random(); cl = gen_centerline(); obs_world = []; idx = 0
while idx < len(cl):
    cx, cy = cl[idx]; wx, wy = cx*SCALE, cy*SCALE
    obs_world.append((wx, wy+rng.uniform(-2.0,2.0))); idx += rng.randint(3,8)
obs_world = [(x,y) for x,y in obs_world if math.hypot(x-10,y-5)>5.0]
OBS_R = 1.0; OBS_CLEAR = OBS_R+SAFE_R

# 127个中心线点 = 全部导航点 (世界坐标)
nav_wps = [(x*SCALE, y*SCALE) for x,y in cl]
FINISH = nav_wps[-1]  # 终点

def sample_hf(wx, wy):
    mx, my = wx/SCALE, wy/SCALE
    px, py = int(mx*PIX_PER_M), HF_RES-1-int(my*PIX_PER_M)
    return int(hf[py,px]) if 0<=px<HF_RES and 0<=py<HF_RES else -1

is_wall = lambda wx, wy: sample_hf(wx, wy) != ROAD_PIX
is_obs = lambda wx, wy: any(math.hypot(wx-ox, wy-oy) < OBS_CLEAR for ox, oy in obs_world)
blocked = lambda wx, wy: is_wall(wx, wy) or is_obs(wx, wy)
walkable = lambda vx, vy: 0<=vx<W and 0<=vy<W and vox[vy,vx] in (FREE, VISITED)

def get_closest_wp(bx, by):
    """返回离机器人最近的导航点索引"""
    best_i, best_d = 0, 99999
    for i, (wx, wy) in enumerate(nav_wps):
        d = math.hypot(wx-bx, wy-by)
        if d < best_d: best_d, best_i = d, i
    return best_i

def target_yaw_dir(bx, by):
    """方向: 从当前位置指向终点"""
    dx, dy = FINISH[0]-bx, FINISH[1]-by
    d = math.hypot(dx, dy)
    return (dx/d, dy/d) if d > 0.01 else (1,0)

def scan_voxels(bx, by):
    for a in np.linspace(0, 2*np.pi, 120):
        cos_a, sin_a = math.cos(a), math.sin(a)
        for d in np.arange(0.5, LIDAR_RANGE+0.1, 0.5):
            wx, wy = bx+cos_a*d, by+sin_a*d
            vx, vy = int(wx/VOXEL), int(wy/VOXEL)
            if not (0 <= vx < W and 0 <= vy < W): break
            if vox[vy, vx] == WALL: break
            if blocked(wx, wy):
                vox[vy, vx] = WALL; break
            vox[vy, vx] = max(vox[vy, vx], FREE)

def find_gate_path(bx, by):
    """A*在FREE+VISITED里搜, 目标是FREE体素(朝终点方向+有UNKNOWN邻居优先)"""
    sx, sy = int(bx/VOXEL), int(by/VOXEL)
    if sx<0 or sx>=W or sy<0 or sy>=W or vox[sy,sx]==WALL: return None
    
    wp_dx, wp_dy = target_yaw_dir(bx, by)
    tx, ty = FINISH
    
    open_set = []
    heapq.heappush(open_set, (0, sx, sy))
    came_from = {}
    g_score = {(sx, sy): 0}
    best_gate = None; best_gate_score = -9999
    visited_a_star = set()
    
    while open_set and len(came_from) < 20000:
        _, cx, cy = heapq.heappop(open_set)
        if (cx, cy) in visited_a_star: continue
        visited_a_star.add((cx, cy))
        cg = g_score.get((cx, cy), 9999)
        
        # 出口=FREE体素(扫描过但没走过的)
        if vox[cy, cx] == FREE:
            dot = ((cx-sx)*wp_dx + (cy-sy)*wp_dy) / max(1, math.hypot(cx-sx, cy-sy))
            score = -cg  # 越近越好
            score += (dot+1)*30  # 朝WP加分 (负dot自动扣分, 不硬过滤)
            # UNKNOWN邻居加分(鼓励去新区附近)
            unk_nb = sum(1 for ndy in (-1,0,1) for ndx in (-1,0,1)
                        if 0<=cx+ndx<W and 0<=cy+ndy<W and vox[cy+ndy,cx+ndx]==UNKNOWN)
            score += unk_nb * 20
            # gate自身离墙距离——太近扣重分, 防选墙边死角
            gate_wd = 999
            for ndy2 in range(-5, 6):
                for ndx2 in range(-5, 6):
                    tnx, tny = cx+ndx2, cy+ndy2
                    if 0<=tnx<W and 0<=tny<W and vox[tny,tnx]==WALL:
                        dd = math.hypot(ndx2, ndy2)
                        if dd < gate_wd: gate_wd = dd
            if gate_wd < 1.5:
                score -= (1.5-gate_wd) * 20
            if score > best_gate_score:
                best_gate_score = score
                best_gate = (cx, cy)
        
        for ndx, ndy in [(0,-1),(0,1),(-1,0),(1,0)]:
            nx, ny = cx+ndx, cy+ndy
            if not walkable(nx, ny): continue
            # wall距离惩罚
            wd = 999.0; wx2, wy2 = (nx+0.5)*VOXEL, (ny+0.5)*VOXEL
            for ndy2 in range(-6, 7):
                for ndx2 in range(-6, 7):
                    tnx, tny = nx+ndx2, ny+ndy2
                    if 0<=tnx<W and 0<=tny<W and vox[tny,tnx]==WALL:
                        d = math.hypot(wx2-(tnx+0.5)*VOXEL, wy2-(tny+0.5)*VOXEL)
                        if d < wd: wd = d
            penalty = max(0, 2.0 - wd) * 3
            ng = cg + 1 + penalty
            if (nx, ny) not in g_score or ng < g_score[(nx, ny)]:
                g_score[(nx, ny)] = ng
                came_from[(nx, ny)] = (cx, cy)
                h = math.hypot(tx-(nx+0.5)*VOXEL, ty-(ny+0.5)*VOXEL) * 0.1
                heapq.heappush(open_set, (ng+h, nx, ny))
    
    if best_gate is None:
        return None
    
    # 回溯路径(只回溯, 不标记)
    path = []
    cur = best_gate
    while cur != (sx, sy):
        path.append(cur)
        if cur not in came_from: break
        cur = came_from[cur]
    path.reverse()
    
    world_path = [((px+0.5)*VOXEL, (py+0.5)*VOXEL) for px, py in path]
    return world_path

def seek_farthest_visited(bx, by):
    """v6风格fallback: A*在VISITED空间搜朝终点方向最远的体素, 不回溯"""
    sx, sy = int(bx/VOXEL), int(by/VOXEL)
    if not (0<=sx<W and 0<=sy<W and vox[sy,sx]==VISITED): return None
    wp_dx, wp_dy = target_yaw_dir(bx, by)
    best_v = None; best_score = -9999
    open_set = [(0, sx, sy)]; came = {}; vin = set()
    while open_set and len(came) < 10000:
        _, cx, cy = heapq.heappop(open_set)
        if (cx,cy) in vin: continue
        vin.add((cx,cy))
        if vox[cy,cx] == VISITED:
            dot = ((cx-sx)*wp_dx + (cy-sy)*wp_dy) / max(1, math.hypot(cx-sx, cy-sy))
            score = math.hypot(cx-sx, cy-sy)*0.5 + dot*50
            if score > best_score:
                best_score = score; best_v = (cx, cy)
        for ndx, ndy in [(0,-1),(0,1),(-1,0),(1,0)]:
            nx, ny = cx+ndx, cy+ndy
            if 0<=nx<W and 0<=ny<W and vox[ny,nx] in (VISITED, FREE) and (nx,ny) not in vin:
                came[(nx,ny)] = (cx,cy)
                heapq.heappush(open_set, (0, nx, ny))
    if best_v is None: return None
    path = []; cur = best_v
    while cur != (sx,sy):
        path.append(cur)
        if cur not in came: break
        cur = came[cur]
    path.reverse()
    return [((px+0.5)*VOXEL, (py+0.5)*VOXEL) for px, py in path]

def path_to_unk(bx, by):
    """A*在VISITED空间找最近UNKNOWN邻居, 返回完整路径"""
    sx, sy = int(bx/VOXEL), int(by/VOXEL)
    if vox[sy,sx]==WALL: return None
    open_set = [(0, sx, sy)]
    came_from = {}; g_score = {(sx,sy): 0}
    best_unk = None; best_g = 99999
    while open_set and len(came_from) < 5000:
        _, cx, cy = heapq.heappop(open_set)
        cg = g_score.get((cx,cy), 9999)
        has_unk = any(vox[cy+ndy,cx+ndx]==UNKNOWN
                     for ndy in (-1,0,1) for ndx in (-1,0,1)
                     if 0<=cx+ndx<W and 0<=cy+ndy<W)
        if has_unk and cg < best_g:
            best_g = cg; best_unk = (cx, cy)
        for ndx, ndy in [(0,-1),(0,1),(-1,0),(1,0)]:
            nx, ny = cx+ndx, cy+ndy
            if not (0<=nx<W and 0<=ny<W and vox[ny,nx]==VISITED): continue
            ng = cg + 1
            if (nx,ny) not in g_score or ng < g_score[(nx,ny)]:
                g_score[(nx,ny)] = ng
                came_from[(nx,ny)] = (cx,cy)
                heapq.heappush(open_set, (ng, nx, ny))
    if best_unk is None: return None
    path = []; cur = best_unk
    while cur != (sx,sy):
        path.append(cur)
        if cur not in came_from: break
        cur = came_from[cur]
    path.reverse()
    return [((px+0.5)*VOXEL, (py+0.5)*VOXEL) for px, py in path]

# ── XML & Mover (不变) ──
CP_XML = f'<body mocap="true" pos="{FINISH[0]} {FINISH[1]} 2"><geom type="sphere" size="1.5" rgba="0.2 1.0 0.2 0.8"/></body>'  # 只有终点绿球
OBS_XML = "".join(f'<body name="obs{i}" pos="{x:.1f} {y:.1f} 2.0"><geom type="cylinder" size="1.0 2.0" rgba="0.9 0.2 0.2 0.9"/></body>' for i,(x,y) in enumerate(obs_world))

xml = f"""<mujoco>
  <compiler angle="radian"/><option timestep="0.005"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset><hfield name="track" size="50.0 50.0 4.0 2.0" file="{MAP}"/></asset>
  <worldbody>
    <light pos="50 50 80" dir="0 0 -1"/>{CP_XML}{OBS_XML}
    <geom type="hfield" hfield="track" pos="50 50 0.0" rgba="0.25 0.30 0.35 1.0" friction="0 0 0"/>
    <body name="bot" pos="0 0 0.5">
      <joint type="slide" axis="1 0 0" damping="0"/>
      <joint type="slide" axis="0 1 0" damping="0"/>
      <geom type="cylinder" size="0.5 0.5" rgba="1 0.3 0 1" friction="0 0 0"/>
    </body>
  </worldbody>
</mujoco>"""

class Mover:
    def __init__(self, m, d):
        self.m, self.d = m, d
        self.yaw = 0.0; self.speed = SPEED; self.bounce = 0
        self.force = 0; self.escaping = False
        self.stuck_t = 0; self.stuck_x = 0.0; self.stuck_y = 0.0
    
    def step(self, tx, ty, step):
        bx, by = self.d.qpos[0], self.d.qpos[1]
        dt = self.m.opt.timestep
        if not self.escaping:
            tgt_yaw = math.atan2(ty-by, tx-bx)
            err = (tgt_yaw-self.yaw+math.pi)%(2*math.pi)-math.pi
            dyaw = max(-YAW_RATE*dt, min(YAW_RATE*dt, err))
            self.yaw += dyaw
            clear = math.hypot(tx-bx, ty-by)
            self.speed = max(1.5, min(SPEED_MAX, clear*0.5))
        vx = math.cos(self.yaw)*self.speed
        vy = math.sin(self.yaw)*self.speed
        nx, ny = bx+vx*dt, by+vy*dt
        if step-self.stuck_t > 300:
            if math.hypot(bx-self.stuck_x, by-self.stuck_y) < 0.5:
                self._bounce(90, 180)
            self.stuck_t = step; self.stuck_x = bx; self.stuck_y = by
        if self.force > 0:
            self.force -= 1
            self.d.qvel[0] = vx; self.d.qvel[1] = vy
        elif blocked(nx, ny):
            self._bounce(45, 120)
        else:
            self.escaping = False
            self.d.qvel[0] = vx; self.d.qvel[1] = vy
        mujoco.mj_step(self.m, self.d); return True
    
    def _bounce(self, lo, hi):
        if not self.escaping:
            self.bounce += 1; self.escaping = True
            bx, by = self.d.qpos[0], self.d.qpos[1]
            if self.bounce % 5 == 0:
                print(f"  💥 bounce#{self.bounce} @({bx:.1f},{by:.1f})", flush=True)
        deg = random.uniform(lo, hi)*random.choice([-1,1])
        self.yaw += math.radians(deg)
        self.d.qvel[:] = 0
        self.force = int(0.3/(SPEED*self.m.opt.timestep))

# ── 主循环 ──
m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
d.qpos[0]=10; d.qpos[1]=5; mujoco.mj_forward(m,d)

mv = Mover(m, d)
wp_idx=0; step=0; t0=time.time(); RENDER_SKIP=20
path = None; path_idx = 0
last_progress_step = 0; last_progress_wp = 0

print(f"=== 萤火算法 Firefly v17 === 127中心线点 + 动态选点 + 跳过不可达", flush=True)

with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as v:
    v.cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    v.cam.distance=25; v.cam.elevation=-35; v.cam.azimuth=180

    LIDAR_TICK = 20

    # 初始扫描(恢复v6量级, FREE铺够才能过CP2窄弯)
    print("  🔍 initial scan (200 steps)...", flush=True)
    for _ in range(200):
        bx, by = d.qpos[0], d.qpos[1]
        vx, vy = int(bx/VOXEL), int(by/VOXEL)
        if 0 <= vx < W and 0 <= vy < W: vox[vy, vx] = VISITED
        if _ % LIDAR_TICK == 0: scan_voxels(bx, by)
        mujoco.mj_step(m, d)
    print(f"  ✅ F{int(np.sum(vox==FREE))} W{int(np.sum(vox==WALL))}", flush=True)
    
    path = find_gate_path(d.qpos[0], d.qpos[1])
    if path:
        gate = path[-1]
        print(f"  🚪 first gate=({gate[0]:.0f},{gate[1]:.0f}) len={len(path)}", flush=True)

    while v.is_running() and wp_idx < len(nav_wps)-1:
        bx, by = d.qpos[0], d.qpos[1]
        if bx<1 or bx>99 or by<1 or by>99:
            d.qpos[0]=max(1,min(99,bx)); d.qpos[1]=max(1,min(99,by))
            d.qvel[:]=0; mv.yaw=random.uniform(0,2*math.pi)
        v.cam.lookat[:]=np.array([bx, by, 0.5], dtype=np.float64)

        # 动态选点: 找最近的导航点, 沿中心线往终点跳
        closest = get_closest_wp(bx, by)
        # 看看能跳到多远: 从closest往后找第一个可达的
        new_wp = wp_idx
        for i in range(closest, len(nav_wps)):
            wx, wy = nav_wps[i]
            # 检查这个点是否被墙/障碍挡住
            vx, vy = int(wx/VOXEL), int(wy/VOXEL)
            if blocked(wx, wy): continue  # 不可达, 跳过
            new_wp = max(new_wp, i)
            if i >= closest + 3: break  # 跳太多步不放心, 先跳到近的
        if new_wp > wp_idx:
            wp_idx = new_wp
            last_progress_step = step; last_progress_wp = wp_idx
            progress_pct = wp_idx*100.0/len(nav_wps)
            print(f"  🏁 wp={wp_idx}/{len(nav_wps)}({progress_pct:.0f}%) @({bx:.1f},{by:.1f}) step={step} V{int(np.sum(vox==VISITED))}", flush=True)
            if wp_idx >= len(nav_wps)-1:
                print(f"🏁 FINISH step={step} time={time.time()-t0:.1f}s bounce={mv.bounce}", flush=True)
                break
            continue

        # 只标记实际到达为OLD
        vx, vy = int(bx/VOXEL), int(by/VOXEL)
        if 0 <= vx < W and 0 <= vy < W: vox[vy, vx] = VISITED
        if step % LIDAR_TICK == 0:
            scan_voxels(bx, by)

        # 路径规划/执行 — 目标始终是终点方向
        if path is None or path_idx >= len(path):
            # 卡死检测: 2000步未进→强制path_to_unk
            if step - last_progress_step > 2000 and wp_idx == last_progress_wp:
                print(f"  ⚠️  [{step}] stuck {step-last_progress_step} steps! force path_to_unk", flush=True)
                path = path_to_unk(bx, by)
                if not path:
                    path = find_gate_path(bx, by)
                path_idx = 0
                last_progress_step = step
            else:
                path = find_gate_path(bx, by)
                path_idx = 0
            if path:
                gate = path[-1]
                print(f"  🚪 [{step}] gate=({gate[0]:.0f},{gate[1]:.0f}) len={len(path)} F{int(np.sum(vox==FREE))}", flush=True)
            else:
                fwd_path = seek_farthest_visited(bx, by)
                if fwd_path:
                    path = fwd_path; path_idx = 0
                    gate = path[-1]
                    print(f"  🧭 [{step}] no gate, seek_fwd→({gate[0]:.0f},{gate[1]:.0f}) len={len(path)}", flush=True)
                else:
                    bk_path = path_to_unk(bx, by)
                    if bk_path:
                        path = bk_path; path_idx = 0
                        gate = path[-1]
                        print(f"  🔍 [{step}] no gate, path_to_unk→({gate[0]:.0f},{gate[1]:.0f}) len={len(path)}", flush=True)
                    else:
                        mv._bounce(90, 180)
        
        if path is not None and path_idx < len(path):
            tx, ty = path[path_idx]
            if math.hypot(tx-bx, ty-by) < 1.0:
                path_idx += 1
                if path_idx >= len(path):
                    gate_reached = path[-1]
                    print(f"  ✅ [{step}] at gate=({gate_reached[0]:.0f},{gate_reached[1]:.0f}), scan...", flush=True)
                    for _ in range(200):
                        if _ % LIDAR_TICK == 0: scan_voxels(d.qpos[0], d.qpos[1])
                        mujoco.mj_step(m, d); step += 1
                        if step % RENDER_SKIP == 0: v.sync()
                    new_path = find_gate_path(d.qpos[0], d.qpos[1])
                    if new_path:
                        ng = new_path[-1]
                        if math.hypot(ng[0]-gate_reached[0], ng[1]-gate_reached[1]) < 3.0:
                            gvx, gvy = int(ng[0]/VOXEL), int(ng[1]/VOXEL)
                            for dy2 in range(-2, 3):
                                for dx2 in range(-2, 3):
                                    tx2, ty2 = gvx+dx2, gvy+dy2
                                    if 0<=tx2<W and 0<=ty2<W and vox[ty2,tx2]==FREE:
                                        vox[ty2,tx2] = VISITED
                            stale = int(np.sum(vox==VISITED))
                            print(f"  [sterile] gate still same, marked {stale}V", flush=True)
                            path = None; path_idx = 0
                        else:
                            path = new_path; path_idx = 0
                    else:
                        path = None; path_idx = 0
            else:
                mv.step(tx, ty, step)
        else:
            mv._bounce(90, 180)
        
        step += 1
        if step % 1000 == 0:
            pct = wp_idx*100.0/len(nav_wps)
            print(f"  ... step={step} V{int(np.sum(vox==VISITED))} F{int(np.sum(vox==FREE))} wp={wp_idx}({pct:.0f}%)", flush=True)
        if step % RENDER_SKIP == 0: v.sync()

    print(f"done: {wp_idx}/{len(nav_wps)-1} step={step} time={time.time()-t0:.1f}s", flush=True)
