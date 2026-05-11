#!/usr/bin/env python3
"""G1 protrain + Tangent Arc — track_hd 半径5米蛇形赛道 + lidar日志"""
import os, sys, time, json, numpy as np, mujoco, mujoco.viewer, torch, yaml
from xml.etree import ElementTree as ET

LEGGED_GYM_ROOT = os.path.expanduser("~/unitree_rl_gym")
sys.path.insert(0, os.path.expanduser("~/navigation"))
from tangent_arc_planner import TangentArcPlanner, TangentArcConfig

POLICY_PATH = f"{LEGGED_GYM_ROOT}/deploy/pre_train/g1/motion.pt"
G1_XML = f"{LEGGED_GYM_ROOT}/resources/robots/g1_description/scene.xml"
TRACK_PNG = "/tmp/track_hd.png"

with open(f"{LEGGED_GYM_ROOT}/deploy/deploy_mujoco/configs/g1.yaml") as f:
    C = yaml.load(f, Loader=yaml.FullLoader)
kps = np.array(C["kps"], np.float32); kds = np.array(C["kds"], np.float32)
default_angles = np.array(C["default_angles"], np.float32)
action_scale, num_obs = C["action_scale"], C["num_obs"]
SIM_DT, DECIMATION, num_actions = 0.002, 10, 12

# ── 赛道: 10段直道(5→45, 40m) + R=5m U型弯 (trackgen_v2实际参数) ──
TURN_R, STRAIGHT_LEN = 5.0, 40.0
START_X, START_Y, MAP_SZ = 5.0, 45.0, 50.0
N_STRAIGHTS = 10
cx, cy = [], []
x, y = START_X, START_Y
for i in range(N_STRAIGHTS):
    if i % 2 == 0:
        for j in range(int(STRAIGHT_LEN/0.5)):
            cx.append(min(x + j*0.5, MAP_SZ-5)); cy.append(y)
        x = cx[-1]
        # 右U型弯
        n_arc = max(10, int(np.pi*TURN_R/0.25))
        for j in range(1, n_arc+1):
            a = np.pi/2 * j/n_arc
            cx.append(x + TURN_R*(1-np.cos(a)))
            cy.append(y - TURN_R*np.sin(a))
        x = cx[-1]; y = cy[-1] - TURN_R*2
    else:
        for j in range(int(STRAIGHT_LEN/0.5)):
            cx.append(max(x - j*0.5, 5.0)); cy.append(y)
        x = cx[-1]
        n_arc = max(10, int(np.pi*TURN_R/0.25))
        for j in range(1, n_arc+1):
            a = np.pi/2 * j/n_arc
            cx.append(x - TURN_R*(1-np.cos(a)))
            cy.append(y - TURN_R*np.sin(a))
        x = cx[-1]; y = cy[-1] - TURN_R*2
cx, cy = np.array(cx), np.array(cy)
track_len = np.sum(np.hypot(np.diff(cx), np.diff(cy)))
# Waypoints 每5m
wp_dists = np.insert(np.cumsum(np.hypot(np.diff(cx), np.diff(cy))), 0, 0)
waypoints, nd = [], 5.0
for i in range(len(cx)):
    if wp_dists[i] >= nd: waypoints.append((float(cx[i]),float(cy[i]))); nd += 5.0
if not waypoints or waypoints[-1] != (cx[-1],cy[-1]): waypoints.append((float(cx[-1]),float(cy[-1])))
print(f"Track: ~{track_len:.0f}m R={TURN_R}m, {len(waypoints)} waypoints")

# ── 场景 ──
g1_tree = ET.parse(G1_XML); g1_root = g1_tree.getroot()
g1_asset = g1_root.find("asset")
if g1_asset is None: g1_asset = ET.SubElement(g1_root, "asset")
ET.SubElement(g1_asset, "hfield", {"name":"track","size":"25.0 25.0 4.0 2.0","file":TRACK_PNG})
ET.SubElement(g1_asset, "material", {"name":"track_vis","rgba":"0.25 0.30 0.35 1.0"})
ET.SubElement(g1_asset, "material", {"name":"invis","rgba":"0.25 0.30 0.35 0.0"})
g1_wb = g1_root.find("worldbody")
ET.SubElement(g1_wb, "geom", {"type":"hfield","hfield":"track","pos":"25 25 0.0","material":"track_vis","contype":"0","conaffinity":"0"})
ET.SubElement(g1_wb, "geom", {"type":"plane","size":"0 0 0.05","material":"invis"})
merged_path = os.path.join(os.path.dirname(G1_XML), "scene_snake_nav.xml")
with open(merged_path, "w") as f: f.write(ET.tostring(g1_root, encoding="unicode"))

# ── Planner (转弯减速) ──
planner = TangentArcPlanner(TangentArcConfig(
    robot_radius=0.35, max_speed=1.0, min_speed=0.05, max_yaw_rate=1.5,
    goal_tolerance=1.2, arc_samples=12, safety_margin=0.15, predict_time=3.0))
LIDAR_RAYS, LIDAR_RANGE = 72, 6.0

def quat2euler(q):
    w,x,y,z=q; return(np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)),
                     np.arcsin(np.clip(2*(w*y-z*x),-1,1)),
                     np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
def gravity_vec(q):
    w,x,y,z=q; return np.array([2*(-z*x+w*y),-2*(z*y+w*x),1-2*(w*w+z*z)])
def lidar_scan(m,d,site_id,quat):
    pos=d.site_xpos[site_id].copy()
    r,p,y=quat2euler(quat)
    cr,sr=np.cos(r),np.sin(r); cp,sp=np.cos(p),np.sin(p); cy,sy=np.cos(y),np.sin(y)
    R=np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],
                [-sp,cp*sr,cp*cr]])
    pts=[]; gid=np.array([-1],np.int32)
    for i in range(LIDAR_RAYS):
        a=2*np.pi*i/LIDAR_RAYS
        dloc=np.array([np.cos(a),np.sin(a),-0.4])
        dw=R@(dloc/np.linalg.norm(dloc))
        dist=mujoco.mj_ray(m,d,pos,dw,None,1,1,gid)
        if gid[0]>=0 and 0<dist<LIDAR_RANGE:
            h=pos+dw*dist
            if h[2]>0.1: pts.append((h[0],h[1]))
    return pts,pos

# ── 预计算转弯段 ──
turn_segments = set()
for i in range(1, len(waypoints)):
    prev, curr = waypoints[i-1], waypoints[i]
    # 简单判断：y变化说明在弯道区域
    if abs(curr[1] - prev[1]) > 2.0:
        turn_segments.add(i)

# ── Load ──
print("Loading...")
m=mujoco.MjModel.from_xml_path(merged_path); d=mujoco.MjData(m); m.opt.timestep=SIM_DT
policy=torch.jit.load(POLICY_PATH)
d.qpos[0:3]=[START_X,START_Y,0.8]; d.qpos[3:7]=[1,0,0,0]
lidar_id=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,"lidar_mount")
action=np.zeros(num_actions,np.float32); target=default_angles.copy()
for _ in range(300):
    d.ctrl[:12]=kps*(default_angles-d.qpos[7:19])-kds*d.qvel[6:18]; mujoco.mj_step(m,d)
print(f"Start: ({d.qpos[0]:.1f},{d.qpos[1]:.1f}) z={d.qpos[2]:.3f}")

# ── Main ──
cnt=0; t0=time.time(); current_wp=0; goal=waypoints[0]
lidar_log = []  # (t_sim, x, y, n_pts, heading, wp_id)

with mujoco.viewer.launch_passive(m,d) as v:
    v.cam.azimuth=90; v.cam.elevation=-30; v.cam.distance=30; v.cam.lookat=(25,25,0)
    while v.is_running():
        d.ctrl[:12]=kps*(target-d.qpos[7:19])-kds*d.qvel[6:18]; mujoco.mj_step(m,d); cnt+=1
        if cnt%DECIMATION!=0: continue
        t_sim=cnt*SIM_DT
        qj=d.qpos[7:19]; dqj=d.qvel[6:18]; quat=d.qpos[3:7]; omega=d.qvel[3:6]
        bp=d.qpos[:3].copy(); _,_,yaw=quat2euler(quat)
        lpts,_=lidar_scan(m,d,lidar_id,quat)
        dist_to_goal=np.hypot(bp[0]-goal[0],bp[1]-goal[1])

        # 转弯减速
        in_turn = current_wp in turn_segments
        planner.cfg.max_speed = 0.5 if in_turn else 1.0

        vc,wc,_=planner.plan(bp[0],bp[1],yaw,0.0,omega[2],goal,lpts)

        # Lidar log
        heading_deg = np.degrees(yaw) % 360
        lidar_log.append((t_sim, bp[0], bp[1], len(lpts), heading_deg, current_wp))

        if dist_to_goal<planner.cfg.goal_tolerance and current_wp<len(waypoints)-1:
            current_wp+=1; goal=waypoints[current_wp]
            print(f"[{t_sim:.0f}s] ▶ WP{current_wp} ({goal[0]:.0f},{goal[1]:.0f})")
        if current_wp>=len(waypoints)-1 and dist_to_goal<1.5:
            avg=track_len/t_sim if t_sim>0 else 0
            print(f"\n✓ ARRIVED sim={t_sim:.0f}s avg={avg:.2f}m/s")
            break
        if bp[2]<0.4: print(f"\n✗ FALL z={bp[2]:.3f}"); break
        cmd=np.array([vc,0.,wc])*np.array([2.,2.,0.25])
        obs=np.zeros(num_obs,np.float32)
        obs[:3]=omega*.25; obs[3:6]=gravity_vec(quat); obs[6:9]=cmd
        obs[9:21]=(qj-default_angles); obs[21:33]=dqj*.05; obs[33:45]=action; obs[45:47]=[0,0]
        action=policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
        target=action*action_scale+default_angles
        if cnt%(DECIMATION*250)==0:
            spd = planner.cfg.max_speed
            print(f"[{t_sim:.0f}s] ({bp[0]:.1f},{bp[1]:.1f}) wp{current_wp}/{len(waypoints)-1} d={dist_to_goal:.1f}m v={vc:.2f} lidar={len(lpts)}{' 🐢' if in_turn else ''}")

t_sim=cnt*SIM_DT; elapsed=time.time()-t0

# ── 保存lidar日志 ──
log_path = "/tmp/g1_lidar_log.jsonl"
with open(log_path, "w") as f:
    for entry in lidar_log:
        f.write(json.dumps({"t":entry[0],"x":round(entry[1],2),"y":round(entry[2],2),
                            "lidar_hits":entry[3],"heading":round(entry[4],1),"wp":entry[5]})+"\n")
print(f"Lidar log: {log_path} ({len(lidar_log)} frames)")

# ── 统计 ──
hits_arr = np.array([e[3] for e in lidar_log])
print(f"Lidar stats: mean={hits_arr.mean():.1f} max={hits_arr.max()} min={hits_arr.min()} "
      f"zero_pct={np.sum(hits_arr==0)/len(hits_arr)*100:.1f}%")
print(f"\n=== DONE sim={t_sim:.0f}s wall={elapsed:.0f}s ===")
