#!/usr/bin/env python3
"""圆柱体+独立雷达模块(10Hz 多线可调)+VO避障 — 障碍物沿赛道中轴线生成"""
import os,sys,time,random,math,numpy as np,mujoco,mujoco.viewer
sys.path.insert(0,os.path.expanduser("~/workspace"))
from vo_algorithm import DebounceState,choose_optimal_heading
from lidar_sensor import LidarSensor

MAP_SZ=50;TURN_R=5;START_X,START_Y=5,45
SIM_DT=0.008;R_ROBOT=0.25;OBS_R=0.3
MAX_SPEED=float(sys.argv[1]) if len(sys.argv)>1 else 2.0
LIDAR_RAYS=120;LIDAR_LINES=3;LIDAR_RANGE=15.0
LOG=open("/tmp/cyl_real.log","w")

def log(msg):print(msg,flush=True);LOG.write(msg+"\n");LOG.flush()
log(f"LIDAR_MODULE rays={LIDAR_RAYS} lines={LIDAR_LINES} range={LIDAR_RANGE}m speed={MAX_SPEED:.1f}")

# 中心线+waypoint
cx,cy=[],[];x,y=START_X,START_Y
for i in range(10):
    if i%2==0:
        for j in range(80):cx.append(x+j*0.5);cy.append(y)
        x=cx[-1];n=int(math.pi*TURN_R/0.25)
        for j in range(1,n+1):a=math.pi/2*j/n;cx.append(x+TURN_R*(1-math.cos(a)));cy.append(y-TURN_R*math.sin(a))
        x=cx[-1];y=cy[-1]-TURN_R*2
    else:
        for j in range(80):cx.append(x-j*0.5);cy.append(y)
        x=cx[-1];n=int(math.pi*TURN_R/0.25)
        for j in range(1,n+1):a=math.pi/2*j/n;cx.append(x-TURN_R*(1-math.cos(a)));cy.append(y-TURN_R*math.sin(a))
        x=cx[-1];y=cy[-1]-TURN_R*2
cx,cy=np.array(cx),np.array(cy)
cum=np.insert(np.cumsum(np.hypot(np.diff(cx),np.diff(cy))),0,0)
track_len=cum[-1]
wp=[];nd=8.0
for i in range(len(cx)):
    if cum[i]>=nd:wp.append((cx[i],cy[i]));nd+=8.0
if not wp or wp[-1]!=(cx[-1],cy[-1]):wp.append((cx[-1],cy[-1]))
log(f"Track:{track_len:.0f}m {len(wp)}wp")

# ====== 新障碍物生成：沿赛道中轴线 ======
rng=random.Random()  # 每次运行随机种子
obs_list=[]
d=5.0  # 起点+5米
while d<track_len-5:
    inc=round(rng.uniform(4,8),1)        # 沿赛道距离增量 4-8
    offset=round(rng.uniform(0.5,4.5),1)  # 横向偏移 0.5-4.5
    side=rng.choice([-1,1])               # 随机左/右

    idx=np.searchsorted(cum,d)
    if idx>=len(cx)-1:break
    dx=cx[idx+1]-cx[idx];dy=cy[idx+1]-cy[idx]
    mag=math.hypot(dx,dy)
    if mag<0.001:d+=inc;continue
    nx=-dy/mag;ny=dx/mag  # 法线(逆时针90°)
    ox=cx[idx]+side*offset*nx
    oy=cy[idx]+side*offset*ny
    obs_list.append((ox,oy))
    d+=inc

obs_xml=""
for i,(ox,oy) in enumerate(obs_list):
    obs_xml+='<body name="o%d" pos="%.1f %.1f %.1f"><geom type="cylinder" size="%.1f %.1f" rgba="0.9 0.3 0.3 0.8"/></body>\n'%(i,ox,oy,OBS_R,OBS_R,OBS_R)
log(f"Obstacles:{len(obs_list)} along {track_len:.0f}m centerline")
# ====== 结束 ======

# 场景: hfield碰撞+plane+障碍物+机器人顶部site
scene=f'''<mujoco>
<compiler angle="radian"/><option timestep="{SIM_DT}"/>
<visual><global offwidth="1280" offheight="720"/></visual>
<asset><hfield name="h" size="25 25 6 3" file="/tmp/track_hd.png"/>
<material name="v" rgba="0.25 0.30 0.35 1"/><material name="i" rgba="0.25 0.30 0.35 0"/></asset>
<worldbody>
<light pos="25 25 80" dir="0 0 -1" diffuse="1.5 1.5 1.5" specular="0.5 0.5 0.5"/>
<geom type="hfield" hfield="h" pos="25 25 0" material="v"/>
<geom type="plane" size="0 0 0.05" material="i"/>
<body name="r" pos="{START_X} {START_Y} 0.5">
<inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
<joint name="x" type="slide" axis="1 0 0" damping="0"/>
<joint name="y" type="slide" axis="0 1 0" damping="0"/>
<joint name="z" type="hinge" axis="0 0 1" damping="0"/>
<geom type="cylinder" size="{R_ROBOT} 0.3" rgba="0.2 0.8 0.2 0.9"/>
<site name="lidar_top" pos="0 0 0.5" size="0.02"/></body>
{obs_xml}</worldbody></mujoco>'''
with open(os.path.expanduser("~/cyl_real.xml"),"w") as f:f.write(scene)

m=mujoco.MjModel.from_xml_path(os.path.expanduser("~/cyl_real.xml"));d=mujoco.MjData(m)
d.qpos[0:2]=[START_X,START_Y]

# 独立雷达模块 — 装在机器人顶部避开自身体积
lidar=LidarSensor(m,d,position=None,
                  rays=LIDAR_RAYS,lines=LIDAR_LINES,range_m=LIDAR_RANGE,hz=10)
log(f"Lidar: {lidar.rays}rays x {lidar.lines}lines, update every {lidar.step_interval} steps")

cnt=0;t0=time.time();cw=0;goal=wp[0];vc=MAX_SPEED;debounce=DebounceState();coll=0
DECISION_EVERY=lidar.step_interval

with mujoco.viewer.launch_passive(m,d) as v:
    v.cam.azimuth=90;v.cam.elevation=-30;v.cam.distance=30;v.cam.lookat=(25,25,0)
    while v.is_running():
        bx,by,yaw=d.qpos[0],d.qpos[1],d.qpos[2]
        if cnt%DECISION_EVERY==0:
            # 10Hz雷达更新→点云→聚类→VO
            lidar.update(bx,by,yaw)
            obstacles=lidar.cluster(grid_size=1.0,min_hits=3)
            dg=math.hypot(bx-goal[0],by-goal[1])
            heading,avoiding=choose_optimal_heading((bx,by),vc,goal,obstacles,debounce)
            yaw=heading
            vc=MAX_SPEED*0.6 if avoiding else MAX_SPEED
            if dg<1.5 and cw<len(wp)-1:
                cw+=1;goal=wp[cw];log(f"[{cnt*SIM_DT:.0f}s] WP{cw} pts={lidar.hit_count} obs={len(obstacles)}")
            if cw>=len(wp)-1 and dg<2.0:log(f"ARRIVED sim={cnt*SIM_DT:.0f}s avg={track_len/(cnt*SIM_DT):.1f}m/s");break
        for i in range(d.ncon):
            if d.contact[i].dist<-0.01:coll+=1;break
        vx,vy=vc*math.cos(yaw),vc*math.sin(yaw)
        d.qvel[0:2]=[vx,vy];d.qvel[2]=0
        mujoco.mj_step(m,d);cnt+=1
        if cnt%500==0:log(f"[{cnt*SIM_DT:.0f}s] ({bx:.1f},{by:.1f}) wp{cw} v={vc:.1f} pts={lidar.hit_count} obs={len(obstacles)} coll={coll}")

log(f"DONE sim={cnt*SIM_DT:.0f}s coll={coll}")
LOG.close()
