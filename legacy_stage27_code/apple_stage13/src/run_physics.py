"""v1.9 experimental force-feedback controller and continuous release. v1.5 grip-clearance and transport-duration sensitivity study. Tests are not executed without PyBullet. Unlike rendering code, tests gravity/contact/motor following.
This module is intentionally NOT claimed as executed without installing PyBullet.
- Apple is never attached to the gripper or teleported after placement.
- Failure is written as FAIL, with actual contact forces and apple positions.
- Camera transforms come from Bullet wrist link frames (NO gaze tracking).
"""
from __future__ import annotations
import sys,math,json,csv,time,argparse,traceback,statistics,random,itertools
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
from mechanism_v09_reference import G,q_at,smootherstep,ik,commanded_wrist
from learned_classifier import APPEARANCES,train,infer,rgb_features
from tray_control import tray_occupied, station_ready, placement_allowed, next_free_nest, TRAY_NAMES
from workbench_geometry import LAYOUT,make_static_geometry
from cushion_texture_assets import (ensure_assets,create_cushion,spawn_textured_apple,
                                    capture_in_cup,CUP_DEPTH_M,CUP_DIAMETER_M)
HERE=Path(__file__).resolve().parents[1]
DT=1/240

@dataclass(frozen=True)
class TrialConfig:
    torque_Nm:float=12.0
    grip_relief_mm:float=0.0
    transport_s:float=0.70
    profile:str='nominal'
    replicate:int=0
    control_mode:str='fixed'  # fixed, force_feedback, force_feedback_v2
    close_duration_s:float=.08  # Same geometry; duration of final 4.3mm inward approach
    scan_duration_s:float=.52  # Stage16: time warp of same camera poses (original 0.52s)
    preloaded_bin:str=''  # v2.2: tray that initially holds a fruit (legacy CLI name)
    human_clear_after_s:float|None=None  # explicit external worker event; NOT robot action
    tray_slot:int=1  # 1,2,3: three receiving nests per grade
    max_worker_wait_s:float=2.0  # finite simulation observation window; actual controller must stay paused
    motion_profile:str='baseline'  # baseline / balanced / fast_trial / guarded_fast
    # Torque is an ACTUATOR setting, NOT a pad contact-force limit.
    def stress_parameters(self):
        """Deterministic but different perturbations per repeat (paired across grip settings).
        Radius +/- 2mm, position <=3mm, friction [0.6,1.0], mass [0.9,1.1].
        No collision/quality guarantee. Identical replicate ID gives identical fruit across conditions.
        """
        rng=random.Random(20261014 + 7919 * self.replicate)
        r=.003*math.sqrt(rng.random()); angle=2*math.pi*rng.random()
        return {'offset_x_m':r*math.cos(angle), 'offset_y_m':r*math.sin(angle),
                'mass_scale':.9+.2*rng.random(),
                'apple_friction':.6+.4*rng.random(),
                'apple_radius_m':G.apple_r+(.004*rng.random()-.002)}
    @property
    def slug(self):
        return f'{self.profile}' + (f'_SC{round(self.scan_duration_s*1000):03d}ms' if abs(self.scan_duration_s-.52)>1e-9 else '') + (f'_V{self.motion_profile}' if self.motion_profile!='baseline' else '') + f'_T{self.torque_Nm:g}_G{self.grip_relief_mm:g}mm_M{self.transport_s:.2f}s' + ({'force_feedback':'_FB','force_feedback_v2':'_FB2'}.get(self.control_mode,'')) + (f'_CL{round(self.close_duration_s*1000):03d}ms' if abs(self.close_duration_s-.08)>1e-9 else '') + (f'_P{self.preloaded_bin}' if self.preloaded_bin else '') + (f'_H{int(self.human_clear_after_s*1000)}' if self.human_clear_after_s is not None else '') + f'_S{self.tray_slot}' + f'_R{self.replicate:02d}'
    @property
    def mass_scale(self):
        return self.stress_parameters()['mass_scale'] if self.profile=='stress_mix' else (1.10 if self.profile=='heavy10' else 1.0)
    @property
    def apple_friction(self):
        return self.stress_parameters()['apple_friction'] if self.profile=='stress_mix' else (.60 if self.profile=='low_mu' else 1.0)
    @property
    def apple_radius_m(self):
        return self.stress_parameters()['apple_radius_m'] if self.profile=='stress_mix' else (G.apple_r + (-.002 if self.profile=='small2' else 0.))
    @property
    def offset_xy(self):
        if self.profile=='stress_mix':
            d=self.stress_parameters();return d['offset_x_m'],d['offset_y_m']
        if self.profile!='shift3': return (0.,0.)
        # deterministic alternate orientations; fresh physical trial per replicate
        angle=2*math.pi*((self.replicate%8)/8.)+math.pi/7.
        return (.003*math.cos(angle), .003*math.sin(angle))

PROFILES=('nominal','shift3','low_mu','heavy10','small2','stress_mix')

# Physical motion experiments: all profiles preserve 100 mm lift, the same 5 cameras,
# the same gripper, support-detected release, and home return within episode time.
# These are MOTOR COMMANDS, not a demonstrated orchard cycle or fruit safety proof.
from dataclasses import dataclass as _motion_dataclass
@_motion_dataclass(frozen=True)
class MotionSettings:
    x_max_velocity:float
    x_effective_rate:float
    preflight_steps:int
    scale_settle_steps:int
    contact_settle_steps:int
    feedback_steps:int
    lift_steps:int
    lower_fast_m_s:float
    lower_slow_m_s:float
    lower_slow_window_m:float
    release_steps:int
    settle_min_steps:int
    settle_max_steps:int

MOTION_SETTINGS={
    'baseline':MotionSettings(.6,.47,50,60,30,40,109,.10,.10,1.0,49,120,120),
    'balanced':MotionSettings(1.0,.75,50,24,18,28,96,.25,.10,.035,43,26,120),
    'fast_trial':MotionSettings(1.25,.95,50,16,14,22,83,.32,.10,.035,39,24,120),
    'guarded_fast':MotionSettings(1.50,1.12,50,16,14,22,83,.36,.10,.025,39,24,120),
    # Experimental tradeoff: a longer slow-approach window and gentler release.
    # Lower contact force is a hypothesis to test, NOT a verified result.
    'guarded_gentle':MotionSettings(1.50,1.12,50,16,14,22,83,.30,.08,.040,50,24,120),
    # Stage09: conservative release (50 steps), measured encoder arrival,
    # shorter input inspection. No change to 100mm lift, full scan or contact gate.
    'precision_fast':MotionSettings(1.50,1.12,8,16,14,22,83,.36,.10,.025,50,24,120),
    # Matched speed probe to isolate release profile effects (39 frames).
    'quick_gate':MotionSettings(1.50,1.12,8,16,14,22,83,.36,.10,.025,39,24,120),
    'contact_gate_39':MotionSettings(1.50,1.12,8,16,14,22,83,.36,.10,.025,39,24,120),
    'contact_gate_50':MotionSettings(1.50,1.12,8,16,14,22,83,.36,.10,.025,50,24,120),
    # Stage11: same quick release, but verify FULL radial clearance and measured motor settling.
    'settled_gate_09':MotionSettings(1.50,1.12,8,16,14,22,83,.36,.10,.025,39,24,120),
    'settled_gate_12':MotionSettings(1.50,1.12,8,16,14,22,83,.36,.10,.025,39,24,120),
}

def motion_settings(cfg):
    if cfg.motion_profile not in MOTION_SETTINGS:
        raise ValueError(f'unknown motion_profile: {cfg.motion_profile}')
    m=MOTION_SETTINGS[cfg.motion_profile]
    # 0.10 m lift is never lowered to optimize simulated throughput.
    assert m.lift_steps>=2 and m.x_max_velocity<=1.5 and m.lower_fast_m_s<=.5
    assert 0 < m.lower_slow_m_s <= .15
    assert m.settle_min_steps<=m.settle_max_steps
    return m

def staged_lower_target(elapsed_s, z_start, touch_z, policy):
    """Two-speed descent; starts slow before the estimated contact plane.

    touch_z is an *estimated* nominal contact stage height, never used to
    release fruit. Actual force contact verifies the support before release.
    """
    slow_start=touch_z + policy.lower_slow_window_m
    if slow_start>=z_start:
        return z_start-policy.lower_slow_m_s*elapsed_s
    fast_dist=max(0.0,z_start-slow_start)
    fast_time=fast_dist/policy.lower_fast_m_s
    if elapsed_s<=fast_time:
        return max(slow_start,z_start-policy.lower_fast_m_s*elapsed_s)
    return slow_start-policy.lower_slow_m_s*(elapsed_s-fast_time)

# Stage20 v4.0d: experimental INITIAL descent acceleration only.
# Preserve original staged_lower_target for all other control modes and test cases.
_stage20_original_staged_lower_target = staged_lower_target

def staged_lower_target(elapsed_s, z_start, touch_z, policy):
    import os as _os
    ramp_s=float(_os.environ.get('STAGE20_V40D_LOWER_RAMP_S', '0'))
    if ramp_s == 0.0:
        return _stage20_original_staged_lower_target(elapsed_s,z_start,touch_z,policy)
    if ramp_s not in (0.08,0.16):
        raise ValueError('Only 0.08s/0.16s controlled descent ramp variants are authorized.')
    # Integrated velocity ramp: v_fast*t/ramp_s until ramp completion.
    # Acceleration is max 4.5 or 2.25 m/s^2 in the COMMANDED profile,
    # NOT an assertion about real axis acceleration or fruit acceleration.
    slow_start=touch_z+policy.lower_slow_window_m
    if slow_start >= z_start:
        return _stage20_original_staged_lower_target(elapsed_s,z_start,touch_z,policy)
    vf=policy.lower_fast_m_s
    fast_dist=max(0., z_start-slow_start)
    ramp_dist=0.5*vf*ramp_s
    if ramp_dist >= fast_dist:
        raise ValueError('Ramp distance overtakes slow approach window.')
    if elapsed_s <= ramp_s:
        return z_start - 0.5*vf*elapsed_s*elapsed_s/ramp_s
    remaining_fast_time=(fast_dist-ramp_dist)/vf
    if elapsed_s <= ramp_s+remaining_fast_time:
        return z_start-ramp_dist-vf*(elapsed_s-ramp_s)
    return slow_start - policy.lower_slow_m_s*(elapsed_s-ramp_s-remaining_fast_time)



def load_scene(p,label,cfg:TrialConfig=TrialConfig()):
    p.resetSimulation();p.setGravity(0,0,-9.81);p.setTimeStep(DT)
    p.setPhysicsEngineParameter(numSolverIterations=180,contactBreakingThreshold=.001)
    # Real bench (2175x550mm), four real-dimension hollow packing crates,
    # and 12 elevated handoff nests. Crates are NEVER robot destinations.
    make_static_geometry(p)
    ensure_assets()
    z_shift=LAYOUT.relative_world_origin_z_m
    plate_col=p.createCollisionShape(p.GEOM_BOX,halfExtents=[LAYOUT.scale_width_m/2,.10,.006])
    plate_vis=p.createVisualShape(p.GEOM_BOX,halfExtents=[LAYOUT.scale_width_m/2,.10,.006],rgbaColor=[.68,.70,.73,1])
    plate=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=plate_col,baseVisualShapeIndex=plate_vis,
                            basePosition=[0,0,G.plate_z-.006+z_shift])
    robot=p.loadURDF(str(HERE/'robot_v14_phase30.urdf'),[0,0,z_shift],useFixedBase=True,
                      flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
    joints={p.getJointInfo(robot,i)[1].decode('utf-8'):i for i in range(p.getNumJoints(robot))}
    q0=q_at(0)
    initial={'stage_x_move':0.,'stage_y_move':0.,'stage_z_move':G.root_z}
    for k in range(3):
       for j,v in zip(('shoulder','elbow','wrist'),q0):
          initial[f'{j}_{k}' if j!='wrist' else f'wrist_{k}_joint']=float(v)
    for name,val in initial.items():p.resetJointState(robot,joints[name],val)
    rgb,mass=APPEARANCES[label]
    ox,oy=cfg.offset_xy
    apple=spawn_textured_apple(p,label,mass*cfg.mass_scale,cfg.apple_radius_m,
          [ox,oy,G.plate_z+cfg.apple_radius_m+.0002+z_shift],cfg.apple_friction)
    p.changeDynamics(plate,-1,lateralFriction=.85)
    for k in range(3):
       p.changeDynamics(robot,joints[f'wrist_{k}_joint'],lateralFriction=1.4,
                        contactStiffness=2000,contactDamping=30)
    # No deep crates in v2.2. Each label gets a separate OPEN handoff tray
    # adjacent to a human packing station. Rimless support permits the 3R
    # fingers to retract sideways without touching neighboring fruit.
    trays={}
    tray_floors={}
    preloads={}
    nest_x={grade:[LAYOUT.nest_centres_x_m[f'{grade}{i}'] for i in (1,2,3)]
            for grade in 'ABCD'}
    for grade in 'ABCD':
      for slot,cx in enumerate(nest_x[grade],start=1):
        name=f'{grade}{slot}'
        cy=0.
        clr={'A':[.24,.63,.29,1],'B':[.93,.77,.25,1],'C':[.88,.31,.28,1],'D':[.36,.66,.77,1]}[grade]
        # Same nominal tray upper-floor datum (0.040+z_shift) as v3.2;
        # a real concave collision mesh rises 15mm at the rim, NOT a flat box.
        tray=create_cushion(p,grade,cx,cy,.040+z_shift)
        tray_floors[name]=tray
        trays[name]=(cx,cy)
        if cfg.preloaded_bin==name:
            old=spawn_textured_apple(p,grade,.18,.040,
                            [cx,cy,.083+z_shift],friction=1.)
            preloads[name]=old
    return robot,apple,plate,joints,trays,tray_floors,preloads


def control(p,robot,joints,goals,cfg:TrialConfig=TrialConfig()):
    for name,v in goals.items():
       jj=joints[name]
       stage=name.startswith('stage')
       m=motion_settings(cfg)
       limit=(.6 if cfg.motion_profile=='baseline' else (m.x_max_velocity if name=='stage_x_move' else (.5 if name=='stage_z_move' else 1.0)))
       p.setJointMotorControl2(robot,jj,p.POSITION_CONTROL,targetPosition=float(v),
           force=500 if stage else cfg.torque_Nm,maxVelocity=limit if stage else 6,
           positionGain=.40 if stage else .65,velocityGain=1.)

def wrist_target_with_close_duration(t,cfg:TrialConfig):
    """Stage16: reparameterize the SAME radial camera poses and grasp end pose.
    Alter scan duration only; retain final close time, IK, physical contact checks.
    Wrist is driven by PyBullet motors, never teleported to a target pose.
    """
    if cfg.close_duration_s<=0:raise ValueError('close duration must be positive')
    if not .28 <= cfg.scan_duration_s <= G.scan_s:
        raise ValueError('stage16 scan duration must be between .28 and .52 s')
    t=float(max(0.,min(cfg.scan_duration_s+cfg.close_duration_s,t)))
    if t<=cfg.scan_duration_s:
        original_t=G.scan_s*t/cfg.scan_duration_s
    else:
        original_t=G.scan_s+G.close_s*(t-cfg.scan_duration_s)/cfg.close_duration_s
    return commanded_wrist(original_t)

def grip_goals(t,cfg:TrialConfig=TrialConfig()):
    """Identical 3R finger structure. Alter the final inward closure by at most mm.
    Positive relief => camera/gripper moves slightly OUTWARD during last 0.08 s.
    No joint is teleported; target joint positions come from analytic IK.
    """
    t=min(cfg.scan_duration_s+cfg.close_duration_s,max(0,t))
    r,z=wrist_target_with_close_duration(t,cfg)
    if t>cfg.scan_duration_s:
        frac=smootherstep((t-cfg.scan_duration_s)/cfg.close_duration_s)
        r += cfg.grip_relief_mm/1000.0 * frac
    q=ik(r,z)
    d={}
    for k in range(3):
        d[f'shoulder_{k}']=float(q[0]);d[f'elbow_{k}']=float(q[1]);d[f'wrist_{k}_joint']=float(q[2])
    return d

def feedback_grip_goals(cfg:TrialConfig, correction_mm):
    """Per-finger 3R IK correction after scan, based ONLY on PyBullet contact feedback.
    correction_mm < 0 closes inward; > 0 relaxes outward. Never reads apple ground-truth radius or offset.
    """
    r,z=commanded_wrist(G.scan_s+G.close_s)
    base=cfg.grip_relief_mm/1000.0
    d={}
    for k in range(3):
        q=ik(r+base+float(correction_mm[k])/1000.,z)
        d[f'shoulder_{k}']=float(q[0])
        d[f'elbow_{k}']=float(q[1])
        d[f'wrist_{k}_joint']=float(q[2])
    return d


def per_finger_wrist_forces(p,robot,apple,joints):
    ids={joints[f'wrist_{k}_joint']:k for k in range(3)}
    force=[0.,0.,0.]
    for cp in p.getContactPoints(robot,apple):
        k=ids.get(cp[3])
        if k is not None:force[k]+=max(0.,float(cp[9]))
    return force

def feedback_step(force_N, correction_mm, mode):
    """One measured contact feedback update in mm. Does NOT use object GT pose."""
    if mode == 'force_feedback_v2':
        if force_N < 1.6: return max(-5.0, correction_mm - .50)
        if force_N > 5.0: return min(1.5, correction_mm + .25)
    else:
        if force_N < 1.6: return max(-3.0, correction_mm - .25)
        if force_N > 5.0: return min(1.5, correction_mm + .15)
    return correction_mm


def release_joint_goals(cfg, correction_mm, progress):
    """For v2, interpolate CONTINUOUSLY from the measured-contact grasp pose to open pose."""
    a=smootherstep(progress)
    grasp=feedback_grip_goals(cfg,correction_mm)
    opened=grip_goals(0,cfg)
    return {name:float((1.-a)*grasp[name]+a*opened[name]) for name in grasp}


def extra_open_joint_goals(clearance_mm: float):
    """Move each existing finger radially OUTWARD after nominal release.

    Positive millimetres are additional camera-wrist clearance, not robot lift.
    Zero retains the original scan-start fully-open pose. No joints are added.
    This is a candidate motor command, NOT proof of physical fruit clearance.
    """
    if not 0.0 <= clearance_mm <= 12.0:
        raise ValueError('clearance_mm must be 0..12')
    r,z=commanded_wrist(0)
    q=ik(r+clearance_mm/1000.0,z)
    return {name:float(q[j]) for k in range(3) for j,name in enumerate((
        f'shoulder_{k}',f'elbow_{k}',f'wrist_{k}_joint'))}


def finger_goal_settled(p, robot, joints, goals, pos_tolerance_rad=.025, velocity_tolerance_rad_s=.16):
    """Bullet encoders must match commanded fully-open finger pose before Z retreat.

    Merely observing zero contacts for 8 frames does not guarantee gripper
    actuators have stopped: Stage10 recontacted at the start of Z retreat.
    """
    finger_names=[name for name in goals if name.startswith(('shoulder_', 'elbow_', 'wrist_'))]
    if len(finger_names)!=9:
        return False
    states=p.getJointStates(robot,[joints[name] for name in finger_names])
    return all(abs(float(state[0])-float(goals[name])) <= pos_tolerance_rad and
               abs(float(state[1])) <= velocity_tolerance_rad_s
               for name,state in zip(finger_names,states))


def clearance_accept(contact_count,supported,center_error_m,speed_m_s,stable_steps,
                     extra_open_mm,min_extra_open_mm,fingers_settled):
    return (extra_open_mm >= min_extra_open_mm and bool(fingers_settled) and
            contact_gate_accept(contact_count,supported,center_error_m,speed_m_s,stable_steps))


def contact_gate_accept(robot_contact_count, supported, center_error_m, speed_m_s, stable_steps):
    """Safety-oriented release gate using simulated observations, no time-only pass."""
    return (int(robot_contact_count)==0 and bool(supported) and
            float(center_error_m)<.020 and float(speed_m_s)<.10 and
            int(stable_steps)>=8)


def images_from_actual_links(p,robot,joints,side=144):
    frames=[]
    for k in range(3):
       _,_,_,_,position,orientation=p.getLinkState(robot,joints[f'finger_camera_{k}_fixed'],computeForwardKinematics=True)
       pos=np.array(position);R=np.array(p.getMatrixFromQuaternion(orientation)).reshape(3,3)
       frames.append((f'finger_{k+1}',pos,pos+R@np.array([.20,0,0]),R@np.array([0,0,1])))
    _,_,_,_,hp,hq=p.getLinkState(robot,joints['stage_z_move'],computeForwardKinematics=True)
    hp,hq=p.getLinkState(robot,joints['palm_camera_fixed'],computeForwardKinematics=True)[4:6]
    hp=np.array(hp);R=np.array(p.getMatrixFromQuaternion(hq)).reshape(3,3)
    palm=hp
    frames.append(('palm',palm,palm+R@np.array([0,0,-.30]),R@np.array([0,1,0])))
    eye=np.array([0.,0.,G.plate_z+.0005+LAYOUT.relative_world_origin_z_m]);frames.append(('plate',eye,eye+np.array([0,0,.30]),np.array([0.,1.,0.])))
    out=[]
    for name,eye,target,up in frames:
       vm=p.computeViewMatrix(eye.tolist(),target.tolist(),up.tolist())
       proj=p.computeProjectionMatrixFOV(fov=G.camera_fov_deg,aspect=1,nearVal=.005,farVal=.8)
       _,_,rgb,_,_=p.getCameraImage(side,side,viewMatrix=vm,projectionMatrix=proj,renderer=p.ER_TINY_RENDERER)
       out.append((name,np.asarray(rgb,dtype=np.uint8).reshape(side,side,4)[:,:,:3]))
    return out

def contacts(p,robot,apple,joints):
    ids={joints[f'wrist_{k}_joint'] for k in range(3)}
    pts=p.getContactPoints(robot,apple)
    active={cp[3] for cp in pts if cp[3] in ids and cp[9]>.002}
    force=sum(max(0.,cp[9]) for cp in pts if cp[3] in ids)
    return len(active),float(force),len(pts)

def post_return_placement_check(actual_xyz, dest_xy, has_support, speed_m_s, *, tolerance_m=.020):
    """Validate placement AFTER final home return, not just before the retreat.

    Must operate only on physically sampled apple pose/contact/velocity. Returns
    (pass, reason, xy_error_m). No object teleportation or saved pre-return flag.
    """
    err=math.hypot(float(actual_xyz[0])-float(dest_xy[0]),
                   float(actual_xyz[1])-float(dest_xy[1]))
    if err>=tolerance_m:return False,'POST_RETURN_POSITION_DRIFT',err
    if not has_support:return False,'POST_RETURN_LOST_SUPPORT',err
    if float(speed_m_s)>=.10:return False,'POST_RETURN_FRUIT_STILL_MOVING',err
    return True,'POST_RETURN_OK',err


def place_next_fruit_on_scale(p, label, cfg, plate):
    """Worker A input abstraction: create NEW fruit only; never reposition existing fruit.

    This is NOT simulated human motor motion. Bullet world, robot and all
    previously delivered fruits remain active across episodes.
    """
    if label not in APPEARANCES:
        raise ValueError('unknown fruit appearance label')
    if p.getContactPoints(bodyA=plate):
        # There should be no body-supported fruit left on the input plate.
        # Contact checks exclude proximity-only checks and may require rest state.
        plate_contacts=[cp for cp in p.getContactPoints(bodyA=plate)
                        if cp[2]!=plate and float(cp[9])>.05]
        if plate_contacts:
            raise RuntimeError('INPUT_PLATE_STILL_OCCUPIED')
    rgb,mass=APPEARANCES[label]
    ox,oy=cfg.offset_xy
    apple=spawn_textured_apple(p,label,mass*cfg.mass_scale,cfg.apple_radius_m,
                [ox,oy,G.plate_z+cfg.apple_radius_m+.0002
                 +LAYOUT.relative_world_origin_z_m],cfg.apple_friction)
    return apple


def inspect_retained_fruits(p, scene):
    """Inspect ALL previously placed fruit after each subsequent robot return.

    Direct fruit coordinates are used ONLY by the post-run verifier, never to
    select a destination or control gripper motion.
    """
    out=[]
    for name,apple_id in scene['placed_fruits'].items():
        floor=scene['tray_floors'][name]
        xyz=p.getBasePositionAndOrientation(apple_id)[0]
        vel=float(np.linalg.norm(p.getBaseVelocity(apple_id)[0]))
        supported=bool(p.getContactPoints(apple_id,floor))
        ok,reason,dxy=post_return_placement_check(xyz,scene['trays'][name],supported,vel)
        cup_ok,cup_error=capture_in_cup(scene['trays'][name],xyz,supported,vel)
        out.append({'nest':name,'apple_body_id':apple_id,'ok':bool(ok),
                    'cup_capture_provisional':bool(cup_ok),
                    'cup_capture_error_m':round(cup_error,6),
                    'cup_capture_tolerance_m':.030,
                    'operational_center_tolerance_m':.020,
                    'status':reason,'xy_error_m':round(dxy,6),
                    'supported':bool(supported),'speed_m_s':round(vel,6)})
    return out


def run_trial(p,label,out,record_video=False,cfg:TrialConfig=TrialConfig(),
              *,persistent_scene=None,return_scene=False):
    loc=out/f'{label}_{cfg.slug}';loc.mkdir(parents=True,exist_ok=True)
    if persistent_scene is None:
        robot,apple,plate,joints,trays,tray_floors,preloads=load_scene(p,label,cfg)
        scene={'robot':robot,'plate':plate,'joints':joints,
               'trays':trays,'tray_floors':tray_floors,'preloads':preloads,
               'placed_fruits':{}}
    else:
        scene=persistent_scene
        robot,plate,joints=scene['robot'],scene['plate'],scene['joints']
        trays,tray_floors,preloads=scene['trays'],scene['tray_floors'],scene['preloads']
        apple=place_next_fruit_on_scale(p,label,cfg,plate)
    if persistent_scene is not None and not return_scene:
        raise ValueError('persistent runs must return their scene to verify history')
    # Episode stopwatch starts at placement; scene resets between episodes.
    # This is NOT one uninterrupted ten-apple physical run.
    wall_started=time.perf_counter()
    n=0;history=[];captured=[];scale_mass=None
    policy=motion_settings(cfg)
    diagnostics={'lift_contact_samples':0,'lift_samples':0,'lift_multi_contact_samples':0,
        'transport_samples':0,'transport_multi_contact_samples':0,'max_pad_normal_force_N':0.,
        'max_single_contact_normal_N':0.,'max_joint_tracking_error_rad':0.,
        'max_stage_tracking_error_m':0.,'collision_points_not_in_pads':0,
        'max_non_wrist_link_contact_force_N':0., 'max_wrist_contact_normal_force_N':0.,
        'max_retained_fruit_robot_contact_N':0., 'retained_fruit_robot_contact_steps':0,
        'first_previous_fruit_contact':None,'max_previous_fruit_contact_detail':None,
        'max_previous_fruit_contact_by_nest_N':{},'closest_previous_fruit_approach':None,
        'max_gripper_applied_motor_torque_Nm':0.,'motor_near_limit_samples':0,
        'motor_effort_samples':0, 'stage_force_max_N':0.0, 'peak_force_by_stage_N':{}, 'feedback_correction_final_mm':[0.,0.,0.], 'feedback_end_wrist_forces_N':[0.,0.,0.]}
    force_samples=[]
    last_goals={}
    stage_steps={}
    recorder = None
    focus_target = None
    if record_video:
        from video_capture import VideoCapture
        recorder = VideoCapture(p, loc / 'physics_trial.mp4', label, apple_body_id=apple, z_shift=LAYOUT.relative_world_origin_z_m)
    def send(goals):
       last_goals.clear();last_goals.update(goals)
       control(p,robot,joints,goals,cfg)
    def step_and_trace(stage):
       nonlocal n
       p.stepSimulation();n+=1
       cc,ff,nn=contacts(p,robot,apple,joints)
       observer=globals().get('_STAGE_CONTACT_OBSERVER')
       if observer is not None:
           observer(p,robot,apple,plate,joints,stage,n*DT,cfg)
       if return_scene:
           for _nest,_prior in scene['placed_fruits'].items():
               _prev_contacts=p.getContactPoints(robot,_prior)
               _prev_force=sum(max(0.,float(cp[9])) for cp in _prev_contacts)
               if stage in ('move_to_bin','destination_arrival_verify','gentle_lower') and n%4==0:
                   nearby=p.getClosestPoints(robot,_prior,distance=.055)
                   if nearby:
                       near=min(nearby,key=lambda c:float(c[8]))
                       rec={'nest':_nest,'stage':stage,'time_s':round(n*DT,5),
                            'gap_m':round(float(near[8]),5),'link_index':int(near[3])}
                       prev=diagnostics['closest_previous_fruit_approach']
                       if prev is None or rec['gap_m']<prev['gap_m']:
                           diagnostics['closest_previous_fruit_approach']=rec
               if _prev_contacts:
                   bynest=diagnostics['max_previous_fruit_contact_by_nest_N']
                   bynest[_nest]=max(float(bynest.get(_nest,0)),_prev_force)
                   cp=max(_prev_contacts,key=lambda c:float(c[9]))
                   li=int(cp[3])
                   detail={
                     'nest':_nest,'stage':stage,'time_s':round(n*DT,5),
                     'total_contact_force_N':round(_prev_force,5),
                     'robot_link_index':li,
                     'robot_link_name':(p.getJointInfo(robot,li)[12].decode('utf8') if li>=0 else 'base'),
                     'robot_x_m':round(float(p.getJointState(robot,joints['stage_x_move'])[0]),5),
                     'robot_z_m':round(float(p.getJointState(robot,joints['stage_z_move'])[0]),5),
                     'new_apple_xyz':[round(float(x),5) for x in p.getBasePositionAndOrientation(apple)[0]],
                     'previous_apple_xyz':[round(float(x),5) for x in p.getBasePositionAndOrientation(_prior)[0]]}
                   if diagnostics['first_previous_fruit_contact'] is None:
                       diagnostics['first_previous_fruit_contact']=detail
                   if _prev_force>diagnostics['max_retained_fruit_robot_contact_N']:
                       diagnostics['max_previous_fruit_contact_detail']=detail
               diagnostics['max_retained_fruit_robot_contact_N']=max(
                   diagnostics['max_retained_fruit_robot_contact_N'],_prev_force)
               if _prev_force>.05:
                   diagnostics['retained_fruit_robot_contact_steps']+=1
       # Bullet reports contacts by link, not by geometric primitive. Each wrist
       # link contains both the 18mm orange pad and a wrist housing; this is a
       # WRIST-LINK aggregate, NOT a measured fruit pressure or pad-only force.
       wrist_ids={joints[f'wrist_{k}_joint'] for k in range(3)}
       other_force=sum(max(0.,c[9]) for c in p.getContactPoints(robot,apple) if c[3] not in wrist_ids)
       diagnostics['max_non_wrist_link_contact_force_N']=max(diagnostics['max_non_wrist_link_contact_force_N'],float(other_force))
       diagnostics['max_wrist_contact_normal_force_N']=max(diagnostics['max_wrist_contact_normal_force_N'],float(ff))
       diag_stage=diagnostics['peak_force_by_stage_N']
       diag_stage[stage]=max(diag_stage.get(stage,0.0),float(ff))
       if n%4==0: # every 16.7ms: motor effort/saturation diagnostic
           jnames=[f'{axis}_{k}' if axis!='wrist' else f'wrist_{k}_joint' for k in range(3) for axis in ('shoulder','elbow','wrist')]
           torque_states=p.getJointStates(robot,[joints[j] for j in jnames])
           applied=[abs(float(z[3])) for z in torque_states]
           diagnostics['max_gripper_applied_motor_torque_Nm']=max(diagnostics['max_gripper_applied_motor_torque_Nm'],max(applied))
           diagnostics['motor_near_limit_samples']+=int(any(a>cfg.torque_Nm*.98 for a in applied))
           diagnostics['motor_effort_samples']+=1
       if stage in ('scan_and_close','contact_settle','contact_feedback','gravity_lift','move_to_bin','supported_release'):
        force_samples.append(float(ff))
        diagnostics['max_pad_normal_force_N']=max(diagnostics['max_pad_normal_force_N'],float(ff))
        ids={joints[f'wrist_{k}_joint'] for k in range(3)}
        vals=[cp[9] for cp in p.getContactPoints(robot,apple) if cp[3] in ids]
        if vals:diagnostics['max_single_contact_normal_N']=max(diagnostics['max_single_contact_normal_N'],max(vals))
       if stage=='gravity_lift':
        diagnostics['lift_samples']+=1
        diagnostics['lift_contact_samples']+=int(cc>=1)
        diagnostics['lift_multi_contact_samples']+=int(cc>=2)
        # Non-wrist link contacts, not the old incorrect count: total points - active pads.
       diagnostics['collision_points_not_in_pads']+=sum(1 for cp in p.getContactPoints(robot,apple) if cp[3] not in wrist_ids)
       if stage=='move_to_bin':
        diagnostics['transport_samples']+=1
        diagnostics['transport_multi_contact_samples']+=int(cc>=2)
       if n%12==0:
        for jname,goal in last_goals.items():
         measured=p.getJointState(robot,joints[jname])[0]
         k='max_stage_tracking_error_m' if jname.startswith('stage') else 'max_joint_tracking_error_rad'
         diagnostics[k]=max(diagnostics[k],abs(float(measured)-float(goal)))
       stage_steps[stage]=stage_steps.get(stage,0)+1
       if recorder and n%12==0:
        recorder.add_frame(n*DT, stage, focus_target)
       if n%12==0:
        xyz=p.getBasePositionAndOrientation(apple)[0]
        c,f,all_contacts=cc,ff,nn
        history.append({'time_s':round(n*DT,6),'stage':stage,'apple_x':xyz[0],
                        'apple_y':xyz[1],'apple_z':xyz[2],'contact_pad_count':c,
                        'pad_normal_force_N':f,'all_robot_contacts':all_contacts})
    # Realistic handshake: empty-tray interlock BEFORE gripping the next
    # apple. Conservative: wait until all twelve handoff nests are empty.
    # This is a modeled operational interlock, NOT a safety-rated human presence detector.
    # Worker removal is an explicitly modeled *external action*, never a
    # robot pick and never a fruit coordinate teleport.
    init=grip_goals(0,cfg);init.update({'stage_x_move':0,'stage_y_move':0,'stage_z_move':G.root_z})
    station_history=[]
    worker_removed=[]
    cleared=False
    last_states={}
    wait_max=max(policy.preflight_steps,int(cfg.max_worker_wait_s/DT))
    for tick in range(wait_max+1):
        send(init);step_and_trace('station_preflight' if tick<policy.preflight_steps else 'wait_for_worker')
        if cfg.human_clear_after_s is not None and n*DT>=cfg.human_clear_after_s:
            for tray_name,body_id in list(preloads.items()):
                # Deliberate abstraction of worker C removing this fruit.
                # NOT a simulated robot operation, and NOT counted as pick.
                p.removeBody(body_id)
                del preloads[tray_name]
                worker_removed.append({'tray':tray_name,'time_s':round(n*DT,5),
                                       'external_actor':'worker_C','mode':'abstract_scene_removal'})
        if tick<policy.preflight_steps-1:
            continue  # allow the initial dynamic sphere to settle onto tray
        states={key:tray_occupied(p,floor,robot) for key,floor in tray_floors.items()}
        last_states=states
        ready=(tick>=policy.preflight_steps-1) if return_scene else station_ready(states)
        if tick==policy.preflight_steps-1 or tick%24==0 or ready:
            station_history.append({'time_s':round(n*DT,5), 'occupied':states})
        if ready:
            cleared=True
            break
    if not cleared:
        result={'label_synthetic_ground_truth':label,'status':'PAUSED_WAITING_FOR_WORKER',
           'success_physical_and_ai':False,'sort_pass':False,
           'simulated_time_s':round(n*DT,5),'wall_clock_time_including_rendering_and_ai_s':round(time.perf_counter()-wall_started,4),
           'meets_3s_simulated_clock':False,
           'real_hardware_cycle_3s_confirmed':False,
           'classification':{'status':'NOT_STARTED','class':None},
           'placement':{'status':'WAIT_TRAYS_NOT_EMPTY','verified_support':False},
           'station_history':station_history,'worker_removal_events':worker_removed,
           'station_states_end':last_states,
           'experiment':{'profile':cfg.profile,'replicate':cfg.replicate,
              'motor_joint_torque_limit_Nm':cfg.torque_Nm,'grip_relief_mm':cfg.grip_relief_mm,
              'transport_duration_target_s':cfg.transport_s,'control_mode':cfg.control_mode,
              'close_duration_s':cfg.close_duration_s,'scan_duration_s':cfg.scan_duration_s,'preloaded_bin':cfg.preloaded_bin,
              'human_clear_after_s':cfg.human_clear_after_s},
           'message':'Robot remained stopped; existing fruit must be removed by worker C before scanning next fruit.'}
        (loc/'trial.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        if recorder: recorder.close(False,n*DT)
        return (result,scene) if return_scene else result
    for _ in range(policy.scale_settle_steps):send(init);step_and_trace('scale_settle')
    plate_norm=sum(max(0,cp[9]) for cp in p.getContactPoints(apple,plate))
    scale_mass=plate_norm/9.81 if plate_norm>0 else None
    # Preserve three full 4-camera scan observations at equivalent wrist poses.
    scan_capture_ticks=tuple(round(k*cfg.scan_duration_s/G.scan_s) for k in (42,84,124))
    assert len(set(scan_capture_ticks))==3
    for i in range(math.ceil((cfg.scan_duration_s+cfg.close_duration_s)/DT)+1):
        t=min(i*DT,cfg.scan_duration_s+cfg.close_duration_s)
        send(grip_goals(t,cfg));step_and_trace('scan_and_close')
        if i in scan_capture_ticks:
            for name,frame in images_from_actual_links(p,robot,joints)[:4]:
                captured.append(frame)
                if record_video: Image.fromarray(frame).save(loc/f'{name}_scan_{i:03}.png')
    # Allow motors to reach the final pose briefly, without moving apple directly.
    for _ in range(policy.contact_settle_steps):send(grip_goals(cfg.scan_duration_s+cfg.close_duration_s,cfg));step_and_trace('contact_settle')
    correction_mm=[0.,0.,0.]
    if cfg.control_mode in ('force_feedback','force_feedback_v2'):
        # Keep physical simulation running for another 0.1667 seconds to acquire
        # three measured contacts. This 0.1667 s is INCLUDED in simulated cycle time.
        # The contact is a Bullet wrist-link aggregate, not fruit-pressure measurement.
        for t in range(policy.feedback_steps):
            if t%4==0:
                forces=per_finger_wrist_forces(p,robot,apple,joints)
                for k,force_k in enumerate(forces):
                    correction_mm[k]=feedback_step(force_k,correction_mm[k],cfg.control_mode)
            send(feedback_grip_goals(cfg,correction_mm))
            step_and_trace('contact_feedback')
    diagnostics['feedback_correction_final_mm']=list(correction_mm)
    diagnostics['feedback_end_wrist_forces_N']=per_finger_wrist_forces(p,robot,apple,joints)
    padcount,force,_=contacts(p,robot,apple,joints)
    z0=p.getBasePositionAndOrientation(apple)[0][2]
    for i in range(policy.lift_steps):
        s=smootherstep(i/(policy.lift_steps-1))
        cmd=(feedback_grip_goals(cfg,correction_mm) if cfg.control_mode in ('force_feedback','force_feedback_v2') else grip_goals(cfg.scan_duration_s+cfg.close_duration_s,cfg));cmd['stage_z_move']=G.root_z+.100*s
        send(cmd);step_and_trace('gravity_lift')
    # Accelerated profiles must physically reach the same 100mm travel height.
    # A command alone is not a clearance guarantee. Additional time is counted.
    if cfg.motion_profile != 'baseline':
        _lift_goal=G.root_z+.100
        for _wait in range(96):
            _z_actual=float(p.getJointState(robot,joints['stage_z_move'])[0])
            _apple_z=float(p.getBasePositionAndOrientation(apple)[0][2])
            if _z_actual>=_lift_goal-.005 and _apple_z-z0>=.090:
                break
            _g=(feedback_grip_goals(cfg,correction_mm) if cfg.control_mode in ('force_feedback','force_feedback_v2') else grip_goals(cfg.scan_duration_s+cfg.close_duration_s,cfg))
            _g['stage_z_move']=_lift_goal
            send(_g);step_and_trace('lift_clearance_verify')
        else:
            raise RuntimeError('LIFT_CLEARANCE_NOT_REACHED: 100mm command not followed; horizontal travel aborted')
    z1=p.getBasePositionAndOrientation(apple)[0][2]
    lift_pass=(z1-z0)>.05 and diagnostics['lift_multi_contact_samples']>=int(diagnostics['lift_samples']*.30)
    frames=images_from_actual_links(p,robot,joints)
    for name,frame in frames:
        captured.append(frame)
        if record_video:Image.fromarray(frame).save(loc/f'{name}_lift.png')
    valid_rgb_frames=sum(1 for img in captured if rgb_features(img) is not None)
    learned=train();result=infer(learned,captured,scale_mass)
    if return_scene:
        sensor_states={key:tray_occupied(p,body,robot) for key,body in tray_floors.items()}
        # Pick the first available nest based on actual Bullet support-sensor input.
        # Grade comes ONLY from the image+mass inference `result`.
        grade=result.get('class')
        # Stage23 v4.3: after actual learned camera+scale inference, an optional
        # operator supervisor checks the PHYSICAL 12-nest load sensors.  Only when
        # same-grade three nests are full does it pause motion, request an
        # explicitly scripted worker-C removal, recheck sensor, and reset.
        # This hook never changes IK, grasp, lower, release or home motion.
        _s23_hook=globals().get('STAGE23_AFTER_INFERENCE_HOOK')
        if _s23_hook is not None:
            def _s23_hold_step():
                _g=(feedback_grip_goals(cfg,correction_mm) if cfg.control_mode in
                    ('force_feedback','force_feedback_v2') else
                    grip_goals(cfg.scan_duration_s+cfg.close_duration_s,cfg))
                _g.update({'stage_x_move':0.,'stage_y_move':0.,
                           'stage_z_move':G.root_z+.100})
                send(_g)
                step_and_trace('stage23_full_pause_hold')
                _count,_,_=contacts(p,robot,apple,joints)
                return _count
            _s23_hook(p=p,scene=scene,grade=grade,inference=result,
                      robot=robot,apple=apple,hold_step=_s23_hold_step,
                      pre_states=dict(sensor_states))
            sensor_states={key:tray_occupied(p,body,robot)
                           for key,body in tray_floors.items()}
        dest=next_free_nest(grade,sensor_states) if isinstance(grade,str) and grade in 'ABCD' else None
    else:
        dest=(f"{result['class']}{cfg.tray_slot}" if result.get('class') in 'ABCD' else None)
    # The AI CLASS decides grade, and configured slot chooses 1/2/3 of that grade. Each recipient
    # is a single-fruit open support tray rather than a deep container.
    # Occupancy is a separate contact-load sensor, NOT inferred by AI.
    placement={'status':'NOT_ATTEMPTED','verified_support':False,
               'support_contact_count':0,'impact_vertical_speed_m_s':None,
               'max_floor_contact_N':0.,'settled_on_floor':False}
    tray_states={key:tray_occupied(p,body,robot) for key,body in tray_floors.items()}
    if dest in trays and placement_allowed(dest,tray_states):
       tx,ty=trays[dest]
       focus_target=(tx,ty)
       # Respect X motor command maxVelocity + actual encoder arrival feedback.
       move_duration=max(cfg.transport_s if cfg.motion_profile=='baseline' else .28,abs(tx)/policy.x_effective_rate)
       move_steps=max(2,round(move_duration/DT))
       for i in range(move_steps):
           v=smootherstep(i/(move_steps-1))
           g=(feedback_grip_goals(cfg,correction_mm) if cfg.control_mode in ('force_feedback','force_feedback_v2') else grip_goals(cfg.scan_duration_s+cfg.close_duration_s,cfg))
           g.update({'stage_x_move':tx*v,'stage_y_move':ty*v,'stage_z_move':G.root_z+.100})
           send(g);step_and_trace('move_to_bin')
       if cfg.motion_profile != 'baseline':
           # Speedup is allowed only if the moving gripper has actually arrived.
           # Count arrival settling instead of omitting motor-following delay.
           _hold=(feedback_grip_goals(cfg,correction_mm) if cfg.control_mode in ('force_feedback','force_feedback_v2') else grip_goals(cfg.scan_duration_s+cfg.close_duration_s,cfg))
           _hold.update({'stage_x_move':tx,'stage_y_move':ty,'stage_z_move':G.root_z+.100})
           # v4.0b: for the Stage20 tripod diagnostic only, never start Z lowering
           # based on position alone. Observe ACTUAL X velocity, apple velocity,
           # at least 2 wrist contacts, and continuous settling for 10 steps.
           _arrival_stable=0
           _last_arrival={}
           for _ in range(192):
               send(_hold);step_and_trace('destination_arrival_verify')
               _x_joint=p.getJointState(robot,joints['stage_x_move'])
               _x_actual=float(_x_joint[0]);_x_speed=float(_x_joint[1])
               _fruit_xyz=p.getBasePositionAndOrientation(apple)[0]
               _fruit_vel=p.getBaseVelocity(apple)[0]
               _grip_count,_,_=contacts(p,robot,apple,joints)
               _pos_error=abs(_x_actual-tx)
               _fruit_rel_x=float(_fruit_xyz[0])-_x_actual
               _last_arrival={'x_error_m':_pos_error,'stage_x_speed_m_s':_x_speed,
                   'apple_vx_m_s':float(_fruit_vel[0]),'fruit_relative_x_m':_fruit_rel_x,
                   'grip_links':_grip_count,'consecutive_stable_steps':_arrival_stable}
               # original safety requirements are kept and strengthened, not bypassed
               _stable=(_pos_error<=.005 and abs(_x_speed)<=.06 and
                        abs(float(_fruit_vel[0]))<=.08 and
                        abs(_fruit_rel_x)<=.018 and _grip_count>=2)
               _arrival_stable=(_arrival_stable+1) if _stable else 0
               if _arrival_stable>=10:
                   break
           else:
               placement['status']='ABORT_DESTINATION_NOT_STABLE'
               placement['arrival_guard_diagnostic']=_last_arrival
               raise RuntimeError('DESTINATION_NOT_STABLE: ' + json.dumps(_last_arrival))
       # Lower while maintaining grasp. Do not open until the crate physically
       # supports the apple. Floor contact is a simulated tactile sensor.
       floor_id=tray_floors[dest]
       lowering_count=0
       impact_v=None
       floor_contact_peak=0.
       support_consecutive=0
       # Cartesian goal is converted into a REAL prismatic stage joint command.
       # A nominal 80mm apple resting on the crate floor has root_z~0.165m.
       # Force termination, rather than a fixed release height, is mandatory.
       z_start=G.root_z+.100
       z_min=.151  # greater than URDF joint lower limit 0.150m
       # Nominal first-contact joint height from baseline trial (m). The fruit
       # is NEVER released at this estimate; only real floor contacts allow that.
       contact_stage_nominal=.165
       _lower_contact_loss_frames=0
       for i in range(360):
           z_target=max(z_min,staged_lower_target(i*DT,z_start,contact_stage_nominal,policy))
           g=(feedback_grip_goals(cfg,correction_mm) if cfg.control_mode in ('force_feedback','force_feedback_v2') else grip_goals(cfg.scan_duration_s+cfg.close_duration_s,cfg))
           g.update({'stage_x_move':tx,'stage_y_move':ty,'stage_z_move':z_target})
           send(g)
           vz=float(p.getBaseVelocity(apple)[0][2])
           step_and_trace('gentle_lower')
           lowering_count+=1
           floor_contacts=p.getContactPoints(apple,floor_id)
           floor_force=sum(max(0.,float(cp[9])) for cp in floor_contacts)
           floor_contact_peak=max(floor_contact_peak,floor_force)
           # v4.0b: do not continue lowering if the under-support is lost
           # before the fruit reaches the cushion. This is a STOP, not a recovery.
           _current_stage_z=float(p.getJointState(robot,joints['stage_z_move'])[0])
           _lower_contacts,_,_=contacts(p,robot,apple,joints)
           if _current_stage_z>.185 and floor_force<=.05 and _lower_contacts<2:
               _lower_contact_loss_frames+=1
           else:
               _lower_contact_loss_frames=0
           if _lower_contact_loss_frames>=3:
               placement['status']='ABORT_LOWER_GRASP_SUPPORT_LOST'
               placement['lower_grip_contacts']=_lower_contacts
               sort_pass=False
               break
           # Require a short sustained contact, not one transient collision.
           if floor_contacts and floor_force>.05:
               if support_consecutive==0:
                   impact_v=abs(vz)  # first measured impact, not four steps later
               support_consecutive+=1
           else:
               support_consecutive=0
               impact_v=None
           if support_consecutive>=4:
               placement['verified_support']=True
               placement['support_contact_count']=len(floor_contacts)
               placement['status']='SUPPORT_CONTACT_DETECTED'
               break
       placement['max_floor_contact_N']=round(floor_contact_peak,4)
       placement['impact_vertical_speed_m_s']=round(float(impact_v),5) if impact_v is not None else None
       if placement['verified_support'] and impact_v is not None and impact_v>.15:
           placement['status']='ABORT_EXCESSIVE_IMPACT_SPEED'
           sort_pass=False
       elif placement['verified_support']:
           # Hold current measured stage height; open continuously from measured
           # finger feedback pose. No release high above contents.
           actual_z=float(p.getJointState(robot,joints['stage_z_move'])[0])
           for i in range(policy.release_steps):
               if cfg.control_mode=='force_feedback_v2':
                   g=release_joint_goals(cfg,correction_mm,i/(policy.release_steps-1))
               else:
                   t=(cfg.scan_duration_s+cfg.close_duration_s)*(1-smootherstep(i/(policy.release_steps-1)))
                   g=grip_goals(t,cfg)
               g.update({'stage_x_move':tx,'stage_y_move':ty,'stage_z_move':actual_z})
               send(g);step_and_trace('supported_release')
           # Wait until the fruit is REALLY supported and slow enough, with
           # >=policy.settle_min_steps observations; do not fake early success.
           # If not stable, continue through the full original 0.5 s window.
           for settle_tick in range(policy.settle_max_steps):
               step_and_trace('settle_in_bin')
               if settle_tick+1 < policy.settle_min_steps: continue
               _xyz=p.getBasePositionAndOrientation(apple)[0]
               _vel=np.linalg.norm(p.getBaseVelocity(apple)[0])
               _floor=bool(p.getContactPoints(apple,floor_id))
               if (_floor and _vel < .08 and
                   math.hypot(_xyz[0]-tx,_xyz[1]-ty)<.020):
                   break
           x,y,z=p.getBasePositionAndOrientation(apple)[0]
           dx=abs(x-tx);dy=abs(y-ty)
           still_supported=bool(p.getContactPoints(apple,floor_id))
           v_final=np.linalg.norm(p.getBaseVelocity(apple)[0])
           placement['settled_on_floor']=still_supported
           placement['final_speed_m_s']=round(float(v_final),5)
           placement['status']='PLACED_ON_SUPPORT' if still_supported else 'LOST_SUPPORT_AFTER_RELEASE'
           # Simulated impact bound is only an experimental acceptance criterion;
           # fruit compression/bruising is NOT simulated.
           placement['impact_within_exploratory_bound']=bool(impact_v is not None and impact_v<=.15)
           sort_pass=(math.hypot(dx,dy)<.020 and .04<z-LAYOUT.relative_world_origin_z_m<.14 and
                      still_supported and v_final<.10 and placement['impact_within_exploratory_bound'])
       else:
           if placement['status']=='NOT_ATTEMPTED':
               placement['status']='ABORT_NO_SUPPORT_DETECTED'
           sort_pass=False
    else:
       placement['status']=('PAUSED_GRADE_THREE_NESTS_FULL' if return_scene and isinstance(result.get('class'),str) and result.get('class') in 'ABCD' and dest is None
                            else 'PAUSED_DESTINATION_TRAY_OCCUPIED' if dest in trays else 'ABORT_UNKNOWN_CLASS')
       sort_pass=False
    # Stage11: the Stage10 contact-only gate reported clearance but then
    # immediately recontacted during vertical retreat. For the NEW profiles,
    # insist on at least 9/12 mm further radial opening, actual finger encoder
    # convergence, low finger joint speeds, and eight stable contact-free steps.
    # Preserve legacy Stage10 gate for reproducible historical comparisons.
    gate_profiles=('contact_gate_39','contact_gate_50','settled_gate_09','settled_gate_12')
    settled_profiles=('settled_gate_09','settled_gate_12')
    if placement['status']=='PLACED_ON_SUPPORT' and cfg.motion_profile in gate_profiles:
        release_gate_log=[]
        release_safe=False
        extra_mm=0.
        min_extra_mm=(9. if cfg.motion_profile=='settled_gate_09' else
                      12. if cfg.motion_profile=='settled_gate_12' else 0.)
        # Do not continuously change Z holding target to measured Z while
        # waiting for fingers; that could make the whole wrist creep down.
        fixed_hold_z=float(p.getJointState(robot,joints['stage_z_move'])[0])
        for extra_mm in (0.,3.,6.,9.,12.):
            clear_count=0  # A new motor target invalidates prior stable samples.
            open_goal=extra_open_joint_goals(extra_mm)
            open_goal.update({'stage_x_move':tx,'stage_y_move':ty,
                              'stage_z_move':fixed_hold_z})
            # For new profiles move through small increments. Convergence may
            # take more than the old 24 steps; all this time is counted.
            max_steps=(96 if cfg.motion_profile in settled_profiles and
                       extra_mm>=min_extra_mm else
                       12 if cfg.motion_profile in settled_profiles else 24)
            for k in range(max_steps):
                send(open_goal);step_and_trace('release_contact_clearance')
                pos=p.getBasePositionAndOrientation(apple)[0]
                speed=float(np.linalg.norm(p.getBaseVelocity(apple)[0]))
                err=math.hypot(pos[0]-tx,pos[1]-ty)
                cps=p.getContactPoints(robot,apple)
                is_supported=bool(p.getContactPoints(apple,floor_id))
                settled=(finger_goal_settled(p,robot,joints,open_goal)
                         if cfg.motion_profile in settled_profiles else True)
                if len(cps)==0 and settled:
                    clear_count+=1
                else:
                    clear_count=0
                accepted=clearance_accept(len(cps),is_supported,err,speed,clear_count,
                                           extra_mm,min_extra_mm,settled)
                if k%12==0 or accepted:
                    release_gate_log.append({'extra_open_mm':extra_mm,
                        't_s':round(n*DT,5),'contact_count':len(cps),
                        'support':is_supported,'center_error_mm':round(err*1000,3),
                        'speed_m_s':round(speed,5),'fingers_settled':settled,
                        'clear_count':clear_count,
                        'z_hold_error_mm':round((float(p.getJointState(robot,joints['stage_z_move'])[0])-fixed_hold_z)*1000,3)})
                if err>=.020 or not is_supported:
                    placement['status']='ABORT_RELEASE_MOVED_FRUIT'
                    break
                if accepted:
                    release_safe=True
                    break
            if release_safe or placement['status']!='PLACED_ON_SUPPORT':break
        placement['release_gate']={'contact_free_before_lift':release_safe,
            'minimum_open_mm':min_extra_mm,
            'actual_open_mm':extra_mm,
            'finger_encoder_verified':bool(release_safe and cfg.motion_profile in settled_profiles),
            'final_center_error_mm':round(math.hypot(
              p.getBasePositionAndOrientation(apple)[0][0]-tx,
              p.getBasePositionAndOrientation(apple)[0][1]-ty)*1000,3),
            'observations':release_gate_log,'extra_open_max_mm':extra_mm}
        if not release_safe:
            if placement['status']=='PLACED_ON_SUPPORT':
                placement['status']='ABORT_RELEASE_CONTACT_PERSISTED'
            sort_pass=False
    # Return towards the scale after placement, with joints actually driven by Bullet.
    # Return is part of the cycle; failure is recorded, never assumed away.
    if placement['status']=='PLACED_ON_SUPPORT':
       actual_x=float(p.getJointState(robot,joints['stage_x_move'])[0])
       # A released apple can be struck when X starts before the empty fingers
       # have lifted clear. First retract VERTICALLY in place, keeping X fixed.
       # This real Bullet joint motion is counted in cycle time.
       _retreat=(extra_open_joint_goals(extra_mm) if cfg.motion_profile in ('contact_gate_39','contact_gate_50','settled_gate_09','settled_gate_12') else grip_goals(0,cfg))
       _retreat.update({'stage_x_move':actual_x,'stage_y_move':ty,
                        'stage_z_move':G.root_z+.100})
       cleared=False
       for _retry in range(144):
           send(_retreat);step_and_trace('vertical_clearance_before_return')
           if cfg.motion_profile in ('contact_gate_39','contact_gate_50','settled_gate_09','settled_gate_12') and p.getContactPoints(robot,apple):
               # Do not silently classify a recontact as success. Report
               # involved link IDs and normal loads for follow-up diagnosis.
               _rc=p.getContactPoints(robot,apple)
               placement['retreat_recontact']={
                   'at_s':round(n*DT,6),'step_index':int(_retry),
                   'contact_count':len(_rc),
                   'max_contact_force_N':round(max((max(0.,float(c[9])) for c in _rc),default=0.),5),
                   'contact_links':[int(c[3]) for c in _rc],
                   'apple_xy_error_mm':round(math.hypot(
                       p.getBasePositionAndOrientation(apple)[0][0]-tx,
                       p.getBasePositionAndOrientation(apple)[0][1]-ty)*1000,4)}
               placement['status']='ABORT_RETREAT_RECONTACT'
               sort_pass=False
               break
           _z_actual=float(p.getJointState(robot,joints['stage_z_move'])[0])
           if _z_actual>=G.root_z+.095:
               cleared=True
               break
       if not cleared:
           if placement['status']=='PLACED_ON_SUPPORT':
               placement['status']='ABORT_RETURN_CLEARANCE_NOT_REACHED'
           sort_pass=False
       else:
           ret_steps=max(2,round(max(.20,abs(actual_x)/policy.x_effective_rate*float(__import__("os").environ.get("STAGE22_V42_RETURN_SCALE","1.0")))/DT))
           for i in range(ret_steps):
               u=smootherstep(i/(ret_steps-1))
               g=grip_goals(0,cfg)
               g.update({'stage_x_move':actual_x*(1-u),
                         'stage_y_move':0.,'stage_z_move':G.root_z+.100})
               send(g);step_and_trace('return_to_scale')
           if cfg.motion_profile != 'baseline':
               _home=grip_goals(0,cfg)
               _home.update({'stage_x_move':0.,'stage_y_move':0.,'stage_z_move':G.root_z+.100})
               for _ in range(96):
                   if abs(float(p.getJointState(robot,joints['stage_x_move'])[0]))<.030: break
                   send(_home);step_and_trace('home_arrival_verify')
    home_x=float(p.getJointState(robot,joints['stage_x_move'])[0])
    home_ok=abs(home_x)<.065
    post_return_check=None
    if placement.get('verified_support') and dest in trays and placement['status'] not in ('ABORT_RELEASE_CONTACT_PERSISTED','ABORT_RELEASE_MOVED_FRUIT','ABORT_RETREAT_RECONTACT'):
       actual_apple_pos=p.getBasePositionAndOrientation(apple)[0]
       actual_apple_speed=float(np.linalg.norm(p.getBaseVelocity(apple)[0]))
       actual_final_support=bool(p.getContactPoints(apple,tray_floors[dest]))
       post_ok,post_reason,post_xy_err=post_return_placement_check(
           actual_apple_pos,trays[dest],actual_final_support,actual_apple_speed)
       cup_ok,cup_err=capture_in_cup(trays[dest],actual_apple_pos,
                                          actual_final_support,actual_apple_speed)
       post_return_check={'checked_after_home_return':True,'ok':post_ok,
            'cup_capture_provisional':bool(cup_ok),
            'cup_capture_error_m':round(cup_err,6),
            'cup_capture_tolerance_m':.030,
            'operational_center_tolerance_m':.020,
            'fruit_nominal_radius_m':float(cfg.apple_radius_m),
            'nest_halfwidth_m':float(LAYOUT.nest_width_m/2),
            'full_footprint_inside_nest':bool(
                abs(actual_apple_pos[0]-trays[dest][0])+cfg.apple_radius_m<=LAYOUT.nest_width_m/2 and
                abs(actual_apple_pos[1]-trays[dest][1])+cfg.apple_radius_m<=.050),
            'status':post_reason,'apple_position_xyz_m':list(actual_apple_pos),
            'target_xy_m':list(trays[dest]),'target_xy_error_m':round(post_xy_err,6),
            'on_assigned_support':actual_final_support,
            'apple_speed_m_s':round(actual_apple_speed,6)}
       if not post_ok:
           placement['status']=post_reason
           sort_pass=False
    report={'world_geometry':{'bench_length_m':LAYOUT.bench_length_m,'crate_centres_x_m':LAYOUT.crate_centres_x_m,'nest_centres_x_m':LAYOUT.nest_centres_x_m,'nest_height_m':LAYOUT.nest_top_z_m,'crate_rim_height_m':LAYOUT.rim_z_m,'scale_width_m':LAYOUT.scale_width_m,'raised_height_unverified':True},'label_synthetic_ground_truth':label,'scale_mass_kg_measured':scale_mass,
       'camera_frames_captured':len(captured),'camera_frames_with_detectable_fruit':valid_rgb_frames,
       'contact_pad_count_before_lift':padcount,'normal_force_N_before_lift':force,
       'apple_lift_delta_m':z1-z0,'lift_pass':lift_pass,'classification':result,
       'sort_pass':sort_pass,'placement':placement,'post_return_check':post_return_check,'return_home_pass':home_ok,
       'stage_x_home_final_m':home_x,'assigned_nest':dest,'station_preflight_s':round((sum(stage_steps.get(k,0) for k in ('station_preflight','wait_for_worker')))*DT,5),
       'station_history':station_history,'worker_removal_events':worker_removed,'station_states_before_placement':tray_states,'success_physical_and_ai':bool(lift_pass and sort_pass and home_ok and padcount>=2 and result.get('class')==label),
       'simulated_time_s':n*DT,'simulation_stage_durations_s':{k:round(v*DT,5) for k,v in stage_steps.items()},
       'wall_clock_time_including_rendering_and_ai_s':round(time.perf_counter()-wall_started,4),
       'meets_3s_simulated_clock':n*DT<=3,
       'meets_2_to_4_target':2<=n*DT<=4,
       'motion_profile':cfg.motion_profile,'lift_commanded_height_m':.100,
       'motor_x_max_velocity_m_s':policy.x_max_velocity,
       'stage13_visual_mesh':'UV sphere + A/B/C/D png textures',
       'stage13_nest_collision_mesh':'15mm concave static trimesh',
       'stage13_cushion_is_soft_body':False,
       'cushion_depth_m':CUP_DEPTH_M,'cushion_diameter_m':CUP_DIAMETER_M,
       'strict_20mm_placement_check':True,
       'physical_contact_support_gate_retained':True,
       'real_hardware_cycle_3s_confirmed':False,'no_drop_release':bool(placement['verified_support']),'diagnostics':diagnostics,'lift_criterion':'delta_z>50mm AND >=2 distinct gripping contacts in >=30% of measured lift steps',
        'experiment':{'profile':cfg.profile,'replicate':cfg.replicate,'motor_joint_torque_limit_Nm':cfg.torque_Nm,
           'grip_relief_mm':cfg.grip_relief_mm, 'transport_duration_target_s':cfg.transport_s,'control_mode':cfg.control_mode,
           'close_duration_s':cfg.close_duration_s,'scan_duration_s':cfg.scan_duration_s,'preloaded_bin':cfg.preloaded_bin,
           'human_clear_after_s':cfg.human_clear_after_s,'motion_profile':cfg.motion_profile,'apple_mass_scale':cfg.mass_scale,'apple_friction':cfg.apple_friction,
           'apple_radius_m':cfg.apple_radius_m,'initial_xy_offset_m':list(cfg.offset_xy),'effective_mass_kg':APPEARANCES[label][1]*cfg.mass_scale},
       'force_warning_exceeds_exploratory_20N':diagnostics['max_pad_normal_force_N']>20.,
       'exploratory_20N_is_a_fruit_safety_limit':False,
       'transport_multi_contact_fraction':diagnostics['transport_multi_contact_samples']/max(1,diagnostics['transport_samples']),
       'force_mean_N':float(np.mean(force_samples)) if force_samples else None,
       'force_p95_N':float(np.percentile(force_samples,95)) if force_samples else None,
       'no_apple_teleportation':True,'mechanism':'v3.3: old workbench rig, 12 static concave cushion meshes, four UV-appearance fruit types; 3R and 5 cameras maintained'  ,
       'contact_force_qualification':'Normal force is SUM of contacts on each wrist link. Wrist includes pad AND housing. Not pure pad force, pressure, or a fruit-safety claim.' }
    if return_scene:
        report['persistent_world']=True
        report['existing_fruits_before']=len(scene['placed_fruits'])
        if report['success_physical_and_ai'] and dest is not None:
            scene['placed_fruits'][dest]=apple
        retained=inspect_retained_fruits(p,scene)
        report['retained_nests_after_return']=retained
        report['all_retained_fruits_ok']=bool(all(x['ok'] for x in retained))
        if diagnostics['retained_fruit_robot_contact_steps']:
            report['success_physical_and_ai']=False
            report['status']='ROBOT_COLLISION_WITH_PREVIOUS_FRUIT'
        if not report['all_retained_fruits_ok']:
            report['success_physical_and_ai']=False
            report['status']='PREVIOUS_FRUIT_DISPLACED'
        report['placed_fruit_count']=len(scene['placed_fruits'])
    (loc/'trial.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,default=float),encoding='utf-8')
    with (loc/'history.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
    if recorder: recorder.close(bool(report['success_physical_and_ai']), report['simulated_time_s'])
    return (report,scene) if return_scene else report

def build_trial_plan(labels, torques, profiles, grip_reliefs_mm, transport_times, modes, repeats, replicate_ids=None, close_durations_s=(.08,)):
    """Return *every* requested trial as (label, config); no dependency on PyBullet.

    v1.7 mistakenly executed only the last repetition. Keeping this as a pure
    function lets tests enforce the complete 42-trial plan before physics runs.
    """
    reps = list(range(repeats)) if replicate_ids is None else list(replicate_ids)
    if not reps or min(reps) < 0 or len(reps) != len(set(reps)):
        raise ValueError('replicate IDs must be a nonempty sequence of distinct nonnegative integers')
    plan=[]
    for label, torque, profile, relief, move_time, mode, close_dur, rep in itertools.product(
            labels, torques, profiles, grip_reliefs_mm, transport_times, modes, close_durations_s, reps):
        cfg=TrialConfig(torque_Nm=torque, profile=profile, replicate=rep,
                        grip_relief_mm=relief, transport_s=move_time, control_mode=mode, close_duration_s=close_dur)
        plan.append((label,cfg))
    keys=[f'{label}_{cfg.slug}' for label,cfg in plan]
    if len(keys)!=len(set(keys)):
        raise ValueError('repeated experiment configurations would overwrite results')
    return plan


def main():
    ap=argparse.ArgumentParser(description='Stage03 v2.3 12-nest four-class physics episode')
    ap.add_argument('--labels',nargs='+',choices=['A','B','C','D'],default=['A','B','C','D'])
    ap.add_argument('--torques',nargs='+',type=float,default=[12.],help='JOINT TORQUE limits in N*m; not pad force')
    ap.add_argument('--grip-reliefs-mm',nargs='+',type=float,default=[.75,1.50])
    ap.add_argument('--transport-times',nargs='+',type=float,default=[.70])
    ap.add_argument('--close-durations',nargs='+',type=float,default=[.08],help='Final inward closure duration, seconds (.08,.12,.16), same scan path')
    ap.add_argument('--profiles',nargs='+',choices=PROFILES,default=['stress_mix'])
    ap.add_argument('--repeats',type=int,default=1)
    ap.add_argument('--replicate-ids',type=int,nargs='+',default=None,
                    help='optional specific IDs, e.g. --replicate-ids 3 5 for problem cases')
    ap.add_argument('--control-modes',nargs='+',choices=['fixed','force_feedback','force_feedback_v2'],default=['fixed'])
    ap.add_argument('--gui',action='store_true')
    ap.add_argument('--record-nominal',action='store_true')
    ap.add_argument('--record-chosen',action='store_true')
    ap.add_argument('--preloaded-bin','--preloaded-tray',dest='preloaded_bin',choices=list(TRAY_NAMES),default='',help='handoff tray initially occupied by an apple')
    ap.add_argument('--human-clear-after-s',type=float,default=None,help='worker C empties preloaded tray after N simulated seconds; explicit external event')
    ap.add_argument('--max-worker-wait-s',type=float,default=2.,help='finite simulation window for a blocked workcell')
    ap.add_argument('--dry-plan',action='store_true',help='validate and print planned trials without PyBullet')
    ap.add_argument('--output',default='output/physics_stage03')
    args=ap.parse_args()
    if args.repeats<1 or any(x<=0 or x>50 for x in args.torques) or any(x<0 or x>3 for x in args.grip_reliefs_mm) or any(x<.45 or x>2 for x in args.transport_times) or any(x<.08 or x>.16 for x in args.close_durations):
        ap.error('invalid repeats/torques/grip-reliefs/transport-times')
    try:
        plan=build_trial_plan(args.labels,args.torques,args.profiles,args.grip_reliefs_mm,
                              args.transport_times,args.control_modes,args.repeats,args.replicate_ids,args.close_durations)
    except ValueError as exc: ap.error(str(exc))
    if args.human_clear_after_s is not None and (args.human_clear_after_s<0 or not args.preloaded_bin):
        ap.error('human clear time requires --preloaded-tray and time >= 0')
    if args.max_worker_wait_s <= 0: ap.error('max worker wait must be positive')
    from dataclasses import replace
    plan=[(label,replace(cfg,preloaded_bin=args.preloaded_bin,
                    human_clear_after_s=args.human_clear_after_s,
                    max_worker_wait_s=args.max_worker_wait_s)) for label,cfg in plan]
    expected=[f'{label}_{cfg.slug}' for label,cfg in plan]
    print('PLANNED_TRIALS:',len(plan))
    print('REPLICATE_IDS:',sorted({cfg.replicate for _,cfg in plan}))
    if args.dry_plan:
        for key in expected: print('PLAN',key)
        return
    out=HERE/args.output;out.mkdir(parents=True,exist_ok=True)
    (out/'trial_manifest.json').write_text(json.dumps({'planned_count':len(plan), 'planned_trial_ids':expected},indent=2),encoding='utf-8')
    try: import pybullet as p
    except ImportError as exc:
        (out/'execution_status.json').write_text(json.dumps({'real_pybullet_was_executed':False,'status':'ENGINE_UNAVAILABLE','error':str(exc)},indent=2),encoding='utf-8')
        raise SystemExit(2)
    client=p.connect(p.GUI if args.gui else p.DIRECT)
    if client<0:raise RuntimeError('PyBullet connection failed')
    p.setRealTimeSimulation(0)
    all_trials=[]
    try:
        for idx,(label,cfg) in enumerate(plan,1):
            rep=cfg.replicate
            try:
                record=((args.record_nominal and cfg.torque_Nm==12. and cfg.profile=='nominal' and cfg.grip_relief_mm==0. and abs(cfg.transport_s-.7)<1e-9 and rep==0)
                    or (args.record_chosen and cfg.control_mode in ('force_feedback','force_feedback_v2') and cfg.torque_Nm==12. and cfg.profile=='nominal' and abs(cfg.grip_relief_mm-1.5)<1e-9 and abs(cfg.transport_s-.7)<1e-9 and rep==0 and abs(cfg.close_duration_s-.12)<1e-9))
                row=run_trial(p,label,out,record_video=record,cfg=cfg)
                print(f'[{idx}/{len(plan)}] {label}/{cfg.slug} success={row.get("success_physical_and_ai")} wrist_normal_sum={row.get("diagnostics",{}).get("max_pad_normal_force_N",0):.2f}N')
            except Exception as exc:
                loc=out/f'{label}_{cfg.slug}';loc.mkdir(parents=True,exist_ok=True)
                row={'label_synthetic_ground_truth':label,'experiment':{'profile':cfg.profile,'replicate':rep,
                     'motor_joint_torque_limit_Nm':cfg.torque_Nm,'grip_relief_mm':cfg.grip_relief_mm,
                     'transport_duration_target_s':cfg.transport_s,'control_mode':cfg.control_mode,
                     'close_duration_s':cfg.close_duration_s,'scan_duration_s':cfg.scan_duration_s,'preloaded_bin':cfg.preloaded_bin,
              'human_clear_after_s':cfg.human_clear_after_s},
                     'status':'ERROR','error':repr(exc),'traceback':traceback.format_exc(),
                     'success_physical_and_ai':False,'meets_3s_simulated_clock':False}
                (loc/'error.json').write_text(json.dumps(row,indent=2,ensure_ascii=False),encoding='utf-8')
                print('ERROR',label,cfg.slug,repr(exc))
            all_trials.append(row)
            write_report(out,all_trials)
        report=write_report(out,all_trials)
        def recorded_id(x):
            try:
                e=x['experiment']
                c=TrialConfig(torque_Nm=e['motor_joint_torque_limit_Nm'],profile=e['profile'],
                              replicate=e['replicate'],grip_relief_mm=e['grip_relief_mm'],
                              transport_s=e['transport_duration_target_s'],
                              control_mode=e.get('control_mode','fixed'),close_duration_s=e.get('close_duration_s',.08),preloaded_bin=e.get('preloaded_bin',''),human_clear_after_s=e.get('human_clear_after_s'))
                return f"{x['label_synthetic_ground_truth']}_{c.slug}"
            except (KeyError,TypeError,ValueError):
                return '__INCOMPLETE_REPORT__'
        recorded=[recorded_id(x) for x in all_trials]
        missing=sorted(set(expected)-set(recorded))
        extras=sorted(set(recorded)-set(expected))
        coverage={'planned_count':len(expected),'recorded_count':len(recorded),'all_expected_conditions_recorded':not(missing or extras or len(recorded)!=len(expected)), 'missing_trial_ids':missing,'unexpected_trial_ids':extras, 'executed_replicate_ids':sorted({x['experiment']['replicate'] for x in all_trials}), 'success_n':report['overall_success_n']}
        (out/'coverage_audit.json').write_text(json.dumps(coverage,indent=2),encoding='utf8')
        print('COVERAGE_AUDIT',json.dumps(coverage,ensure_ascii=False))
        print('Completed',len(all_trials),'trials; review',out/'benchmark_summary.csv')
        print('NO real-apple safety or real hardware 3-second guarantee.')
        if not coverage['all_expected_conditions_recorded']:
            raise SystemExit(3)
        if report['overall_success_n']<report['trial_count']:
            raise SystemExit(1)
    finally: p.disconnect()

def aggregate_rows(trials):
    groups={}
    for row in trials:
        exp=row.get('experiment',{})
        key=(exp.get('motor_joint_torque_limit_Nm'),exp.get('profile'),exp.get('grip_relief_mm'),exp.get('transport_duration_target_s'),exp.get('control_mode','fixed'),exp.get('close_duration_s',.08))
        groups.setdefault(key,[]).append(row)
    out=[]
    for (torque,profile,relief,move_s,mode,close_s),rows in sorted(groups.items(),key=lambda kv:(-float(kv[0][0] or 0),kv[0][1] or '',kv[0][2] or 0.,kv[0][3] or 0.,kv[0][4])):
        n=len(rows)
        good=[x for x in rows if x.get('success_physical_and_ai')]
        valid=[x for x in rows if x.get('diagnostics')]
        allmax=[x['diagnostics']['max_pad_normal_force_N'] for x in valid]
        with_time=[x for x in rows if x.get('meets_3s_simulated_clock') and x.get('success_physical_and_ai')]
        out.append({'torque_limit_Nm':torque,'profile':profile,'grip_relief_mm':relief,'transport_target_s':move_s,'control_mode':mode,'close_duration_s':close_s,'n':n,'success_n':len(good),
         'success_rate':len(good)/n,'success_within_3s_n':len(with_time),
         'success_within_3s_rate':len(with_time)/n,
         'peak_normal_force_max_N':max(allmax) if allmax else None,
         'peak_normal_force_median_N':statistics.median(allmax) if allmax else None,
         'mean_wall_clock_s':statistics.mean(x['wall_clock_time_including_rendering_and_ai_s'] for x in rows if x.get('wall_clock_time_including_rendering_and_ai_s') is not None) if any(x.get('wall_clock_time_including_rendering_and_ai_s') is not None for x in rows) else None,
         'engine_executed':True})
    return out

def write_report(out,trials):
    out.mkdir(parents=True,exist_ok=True)
    groups=aggregate_rows(trials)
    for name,rows in [('all_trials.csv',[flatten_trial(x) for x in trials]),('benchmark_summary.csv',groups)]:
        if not rows:continue
        with (out/name).open('w',encoding='utf8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'real_pybullet_was_executed':bool(trials),'trial_count':len(trials),'error_count':sum(x.get('status')=='ERROR' for x in trials),
      'results':groups,'overall_success_n':sum(x.get('success_physical_and_ai',False) for x in trials),
      'important':'v2.2 open handoff tray is NOT validated in PyBullet here. Load-contact sensor interlock, human removal event, descent, support-release require physics validation. Nm is joint torque; fruit damage not simulated. ' +
                  'No claimed safe-force threshold. Varying initial conditions does not prove field reliability.'}
    (out/'acceptance.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    return summary

def flatten_trial(row):
    exp=row.get('experiment',{});d=row.get('diagnostics',{})
    return {'label':row.get('label_synthetic_ground_truth'),'torque_limit_Nm':exp.get('motor_joint_torque_limit_Nm'),
      'grip_relief_mm':exp.get('grip_relief_mm'),'transport_target_s':exp.get('transport_duration_target_s'),'control_mode':exp.get('control_mode','fixed'),
      'close_duration_s':exp.get('close_duration_s',.08),
      'profile':exp.get('profile'),'replicate':exp.get('replicate'),'success':row.get('success_physical_and_ai',False),
      'lift_pass':row.get('lift_pass'),'sort_pass':row.get('sort_pass'),
      'class_predicted':(row.get('classification') or {}).get('class'),
      'max_pad_normal_force_N':d.get('max_pad_normal_force_N'),
      'transport_peak_normal_force_N':(d.get('peak_force_by_stage_N') or {}).get('move_to_bin'),
      'max_non_wrist_link_contact_force_N':d.get('max_non_wrist_link_contact_force_N'),
      'max_gripper_applied_motor_torque_Nm':d.get('max_gripper_applied_motor_torque_Nm'),
      'motor_near_limit_fraction':(d.get('motor_near_limit_samples',0) / max(1,d.get('motor_effort_samples',0))),
      'max_single_contact_normal_N':d.get('max_single_contact_normal_N'),
      'force_p95_N':row.get('force_p95_N'),'simulated_time_s':row.get('simulated_time_s'),
      'peak_scan_N':(d.get('peak_force_by_stage_N') or {}).get('scan_and_close'),
      'peak_feedback_N':(d.get('peak_force_by_stage_N') or {}).get('contact_feedback'),
      'peak_lift_N':(d.get('peak_force_by_stage_N') or {}).get('gravity_lift'),
      'peak_release_N':(d.get('peak_force_by_stage_N') or {}).get('release'),
      'feedback_correction_final_mm':json.dumps(d.get('feedback_correction_final_mm')), 'feedback_end_wrist_forces_N':json.dumps(d.get('feedback_end_wrist_forces_N')),
      'wall_clock_s':row.get('wall_clock_time_including_rendering_and_ai_s'),
      'meets_3s_simulated_clock':row.get('meets_3s_simulated_clock'),'placement_status':(row.get('placement') or {}).get('status'), 'support_verified':(row.get('placement') or {}).get('verified_support'), 'landing_impact_speed_m_s':(row.get('placement') or {}).get('impact_vertical_speed_m_s'), 'preinspection_time_s':row.get('station_preflight_s'), 'error':row.get('error')}

if __name__=='__main__':main()
