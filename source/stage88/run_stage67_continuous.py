#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage51: post-object-gate scan/load-settling overlap with Stage50 landing, retaining Stage48 A2 scenario.

Reuses the Stage39 *real* camera-inference -> physical grasp/transport core.
Worker A/C hands and the supply baskets are visibly scripted geometric proxies.
Human intrusion guard uses a virtual palm-to-hand distance, NOT safety hardware.
Worker C's fruit handling is a scripted kinematic transfer, NOT human grasp physics.
Output is one unspliced 1.0x MP4 plus machine-readable evidence. No fake AI grades.
This is a Colab physics experiment; no pass is claimed prior to execution.
"""
from __future__ import annotations
import argparse, csv, datetime, hashlib, json, math, os, sys, time, zipfile, importlib.util
from dataclasses import replace
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
import joblib

import run_stage35_motion as s35
import object_gate_ai as object_ai
import visual_grade_ai as grade_ai
import ball_scene_assets as ball_ai
from stage67_camera_ai import Stage67CameraAI
from stage62_workcell_video import Stage62Recorder
from stage62_engine_patch import apply_stage62_engine_patch
from stage66_grasp_guard import apply_stage66_grasp_guard

ROOT=Path(__file__).resolve().parent
# In this one continuous demonstration, B3 is used first and B2 next.
# B1 must remain empty.  No unsupported backlog apple is inserted mid-film.
CASES=[('apple','A','A3'),('apple','B','B3'),('apple','C','C1'),
       ('apple','D','D1'),('apple','B','B2'),('apple','A','A2'),
       ('apple','A','A1'),('apple','A','A2'),('tennis','tennis',None),
       ('apple','C','C2')]
UTC9=datetime.timezone(datetime.timedelta(hours=9))
def stamp():return datetime.datetime.now(UTC9).strftime('%Y/%m/%d %H:%M:%S JST')
def save_json(path,thing):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(thing,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def ensure_engine(result_dir):
    if importlib.util.find_spec('pybullet') is None:
        raise RuntimeError('PHYSICS_NOT_EXECUTED: PyBullet missing; run in Colab')
    engine=s35.clone_engine(result_dir)
    s35.modify_recording_overlay(engine)
    # Exactly the successful Stage38/39 recording-camera fix in a cloned engine.
    f=engine/'apple_stage13'/'src'/'video_capture.py'
    t=f.read_text(encoding='utf8')
    old=("  if focus_xy is not None and stage in ('gentle_lower','supported_release','settle_in_bin'):\n"
         "   cx,cy=focus_xy\n"
         "   eye=[cx+.23,cy-.29,self.z_shift+.25];target=[cx,cy,self.z_shift+.075]\n"
         "  elif self.apple_body_id is not None:\n")
    if t.count(old)!=1:raise RuntimeError('CAMERA_PATCH_SOURCE_MISMATCH')
    u=t.replace(old,'  if self.apple_body_id is not None:\n',1)
    compile(u,str(f),'exec');f.write_text(u,encoding='utf8')
    save_json(result_dir/'engine_camera_manifest.json',{'stage39_stable_camera_patch':True,
        'original_sha256':hashlib.sha256(t.encode()).hexdigest(),
        'patched_sha256':sha(f),'robot_or_AI_controller_changed':False})
    # ONLY input staging: after the current apple has cleared the inspection
    # plate, A may put the NEXT physical fruit on it during B's transport.
    # The next run_trial consumes that exact body ID. Never spawn a duplicate.
    physics_file=engine/'apple_stage13'/'src'/'run_physics.py'
    pt=physics_file.read_text(encoding='utf-8')
    old='        apple=place_next_fruit_on_scale(p,label,cfg,plate)'
    assert pt.count(old)==1, 'STAGE44_INPUT_PATCH_ENGINE_SOURCE_MISMATCH'
    new=('        apple=scene.pop("stage44_prepositioned_input_body",None)\n'
         '        if apple is None:apple=place_next_fruit_on_scale(p,label,cfg,plate)')
    patched=pt.replace(old,new,1)
    compile(patched,str(physics_file),'exec')
    physics_file.write_text(patched,encoding='utf-8')
    save_json(result_dir/'stage44_input_engine_patch.json',{
        'unmodified_physics_engine_sha256':hashlib.sha256(pt.encode()).hexdigest(),
        'input_prefetch_only_patch_sha256':sha(physics_file),
        'original_input_spawn_replaced_once':True,
        'robot_grasp_controller_changed':False,'AI_changed':False})
    # Stage48: the OPERATOR chooses which occupied A nest to clear.
    # The robot's post-clear destination is still chosen from PHYSICAL nest-load
    # readings by next_free_nest(); no special A2 route is hardcoded into B.
    supervisor_path=engine/'apple_stage13'/'src'/'stage23_full_supervisor.py'
    old_sup=supervisor_path.read_text(encoding='utf-8')
    anchor="        if target not in scene['placed_fruits']:\n"
    assert old_sup.count(anchor)==1, 'STAGE48_SUPERVISOR_PATCH_POINT_MISMATCH'
    adjustment=(
        "        if grade=='A' and os.environ.get('STAGE48_WORKER_C_MIDDLE_A')=='1':\n"
        "            target='A2'\n"
        "            self._event('WORKER_C_SELECTS_MIDDLE_A2',grade,nest=target,\n"
        "                        operator_choice=True,robot_route_overridden=False)\n"
    )
    patched=old_sup.replace(anchor,adjustment+anchor,1)
    compile(patched,str(supervisor_path),'exec')
    supervisor_path.write_text(patched,encoding='utf-8')
    save_json(result_dir/'stage48_supervisor_patch.json',{
        'original_sha256':hashlib.sha256(old_sup.encode()).hexdigest(),
        'patched_sha256':sha(supervisor_path),
        'robot_route_controller_changed':False,
        'worker_C_A_full_clearing_choice':'A2',
        'post_clear_route_from_real_nest_load_sensor':True})
    _stage49_patch_motion_engine(engine,result_dir)
    _stage51_patch_scan_overlap(engine,result_dir)
    apply_stage62_engine_patch(engine,result_dir)
    apply_stage66_grasp_guard(engine,result_dir)
    return engine

def _stage51_patch_scan_overlap(engine,result_dir):
    """Stage51 corrected: retain the REAL Stage32 apple-vs-ball gate.

    Stage32 inserts its own gate *between* scale_mass and scan. Never delete
    or skip it. Probe plate load/fruit motion before the gate (max 16 ticks),
    require four consecutive valid contact ticks, and then run the unchanged
    Stage32 3-sample fail-closed gate. Only *after* its positive decision may
    spare stabilization ticks overlap open-finger scan commands. Photos remain
    at the original scan ticks, after the full stabilization window.
    """
    pf=engine/'apple_stage13'/'src'/'run_physics.py'
    before=pf.read_text(encoding='utf-8')
    # This is the actual Stage32 patched physical engine, not the older
    # Stage27 sequence. Hard guards prevent accidental removal of the gate.
    settle="    for _ in range(policy.scale_settle_steps):send(init);step_and_trace('scale_settle')\n"
    gate="    _stage32_gate_cb=globals().get('STAGE32_PREGRIP_GATE')\n"
    scan="    # Preserve three full 4-camera scan observations at equivalent wrist poses.\n"
    scan_loop="    for i in range(math.ceil((cfg.scan_duration_s+cfg.close_duration_s)/DT)+1):\n"
    if any(before.count(x)!=1 for x in (settle,gate,scan,scan_loop)):
        raise RuntimeError('STAGE51_V2_UNEXPECTED_STAGE32_ENGINE_LAYOUT')
    start=before.index(settle)
    gpos=before.index(gate)
    scanpos=before.index(scan)
    looppos=before.index(scan_loop)
    if not (start<gpos<scanpos<looppos):
        raise RuntimeError('STAGE51_V2_OBJECT_GATE_ORDER_UNEXPECTED')
    head=before[start:gpos]
    if 'plate_norm=sum(max(0,cp[9])' not in head or 'scale_mass=plate_norm/9.81' not in head:
        raise RuntimeError('STAGE51_V2_MISSING_ORIGINAL_LOAD_READ')
    preserved_gate=before[gpos:scanpos]
    # Protect the negative-case early return and its event provenance.
    essential=("_stage32_gate_cb", "_stage32_stable_load", "if not _stage32_gate_record['allow_descent']:",
               "STAGE32_REJECTED_BEFORE_DESCENT", "descent_executed':False", "return (rejected,scene)")
    if not all(s in preserved_gate for s in essential):
        raise RuntimeError('STAGE51_V2_GATE_REJECTION_CONTRACT_CHANGED')
    preserved_ticks=before[scanpos:looppos]
    if 'scan_capture_ticks=tuple' not in preserved_ticks:
        raise RuntimeError('STAGE51_V2_CAMERA_TIMING_MISSING')
    new_head=(
        "    # Stage51v2: measured readiness. Stage32 object gate STILL runs before finger scan.\n"
        "    _s51_probe_ticks=0\n"
        "    _s51_stable_contacts=0\n"
        "    _s51_early_load_stable=False\n"
        "    for _s51_j in range(int(policy.scale_settle_steps)):\n"
        "        send(init);step_and_trace('stage51_plate_probe')\n"
        "        _s51_probe_ticks+=1\n"
        "        _s51_nf=sum(max(0,cp[9]) for cp in p.getContactPoints(apple,plate))\n"
        "        _s51_speed=np.linalg.norm(p.getBaseVelocity(apple)[0])\n"
        "        _s51_omega=np.linalg.norm(p.getBaseVelocity(apple)[1])\n"
        "        if _s51_nf/9.81>=.025 and _s51_speed<=.08 and _s51_omega<=1.5:\n"
        "            _s51_stable_contacts+=1\n"
        "        else:\n"
        "            _s51_stable_contacts=0\n"
        "        if _s51_stable_contacts>=4:\n"
        "            _s51_early_load_stable=True\n"
        "            break\n"
        "    # Gate load and its two further samples are still real Bullet contacts.\n"
        "    plate_norm=sum(max(0,cp[9]) for cp in p.getContactPoints(apple,plate))\n"
        "    scale_mass=plate_norm/9.81 if plate_norm>0 else None\n"
    )
    post_gate=(
        "    # The unchanged Stage32 block above returns immediately for nonapples.\n"
        "    # Only here can the Stage51 finger motors advance.\n"
        "    _s51_gate_debounce_ticks=2 if _stage32_gate_cb is not None else 0\n"
        "    _s51_overlap=max(0,int(policy.scale_settle_steps)-_s51_probe_ticks-_s51_gate_debounce_ticks)\n"
        "    if _s51_overlap>=min(scan_capture_ticks):\n"
        "        raise RuntimeError('STAGE51_V2_EARLY_CAMERA_BEFORE_SETTLE')\n"
        "    for _s51_i in range(_s51_overlap):\n"
        "        _s51_t=min(_s51_i*DT,cfg.scan_duration_s+cfg.close_duration_s)\n"
        "        _s51_cmd=grip_goals(_s51_t,cfg)\n"
        "        _s51_cmd.update({'stage_x_move':0.,'stage_y_move':0.,'stage_z_move':G.root_z})\n"
        "        send(_s51_cmd);step_and_trace('scale_settle_scan_overlap')\n"
        "        if p.getContactPoints(robot,apple):\n"
        "            raise RuntimeError('STAGE51_V2_PREMATURE_ROBOT_APPLE_CONTACT')\n"
        "    # The original scan loop resumes at its correct camera-clock tick.\n"
    )
    # Keep Stage32 callback and its early return byte-identical.
    new_loop="    for i in range(_s51_overlap,math.ceil((cfg.scan_duration_s+cfg.close_duration_s)/DT)+1):\n"
    after=before[:start]+new_head+preserved_gate+preserved_ticks+post_gate+new_loop+before[looppos+len(scan_loop):]
    if after.count('STAGE32_REJECTED_BEFORE_DESCENT')!=1 or after.count('allow_descent')<1:
        raise RuntimeError('STAGE51_V2_OBJECT_GATE_NOT_PRESERVED')
    if after.count('for i in range(_s51_overlap,math.ceil(')!=1:
        raise RuntimeError('STAGE51_V2_SCAN_LOOP_DUPLICATED')
    compile(after,str(pf),'exec')
    pf.write_text(after,encoding='utf-8')
    save_json(result_dir/'stage51_engine_overlap_patch.json',{
        'date_jst':stamp(),'stage':51,'revision':'v2 actual Stage32 gate compatible',
        'pregrip_object_gate':'unchanged STAGE32_PREGRIP_GATE, never starts finger scan on rejection',
        'probe_rule':'up to original 16 ticks, four consecutive positive contact/velocity measurements',
        'if_stability_not_ready':'fall back to full 16-tick probe before unchanged gate',
        'early_probe_can_shift_gate_camera':'YES - potential classification change requires physical retest',
        'same_3_camera_capture_ticks':True,
        'overlap_ticks_dynamic':'max(0,16-probe_ticks-2_gate_debounce_ticks)',
        'max_possible_overlap_ticks':10,
        'no_camera_capture_before_16tick_combined_stabilization':True,
        'stage32_gate_block_preserved_byte_identical':True,
        'original_engine_sha256':hashlib.sha256(before.encode()).hexdigest(),
        'stage51_engine_sha256':sha(pf),
        'not_a_Z_axis_speed_increase':True,
        'physical_success_not_verified':True})

def _stage49_patch_motion_engine(engine,result_dir):
    pf=engine/'apple_stage13'/'src'/'run_physics.py'
    original=pf.read_text(encoding='utf-8')
    anchor='def grip_goals(t,cfg:TrialConfig=TrialConfig()):'
    assert original.count(anchor)==1,'STAGE49_WRIST_ANCHOR_CHANGED'
    wrist_code = '# Stage49 experiment: unified, C1-continuous camera scan + finger closure.\n# Physically commanded from the original three-finger 3R joints, not a render-only edit.\n# Preserve wrist START, SCAN MID, SCAN END, and final grasp geometry.\n_stage49_unmodified_wrist_target=wrist_target_with_close_duration\n\ndef _stage49_hermite_piece(v0,v1,d0,d1,dt,u):\n    u=max(0.,min(1.,float(u)))\n    uu=u*u;uuu=uu*u\n    return ((2*uuu-3*uu+1)*v0 + (uuu-2*uu+u)*dt*d0\n            + (-2*uuu+3*uu)*v1 + (uuu-uu)*dt*d1)\n\ndef wrist_target_with_close_duration(t,cfg:TrialConfig):\n    if abs(cfg.scan_duration_s-.28)>1e-9 or abs(cfg.close_duration_s-.08)>1e-9:\n        return _stage49_unmodified_wrist_target(t,cfg)\n    scan=float(cfg.scan_duration_s);close=float(cfg.close_duration_s)\n    t=min(scan+close,max(0.,float(t)))\n    tt=[0.,scan*.5,scan,scan+close]\n    rr=[G.scan_start_r,(G.scan_start_r+G.scan_end_r)*.5,G.scan_end_r,G.grasp_camera_r]\n    zz=[G.scan_start_z,(G.scan_start_z+G.scan_end_z)*.5,G.scan_end_z,G.grasp_camera_z]\n    dr=[0.,(rr[2]-rr[0])/scan,(rr[3]-rr[1])/(close+scan*.5),0.]\n    dz=[0.,(zz[2]-zz[0])/scan,0.,0.]\n    i=0 if t<=tt[1] else (1 if t<=tt[2] else 2)\n    d=tt[i+1]-tt[i]\n    u=(t-tt[i])/d\n    return (_stage49_hermite_piece(rr[i],rr[i+1],dr[i],dr[i+1],d,u),\n            _stage49_hermite_piece(zz[i],zz[i+1],dz[i],dz[i+1],d,u))\n\n'
    updated=original.replace(anchor,wrist_code+'\n'+anchor,1)
    old_init='    station_history=[]\n    worker_removed=[]\n'
    assert updated.count(old_init)==1,'STAGE49_PREFLIGHT_INIT_CHANGED'
    updated=updated.replace(old_init,'    station_history=[]\n    _stage49_stable_pose_ticks=0\n    worker_removed=[]\n',1)
    old_gate=('        if tick<policy.preflight_steps-1:\n'
        '            continue  # allow the initial dynamic sphere to settle onto tray\n')
    assert updated.count(old_gate)==1,'STAGE49_PREFLIGHT_GATE_CHANGED'
    preflight_code="        # Stage49: do not begin imaging/scan until measured robot pose is stable.\n        # The former fixed 96 ticks is replaced by dynamic 12-tick stable-state\n        # verification, with at least 24 ticks.  Unlike Stage36's 8-tick trial,\n        # early release based only on elapsed time is prohibited.\n        _stage49_pose_ok=True\n        for _axis,_command in init.items():\n            _state=p.getJointState(robot,joints[_axis])\n            _err=abs(float(_state[0])-float(_command))\n            _vel=abs(float(_state[1]))\n            if _axis.startswith('stage'):\n                _pos_limit=.0045 if _axis=='stage_z_move' else .0050\n                _vel_limit=.075\n            else:\n                _pos_limit=.10\n                _vel_limit=.65\n            if _err>_pos_limit or _vel>_vel_limit:\n                _stage49_pose_ok=False\n                break\n        _stage49_stable_pose_ticks=(_stage49_stable_pose_ticks+1\n                                   if _stage49_pose_ok else 0)\n        _stage49_station_ready=(tick>=23 and _stage49_stable_pose_ticks>=12)\n        if not _stage49_station_ready:\n            continue\n"
    updated=updated.replace(old_gate,preflight_code,1)
    old_ready='        ready=(tick>=policy.preflight_steps-1) if return_scene else station_ready(states)'
    assert updated.count(old_ready)==1,'STAGE49_PREFLIGHT_READY_CHANGED'
    updated=updated.replace(old_ready,'        ready=_stage49_station_ready if return_scene else station_ready(states)',1)
    compile(updated,str(pf),'exec')
    pf.write_text(updated,encoding='utf-8')
    save_json(result_dir/'stage49_motion_patch_manifest_preserved.json',{
        'date_jst':stamp(),'stage':50,
        'cloned_parent_physics_sha256':hashlib.sha256(original.encode()).hexdigest(),
        'experiment_physics_sha256':sha(pf),
        'scan_s_stage48':.32,'scan_s_stage49':.28,
        'close_s_stage49':.08,
        'scan_start_mid_end_pose_preserved':True,
        'grasp_final_pose_preserved':True,'four_camera_snapshots_per_trial':3,
        'original_12_nests_and_robot_state':'unchanged',
        'preflight_measured_position_velocity_minimum_steps':24,
        'preflight_required_stable_steps':12,
        'motion_timing':'actual PyBullet steps; no video speed change',
        'real_hardware_3s_per_fruit':'NOT_VERIFIED',
        'stage_Z_grasp_motion':'unchanged; only 3R finger motion is blended',
        'no_claim_of_physical_pass_before_colab':True})

def _stage40_overlay_font(size):
    # Colab runtimes do not guarantee any /usr/share/fonts/... pathname.
    # This is a presentation-only fallback; AI and robot control are untouched.
    candidates=(
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
        'DejaVuSans.ttf',
        'LiberationSans-Regular.ttf',
    )
    for pathname in candidates:
        try:
            return ImageFont.truetype(pathname,size)
        except (OSError,ValueError):
            continue
    # Pillow ships an embedded, size-configurable Unicode-capable font.
    try:
        return ImageFont.load_default(size=size)
    except (TypeError,OSError):
        return ImageFont.load_default()

class ChronologicalRecorder:
    """One writer, one simulation clock, same Bullet world, no splicing."""
    def __init__(self,p,out,fps=20,w=800,h=480):
        self.p=p;self.w=w;self.h=h;self.fps=fps;self.count=0;self.steps=0
        self.writer=imageio.get_writer(str(out),fps=fps,codec='libx264',quality=7)
        self.font=_stage40_overlay_font(18)
        self.small=_stage40_overlay_font(13)
        self.state='SETUP';self.pred='Awaiting camera';self.destination='—'
        self.occupied='—';self.input_count=0;self.proxy_note='Blue: Worker A | Green: Worker C | scripted non-colliding workers'
        self.date=stamp()
    def frame(self,phase,force=False):
        self.steps+=1
        if not force and self.steps%12!=0:return
        p=self.p;w=self.w;h=self.h
        view=p.computeViewMatrix(cameraEyePosition=[1.30,-2.05,2.22],
          cameraTargetPosition=[0.,.12,1.04],cameraUpVector=[0,0,1])
        proj=p.computeProjectionMatrixFOV(fov=64,aspect=w/h,nearVal=.02,farVal=6)
        _,_,rgba,_,_=p.getCameraImage(w,h,viewMatrix=view,projectionMatrix=proj,
                                     renderer=p.ER_TINY_RENDERER)
        a=np.asarray(rgba,dtype=np.uint8).reshape(h,w,4)[:,:,:3]
        im=Image.fromarray(a);d=ImageDraw.Draw(im)
        d.rectangle([0,0,w,77],fill=(12,27,47))
        d.text((12,7),'PAI APPLE WORK-CELL  |  continuous PyBullet physics',font=self.font,fill='white')
        d.text((12,34),f'CASE: {self.state}   PHASE: {phase}',font=self.small,fill=(212,223,244))
        d.text((12,54),f'AI: {self.pred}   ROUTE: {self.destination}   A baskets: 3 x 30; supplied: {self.input_count}',
               font=self.small,fill=(199,236,200))
        d.rectangle([0,h-45,w,h],fill=(12,27,47))
        d.text((10,h-40),self.proxy_note,font=self.small,fill=(255,222,168))
        d.text((10,h-21),f'{self.date}  |  1.0x  |  no edits/splicing  |  sim T={self.steps/240:.2f}s',
               font=self.small,fill=(210,218,235))
        self.writer.append_data(np.array(im));self.count+=1
    def close(self):self.writer.close()

class VisualActors:
    """No collision: visual representations are not anthropomorphic control evidence."""
    def __init__(self,p,rec):
        self.p=p;self.rec=rec;self.baskets=[];self.items=[];self.hands={};self.hand_parts={}
        self.setup()
    def body(self,kind,dimensions,rgba,pos):
        p=self.p
        if kind=='sphere':v=p.createVisualShape(p.GEOM_SPHERE,radius=dimensions[0],rgbaColor=rgba)
        else:v=p.createVisualShape(p.GEOM_BOX,halfExtents=dimensions,rgbaColor=rgba)
        return p.createMultiBody(baseMass=0,baseCollisionShapeIndex=-1,
                                 baseVisualShapeIndex=v,basePosition=pos)
    def _make_hand(self,which,pos):
        # Stage41 visual-only proxy. Palm is still the same-position body ID used
        # by the existing virtual proximity test; all extra parts have NO collision.
        side=1 if which=='A' else -1  # A reaches from rear, C from near side.
        skin=[.96,.78,.60,1]
        sleeve=[.13,.34,.76,1] if which=='A' else [.15,.56,.35,1]
        palm=self.body('box',[.057,.035,.014],skin,pos)
        self.hands[which]=palm
        parts=[]
        def piece(half,colour,delta):
            xyz=[pos[k]+delta[k] for k in range(3)]
            obj=self.body('box',half,colour,xyz)
            parts.append((obj,delta))
        # Solid, conspicuously coloured forearm and sleeve, not an apple sphere.
        piece([.036,.225,.038],sleeve,[0.,side*.350,.010])
        piece([.042,.035,.034],sleeve,[0.,side*.095,.010])
        # 4 slender fingers with visibly different lengths plus an offset thumb.
        for dx,length in ((-.040,.055),(-.014,.067),(.013,.062),(.040,.048)):
            piece([.009,length/2,.009],skin,[dx,-side*(.035+length*.43),-.006])
        piece([.018,.032,.010],skin,[.071,-side*.035,-.009])
        # Dark wrist stripe gives a hand/arm silhouette even in a wide shot.
        piece([.045,.009,.039],[.12,.17,.22,1],[0.,side*.112,.010])
        self.hand_parts[which]=parts
    def setup(self):
        # 90 apple proxies: 3 x 30 visually staged at foot level behind bench.
        for basket in range(3):
            bx=(basket-1)*.36;by=.85
            self.baskets.append(self.body('box',[.155,.155,.008],[.50,.34,.13,1],[bx,by,.05]))
            for wall in (-1,1):
                self.body('box',[.009,.155,.115],[.53,.35,.17,.8],[bx+wall*.148,by,.16])
                self.body('box',[.155,.009,.115],[.53,.35,.17,.8],[bx,by+wall*.148,.16])
            entries=[]
            for i in range(30):
                layer=i//9; k=i%9
                x=bx+(k%3-1)*.087;y=by+(k//3-1)*.087
                z=.095+layer*.082
                entries.append(self.body('sphere',[.039],[.77,.13+.018*(i%4),.09,1],[x,y,z]))
            self.items.append(entries)
        self._make_hand('A',[0.,.73,1.27])
        self._make_hand('C',[1.05,-.62,1.27])
    def place_hand(self,which,at):
        self.p.resetBasePositionAndOrientation(self.hands[which],at,[0,0,0,1])
        for part,delta in self.hand_parts[which]:
            self.p.resetBasePositionAndOrientation(
                part,[at[k]+delta[k] for k in range(3)],[0,0,0,1])
    def move_hand(self,which,start,end,steps,tag):
        for i in range(steps):
            a=(i+1)/steps
            xyz=[start[j]*(1-a)+end[j]*a for j in range(3)]
            self.place_hand(which,xyz)
            self.p.stepSimulation()
            self.rec.frame(tag)
    def next_input(self):
        idx=self.rec.input_count
        # 3 baskets of 30 visually staged. Each supplied item removed from display.
        basket=min(2,idx//30)
        slot=idx%30
        if idx>=90:raise RuntimeError('A_BASKET_INPUT_EXHAUSTED')
        self.p.removeBody(self.items[basket][slot]);self.rec.input_count+=1
    def a_supply(self):
        self.next_input()
        start=[0.,.73,1.27];end=[0.,.09,1.27]
        self.move_hand('A',start,end,18,'WORKER A: visual scripted supply')
        self.move_hand('A',end,start,18,'WORKER A: hand withdraws')
    def c_transfer(self,body,nest,grade,xy,crate_xy):
        p=self.p;z=1.28
        src=[xy[0],xy[1]-.015,z]
        idle=[1.05,-.62,z]
        self.move_hand('C',idle,src,24,'WORKER C: reach to occupied nest')
        # Kinematic hand-carried fruit: NOT grip forces or human physics.
        end=[crate_xy[0],-.03,.80]
        for i in range(48):
            a=(i+1)/48
            hand=[src[j]*(1-a)+end[j]*a for j in range(3)]
            self.place_hand('C',hand)
            fruit=[hand[0],hand[1],hand[2]-.045]
            p.resetBasePositionAndOrientation(body,fruit,[0,0,0,1])
            p.resetBaseVelocity(body,[0.,0.,0.],[0.,0.,0.])
            p.stepSimulation();self.rec.frame('WORKER C: scripted kinematic crate transfer')
        p.resetBasePositionAndOrientation(body,[crate_xy[0],0.,.745],[0,0,0,1])
        p.resetBaseVelocity(body,[0,0,0],[0,0,0])
        for _ in range(60):p.stepSimulation();self.rec.frame('WORKER C: place in grade '+grade+' crate')
        self.move_hand('C',end,idle,22,'WORKER C: return to station')


# --------------------------------------------------------------------------
# Stage42: CONNECTED worker visual proxies; does not participate in physics.
# The previous Stage41 actor, AI inference, and grasp-control code are kept.
# --------------------------------------------------------------------------
class ConnectedWorkerVisualActors(VisualActors):
    """Stage42 worker displays: visible torso, head, legs, shoulder and linked arms.

    The geometry is collision-free; motion is kinematically scripted.  It is
    deliberately not a true human, a grasp simulation, or a safety mechanism.
    The C worker's body translates along the side of the workbench when their
    hand reaches far along X; this is a graphical proxy, not walking physics.
    """
    def _make_hand(self,which,pos):
        # Keep SAME palm body-id (the stage40 virtual near-hand monitor reads it).
        # Replace the Stage41 long unconnected sleeve with a short wrist cuff;
        # dynamically updated links will connect cuff to the visible shoulder.
        side=1 if which=='A' else -1
        skin=[.96,.78,.60,1]
        sleeve=[.13,.34,.76,1] if which=='A' else [.15,.56,.35,1]
        palm=self.body('box',[.057,.035,.014],skin,pos)
        self.hands[which]=palm
        parts=[]
        def piece(half,colour,delta):
            xyz=[pos[k]+delta[k] for k in range(3)]
            obj=self.body('box',half,colour,xyz)
            parts.append((obj,delta))
        # Short wrist cuff plus the same 4 fingers and thumb as Stage41.
        piece([.045,.050,.032],sleeve,[0.,side*.093,.010])
        piece([.044,.007,.030],[.12,.17,.22,1],[0.,side*.049,.011])
        for dx,length in ((-.040,.055),(-.014,.067),(.013,.062),(.040,.048)):
            piece([.009,length/2,.009],skin,[dx,-side*(.035+length*.43),-.006])
        piece([.018,.032,.010],skin,[.071,-side*.035,-.009])
        self.hand_parts[which]=parts

    def setup(self):
        super().setup()  # Preserves all 3 x 30 fruit proxy baskets unchanged.
        self.avatars={}
        self._make_avatar('A',[0.,.73,1.27])
        self._make_avatar('C',[1.05,-.62,1.27])
        self.place_hand('A',[0.,.73,1.27])
        self.place_hand('C',[1.05,-.62,1.27])

    def _make_avatar(self,which,at):
        blue=which=='A'
        sleeve=[.13,.34,.76,1] if blue else [.15,.56,.35,1]
        pants=[.16,.19,.24,1]
        skin=[.96,.78,.60,1]
        dark=[.15,.18,.24,1]
        # Build once. Only visual bodies, never create collision geometry.
        ids={}
        for name,kind,half,colour in (
          ('torso','box',[.125,.090,.235],sleeve),
          ('hips','box',[.119,.082,.075],pants),
          ('neck','box',[.043,.040,.048],skin),
          ('head','sphere',[.097],skin),
          ('hair','box',[.097,.068,.021],dark),
          ('left_leg','box',[.047,.053,.32],pants),
          ('right_leg','box',[.047,.053,.32],pants),
          ('left_foot','box',[.062,.10,.036],dark),
          ('right_foot','box',[.062,.10,.036],dark),
          ('shoulder','sphere',[.051],sleeve),
          ('elbow','sphere',[.040],sleeve),
          ('wrist','sphere',[.030],sleeve),
        ):
            ids[name]=self.body(kind,half,colour,[0.,0.,-5.])
        # Four overlapping oriented sleeve sections on upper and lower arm.
        ids['upper']=[self.body('box',[.039,.122,.037],sleeve,[0,0,-5]) for _ in range(4)]
        ids['fore']=[self.body('box',[.032,.122,.031],sleeve,[0,0,-5]) for _ in range(4)]
        self.avatars[which]=ids

    @staticmethod
    def _link_quat(source,target):
        # Align each box's local +Y axis with shoulder→elbow or elbow→wrist.
        dx=float(target[0]-source[0]);dy=float(target[1]-source[1]);dz=float(target[2]-source[2])
        v=(dx*dx+dy*dy+dz*dz)**.5
        if v<1e-8:return [0.,0.,0.,1.]
        ux,uy,uz=dx/v,dy/v,dz/v
        # q rotates (0,1,0) onto (ux,uy,uz); cross is (uz,0,-ux).
        if uy < -.999999:return [1.,0.,0.,0.]
        a,b,c,w=uz,0.,-ux,1.+uy
        norm=(a*a+b*b+c*c+w*w)**.5
        return [a/norm,b/norm,c/norm,w/norm]

    def _put(self,body,where,orientation=None):
        self.p.resetBasePositionAndOrientation(body,where,
            orientation if orientation is not None else [0,0,0,1])

    def _place_chain(self,bodies,src,dst):
        q=self._link_quat(src,dst)
        for i,body in enumerate(bodies):
            t=(i+.5)/len(bodies)
            xyz=[src[k]*(1.-t)+dst[k]*t for k in range(3)]
            self._put(body,xyz,q)

    def place_hand(self,which,at):
        super().place_hand(which,at)
        if not hasattr(self,'avatars') or which not in self.avatars:return
        ids=self.avatars[which]
        # Camera-facing worker stations are on opposite sides of workbench.
        # C translates along X to reach distant grades; visuals only.
        if which=='A':
            cx,cy=-.22,.94
            shoulder=[cx+.115,.765,1.335]
            elbow=[(shoulder[0]+at[0])*.5-.035,(shoulder[1]+at[1])*.5,1.225]
            wrist=[at[0],at[1]+.090,at[2]+.010]
        else:
            cx=max(-.95,min(1.10,at[0]+.11))
            cy=-.86
            shoulder=[cx-.105,-.675,1.335]
            elbow=[(shoulder[0]+at[0])*.5+.045,(shoulder[1]+at[1])*.5,1.205]
            wrist=[at[0],at[1]-.090,at[2]+.010]
        # A human-shaped proxy at each work station with non-interacting feet.
        coord={
            'torso':[cx,cy,1.085],
            'hips':[cx,cy,.788],
            'neck':[cx,cy,1.368],
            'head':[cx,cy,1.510],
            'hair':[cx,cy,1.595],
            'left_leg':[cx-.066,cy,.428],
            'right_leg':[cx+.066,cy,.428],
            'left_foot':[cx-.066,cy-.065,.080],
            'right_foot':[cx+.066,cy-.065,.080],
            'shoulder':shoulder,'elbow':elbow,'wrist':wrist,
        }
        for key,pt in coord.items():self._put(ids[key],pt)
        self._place_chain(ids['upper'],shoulder,elbow)
        self._place_chain(ids['fore'],elbow,wrist)

# Preserve Stage41's exact external interface used by run(), including
# actor.hands, actor.move_hand(), actor.a_supply(), actor.c_transfer().
VisualActors=ConnectedWorkerVisualActors


# --------------------------------------------------------------------------
# Stage43: camera and encoding only.  All Stage42 AI, Bullet dynamics, robotic
# control, scripted A/C actions, and virtual interlock remain byte-identical.
# The overlay is NOT an AI input.  These visual-only proxies are not safety
# hardware or physically grasping humans.
# --------------------------------------------------------------------------
class Stage43ReadableRecorder(ChronologicalRecorder):
    # Higher oblique, no automated zoom or hard cuts.  Captures both avatars,
    # the four packing crates, center inspection platform and twelve nests.
    CAMERA_EYE=(0.60,-3.10,3.00)
    CAMERA_TARGET=(0.00,0.10,0.72)
    CAMERA_FOV_DEG=50.0
    RESOLUTION=(1280,720)

    def __init__(self,p,out,fps=20,w=1280,h=720):
        super().__init__(p,out,fps=fps,w=w,h=h)
        self.font=_stage40_overlay_font(25)
        self.small=_stage40_overlay_font(18)
        self.proxy_note=(
            'BLUE: Worker A  |  GREEN: Worker C  |  human actions: scripted, non-colliding')

    def frame(self,phase,force=False):
        # Same recording tick + 20 fps as Stage42.  Do not skip any new frames
        # or change physics stepping / process speed.
        self.steps+=1
        if not force and self.steps%12!=0:return
        p=self.p;w=self.w;h=self.h
        view=p.computeViewMatrix(cameraEyePosition=list(self.CAMERA_EYE),
            cameraTargetPosition=list(self.CAMERA_TARGET),cameraUpVector=[0,0,1])
        proj=p.computeProjectionMatrixFOV(
            fov=self.CAMERA_FOV_DEG,aspect=w/h,nearVal=.02,farVal=6)
        _,_,rgba,_,_=p.getCameraImage(w,h,viewMatrix=view,
                                     projectionMatrix=proj,
                                     renderer=p.ER_TINY_RENDERER)
        rgb=np.asarray(rgba,dtype=np.uint8).reshape(h,w,4)[:,:,:3]
        im=Image.fromarray(rgb)
        # Stage46: optional LIVE Bullet top view of the same physical A crate.
        # Not a pre-made clip, composited fruit, or second simulated world.
        if getattr(self,'packed_A_body',None) is not None:
            cx=float(self.crate_A_actual_cx)  # value from LAYOUT at packing
            inset_w,inset_h=290,200
            inset_view=p.computeViewMatrix(
                cameraEyePosition=[cx,-.08,1.82],
                cameraTargetPosition=[cx,0.,.75],cameraUpVector=[0,1,0])
            inset_proj=p.computeProjectionMatrixFOV(
                fov=44.,aspect=inset_w/inset_h,nearVal=.02,farVal=4.)
            _,_,ibox,_,_=p.getCameraImage(inset_w,inset_h,
                 viewMatrix=inset_view,projectionMatrix=inset_proj,
                 renderer=p.ER_TINY_RENDERER)
            inside=Image.fromarray(np.asarray(ibox,dtype=np.uint8).reshape(inset_h,inset_w,4)[:,:,:3])
            ix,iy=w-inset_w-25,150
            im.paste(inside,(ix,iy))
        d=ImageDraw.Draw(im)
        if getattr(self,'packed_A_body',None) is not None:
            ix,iy=w-290-25,150
            d.rectangle([ix-2,iy-2,ix+291,iy+201],outline=(25,35,50),width=3)
            d.rectangle([ix,iy-29,ix+290,iy],fill=(12,27,47))
            d.text((ix+8,iy-26),'LIVE A CRATE: physical apple inside',
                   font=self.small,fill='white')
        # Smaller HUD percentage than Stage42; no cover over avatar heads.
        d.rectangle([0,0,w,104],fill=(12,27,47))
        d.text((18,10),'PAI APPLE WORK-CELL | continuous PyBullet physics',
               font=self.font,fill='white')
        d.text((18,48),f'CASE: {self.state}   PHASE: {phase}',
               font=self.small,fill=(212,223,244))
        d.text((18,77),
               f'AI: {self.pred}  |  ROUTE: {self.destination}  |  A baskets: 3 x 30; supplied: {self.input_count}',
               font=self.small,fill=(199,236,200))
        d.rectangle([0,h-51,w,h],fill=(12,27,47))
        d.text((18,h-48),self.proxy_note,font=self.small,fill=(255,222,168))
        d.text((18,h-24),
               f'{self.date}  |  1.0x  |  single unspliced PyBullet recording  |  sim T={self.steps/240:.2f}s',
               font=self.small,fill=(210,218,235))
        self.writer.append_data(np.array(im));self.count+=1

# The existing workcell entry point calls ChronologicalRecorder by name.
# This substitution changes only the recorder, not the simulation.
ChronologicalRecorder=Stage43ReadableRecorder


# --------------------------------------------------------------------------
# Stage44: corrected worker A/B overlap and worker C connected behaviour.
# All worker parts remain SCRIPTED, COLLISION-FREE graphics, NOT human physics.
# --------------------------------------------------------------------------
class Stage44WorkerActors(ConnectedWorkerVisualActors):
    """A remains at the supply station. C has an independent walking station."""
    def setup(self):
        self.c_station_x=1.05
        self.c_hand_xyz=[1.05,-.62,1.27]
        self.c_station_before=self.c_station_x
        self.max_c_station_step_m=0.0
        self.c_a2_approach_start=None
        self.c_a2_transfer_finished=False
        super().setup()

    def _set_c_station(self,x):
        x=float(x)
        step=abs(x-self.c_station_x)
        self.max_c_station_step_m=max(self.max_c_station_step_m,step)
        # No discontinuous full-body jump; walking is a moving visual proxy.
        if step>.035:raise RuntimeError('STAGE44_WORKER_C_UNNATURAL_POSITION_JUMP: '+str(step))
        self.c_station_x=x

    def c_idle(self):return [self.c_station_x,-.62,1.27]

    def place_hand(self,which,at):
        # Bypass Stage42's hand-derived moving-torso position; update the
        # identical hand/arm visual geometry, with a separately controlled
        # torso station. All bodies are collision disabled.
        super(ConnectedWorkerVisualActors,self).place_hand(which,at)
        if which=='C':self.c_hand_xyz=[float(v) for v in at]
        if not hasattr(self,'avatars') or which not in self.avatars:return
        ids=self.avatars[which]
        if which=='A':
            cx,cy=-.22,.94
            shoulder=[cx+.115,.765,1.335]
            elbow=[(shoulder[0]+at[0])*.5-.035,(shoulder[1]+at[1])*.5,1.225]
            wrist=[at[0],at[1]+.090,at[2]+.010]
        else:
            cx,cy=self.c_station_x,-.86
            shoulder=[cx-.105,-.675,1.335]
            elbow=[(shoulder[0]+at[0])*.5+.045,(shoulder[1]+at[1])*.5,1.205]
            wrist=[at[0],at[1]-.090,at[2]+.010]
        coord={
            'torso':[cx,cy,1.085], 'hips':[cx,cy,.788],
            'neck':[cx,cy,1.368], 'head':[cx,cy,1.510],
            'hair':[cx,cy,1.595],
            'left_leg':[cx-.066,cy,.428], 'right_leg':[cx+.066,cy,.428],
            'left_foot':[cx-.066,cy-.065,.080],
            'right_foot':[cx+.066,cy-.065,.080],
            'shoulder':shoulder,'elbow':elbow,'wrist':wrist}
        for name,pt in coord.items():self._put(ids[name],pt)
        self._place_chain(ids['upper'],shoulder,elbow)
        self._place_chain(ids['fore'],elbow,wrist)

    def walk_C_to(self,target_x,steps=84,tag='WORKER C: scripted walk along operator aisle'):
        start=self.c_station_x
        n=max(1,int(steps))
        for i in range(n):
            alpha=(i+1)/n
            x=start+(target_x-start)*alpha
            self._set_c_station(x)
            self.place_hand('C',self.c_idle())
            self.p.stepSimulation();self.rec.frame(tag)

    def approach_A2_during_hold(self,u,xy):
        # Called by the original full-stop supervisor, in its own physics ticks.
        # First walk LEFT along the front aisle, THEN reach for middle A2.
        if self.c_a2_approach_start is None:
            self.c_a2_approach_start=self.c_station_x
        first=min(1.,u/.72)
        if first<1.:
            self._set_c_station(self.c_a2_approach_start+(-.72-self.c_a2_approach_start)*first)
            self.place_hand('C',self.c_idle())
        else:
            alpha=min(1.,(u-.72)/.28)
            home=self.c_idle()
            src=[xy[0],xy[1]-.015,1.28]
            self.place_hand('C',[home[k]*(1.-alpha)+src[k]*alpha for k in range(3)])

    def c_transfer(self,body,nest,grade,xy,crate_xy):
        # Natural scripted sequence. A2: from the hand already waiting at A2;
        # do not jump BACK to the old C position at right end before pickup.
        p=self.p
        target_x=float(crate_xy[0])
        if abs(self.c_station_x-target_x)>.55:
            self.walk_C_to(target_x,steps=max(72,math.ceil(abs(target_x-self.c_station_x)*65)))
        src=[xy[0],xy[1]-.015,1.28]
        current=list(self.c_hand_xyz)
        if math.dist(current,src)>.025:
            self.move_hand('C',current,src,24,'WORKER C: reach to '+nest)
        # The fruit remains a real Bullet body but is kinematically moved by
        # the SCRIPTED worker. There is no claim of human physical grasp.
        waypoints=[
            ('lift fruit clear of nest',[src[0],src[1],1.38],16),
            ('move above '+grade+' crate',[target_x,-.03,1.15],34),
            ('lower into '+grade+' crate',[target_x,-.03,.83],22),
        ]
        last=src
        for tag,target,steps in waypoints:
            for i in range(steps):
                a=(i+1)/steps
                hand=[last[j]*(1-a)+target[j]*a for j in range(3)]
                self.place_hand('C',hand)
                p.resetBasePositionAndOrientation(body,
                    [hand[0],hand[1],hand[2]-.045],[0,0,0,1])
                p.resetBaseVelocity(body,[0,0,0],[0,0,0])
                p.stepSimulation();self.rec.frame('WORKER C: scripted '+tag)
            last=target
        # Same known final packing datum as Stage43, without teleporting C.
        p.resetBasePositionAndOrientation(body,[target_x,0.,.745],[0,0,0,1])
        p.resetBaseVelocity(body,[0,0,0],[0,0,0])
        for _ in range(30):p.stepSimulation();self.rec.frame('WORKER C: fruit in crate')
        upper=[target_x,-.03,1.24]
        self.move_hand('C',last,upper,25,'WORKER C: lift empty hand above rim')
        self.move_hand('C',upper,self.c_idle(),26,'WORKER C: return to local standby')
        if nest=='A2':self.c_a2_transfer_finished=True
        # No snap to x=1.05 after A2. C remains at the A-side workstation.

VisualActors=Stage44WorkerActors

class ProxyBullet:
    """The supervisor's removeBody marks C's scripted transfer (not physical pick)."""
    def __init__(self,p,actions,scene,grade):
        self.real=p;self.actions=actions;self.scene=scene;self.grade=grade
    def __getattr__(self,k):return getattr(self.real,k)
    def removeBody(self,body):
        nest=next((k for k,v in self.scene['placed_fruits'].items() if v==body),None)
        if nest is None:raise RuntimeError('SCRIPTED_C_CLEAR_NEST_NOT_FOUND')
        from workbench_geometry import LAYOUT
        from tray_control import tray_occupied
        if self.grade=='A':
            if nest!='A2':raise RuntimeError('STAGE48_WORKER_C_MUST_CLEAR_A2')
            self.stage48_A_before={k:bool(tray_occupied(self.real,
                self.scene['tray_floors'][k],self.scene['robot'])) for k in ('A1','A2','A3')}
            if not all(self.stage48_A_before.values()):
                raise RuntimeError('STAGE48_A_NOT_THREE_FULL_BEFORE_WORKER '+str(self.stage48_A_before))
        self.actions.c_transfer(body,nest,self.grade,
           self.scene['trays'][nest],(LAYOUT.crate_centres_x_m[self.grade],0.))
        if self.grade=='A':
            self.stage48_A_after={k:bool(tray_occupied(self.real,
                self.scene['tray_floors'][k],self.scene['robot'])) for k in ('A1','A2','A3')}
            if self.stage48_A_after!={'A1':True,'A2':False,'A3':True}:
                raise RuntimeError('STAGE48_A_MIDDLE_GAP_SENSOR_FAIL '+str(self.stage48_A_after))
        # Preserve the SAME Bullet apple body inside the genuine crate.
        # This test cannot be passed by adding an ornamental apple marker.
        if self.grade=='A':
            x,y,z=map(float,self.real.getBasePositionAndOrientation(body)[0])
            from workbench_geometry import LAYOUT
            cx=float(LAYOUT.crate_centres_x_m['A'])
            inside=(abs(x-cx)<LAYOUT.crate_x_m/2-.04 and
                    abs(y)<LAYOUT.crate_y_m/2-.04 and
                    LAYOUT.bench_top_z_m+.035<=z<=LAYOUT.rim_z_m-.01)
            if not inside:
                raise RuntimeError('STAGE46_PACKED_A_BODY_NOT_INSIDE_CRATE '+str((x,y,z)))
            self.actions.rec.packed_A_body=body
            self.actions.rec.packed_A_xyz_at_pack=[x,y,z]
            self.actions.rec.crate_A_actual_cx=cx
        # Physical world retains fruit in crate. Supervisor removes nest mapping.
        self.actions_c_event={'source':'scripted kinematic worker hand',
           'from_nest':nest,'packed_crate':self.grade,'body_kept_in_pybullet_world':True}
        if callable(getattr(self,'on_worker_transfer_complete',None)):
            self.on_worker_transfer_complete(self)

def run(result_dir,engine):
    app=engine/'apple_stage13'
    sys.path[:0]=[str(engine),str(app),str(app/'src')]
    import pybullet as p
    import run_physics as rp
    from run_stage03 import plan
    from stage23_full_supervisor import FullGradeSupervisor
    from tray_control import tray_occupied
    from workbench_geometry import LAYOUT
    from mechanism_v09_reference import G
    if p.connect(p.DIRECT)<0:raise RuntimeError('PyBullet connect failed')
    identifier=joblib.load(ROOT/'models'/'apple_identifier.joblib')
    photo_assets=ROOT/'stage58_apple_photo_assets'/'prepared'
    stage62=Stage67CameraAI(ROOT,result_dir,photo_assets,object_ai)
    grader=joblib.load(ROOT/'models'/'abcd_visual_grade.joblib')
    rp.train=lambda:grader
    rp.infer=lambda model,images,_mass:grade_ai.infer_visual(model,images,mode='normal')
    # Stage50 A/B matched experiment: Stage49 ALREADY has two-speed lowering.
    # The only motor-speed change here is a modest faster far-field motion
    # and a 25 -> 20 mm low-speed contact window. Real contact gates remain.
    _base_lower=rp.MOTION_SETTINGS['settled_gate_09']
    assert abs(_base_lower.lower_fast_m_s-.36)<1e-9
    assert abs(_base_lower.lower_slow_m_s-.10)<1e-9
    assert abs(_base_lower.lower_slow_window_m-.025)<1e-9
    rp.MOTION_SETTINGS['settled_gate_09']=replace(_base_lower,
                   x_effective_rate=1.45,preflight_steps=96,
                   lower_fast_m_s=.40,lower_slow_m_s=.10,
                   lower_slow_window_m=.020)
    save_json(result_dir/'stage50_two_speed_profile.json',{
       'stage':50,'date_jst':stamp(),
       'baseline_stage49':{'fast_m_s':.36,'slow_m_s':.10,'last_slow_window_m':.025},
       'stage50_experiment':{'fast_m_s':.40,'slow_m_s':.10,'last_slow_window_m':.020},
       'stage49_parent_sha256':'fb98095584313d1b762c0aabc274fc46760d96dbc6311b9f3b147738133d364a',
       'contact_support_consecutive_steps_required':4,
       'experimental_max_impact_speed_m_s':.15,
       'physical_safety_certified':False,
       'NOT_modified':['AI_models','object_gate','3_finger_geometry','grip_scan',
                       'contact_and_release_guards','worker_timing','human_intrusion_interlock',
                       '12_cushion_nests','video_speed']})
    os.environ.update({'APPLE_NEST_ORDER':'nearest',
      'STAGE20_V40D_LOWER_RAMP_S':'0.08','STAGE22_V42_RETURN_SCALE':'0.65','STAGE48_WORKER_C_MIDDLE_A':'1'})
    cfgs={g:next(c for _,grade,c in plan('settled_gate_09') if grade==g) for g in 'ABCD'}
    sup=FullGradeSupervisor(pause_steps=360)
    scene=None;actors=None;rec=Stage62Recorder(p,result_dir/'stage67_one_world_continuous_1x.mp4')
    live={'case':'','apple':None,'grade':None,'interruption_done':False,
          'intrusion_start':None,'intrusion_end':None,'case_number':0,
          'prefetch':None}
    staged_next={'body':None,'target_case':None,'label':None,'from_case':None}
    prefetched_count=0
    reused_count=0

    events=[];rows=[]
    emit_original=sup._event
    def stage48_chronological_event(kind,grade,**kw):
        item=emit_original(kind,grade,**kw)
        events.append({'case':live['case'],**item})
        return item
    sup._event=stage48_chronological_event

    # Stage45: the D1 worker-C scripted retrieval is a per-tick coroutine.
    # No p.stepSimulation(), time.sleep(), or camera-frame duplication occurs
    # inside this worker script; the already-running robot physics clock owns
    # every step.  Worker proxy has NO collision geometry and cannot establish
    # real-world human safety.
    d_parallel={'pending_body':None,'task':None,'started':False,'finished':False,
       'clock_ticks':0,'last_phase':None,'min_palm_hand_distance_m':100.,
       'robot_x_samples':[],'robot_z_samples':[],'worker_motion_phases':set()}
    case_stage_ticks={}
    # Stage46: C reaches for the EMPTY C2 cushion during B's gravity-lift.
    # After B begins approaching THIS nest, a virtual guard holds B's joints.
    # Worker avatar has NO collision body and this is NOT safety certification.
    nest_interlock={'case':10,'nest':'C2','reach_ticks':0,'reached':False,
                    'reach_start':None,'nest_hand_xyz':None,
                    'stop_triggered':False,'held_ticks':0,
                    'stop_xy_gap_m':None,'stop_3d_gap_m':None,
                    'stop_robot_x_m':None,'hand_withdrawn':False}

    def scripted_D1_coroutine(body,xy,crate_x):
        x,y=map(float,xy)
        cx=float(crate_x)
        source=[x,y-.015,1.28]
        # C has remained at their original D workstation.  No teleporting.
        if abs(actors.c_station_x-cx)>.55:
            first=actors.c_station_x
            n=max(72,math.ceil(abs(cx-first)*65))
            for k in range(n):
                alpha=(k+1)/n
                actors._set_c_station(first+(cx-first)*alpha)
                actors.place_hand('C',actors.c_idle())
                yield 'worker_C_walk_D'
        before=list(actors.c_hand_xyz)
        for k in range(24):
            a=(k+1)/24
            actors.place_hand('C',[before[j]*(1-a)+source[j]*a for j in range(3)])
            yield 'worker_C_reach_D1'
        # Exclude the fruit from occupied handoff slots only when lifted;
        # the actual Bullet body continues to exist in the world.
        popped=scene['placed_fruits'].pop('D1',None)
        if popped!=body:
            raise RuntimeError('STAGE45_D1_CLAIM_BODY_MISMATCH')
        events.append({'event':'WORKER_C_D1_PICKUP_SCRIPTED',
            'case':live['case'],'body_id':int(body),
            'note':'scripted proxy hand, not physical human grasp'})
        last=source
        for phase,target,n in (
                ('worker_C_lift_D1',[source[0],source[1],1.38],16),
                ('worker_C_carry_to_D_crate',[cx,-.03,1.15],34),
                ('worker_C_lower_into_D_crate',[cx,-.03,.83],22)):
            for k in range(n):
                a=(k+1)/n
                hand=[last[j]*(1-a)+target[j]*a for j in range(3)]
                actors.place_hand('C',hand)
                p.resetBasePositionAndOrientation(body,
                    [hand[0],hand[1],hand[2]-.045],[0,0,0,1])
                p.resetBaseVelocity(body,[0,0,0],[0,0,0])
                yield phase
            last=target
        p.resetBasePositionAndOrientation(body,[cx,0.,.745],[0,0,0,1])
        p.resetBaseVelocity(body,[0,0,0],[0,0,0])
        for _ in range(30):
            yield 'worker_C_D_crate_settle'
        before=last;upper=[cx,-.03,1.24]
        for k in range(25):
            a=(k+1)/25
            actors.place_hand('C',[before[j]*(1-a)+upper[j]*a for j in range(3)])
            yield 'worker_C_raise_empty_hand'
        before=upper;home=actors.c_idle()
        for k in range(26):
            a=(k+1)/26
            actors.place_hand('C',[before[j]*(1-a)+home[j]*a for j in range(3)])
            yield 'worker_C_return_D_standby'
        actors.place_hand('C',actors.c_idle())

    def tick_D1_concurrent(pc,robot,joints,phase):
        task=d_parallel
        # Case 5 is the B2 route after the D1 placement (case 4).
        if task['pending_body'] is None or task['finished']:
            return
        if not task['started']:
            if live['case_number']!=5 or phase!='gravity_lift':
                return
            xy=scene['trays']['D1']
            task['task']=scripted_D1_coroutine(task['pending_body'],xy,
                LAYOUT.crate_centres_x_m['D'])
            task['started']=True
            events.append({'event':'WORKER_C_D1_CONCURRENT_START',
                'case':live['case'],'robot_phase':phase,
                'scripted_worker':True,'robot_not_commanded_to_hold':True})
        try:
            task['last_phase']=next(task['task'])
            task['worker_motion_phases'].add(task['last_phase'])
            task['clock_ticks']+=1
            # Diagnostic separation of the visible hand proxy from robot palm;
            # this is NOT proof of safety with an actual person.
            palm=pc.getLinkState(robot,joints['palm_camera_fixed'],
                                  computeForwardKinematics=True)[4]
            hand=pc.getBasePositionAndOrientation(actors.hands['C'])[0]
            gap=math.dist(palm,hand)
            task['min_palm_hand_distance_m']=min(task['min_palm_hand_distance_m'],gap)
            if gap<.16:
                raise RuntimeError('STAGE45_D1_ROBOT_C_HAND_VISUAL_PROXIMITY_BELOW_160MM')
            task['robot_x_samples'].append(float(pc.getJointState(
                robot,joints['stage_x_move'])[0]))
            task['robot_z_samples'].append(float(pc.getJointState(
                robot,joints['stage_z_move'])[0]))
        except StopIteration:
            task['finished']=True
            tray_clear=not tray_occupied(p,scene['tray_floors']['D1'],scene['robot'])
            moved=max(max(task['robot_x_samples'])-min(task['robot_x_samples']),
                      max(task['robot_z_samples'])-min(task['robot_z_samples']))
            events.append({'event':'WORKER_C_SCRIPTED_PACK_D',
                'case':live['case'],'sensor_D1_now_empty':tray_clear,
                'robot_moved_while_C_was_packing':moved>.015,
                'robot_joint_displacement_range_m':round(moved,5),
                'min_palm_hand_distance_m':round(task['min_palm_hand_distance_m'],5),
                'simultaneous_physics_ticks':task['clock_ticks'],
                'simultaneous_simulated_seconds':round(task['clock_ticks']/240.,5),
                'true_human_physics':False,
                'B_robot_interlock_bypassed':False})
            if not tray_clear:
                raise RuntimeError('STAGE45_D1_REMOVAL_SENSOR_FALSE')
            if moved<=.015:
                raise RuntimeError('STAGE45_C_FINISHED_BUT_B_WAS_STATIONARY')
            if live['case_number']!=5:
                raise RuntimeError('STAGE45_D_PACKING_OUTSIDE_CASE5')

    assets=result_dir/'sports_ball_assets';ball_ai.static_checks()
    original_spawn=rp.spawn_textured_apple
    def gate(p_client,robot,joints,load):
        return stage62.gate(p_client,robot,joints,load,object_ai,s35,identifier,live)
    rp.STAGE32_PREGRIP_GATE=gate
    rp.infer=stage62.grade
    def supervisor_hook(**kwargs):
        grade=kwargs['grade']
        scene_=kwargs['scene'];idx=len(sup.events)
        original_hold = kwargs['hold_step']
        # During full pause, C approaches A2, without affecting AI inference;
        # for a measured image classifier or a safety-rated robot detector.
        def hold_animated():
            n=live.setdefault('full_step',0);live['full_step']=n+1
            if n%4==0 and actors and grade=='A' and not actors.c_a2_transfer_finished:
                x,y=scene_['trays']['A2']
                actors.approach_A2_during_hold(min(1.,n/356),(x,y))
            return original_hold()
        kwargs['hold_step']=hold_animated
        # Stage48 supervisor still uses sensor-gated full/clear/resume; only C's pickup target changes.
        # Here its scripted "removeBody" is represented by visible C transfer.
        proxy=ProxyBullet(p,actors,scene_,grade)
        kwargs['p']=proxy
        def on_transfer_complete(done):
            events.append({'event':'WORKER_C_TRANSFER_COMPLETED',
                'case':live['case'],'nest':done.actions_c_event['from_nest'],
                'crate':done.actions_c_event['packed_crate'],
                'worker_C_station_x_m':actors.c_station_x,
                'worker_C_hand_after_transfer':list(actors.c_hand_xyz),
                'scripted_visual_worker':True,
                'hand_after_C_transfer_is_idle':(
                     math.dist(actors.c_hand_xyz,actors.c_idle())<.025)})
            if done.actions_c_event['packed_crate']=='A':
                events.append({'event':'STAGE48_A2_CLEARED_MIDDLE_GAP',
                    'case':live['case'],'worker_choice':'A2',
                    'sensor_before':done.stage48_A_before,
                    'sensor_after':done.stage48_A_after,
                    'A1_and_A3_still_occupied':True,
                    'robot_A2_route_not_forced':True})
                px,py,pz=map(float,p.getBasePositionAndOrientation(
                    rec.packed_A_body)[0])
                events.append({'event':'STAGE46_REAL_A_APPLE_INSIDE_CRATE',
                    'case':live['case'],'body_id':int(rec.packed_A_body),
                    'physical_body_xyz_m':[px,py,pz],
                    'crate_A_centre_x_m':float(LAYOUT.crate_centres_x_m['A']),
                    'visual_evidence':'same-tick live PyBullet overhead camera',
                    'not_an_added_visual_apple':True})
        proxy.on_worker_transfer_complete=on_transfer_complete
        sup.process(**kwargs)
        rec.destination=next((e.get('destination') for e in reversed(sup.events[idx:])
                              if e.get('event') in ('GO_TO_NEST','RESUME')),None) or 'WAIT'
    rp.STAGE23_AFTER_INFERENCE_HOOK=supervisor_hook

    def a_overlap_tick(pc,robot,apple,plate,joints,stage,cfg):
        # Deterministic physical prefetch after B lifts/clears the scale.
        # Only runs when the next programmed object is an apple.  Tennis is
        # still inserted normally on the next episode (no hidden fake object).
        nonlocal prefetched_count
        ix=live['case_number']
        if ix<1 or ix>=len(CASES):return
        next_type,next_label,_=CASES[ix]
        if next_type!='apple':return
        state=live.get('prefetch')
        if state is None:
            if stage!='move_to_bin':return
            if staged_next['body'] is not None:return
            x=float(pc.getJointState(robot,joints['stage_x_move'])[0])
            z=float(pc.getJointState(robot,joints['stage_z_move'])[0])
            # Do not permit A inside while gripper is low or over the scale.
            if abs(x)<.145 or z < G.root_z+.080:return
            prev_z=float(pc.getBasePositionAndOrientation(apple)[0][2])
            if prev_z < G.plate_z+LAYOUT.relative_world_origin_z_m+.075:return
            # The previous fruit must actually be off the scale.
            if any(float(c[9])>.01 for c in pc.getContactPoints(apple,plate)):return
            actors.next_input()
            state={'tick':0,'target_case':ix+1,'label':next_label,
                   'prior_case':live['case'],'x_clearance_m':abs(x),
                   'z_joint_m':z,'prior_apple_z_m':prev_z,
                   'body':None,'ready':False}
            live['prefetch']=state
            events.append({'event':'WORKER_A_EARLY_SUPPLY_START','from_case':live['case'],
                 'next_case_index':ix+1,'horizontal_clearance_m':abs(x),
                 'robot_joint_z_m':z,'prior_apple_off_plate':True,
                 'worker_model':'scripted, collision-free'})
        if state.get('ready'):return
        state['tick']+=1
        n=state['tick'];origin=[0.,.73,1.27];target=[0.,.09,1.27]
        if n<=18:
            a=n/18
            actors.place_hand('A',[origin[k]*(1-a)+target[k]*a for k in range(3)])
        elif n==19:
            new_cfg=replace(cfgs[next_label],profile='nominal',
                 replicate=4000+ix+1,scan_duration_s=.32,torque_Nm=12.)
            # The original fruit factory creates the next genuine dynamic body.
            body=rp.place_next_fruit_on_scale(pc,next_label,new_cfg,plate)
            stage62.texture_body(pc,body,ix+1)  # real, early-supplied next Bullet body
            state['body']=body
            staged_next.update({'body':body,'target_case':ix+1,
                                'label':next_label,'from_case':live['case']})
            events.append({'event':'WORKER_A_NEXT_REAL_APPLE_ON_SCALE',
                  'from_case':live['case'],'target_case':ix+1,
                  'next_label_input':next_label,'new_bullet_body_id':int(body),
                  'while_robot_transports_previous':True,
                  'actual_physical_dynamic_body':True})
        elif n<=37:
            a=(n-19)/18
            actors.place_hand('A',[target[k]*(1-a)+origin[k]*a for k in range(3)])
        if n>=37:
            actors.place_hand('A',origin)
            state['ready']=True
            prefetched_count+=1
            events.append({'event':'WORKER_A_EARLY_HAND_WITHDRAWN',
                           'from_case':live['case'],'target_case':ix+1})

    def observer(pc,robot,apple,plate,joints,stage,t,cfg):
        nonlocal actors
        stage62.observe(pc,robot,apple,plate,joints,stage,t,cfg,rp)
        case_stage_ticks.setdefault(live['case'],{}).setdefault(stage,0)
        case_stage_ticks[live['case']][stage]+=1
        if actors is None:
            actors=VisualActors(pc,rec)
            actors.a_supply()
            rec.state='01 / first apple supplied by A (scripted)'
            events.append({'event':'WORKER_A_FIRST_SCRIPTED_SUPPLY',
                           'real_hand_physics':False,'3x30_visual_baskets':True})
        # Stage46: C's hand stays over the C2 depressed cushion BEFORE the
        # 3-finger robot arrives. The B claw visibly approaches that nest.
        if live['case_number']==10 and not nest_interlock['stop_triggered']:
            guard=nest_interlock
            if stage=='gravity_lift':
                guard['reach_ticks']+=1
                xy=scene['trays']['C2']
                target=[float(xy[0]),float(xy[1])-.015,
                        float(LAYOUT.nest_top_z_m)+.075]
                if guard['reach_start'] is None:
                    guard['reach_start']=list(actors.c_hand_xyz)
                    guard['nest_hand_xyz']=target
                    events.append({'event':'WORKER_C_REACHES_INTO_C2_NEST',
                        'case':live['case'],'nest':'C2',
                        'hand_target_xyz_m':target,'nest_floor_top_z_m':float(LAYOUT.nest_top_z_m),
                        'scripted_hand_not_sensor':True})
                alpha=min(1.,guard['reach_ticks']/64.)
                origin=guard['reach_start']
                actors.place_hand('C',[origin[j]*(1-alpha)+target[j]*alpha for j in range(3)])
                if alpha>=1.:guard['reached']=True
            elif stage=='move_to_bin':
                if not guard['reached']:
                    raise RuntimeError('STAGE46_C2_HAND_NOT_ON_CUSHION_BEFORE_APPROACH')
                tx,ty=scene['trays']['C2']
                robot_x=float(pc.getJointState(robot,joints['stage_x_move'])[0])
                palm=pc.getLinkState(robot,joints['palm_camera_fixed'],
                      computeForwardKinematics=True)[4]
                hand=pc.getBasePositionAndOrientation(actors.hands['C'])[0]
                gap_xy=math.hypot(float(palm[0]-hand[0]),float(palm[1]-hand[1]))
                gap_3d=math.dist(palm,hand)
                # Require the ROBOT to move toward the occupied C2 nest;
                # never teleport C's hand to the robot at the scale.
                if (robot_x>.14 and abs(robot_x-float(tx))<.135
                        and gap_xy<.22 and gap_3d<.38):
                    guard['stop_triggered']=True
                    guard['stop_robot_x_m']=robot_x
                    guard['stop_xy_gap_m']=gap_xy
                    guard['stop_3d_gap_m']=gap_3d
                    rec.state='WORKER C HAND ON C2 NEST: ROBOT B STOPS BEFORE REACH'
                    events.append({'event':'VIRTUAL_HAND_PROXIMITY_STOP',
                        'case':live['case'],'nest':'C2','hand_on_nest_before_approach':True,
                        'robot_approached_nest_x_m':robot_x,
                        'hand_xyz_m':list(map(float,hand)),
                        'palm_xyz_m':list(map(float,palm)),
                        'xy_distance_m':gap_xy,'distance_m':gap_3d,
                        'safety_rated_sensor':False,'collision_enabled_for_hand':False})
                    pos={axis:pc.getJointState(robot,joints[axis])[0] for axis in
                         ('stage_x_move','stage_y_move','stage_z_move')}
                    before=float(pos['stage_x_move'])
                    for _ in range(240):
                        rp.control(pc,robot,joints,pos,cfg)
                        pc.stepSimulation();rec.frame('HAND OCCUPIES C2 CUSHION: ROBOT B HOLD')
                        guard['held_ticks']+=1
                    after=float(pc.getJointState(robot,joints['stage_x_move'])[0])
                    events.append({'event':'VIRTUAL_HAND_HOLD_MEASURE',
                        'case':live['case'],'nest':'C2',
                        'x_drift_m':float(after-before),'held_ticks':guard['held_ticks']})
                    # Withdraw palm first UP and toward the operator, then
                    # retract to C's stationary work position while B holds.
                    start=list(map(float,hand))
                    via=[start[0],start[1]-.22,start[2]+.18]
                    end=actors.c_idle()
                    for origin,target,n,phase in (
                        (start,via,55,'C2 HAND LIFTS CLEAR'),
                        (via,end,75,'C2 HAND RETREATS TO C')):
                        for i in range(n):
                            a=(i+1)/n
                            actors.place_hand('C',[origin[j]*(1-a)+target[j]*a for j in range(3)])
                            rp.control(pc,robot,joints,pos,cfg)
                            pc.stepSimulation();rec.frame(phase)
                    actors.place_hand('C',end)
                    guard['hand_withdrawn']=True
                    final_hand=pc.getBasePositionAndOrientation(actors.hands['C'])[0]
                    events.append({'event':'VIRTUAL_HAND_CLEARED_RESUME',
                        'case':live['case'],'nest':'C2','hand_withdrawn_from_cushion':True,
                        'withdrawn_hand_xyz_m':list(map(float,final_hand)),
                        'operator_reset_event':True})
        tick_D1_concurrent(pc,robot,joints,stage)
        a_overlap_tick(pc,robot,apple,plate,joints,stage,cfg)
        rec.frame(stage if not (d_parallel['started'] and not d_parallel['finished'])
             else stage+' | C D1 '+str(d_parallel['last_phase']))
    rp._STAGE_CONTACT_OBSERVER=observer

    try:
        for ix,(typ,label,expected) in enumerate(CASES,1):
            name=f'{ix:02d}_{typ}_{label}_{expected or "reject"}'
            live.update({'case':name,'apple':None,'grade':label,'interruption_done':False,
                         'full_step':0,'movement_tick':0,'case_number':ix,
                         'prefetch':None})
            rec.state=f'{ix:02d} / {typ} {label}'
            stage62.start_case(ix,typ,label,rec)
            rec.pred='Awaiting palm camera and visual AI';rec.destination='—'
            cfg=replace(cfgs[label] if typ=='apple' else cfgs['A'],
                profile='nominal',replicate=4000+ix,scan_duration_s=.28,close_duration_s=.08,torque_Nm=12.)
            expected_reused_id=None
            if scene is not None:
                if staged_next['body'] is not None and staged_next['target_case']==ix:
                    if typ!='apple' or staged_next['label']!=label:
                        raise RuntimeError('STAGE44_NEXT_INPUT_WRONG_TYPE')
                    expected_reused_id=staged_next['body']
                    scene['stage44_prepositioned_input_body']=expected_reused_id
                    events.append({'event':'WORKER_A_PREPOSITIONED_BODY_REUSED',
                        'case':name,'bullet_body_id':int(expected_reused_id),
                        'prior_case':staged_next['from_case'],
                        'physical_body_reused':True})
                    staged_next.update({'body':None,'target_case':None,
                                        'label':None,'from_case':None})
                    reused_count+=1
                else:
                    if staged_next['body'] is not None:
                        raise RuntimeError('STAGE44_UNCONSUMED_PREPOSITIONED_BODY')
                    actors.a_supply()
            # Stage47: prove the B2 decision has a genuine, physical sensor
            # history.  B3 was filled by CASE 2, B2 is vacant and B1 has NEVER
            # been filled or preloaded in this continuous demonstration.
            if ix==5:
                occupied_B_pre={k:bool(tray_occupied(p,scene['tray_floors'][k],scene['robot']))
                                for k in ('B1','B2','B3')}
                events.append({'event':'STAGE47_B2_PRE_DECISION_SENSOR',
                    'case':name,'occupied':occupied_B_pre,
                    'claim':'B3 occupied from case 2; B2 and B1 empty',
                    'no_preexisting_B1_apple_inserted':True})
                if occupied_B_pre!={'B1':False,'B2':False,'B3':True}:
                    raise RuntimeError('STAGE47_B2_PHYSICAL_PRECONDITION_FAIL '+str(occupied_B_pre))
                if 'B1' in scene['placed_fruits']:
                    raise RuntimeError('STAGE47_B1_UNEXPECTED_FRUIT_IN_WORLD')
            before_event=len(events)
            orig_spawn=rp.spawn_textured_apple
            if typ=='tennis':
                def spawn_tennis(pc,fruit_label,mass,radius,xyz,friction=1.):
                    position=[float(xyz[0]),float(xyz[1]),float(xyz[2])+
                              ball_ai.SPECS['tennis']['radius_m']-radius]
                    return ball_ai.spawn_ball(pc,'tennis',0,position,assets)
                rp.spawn_textured_apple=spawn_tennis
            current_body=[None]
            def object_observer(*args):
                current_body[0]=args[2]
                observer(*args)
            rp._STAGE_CONTACT_OBSERVER=object_observer
            try:
                row,scene=rp.run_trial(p,label if typ=='apple' else 'A',
                   result_dir/'episodes',record_video=False,cfg=cfg,
                   persistent_scene=scene,return_scene=True)
                if typ=='tennis':
                    if current_body[0] is not None:p.removeBody(current_body[0])
                    for _ in range(12):p.stepSimulation();rec.frame('WORKER A REMOVES REJECTED BALL')
                if expected_reused_id is not None and current_body[0]!=expected_reused_id:
                    raise RuntimeError('STAGE44_PHYSICAL_BODY_REUSE_ID_MISMATCH')
                if live.get('prefetch') is not None and not live['prefetch']['ready']:
                    raise RuntimeError('STAGE44_EARLY_SUPPLY_DID_NOT_FINISH_BEFORE_TRIAL_END')
                stage62_audit=stage62.conclude(name,row,rec)
                result={'case':name,'object':typ,'truth_label':label,'expected_destination':expected,
                    'stage62_five_camera':stage62_audit,
                    'stage44_prepositioned_body_reused':expected_reused_id is not None,
                    'stage44_early_supplied_during_this_case':bool(live.get('prefetch') and live['prefetch']['ready']),
                    'physical_status':row.get('status'),'success_physical_and_ai':bool(row.get('success_physical_and_ai')),
                    'stage32':row.get('stage32_gate'),'stage64_approach':row.get('stage64_approach'),'classification':row.get('classification'),
                    'destination':row.get('assigned_nest'),'stage_time_s':row.get('simulated_time_s'),
                    'stage37_occupancy':{k:bool(tray_occupied(p,v,scene['robot'])) for k,v in scene['tray_floors'].items()},
                    'case_events':events[before_event:]}
                if typ=='apple':
                    passed=(bool(row.get('success_physical_and_ai')) and
                            row.get('classification',{}).get('class')==label and
                            row.get('assigned_nest')==expected and
                            bool(row.get('return_home_pass')))
                else:
                    gate=row.get('stage32_gate') or {}
                    passed=(row.get('status')=='STAGE32_REJECTED_BEFORE_DESCENT'
                        and gate.get('allow_descent') is False
                        and (gate.get('object') or {}).get('predicted_object')=='tennis'
                        and row.get('descent_executed') is False)
                result['pass']=bool(passed)
                rows.append(result);save_json(result_dir/'case_reports'/f'{name}.json',result)
                print('STAGE40_CASE',ix,typ,label,'AI',row.get('classification',{}).get('class'),
                     'dest',row.get('assigned_nest'),'PASS',passed,flush=True)
                if not passed:break
                # Schedule D1 pickup for the NEXT B2 episode instead of
                # stopping B while worker C completes an entire blocking task.
                if ix==4 and 'D1' in scene['placed_fruits']:
                    d_parallel['pending_body']=scene['placed_fruits']['D1']
                    events.append({'event':'WORKER_C_D1_SCHEDULED_FOR_PARALLEL_B2',
                         'after_case':name,'body_id':int(d_parallel['pending_body']),
                         'no_worker_motion_performed_serially':True})
                if ix==5 and not d_parallel['finished']:
                    raise RuntimeError('STAGE45_D1_CONCURRENT_TRANSFER_INCOMPLETE')
                if ix==9:
                    actors.walk_C_to(.42,steps=90,tag='WORKER C: walk to C-class packing station')
                    events.append({'event':'WORKER_C_WALK_TO_NEXT_STATION',
                        'case':name,'station_x_m':actors.c_station_x,
                        'scripted_non_collision':True})
                if ix==5:
                    states={k:bool(tray_occupied(p,v,scene['robot']))
                            for k,v in scene['tray_floors'].items()}
                    events.append({'event':'B2_FREE_CHOICE_CONFIRMATION',
                        'B1_occupied':states['B1'],'B3_occupied':states['B3'],
                        'B2_occupied_after':states['B2'],
                        'logic':'B3 filled in case 2; nearest free B nest is B2',
                        'stage47_no_B1_preload':True})
                    if (states['B1'] or not states['B3'] or not states['B2']
                            or 'B1' in scene['placed_fruits']):
                        raise RuntimeError('STAGE47_B2_POSTCONDITION_FAIL '+str(states))
            finally:
                rp.spawn_textured_apple=original_spawn
        all_pass=len(rows)==len(CASES) and all(r['pass'] for r in rows)
        save_json(result_dir/'stage62_five_camera_summary.json',stage62.history)
        names=[e['event'] for e in events]
        strict={'A_to_C_workcell_scripted':('WORKER_C_SCRIPTED_PACK_D' in names),
            'tennis_reject':any(r['object']=='tennis' and r['pass'] for r in rows),
            'A_three_full_stop':('THREE_FULL_STOP' in names),
            'worker_C_clear_and_resume':all(x in names for x in ('WORKER_C_REMOVED','RESUME')),
            'B3_to_B2_without_B1':any(e['event']=='B2_FREE_CHOICE_CONFIRMATION' and
                not e['B1_occupied'] and e['B3_occupied'] and e['B2_occupied_after']
                for e in events),
            'B1_remains_empty_no_scripted_spawn':(
                'SCRIPTED_B1_BACKLOG_PRELOAD' not in names
                and all(not r['stage37_occupancy']['B1'] for r in rows)
                and not (scene and 'B1' in scene['placed_fruits'])),
            'stage48_A2_middle_gap_physically_sensed':any(
                e['event']=='STAGE48_A2_CLEARED_MIDDLE_GAP' and
                e.get('sensor_before')=={'A1':True,'A2':True,'A3':True} and
                e.get('sensor_after')=={'A1':True,'A2':False,'A3':True} for e in events),
            'stage48_A2_chosen_by_unmodified_robot_priority':any(
                e['event']=='RESUME' and e.get('grade')=='A' and
                e.get('destination')=='A2' for e in events),
            'stage48_A2_final_occupancy':bool(
                len(rows)>=8 and rows[7]['destination']=='A2' and
                all(rows[7]['stage37_occupancy'].get(k) for k in ('A1','A2','A3'))),
            'stage48_stop_worker_remove_resume_chronological':(
                (lambda seq: bool(len(seq)==1 and all(name in seq[0] for name in
                    ('THREE_FULL_STOP','WORKER_C_SELECTS_MIDDLE_A2','WORKER_C_TRANSFER_COMPLETED',
                     'STAGE48_A2_CLEARED_MIDDLE_GAP','WORKER_C_REMOVED','RESUME')) and
                    seq[0].index('THREE_FULL_STOP')<seq[0].index('WORKER_C_SELECTS_MIDDLE_A2')<
                    seq[0].index('WORKER_C_TRANSFER_COMPLETED')<seq[0].index('WORKER_C_REMOVED')<
                    seq[0].index('RESUME')))([
                        [e['event'] for e in events if e.get('case')==r['case']]
                        for r in rows if r['expected_destination']=='A2' and
                        r['case'].startswith('08_')])) ,
            'B2_pre_decision_load_sensor':any(
                e['event']=='STAGE47_B2_PRE_DECISION_SENSOR' and
                e.get('occupied')=={'B1':False,'B2':False,'B3':True}
                for e in events),
            'virtual_hand_approach_stop_resume':all(x in names for x in
                 ('VIRTUAL_HAND_PROXIMITY_STOP','VIRTUAL_HAND_CLEARED_RESUME')),
            'stage46_C_hand_at_C2_before_claw_approach':(
                nest_interlock['reached'] and nest_interlock['stop_triggered']
                and nest_interlock['stop_robot_x_m']>.14),
            'stage46_C_hand_retracted_before_resume':nest_interlock['hand_withdrawn'],
            'stage46_A_crate_contains_original_Bullet_fruit':(
                 getattr(rec,'packed_A_body',None) is not None
                 and any(e.get('event')=='STAGE46_REAL_A_APPLE_INSIDE_CRATE' for e in events)),
            'stage48_C_hand_at_idle_after_A2_transfer':any(
                 e.get('event')=='WORKER_C_TRANSFER_COMPLETED' and
                 e.get('nest')=='A2' and e.get('hand_after_C_transfer_is_idle')
                 for e in events),
            'virtual_hand_bullet_x_hold_below_15mm':any(
                 e.get('event')=='VIRTUAL_HAND_HOLD_MEASURE' and
                 abs(float(e.get('x_drift_m',9)))<.015 for e in events)}
        strict.update({
            'worker_C_D1_B2_overlap_in_one_physics_clock':(
                d_parallel['finished'] and d_parallel['clock_ticks']>=170 and
                any(e.get('event')=='WORKER_C_SCRIPTED_PACK_D' and
                    e.get('robot_moved_while_C_was_packing') for e in events)),
            'worker_C_D1_visual_gap_at_least_160mm':(
                d_parallel['min_palm_hand_distance_m']>=.16),
            'worker_A_early_parallel_input_physical':prefetched_count>=5,
            'same_Bullet_input_body_reused':prefetched_count==reused_count,
            'worker_C_torso_never_jumps':bool(actors and actors.max_c_station_step_m<=.035),
            'worker_C_stays_at_A_after_A2_transfer':any(
                e['event']=='WORKER_C_TRANSFER_COMPLETED' and
                e.get('nest')=='A2' and abs(e.get('worker_C_station_x_m',10)+.72)<.09
                for e in events),
        })
        # Stage50: on actual Bullet trial records, require real support and impact guard.
        _stage49_ref={'01_apple_A_A3': {'duration_s': 3.2624999999999997, 'descent_ticks': 131}, '02_apple_B_B3': {'duration_s': 2.9916666666666667, 'descent_ticks': 133}, '03_apple_C_C1': {'duration_s': 2.9916666666666667, 'descent_ticks': 133}, '04_apple_D_D1': {'duration_s': 3.4208333333333334, 'descent_ticks': 131}, '05_apple_B_B2': {'duration_s': 3.0458333333333334, 'descent_ticks': 131}, '06_apple_A_A2': {'duration_s': 3.595833333333333, 'descent_ticks': 131}, '07_apple_A_A1': {'duration_s': 3.7666666666666666, 'descent_ticks': 131}, '08_apple_A_A2': {'duration_s': 5.145833333333333, 'descent_ticks': 131}, '10_apple_C_C2': {'duration_s': 2.9916666666666667, 'descent_ticks': 136}}
        _lower_cmp=[]
        for _record in rows:
            if _record['object']!='apple':continue
            _name=_record['case']
            _past=_stage49_ref.get(_name)
            _ticks=int(case_stage_ticks.get(_name,{}).get('gentle_lower',0))
            if not _past:raise RuntimeError('STAGE50_REFERENCE_CASE_MISSING '+_name)
            _lower_cmp.append({'case':_name,'baseline_stage49_lower_ticks':_past['descent_ticks'],
               'stage50_lower_ticks':_ticks,'saved_lower_ticks':_past['descent_ticks']-_ticks,
               'baseline_stage49_time_s':_past['duration_s'],
               'stage50_time_s':float(_record['stage_time_s']),
               'saved_total_s':round(_past['duration_s']-float(_record['stage_time_s']),5),
               'ai_and_physical_pass':bool(_record['pass'])})
        _impact_records=[]
        for _trialpath in sorted((result_dir/'episodes').glob('*/trial.json')):
            _tr=json.loads(_trialpath.read_text(encoding='utf-8'))
            _landing=_tr.get('placement') or {}
            if _landing.get('status')!='PLACED_ON_SUPPORT':continue
            _impact_records.append({'trial':_trialpath.parent.name,
                'verified_support':bool(_landing.get('verified_support')),
                'impact_vertical_speed_m_s':_landing.get('impact_vertical_speed_m_s'),
                'within_bound':bool(_landing.get('impact_within_exploratory_bound')),
                'settled_on_floor':bool(_landing.get('settled_on_floor')),
                'release_fingers_contact_free':bool((_landing.get('release_gate') or {}).get('contact_free_before_lift'))})
        _avg_stage50=(sum(x['stage50_time_s'] for x in _lower_cmp)/len(_lower_cmp)
                       if _lower_cmp else None)
        _avg_stage49=(sum(x['baseline_stage49_time_s'] for x in _lower_cmp)/len(_lower_cmp)
                       if _lower_cmp else None)
        _stage50_report={
          'date_jst':stamp(),'stage':50,
          'experiment':'Stage62 current cycle versus historical Stage49; changed AI/scan pose so not a controlled speed comparison',
          'stage49_reference':'verified 2026/10/21 result ZIP, individual trials, no video speed changes',
          'per_case':_lower_cmp,'landing':_impact_records,
          'baseline_average_apple_s':_avg_stage49,'stage50_average_apple_s':_avg_stage50,
          'average_saved_s':((_avg_stage49-_avg_stage50) if _avg_stage49 is not None else None),
          'baseline_average_lower_ticks':sum(x['baseline_stage49_lower_ticks'] for x in _lower_cmp)/len(_lower_cmp) if _lower_cmp else None,
          'stage50_average_lower_ticks':sum(x['stage50_lower_ticks'] for x in _lower_cmp)/len(_lower_cmp) if _lower_cmp else None,
          'stage50_mean_lower_phase_saved_s':(sum(x['saved_lower_ticks'] for x in _lower_cmp)/len(_lower_cmp)/240) if _lower_cmp else None,
          'not_real_hardware_or_fruit_bruise_test':True,'stage62_speed_comparison_noncausal':True}
        save_json(result_dir/'stage50_lowering_comparison.json',_stage50_report)
        strict['stage50_nine_landing_records']=bool(len(_impact_records)==9)
        strict['stage50_impact_release_guards_preserved']=bool(
            len(_impact_records)==9 and all(
              x['verified_support'] and x['within_bound'] and
              x['settled_on_floor'] and x['release_fingers_contact_free']
              for x in _impact_records))
        strict['stage50_lowering_is_faster']=bool(
            len(_lower_cmp)==9 and sum(x['saved_lower_ticks'] for x in _lower_cmp)>0)
        strict['stage50_total_avg_is_faster']=bool(
            _avg_stage49 is not None and _avg_stage50<_avg_stage49)
        pregrip=[case_stage_ticks.get(row['case'],{}) for row in rows if row['object']=='apple']
        timing={
            'stage':50,
            'per_case_station_preflight_s':[round(d.get('station_preflight',0)/240,5) for d in pregrip],
            'per_case_scan_close_s':[round(d.get('scan_and_close',0)/240,5) for d in pregrip],
            'per_case_contact_settle_s':[round(d.get('contact_settle',0)/240,5) for d in pregrip],
            'per_case_contact_feedback_s':[round(d.get('contact_feedback',0)/240,5) for d in pregrip],
            'scan_close_duration_design_s':.36,
            'baseline_stage48_average_apple_physics_s':3.724,
            'no_video_retiming':True}
        save_json(result_dir/'stage49_grasp_timing_comparison_preserved.json',timing)
        strict['stage49_scan_command_length_physics_checked']=bool(
            pregrip and all(80 <= (d.get('scan_and_close',0) +
                 d.get('scale_settle_scan_overlap',0)) <= 92 for d in pregrip))
        strict['stage49_preflight_minimum_pose_check']=bool(
            pregrip and all(d.get('station_preflight',0)>=24 for d in pregrip))
        # Stage51 controlled paired comparison to user's verified Stage50 outputs.
        _stage51_baseline={'01_apple_A_A3': 3.2, '02_apple_B_B3': 2.92916667, '03_apple_C_C1': 2.92916667, '04_apple_D_D1': 3.35833333, '05_apple_B_B2': 2.98333333, '06_apple_A_A2': 3.53333333, '07_apple_A_A1': 3.70833333, '08_apple_A_A2': 5.08333333, '09_tennis_tennis_reject': 0.33333333, '10_apple_C_C2': 2.92916667}
        _stage51_overlap_per_case=[]
        for _row in rows:
            _case=_row['case'];_dur=float(_row['stage_time_s'])
            _ph=case_stage_ticks.get(_case,{})
            _overlap_ticks=int(_ph.get('scale_settle_scan_overlap',0))
            _scan_ticks=int(_ph.get('scan_and_close',0))
            _old=float(_stage51_baseline[_case])
            _stage51_overlap_per_case.append({
                'case':_case,'object':_row['object'],
                'stage50_simulated_s':_old,
                'stage51_simulated_s':_dur,
                'saved_s':round(_old-_dur,6),
                'scan_settle_overlap_ticks':_overlap_ticks,
                'scan_after_settle_ticks':_scan_ticks,
                'expected_first_camera_capture_tick':round(42*.28/.52),
                'result_pass':bool(_row['pass'])})
        _apple_pairs=[x for x in _stage51_overlap_per_case if x['object']=='apple']
        _s51_base_mean=sum(x['stage50_simulated_s'] for x in _apple_pairs)/len(_apple_pairs)
        _s51_new_mean=sum(x['stage51_simulated_s'] for x in _apple_pairs)/len(_apple_pairs)
        strict['stage51_nine_postgate_overlaps']=bool(
            len(_apple_pairs)==9 and all(0<=x['scan_settle_overlap_ticks']<=10 for x in _apple_pairs)
            and sum(x['scan_settle_overlap_ticks'] for x in _apple_pairs)>0)
        strict['stage51_all_10_trials_success']=bool(len(rows)==10 and all(x['pass'] for x in rows))
        strict['stage51_avg_faster_than_verified_stage50']=bool(_s51_new_mean < _s51_base_mean)
        strict['stage51_3_camera_capture_points_after_settle']=bool(
            all(x['expected_first_camera_capture_tick']>x['scan_settle_overlap_ticks'] for x in _apple_pairs))
        save_json(result_dir/'stage51_overlap_comparison.json',{
            'date_jst':stamp(),'stage':51,
            'baseline':'verified Stage50 actual user Colab physics results, same 10 cases',
            'stage50_result_zip_sha256':'e15f906a791fc5d592a018c2930fe0526d9f1edbaff7f77112bc9690024d59f1',
            'overlap_design':'post-Stage32-approval dynamic remaining portion of original 16-tick plate settle + open-finger scan; 4-tick measured probe minimum plus 2-tick gate debounce',
            'same_images_expected_not_hash_identical':True,
            'stage50_mean_apple_s':_s51_base_mean,
            'stage51_mean_apple_s':_s51_new_mean,
            'mean_saved_s':_s51_base_mean-_s51_new_mean,
            'per_case':_stage51_overlap_per_case,
            'all_legacy_physics_guards_still_activated':True,'gate_earlier_than_stage50_when_load_probe_stable':True,
            'earlier_object_gate_requires_measured_four_tick_load_probe':True,
            'not_actual_orchard_or_safety_certification':True})
        # Stage62 intentionally drops overlapping pre-scan movement and adds real
        # camera work; the historic Stage50/51 speed-comparison checks are not
        # success criteria and must not be silently marked true.
        diagnostic_keys=('stage50_total_avg_is_faster','stage50_lowering_is_faster',
            'stage51_nine_postgate_overlaps','stage51_avg_faster_than_verified_stage50',
            'stage51_avg_faster_than_verified_stage50')
        comparison_only={key:strict.pop(key) for key in dict.fromkeys(diagnostic_keys) if key in strict}
        strict['stage62_actual_9_apple_grades_recorded']=bool(sum(x['grade'] is not None for x in stage62.history)==9)
        strict['stage62_all_9_5camera_scans']=bool(sum(x['grade'] is not None and
            x['finger_frames']==12 and x['under_camera_frames']==3 and x['descending']
            for x in stage62.history)==9)
        # Stage67: actually measured finger descent in the Stage62 v3 scan,
        # not the superseded Stage64 pre-gate approach. Don't require the
        # obsolete Stage64 manifest in an otherwise successful physical run.
        strict['stage67_nine_descending_finger_scans_and_raw_grade_palm']=bool(
            sum(h.get('finger_frames')==12 and h.get('descending') and
                h.get('under_camera_frames')==3 and h.get('grade_palm_raw_captured')
                for h in stage62.history)==9)
        all_pass=bool(all_pass and all(strict.values()))
        summary={'date_jst':stamp(),'stage':67,'status':'PHYSICS_EXECUTED',
            'n_episodes':len(rows),'n_expected_episodes':len(CASES),
            'all_pass':all_pass,'scenario_coverage':strict,
            'one_continuous_pybullet_world':True,
            'one_unspliced_recording':True,
            'stage62_continuous_five_camera_trial':True,
            'stage62_historical_comparison_not_success_criteria':comparison_only,
            'stage62_photo_repetition':'one source A photo reused in four A scenes; distinct Bullet bodies',
            'stage62_not_orchard_validated':True,
            'stage65_apple_gate_train_includes_all_eight_known_photos':True,
            'stage65_classification_not_independent_unknown_fruit_accuracy':True,
            'stage65_no_runtime_asset_id_model_switching':True,
            'worker_A_early_input_count':prefetched_count,
            'worker_A_reused_input_count':reused_count,
            'worker_C_max_station_step_m':actors.max_c_station_step_m if actors else None,
            'stage44_changes':'A physical prefetch after clearance; C stationary at reach, scripted walking and crate transfer',
            'stage45_changes':'D1 C-pack and next B2 robot pick run on the same physical clock without serial C blocking',
            'stage46_changes':'hand reaches C2 nest BEFORE claw approach; C A3 hand stays withdrawn; live Bullet A-crate camera',
            'stage47_changes':'remove unscripted B1 fruit spawn after case 4; B2 selected naturally after B3 by physical sensor',
            'stage49_changes':'validated 3R finger scan-to-grasp and measured station readiness',
            'stage50_changes':'only descent profile .40/.10 m/s and 20mm slow window, compared to stage49',
            'stage48_changes':'C selects middle A2 from three-full A; physical A1/A3 stay occupied, B re-reads and chooses A2; supervisor events recorded chronologically',
            'stage62_not_a_true_8_distinct_apple_batch':'Nine fruit bodies; A source photo reused in four A positions; only two C source photos used',
            'stage46_C2_guard':nest_interlock,
            'stage50_comparison_file':'stage50_lowering_comparison.json',
            'stage46_A_crate_physical_fruit_id':getattr(rec,'packed_A_body',None),
            'stage46_A_crate_physical_xyz_at_pack':getattr(rec,'packed_A_xyz_at_pack',None),
            'stage45_D1_C_concurrent_ticks':d_parallel['clock_ticks'],
            'stage45_D1_C_concurrent_s':round(d_parallel['clock_ticks']/240.,5),
            'stage45_D1_min_virtual_hand_to_palm_m':(
                 round(d_parallel['min_palm_hand_distance_m'],5)),
            'stage45_D1_worker_phases':sorted(d_parallel['worker_motion_phases']),
            'stage45_per_case_robot_phase_ticks':case_stage_ticks,
            'video_frames':rec.count,
            'video_fps':rec.fps,'video_seconds':rec.count/rec.fps,
            'workcell_3_baskets_display_30_each':True,
            'stage62_5cam_all_apple_cases':all(x['finger_frames']==12 and x['under_camera_frames']==3 and x['descending'] for x in stage62.history if x['grade'] is not None),
            'stage62_5cam_case_count':sum(x['grade'] is not None for x in stage62.history),
            'actual_90_apples_processed':False,
            'worker_A_and_C': 'scripted collision-free visual proxies, not dynamic people',
            'human_safety':'virtual geometry/interlock demo; NOT safety-rated detector',
            'mass_used_for_grade':False,'ai_synthetic_training_only':True,
            'stage39_code_not_overwritten':True}
        save_json(result_dir/'events.json',events)
        save_json(result_dir/'cases.json',rows)
        save_json(result_dir/'summary.json',summary)
        save_json(result_dir/'stage45_timing_diagnostics.json',{
            'date_jst':stamp(),'stage':50,
            'note':'physics stage tick counts per case. 1 step=1/240 s; renderer can add overhead in real wall time',
            'case_phase_ticks':case_stage_ticks,
            'D1_C_parallel_ticks':d_parallel['clock_ticks'],
            'total_film_s':rec.count/rec.fps,
            'no_speedup_or_video_splicing':True})
        return 0 if all_pass else 2
    finally:
        rec.close()
        if p.isConnected():p.disconnect()

def pack(result):
    target=result.parent/(result.name+'_evidence.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for f in sorted(result.rglob('*')):
            if (f.is_file() and 'engine' not in f.relative_to(result).parts
                    and f.suffix.lower() in ('.json','.csv','.mp4','.png','.txt')):
                z.write(f,f.relative_to(result))
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:raise RuntimeError('ZIP_CRC_FAIL')
    return target

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default='stage67_continuous_results')
    ap.add_argument('--internal',action='store_true')
    ap.add_argument('--engine')
    a=ap.parse_args();output=Path(a.output).resolve()
    if a.internal:return run(output,Path(a.engine).resolve())
    if output.exists():raise FileExistsError('Result must not overwrite: '+str(output))
    if not (ROOT/'models'/'apple_identifier.joblib').exists():
        raise RuntimeError('STAGE39_MODELS_MISSING: run stage39 Colab setup first')
    output.parent.mkdir(parents=True,exist_ok=True)
    engine=ensure_engine(output)
    # subprocess protects import from stale Colab module cache.
    import subprocess
    command=[sys.executable,'-u',str(Path(__file__).resolve()),
             '--internal','--engine',str(engine),'--output',str(output)]
    result=subprocess.run(command,cwd=output)
    if not (output/'summary.json').exists():
        save_json(output/'execution_incomplete.json',{'date_jst':stamp(),
         'stage':67,'status':'NOT_VERIFIED','subprocess_returncode':result.returncode,
         'note':'Find Python traceback in Colab cell output; partial video is not success'})
    save_json(output/'stage65_execution_manifest.json', {
        'stage':67,'date_jst':stamp(),'output_size_px':[1600,900],
        'fps':20,'video_speed':'1.0x','continuous_world':True,
        'single_unspliced_video':True,'camera_eye':[0.60,-3.10,3.00],
        'camera_target':[0.00,0.10,0.72], 'camera_fov_degrees':50.0,
        'fixed_camera_no_mid_trial_zoom':True,
        'parent_stage43_source_sha256':'b22ba383087211bb486f7343e5645522b590db75ca485ae8e64bfd63c500b62c',
        'AI_robot_control':'Stage51 continuity + Stage62 proven wrist-scan descent + Stage67 high RAW shared development gate and low RAW grade palm; no runtime photo-ID model switch', 'worker_A_C_scheduling':'stage45_D_parallel_B2',
        'actor_representation':'scripted collision-free proxies, not human physics',
        'D1_pack_parallel_with_B2':True, 'robot_speed_control':'unchanged_stage44',
        'worker_C_hand_nest_stop':'C2 before B arrival, virtual hold',
        'A_crate_evidence':'live overhead PyBullet inset of real transferred body',
        'hand_after_A2':'remains retracted, no re-approach in hold callback',
        'case5_nest_B2':'B3 occupied, B2 empty before decision; B1 remains empty',
        'removed_artifact':'SCRIPTED_B1_BACKLOG_PRELOAD removed, no apple appears mid-film',
        'stage48_worker_C_A_full_removal':'A2, between still occupied A1 and A3',
        'stage48_robot_A2_destination':'must be determined from physical tray presence sensors',
        'stage48_chronological_event_capture':True
    })
    z=pack(output)
    print('STAGE67_RESULT_ZIP',z,'returncode',result.returncode,flush=True)
    return result.returncode

if __name__=='__main__':sys.exit(main())
