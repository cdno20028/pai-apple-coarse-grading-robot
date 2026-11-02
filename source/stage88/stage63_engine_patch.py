# -*- coding: utf-8 -*-
"""Stage63: monotone downward open-claw approach before gate, then 6mm descent scan.
No Stage59 +44mm upstroke. Preserve Stage51 grasp endpoint and object-gate fail closed.
Requires actual PyBullet run; neither classification nor physical grasp is pre-certified.
"""
from pathlib import Path
import hashlib,json,datetime

def apply_stage63_engine_patch(engine,out):
    p=Path(engine)/'apple_stage13'/'src'/'run_physics.py'
    old=p.read_text(encoding='utf-8')
    changes=[
      ("initial={'stage_x_move':0.,'stage_y_move':0.,'stage_z_move':G.root_z}",
       "initial={'stage_x_move':0.,'stage_y_move':0.,'stage_z_move':G.root_z+.044}  # S63 standby"),
      ("init=grip_goals(0,cfg);init.update({'stage_x_move':0,'stage_y_move':0,'stage_z_move':G.root_z})",
       "init=grip_goals(0,cfg);init.update({'stage_x_move':0,'stage_y_move':0,'stage_z_move':G.root_z+.044})  # S63 standby"),
      ("_s51_overlap=max(0,int(policy.scale_settle_steps)-_s51_probe_ticks-_s51_gate_debounce_ticks)",
       "_s51_overlap=0  # Stage63: no early finger motor motion before object gate"),
    ]
    cur=old
    for target,repl in changes:
        if cur.count(target)!=1:raise RuntimeError('STAGE63_PATCH_ANCHOR_MISMATCH:'+target[:75])
        cur=cur.replace(target,repl,1)
    gate="    _stage32_gate_cb=globals().get('STAGE32_PREGRIP_GATE')\n"
    assert cur.count(gate)==1,'Missing original Stage32 gate'
    approach=(
        "    # S63 open-finger MONOTONE approach, no scanning/gripping before Stage32 gate.\n"
        "    # Stop 6mm above original Stage61 palm image position, not 44mm above it.\n"
        "    _s63_gate_z=G.root_z+.006\n"
        "    _s63_approach_trace=[]\n"
        "    _s63_started_z=float(p.getJointState(robot,joints['stage_z_move'])[0])\n"
        "    if _s63_started_z<_s63_gate_z+.025:\n"
        "        raise RuntimeError('STAGE63_HIGH_STANDBY_NOT_REACHED')\n"
        "    for _s63_i in range(110):\n"
        "        _s63_open=grip_goals(0,cfg)\n"
        "        _s63_open.update({'stage_x_move':0.,'stage_y_move':0.,'stage_z_move':_s63_gate_z})\n"
        "        send(_s63_open);step_and_trace('stage63_open_finger_reference_approach')\n"
        "        _s63_joint_state=p.getJointState(robot,joints['stage_z_move'])\n"
        "        _s63_actual=float(_s63_joint_state[0])\n"
        "        _s63_approach_trace.append(_s63_actual)\n"
        "        if abs(_s63_actual-_s63_gate_z)<.0008 and abs(float(_s63_joint_state[1]))<.035:\n"
        "            break\n"
        "    else:\n"
        "        raise RuntimeError('STAGE63_OPEN_APPROACH_POSE_NOT_REACHED')\n"
        "    if any(b>a+.001 for a,b in zip(_s63_approach_trace,_s63_approach_trace[1:])):\n"
        "        raise RuntimeError('STAGE63_REVERSE_STANDBY_MOTION')\n"
        "    _s63_approach_evidence={'standby_z':_s63_started_z,'target_gate_z':_s63_gate_z,\n"
        "        'actual_gate_z':_s63_actual,'nominal_reference_stage61_z':G.root_z,\n"
        "        'gate_approximate_position_not_exact_match':True,\n"
        "        'approach_ticks':len(_s63_approach_trace),\n"
        "        'z_samples':_s63_approach_trace,'open_fingers_until_gate':True,\n"
        "        'stage59_upstroke_steps':0}\n"
    ).replace('\\n','\n')
    cur=cur.replace(gate,approach+gate,1)
    scan="        send(grip_goals(t,cfg));step_and_trace('scan_and_close')"
    assert cur.count(scan)==1,'Missing original scan command'
    replace_scan=(
        "        _s63_scan=grip_goals(t,cfg)\n"
        "        _s63_f=max(0.,min(1.,t/cfg.scan_duration_s))\n"
        "        _s63_scan['stage_z_move']=G.root_z+.006*(1.-smootherstep(_s63_f))\n"
        "        send(_s63_scan);step_and_trace('scan_and_close')"
    ).replace('\\n','\n')
    cur=cur.replace(scan,replace_scan,1)
    report="report={'stage32_gate':locals().get('_stage32_gate_record'),"
    assert cur.count(report)==1,'Missing Stage32 report field'
    cur=cur.replace(report,"report={'stage63_approach':locals().get('_s63_approach_evidence'),"+"'stage32_gate':locals().get('_stage32_gate_record'),",1)
    for safety in ('STAGE32_REJECTED_BEFORE_DESCENT','STAGE23_AFTER_INFERENCE_HOOK','gravity_lift','return_home_pass','station_preflight'):
        if safety not in cur:raise RuntimeError('Safety control missing '+safety)
    if '_s59_up_step' in cur or 'stage59_pre_scan_clearance' in cur:
        raise RuntimeError('Old Stage59 bounce unexpectedly remains')
    compile(cur,str(p),'exec');p.write_text(cur,encoding='utf8')
    digest=lambda v:hashlib.sha256(v.encode('utf-8')).hexdigest()
    manifest={'stage':63,'JST':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
      'pre_patch_sha256':digest(old),'patched_sha256':digest(cur),'stage59_bob_steps':0,
      'standby_offset_m':.044,'gate_offset_from_stage61_m':.006,
      'scan_offset_start_m':.006,'scan_offset_end_m':0.0,
      'camera_roi_cropping':False,'object_gate_before_finger_scan':True,
      'open_finger_lowering_before_gate':True,
      'reference_camera_height_exactly_equal':False,
      'original_grasp_and_lift_endpoint_preserved':True,
      'no_real_hardware_test':True}
    (Path(out)/'stage63_engine_patch_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest
