# -*- coding: utf-8 -*-
"""Stage62: remove Stage59 prescan up/down bob by starting each scan at clearance.
Only patch a FRESH Stage51-cloned engine. Baseline Stage51/Stage61 source remains unchanged.
Created 2026/10/22 JST. Must execute PyBullet tests before claiming success.
"""
import json,hashlib,datetime
from pathlib import Path

def apply_stage62_engine_patch(engine,out):
    p=Path(engine)/'apple_stage13/src/run_physics.py'
    old=p.read_text(encoding='utf8')
    changes=[
      ("initial={'stage_x_move':0.,'stage_y_move':0.,'stage_z_move':G.root_z}",
       "initial={'stage_x_move':0.,'stage_y_move':0.,'stage_z_move':G.root_z+.044}  # S62 standby at scan-start height"),
      ("init=grip_goals(0,cfg);init.update({'stage_x_move':0,'stage_y_move':0,'stage_z_move':G.root_z})",
       "init=grip_goals(0,cfg);init.update({'stage_x_move':0,'stage_y_move':0,'stage_z_move':G.root_z+.044})  # S62 ready pose"),
      ("_s51_overlap=max(0,int(policy.scale_settle_steps)-_s51_probe_ticks-_s51_gate_debounce_ticks)",
       "_s51_overlap=0  # S62: disable pre-scan motor overlap so no premature Z descent"),
      ("        send(grip_goals(t,cfg));step_and_trace('scan_and_close')",
       "        _s62_command=grip_goals(t,cfg)\n"
       "        _s62_fraction=max(0.,min(1.,t/cfg.scan_duration_s))\n"
       "        _s62_command['stage_z_move']=G.root_z+.044*(1.-smootherstep(_s62_fraction))\n"
       "        send(_s62_command);step_and_trace('scan_and_close')"),
    ]
    cur=old
    for target,repl in changes:
        n=cur.count(target)
        if n!=1:raise RuntimeError(f'STAGE62_PATCH_ANCHOR_MISMATCH: {n} {target[:72]}')
        cur=cur.replace(target,repl,1)
    for safety in ('STAGE32_REJECTED_BEFORE_DESCENT','STAGE23_AFTER_INFERENCE_HOOK',
                   'gravity_lift','return_home_pass','station_preflight'):
        if safety not in cur:raise RuntimeError('Safety control not found: '+safety)
    compile(cur,str(p),'exec')
    p.write_text(cur,encoding='utf8')
    record={'stage':62,'date_jst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
           'source_sha256':hashlib.sha256(old.encode()).hexdigest(),
           'patched_sha256':hashlib.sha256(cur.encode()).hexdigest(),
           'scope':'ONLY isolated Stage51 engine copy',
           'initial_ready_z_offset_m':.044,'prescan_raise_steps':0,
           'scan_descent_offset_m':.044,'prescan_overlap_disabled':True,
           'uncertainty':'not confirmed physically until Colab experiment',
           'untouched':'motor constraints, contact gates, release, return, full stop, worker proxy'}
    (Path(out)/'stage62_engine_patch_manifest.json').write_text(json.dumps(record,indent=2,ensure_ascii=False),encoding='utf8')
    return record
