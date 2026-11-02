#!/usr/bin/env python3
"""Stage25 v4.5a: prove routine worker removes a *middle* occupied nest.

All 12 physical nest positions, robot, 3 fingers, 5 cameras, classifier,
under-support contact dynamics, and inherited speed remain untouched.
This is a separate 5-apple PyBullet trial, not the 16-trial full-capacity test.
"""
from __future__ import annotations
import os,sys,json,traceback,zipfile,hashlib,statistics,shutil
from dataclasses import replace
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import pybullet as p
import run_stage25_v45_centerfirst as base

BASE=Path(__file__).resolve().parent/'apple_stage13'
sys.path[:0]=[str(BASE),str(BASE/'src')]
from run_stage03 import plan
from learned_classifier import LABELS
from tray_control import TRAY_NAMES,tray_occupied
from stage23_full_supervisor import FullGradeSupervisor
import run_physics as rp

SEQUENCE=['A']*5
EXPECTED=['A3','A2','A1','A2','A3']
REMOVAL_BEFORE={4:'A2',5:'A3'}
NOW=lambda:datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y/%m/%d %H:%M JST')

def sense(p,scene):
    return {name:bool(tray_occupied(p,body,scene['robot']))
            for name,body in scene['tray_floors'].items()}

def routine_worker_clear(p,scene,slot,supervisor):
    """Explicit simulated action by worker C; no synthetic fruit translation.

    Removes only one PREVIOUSLY PLACED apple body after robot has returned
    home. Per-nest physical load sensing must confirm the cleared vacancy.
    """
    before=sense(p,scene)
    assert before[slot],f'Expected physically occupied slot {slot}'
    assert slot in scene['placed_fruits'],f'No body recorded at {slot}'
    old=scene['placed_fruits'][slot]
    p.removeBody(old)
    del scene['placed_fruits'][slot]
    for _ in range(12):p.stepSimulation() # let sensor contacts update
    after=sense(p,scene)
    assert after[slot] is False,f'Physical vacancy NOT confirmed: {slot}'
    for name,was in before.items():
        if name!=slot and was:assert after[name],f'Other occupied nest changed: {name}'
    evt={'sequence':len(supervisor.events)+1,
         'event':'WORKER_C_REMOVED_ROUTINE','grade':slot[0],
         'nest':slot,'apple_body_removed':int(old),
         'source':'scripted external operator action between trials; not an animated/safety-rated human',
         'sensor_before':before,'sensor_after':after}
    supervisor.events.append(evt)
    return evt

def run():
    # Guard against unintentional changes to the previous validated physics.
    base.verify()
    os.environ['APPLE_NEST_ORDER']='nearest'
    os.environ['STAGE20_V40D_LOWER_RAMP_S']='0.08'
    os.environ['STAGE22_V42_RETURN_SCALE']='0.65'
    root=BASE/'output'/'stage25_v4_5a_a2_partial_clear'
    if root.exists():
        raise RuntimeError('Existing results must be recovered; refusing overwrite: '+str(root))
    root.mkdir(parents=True)
    cfg0=next(cfg for _,g,cfg in plan('settled_gate_09') if g=='A')
    prev=rp.MOTION_SETTINGS['settled_gate_09']
    rp.MOTION_SETTINGS['settled_gate_09']=replace(prev,x_effective_rate=1.45)
    supervisor=FullGradeSupervisor()
    rp.STAGE23_AFTER_INFERENCE_HOOK=supervisor.process
    scene=None;records=[];removals=[]
    p.connect(p.DIRECT)
    try:
        for idx,grade in enumerate(SEQUENCE,1):
            before_evt=len(supervisor.events)
            if idx in REMOVAL_BEFORE:
                assert scene is not None
                ev=routine_worker_clear(p,scene,REMOVAL_BEFORE[idx],supervisor)
                removals.append(ev)
                print('WORKER_CLEAR',idx,ev['nest'],'detected free',not ev['sensor_after'][ev['nest']],flush=True)
            cfg=replace(cfg0,profile='nominal',replicate=400+idx,
                        scan_duration_s=.32,torque_Nm=12.)
            print('TRIAL',idx,'/ 5 target',EXPECTED[idx-1],flush=True)
            try:
                row,scene=rp.run_trial(p,grade,root/'episodes',record_video=True,
                                    cfg=cfg,persistent_scene=scene,return_scene=True)
                evts=supervisor.events[before_evt:]
                good=(row.get('classification',{}).get('class')==grade
                      and row.get('assigned_nest')==EXPECTED[idx-1]
                      and bool(row.get('success_physical_and_ai'))
                      and bool(row.get('all_retained_fruits_ok'))
                      and bool(row.get('return_home_pass'))
                      and bool((row.get('post_return_check') or {}).get('ok'))
                      and not any(x['event']=='THREE_FULL_STOP' for x in evts))
                r={'episode':idx,'grade':grade,'expected_destination':EXPECTED[idx-1],
                   'actual_destination':row.get('assigned_nest'),
                   'cycle_s':row.get('simulated_time_s'),
                   'physical_success':bool(row.get('success_physical_and_ai')),
                   'all_retained_fruits_ok':bool(row.get('all_retained_fruits_ok')),
                   'success':good,'events':evts,
                   'occupied_after':sense(p,scene)}
                records.append(r)
                print('RESULT',idx,r['actual_destination'],'success',good,
                      'cycle',r['cycle_s'],flush=True)
                if not good:break
            except Exception as e:
                (root/f'exception_ep{idx:02d}.txt').write_text(traceback.format_exc(),encoding='utf8')
                records.append({'episode':idx,'success':False,'error':repr(e)})
                print('ERROR',repr(e),flush=True)
                break
            finally:
                (root/'events_all.json').write_text(json.dumps(supervisor.events,ensure_ascii=False,indent=2),encoding='utf8')
                (root/'progress.json').write_text(json.dumps({'timestamp':NOW(),'results':records},ensure_ascii=False,indent=2),encoding='utf8')
    finally:
        p.disconnect()
        rp.MOTION_SETTINGS['settled_gate_09']=prev
        if hasattr(rp,'STAGE23_AFTER_INFERENCE_HOOK'):del rp.STAGE23_AFTER_INFERENCE_HOOK
        for key in ('APPLE_NEST_ORDER','STAGE20_V40D_LOWER_RAMP_S','STAGE22_V42_RETURN_SCALE'):
            os.environ.pop(key,None)
    summary={'date_jst':NOW(),'stage':'25-v4.5a','scenario':'A2 and then A3 cleared by scripted worker C between physical episodes',
             'expected_destinations':EXPECTED,'records':records,
             'verified_removal_slots':[e['nest'] for e in removals],
             'all_success':len(records)==5 and all(x['success'] for x in records),
             'mean_cycle_s':(statistics.mean(x['cycle_s'] for x in records) if len(records)==5 and all(x['success'] for x in records) else None),
             'scope_note':'synthetic A apples only; offline permutation tests include grades B,C,D; worker C is an external scripted event; no human animation'}
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    archive=BASE/'apple_stage25_v4.5a_partial_clear_results.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for f in sorted(root.rglob('*')):
            if f.is_file(): z.write(f,arcname=str(f.relative_to(BASE)))
        z.writestr('PROVENANCE.txt',NOW()+' | Stage25 v4.5a A2 vacancy test; five separate PyBullet episodes; physical fruit removal only between episodes; no body teleportation of the new apple.\n')
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print('RESULT ZIP',archive,'bytes',archive.stat().st_size,flush=True)
    if not summary['all_success'] or summary['verified_removal_slots']!=['A2','A3']:
        raise RuntimeError('PARTIAL_CLEAR_PHYSICS_FAILED; retain result ZIP for diagnosis')
    print('PASS A2 intermediate opening -> reuse; A3 opening -> reuse;',
          'average',summary['mean_cycle_s'],flush=True)
    return archive

if __name__=='__main__':run()
