#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage30: speed-gated home-return A/B/C/D isolated physics comparison.

Does not modify Stage27 master ZIP. The additional 30mm + 0.10m/s 10-step
criterion is EXPERIMENTAL, NOT real robot safety certification.
"""
from __future__ import annotations
import argparse, csv, datetime, hashlib, importlib.util, json, os, subprocess, sys, zipfile
from dataclasses import replace
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE_ZIP=HERE/'stage27_original_96files.zip'
EXPECTED_SOURCE_SHA='2f9a4daba4ca77e8b5c78d0b6a9aad7a5710a290729f545b7bfde4befaae2dc9'
EXPECTED_URDF_SHA='829f6e6823806ea4d2e3766c29c4330979406cb7ac14ecc79ed009a8f86cd08d'
MODES=('baseline_safe','direct_goal_safe')
LABELS=('A','B','C','D')
SLOTS={'A':'A3','B':'B3','C':'C1','D':'D1'}
OLD_RETURN="""               g.update({'stage_x_move':actual_x*(1-u),
                         'stage_y_move':0.,'stage_z_move':G.root_z+.100})
               send(g);step_and_trace('return_to_scale')"""
DIRECT_RETURN="""               g.update({'stage_x_move':0.0,
                         'stage_y_move':0.,'stage_z_move':G.root_z+.100})
               send(g);step_and_trace('return_to_scale')"""
OLD_GATE="""           if cfg.motion_profile != 'baseline':
               _home=grip_goals(0,cfg)
               _home.update({'stage_x_move':0.,'stage_y_move':0.,'stage_z_move':G.root_z+.100})
               for _ in range(96):
                   if abs(float(p.getJointState(robot,joints['stage_x_move'])[0]))<.030: break
                   send(_home);step_and_trace('home_arrival_verify')
    home_x=float(p.getJointState(robot,joints['stage_x_move'])[0])
    home_ok=abs(home_x)<.065"""
NEW_GATE="""           if cfg.motion_profile != 'baseline':
               _home=grip_goals(0,cfg)
               _home.update({'stage_x_move':0.,'stage_y_move':0.,'stage_z_move':G.root_z+.100})
               _home_stable_frames=0
               _home_stability_gate_pass=False
               # EXPERIMENTAL only: position+speed both must remain in range for 10 steps.
               # A safety gate timeout is a FAIL, not a success.
               for _ in range(240):
                   send(_home);step_and_trace('home_arrival_verify')
                   _state=p.getJointState(robot,joints['stage_x_move'])
                   if abs(float(_state[0]))<.030 and abs(float(_state[1]))<=.100:
                       _home_stable_frames+=1
                       if _home_stable_frames>=10:
                           _home_stability_gate_pass=True
                           break
                   else:
                       _home_stable_frames=0
    home_x=float(p.getJointState(robot,joints['stage_x_move'])[0])
    home_vx=float(p.getJointState(robot,joints['stage_x_move'])[1])
    home_ok=(abs(home_x)<.030 and abs(home_vx)<=.100 and
             bool(locals().get('_home_stability_gate_pass',False)))"""

def sha(b):return hashlib.sha256(b).hexdigest()
def jst():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y/%m/%d %H:%M:%S JST')

def source_clone(mode,out):
    assert mode in MODES
    out=Path(out)
    if out.exists():raise RuntimeError('No overwriting: '+str(out))
    eng=out/'engine';eng.mkdir(parents=True)
    with zipfile.ZipFile(SOURCE_ZIP) as z:
        if z.testzip() is not None:raise RuntimeError('Original ZIP CRC failure')
        z.extractall(eng)
    f=eng/'apple_stage13/src/run_physics.py'
    u=eng/'apple_stage13/robot_v14_phase30.urdf'
    o=f.read_bytes(); 
    if sha(o)!=EXPECTED_SOURCE_SHA or sha(u.read_bytes())!=EXPECTED_URDF_SHA:
        raise RuntimeError('Original Stage27 source and/or URDF SHA mismatch')
    s=o.decode('utf8')
    if s.count(OLD_GATE)!=1:raise RuntimeError('Gate patch not unique')
    if s.count(OLD_RETURN)!=1:raise RuntimeError('Return patch not unique')
    s=s.replace(OLD_GATE,NEW_GATE,1)
    if mode=='direct_goal_safe':s=s.replace(OLD_RETURN,DIRECT_RETURN,1)
    f.write_text(s,encoding='utf8')
    compile(s,str(f),'exec')
    (out/'source_manifest.json').write_text(json.dumps({
      'mode':mode,'time_jst':jst(),'original_src_sha256':sha(o),
      'patched_src_sha256':sha(f.read_bytes()),'unchanged_urdf_sha256':sha(u.read_bytes()),
      'gate':'30mm & <=0.10m/s for 10 successive PyBullet steps, <=240 retries',
      'return':'unchanged smootherstep' if mode=='baseline_safe' else 'direct zero goal (test only)',
      'gate_is_experimental':True},ensure_ascii=False,indent=2),encoding='utf8')
    return eng

def trial_worker(mode,eng):
    if importlib.util.find_spec('pybullet') is None:raise RuntimeError('PyBullet unavailable; use Colab')
    eng=Path(eng).resolve(); app=eng/'apple_stage13'
    sys.path[:0]=[str(eng),str(app),str(app/'src')]
    import pybullet as p
    import run_physics as rp
    from run_stage03 import plan
    from stage23_full_supervisor import FullGradeSupervisor
    rp.MOTION_SETTINGS['settled_gate_09']=replace(rp.MOTION_SETTINGS['settled_gate_09'],x_effective_rate=1.45)
    rp.STAGE23_AFTER_INFERENCE_HOOK=FullGradeSupervisor().process
    os.environ.update({'APPLE_NEST_ORDER':'nearest','STAGE20_V40D_LOWER_RAMP_S':'0.08','STAGE22_V42_RETURN_SCALE':'0.65'})
    res_dir=eng.parent; all_rows=[]
    if p.connect(p.DIRECT)<0:raise RuntimeError('PyBullet connect failed')
    try:
        for i,grade in enumerate(LABELS,1):
            cfg0=next(cfg for _,g,cfg in plan('settled_gate_09') if g==grade)
            cfg=replace(cfg0,profile='nominal',replicate=2700+i,scan_duration_s=.32,torque_Nm=12.)
            try:
                row,sc=rp.run_trial(p,grade,res_dir/'episodes',record_video=False,cfg=cfg,
                                    persistent_scene=None,return_scene=True)
                pos,vel=p.getJointState(sc['robot'],sc['joints']['stage_x_move'])[:2]
                dims=row.get('simulation_stage_durations_s') or {}
                exp_ok=(row.get('classification',{}).get('class')==grade
                        and row.get('assigned_nest')==SLOTS[grade]
                        and bool(row.get('success_physical_and_ai'))
                        and bool(row.get('all_retained_fruits_ok'))
                        and bool(row.get('return_home_pass'))
                        and bool((row.get('post_return_check') or {}).get('ok')))
                speed_gate=abs(pos)<.030 and abs(vel)<=.100
                result={'mode':mode,'grade':grade,'slot':row.get('assigned_nest'),
                        'recorded_success':bool(exp_ok), 'velocity_gate_pass':bool(speed_gate),
                        'cycle_s':row.get('simulated_time_s'),
                        'return_command_s':dims.get('return_to_scale'),
                        'home_verify_s':dims.get('home_arrival_verify',0),
                        'home_x_m':float(pos),'home_vx_m_s':float(vel),
                        'apple_cup_ok':bool((row.get('post_return_check') or {}).get('ok')),
                        'max_floor_contact_N':(row.get('placement') or {}).get('max_floor_contact_N'),
                        'classification':row.get('classification',{}).get('class'),
                        'stages':dims}
            except Exception as e:
                result={'mode':mode,'grade':grade,'recorded_success':False,'velocity_gate_pass':False,'error':repr(e)}
            all_rows.append(result)
            (res_dir/f'case_{grade}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
            print('CASE',mode,grade, 'SUCCESS',result['recorded_success'],
                  'CYCLE',result.get('cycle_s'),'HOME_V',result.get('home_vx_m_s'),flush=True)
    finally:
        if p.isConnected():p.disconnect()
    (res_dir/'worker_results.json').write_text(json.dumps(all_rows,ensure_ascii=False,indent=2),encoding='utf8')
    return 0 if len(all_rows)==4 and all(x['recorded_success'] and x['velocity_gate_pass'] for x in all_rows) else 2

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='stage30_compare_results')
    p.add_argument('--modes',default=','.join(MODES));p.add_argument('--internal',action='store_true')
    p.add_argument('--engine');p.add_argument('--mode');a=p.parse_args()
    if a.internal:return trial_worker(a.mode,a.engine)
    if importlib.util.find_spec('pybullet') is None:
        print('PHYSICS_NOT_RUN: PyBullet unavailable. Run Colab notebook.',file=sys.stderr);return 3
    modes=a.modes.split(',')
    if not modes or any(m not in MODES for m in modes) or len(set(modes))!=len(modes):p.error('bad modes')
    out=Path(a.output).resolve()
    if out.exists():raise RuntimeError('No overwriting existing result folder '+str(out))
    out.mkdir(parents=True)
    results=[];failed=[]
    for mode in modes:
        eng=source_clone(mode,out/mode)
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--internal','--engine',str(eng),'--mode',mode]
        print('===',mode,'===',flush=True)
        rc=subprocess.call(cmd,cwd=out/mode)
        f=out/mode/'worker_results.json'
        if f.is_file():results+=json.loads(f.read_text(encoding='utf8'))
        if rc: failed.append({'mode':mode,'exit_code':rc})
    fields=['mode','grade','slot','recorded_success','velocity_gate_pass','cycle_s','return_command_s','home_verify_s','home_x_m','home_vx_m_s','apple_cup_ok']
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows({k:r.get(k) for k in fields} for r in results)
    summ={'time_jst':jst(),'physics_executed':len(results)>0,'total_trials':len(results),'expected':4*len(modes),
          'source_zip_sha256':sha(SOURCE_ZIP.read_bytes()),'failed_modes':failed,
          'eligible_n':sum(r.get('recorded_success',False) and r.get('velocity_gate_pass',False) for r in results),
          'limitation':'4 isolated apple grades/mode, real hardware not validated. Gate 0.1m/s is provisional.'}
    (out/'summary.json').write_text(json.dumps(summ,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summ,ensure_ascii=False,indent=2),flush=True)
    return 0 if len(results)==4*len(modes) and not failed else 2

if __name__=='__main__':raise SystemExit(main())
