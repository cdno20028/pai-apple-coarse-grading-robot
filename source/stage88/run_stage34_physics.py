#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage34: simulated SPORTS BALLS ON THE ACTUAL PYBULLET PLATE, view from
unchanged Stage33 palm camera; object gate must stop gripper before descent.

6 negative 3D physics episodes: 2 variants each of baseball, tennis, red rubber ball.
Stage33 apples remain 4 separate positive physics regression episodes.
This script NEVER fakes a camera image or overrides an AI classification.
A false acceptance triggers a real negative-test failure and full trace is saved.
"""
from __future__ import annotations
import argparse,csv,datetime,hashlib,importlib.util,json,os,subprocess,sys,time
from dataclasses import replace
from pathlib import Path
import joblib
import numpy as np
from PIL import Image
import run_stage33_physics as previous
import object_gate_ai as obj
import visual_grade_ai as grade
import ball_scene_assets as sports
ROOT=Path(__file__).resolve().parent
CASES=[(k,v) for k in sports.KINDS for v in (0,1)]
SAFE_STAGES={'station_preflight','wait_for_worker','scale_settle','stage32_load_debounce'}


def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y/%m/%d %H:%M:%S JST')
def sha(fp):return hashlib.sha256(Path(fp).read_bytes()).hexdigest()


def worker(engine):
    if importlib.util.find_spec('pybullet') is None:raise RuntimeError('PHYSICS_NOT_RUN: pybullet missing')
    engine=Path(engine).resolve()
    app=engine/'apple_stage13'
    sys.path[:0]=[str(engine),str(app),str(app/'src')]
    import pybullet as p
    import run_physics as rp
    from run_stage03 import plan
    from stage23_full_supervisor import FullGradeSupervisor
    from mechanism_v09_reference import G

    identifier=joblib.load(ROOT/'models'/'apple_identifier.joblib')
    grade_model=joblib.load(ROOT/'models'/'abcd_visual_grade.joblib')
    rp.train=lambda:grade_model
    def infer_normal(model,frames,mass_kg):
        t=time.perf_counter()
        result=grade.infer_visual(model,frames,mode='normal')
        result['grade_infer_wall_ms']=round((time.perf_counter()-t)*1000,4)
        result['mass_ignored_for_grading']=True
        return result
    rp.infer=infer_normal
    rp.MOTION_SETTINGS['settled_gate_09']=replace(rp.MOTION_SETTINGS['settled_gate_09'],x_effective_rate=1.45)
    rp.STAGE23_AFTER_INFERENCE_HOOK=FullGradeSupervisor().process
    os.environ.update({'APPLE_NEST_ORDER':'nearest','STAGE20_V40D_LOWER_RAMP_S':'0.08',
                       'STAGE22_V42_RETURN_SCALE':'0.65'})
    unmodified_spawn=rp.spawn_textured_apple
    out=engine.parent
    sports_folder=out/'sports_ball_assets'
    sports.static_checks()
    rows=[]
    if p.connect(p.DIRECT)<0:raise RuntimeError('cannot connect to PyBullet')
    try:
        for serial,(kind,variant) in enumerate(CASES,1):
            case_name=f'{kind}_v{variant}'
            trial_stages=[]
            captured={'body_id':None,'upper_view_sha256':None}
            # ONLY physical fruit body creation is substituted by a dynamic sphere;
            # robot/plate/trays/controller/vision feature code remain unchanged.
            def spawn_replacement(p_client,label,mass,radius,xyz,friction=1.,*,_kind=kind,_variant=variant):
                xyz_ball=[float(xyz[0]),float(xyz[1]),float(xyz[2])+sports.SPECS[_kind]['radius_m']-radius]
                body=sports.spawn_ball(p_client,_kind,_variant,xyz_ball,sports_folder)
                captured['body_id']=int(body)
                return body
            rp.spawn_textured_apple=spawn_replacement
            def watch_stages(p_client,robot,body,plate,joints,stage,t,cfg):
                trial_stages.append(stage)
            rp._STAGE_CONTACT_OBSERVER=watch_stages
            def gate(p_client,robot,joints,load_kg,*,_case=case_name):
                start=time.perf_counter()
                event=obj.detect_placement(load_kg)
                if not event.triggered:
                    return {'allow_descent':False,'load':event.__dict__,
                       'upper_image_captured':False,'top_capture_wall_ms':0.,'apple_id_infer_wall_ms':0.,
                       'gate_wall_ms':round((time.perf_counter()-start)*1000,4)}
                cstart=time.perf_counter()
                frame=previous.camera_palm_only(p_client,robot,joints)
                capture_ms=(time.perf_counter()-cstart)*1000
                img_path=out/f'upper_view_{_case}.png'
                Image.fromarray(frame).save(img_path)
                captured['upper_view_sha256']=sha(img_path)
                prediction=obj.identify_top_frame(identifier,frame)
                return {'allow_descent':bool(prediction['accepted']),
                    'load':event.__dict__,'upper_image_captured':True,
                    'top_capture_wall_ms':round(capture_ms,4),
                    'apple_id_infer_wall_ms':round(prediction['infer_wall_ms'],4),
                    'object':prediction,'gate_wall_ms':round((time.perf_counter()-start)*1000,4),
                    'source':'PyBullet palm_camera_fixed TinyRenderer 144x144 actual image; no mock injection'}
            rp.STAGE32_PREGRIP_GATE=gate
            # A here denotes only the trial-config mechanical template. The physically
            # spawned object is the negative ball, and the AI sees image pixels only.
            cfg0=next(cfg for _,g,cfg in plan('settled_gate_09') if g=='A')
            cfg=replace(cfg0,profile='nominal',replicate=3400+serial,
                        scan_duration_s=.32,torque_Nm=12.)
            start=time.perf_counter()
            try:
                report,scene=rp.run_trial(p,'A',out/'episodes'/case_name,record_video=False,
                                          cfg=cfg,persistent_scene=None,return_scene=True)
                gaterec=report.get('stage32_gate') or {}
                prediction=gaterec.get('object') or {}
                load=gaterec.get('load') or {}
                actual_stages=sorted(set(trial_stages))
                physical_trigger=bool(load.get('triggered')) and bool(gaterec.get('upper_image_captured'))
                refused=not bool(gaterec.get('allow_descent'))
                stopped=(report.get('status')=='STAGE32_REJECTED_BEFORE_DESCENT'
                        and not bool(report.get('descent_executed',True))
                        and not bool(report.get('gripper_closed',True))
                        and not bool(set(trial_stages)-SAFE_STAGES)
                        and (report.get('classification') or {}).get('status')=='NOT_STARTED')
                nonapple_type=prediction.get('predicted_object') not in (None,'apple')
                fail_closed=physical_trigger and refused and stopped
                # Strict correct class is tracked separately: wrong nonapple label
                # still blocks descent but must not masquerade as class accuracy.
                strict_class=prediction.get('predicted_object')==kind
                body_id=captured['body_id']
                body_mass=p.getDynamicsInfo(body_id,-1)[0] if body_id is not None else None
                actual_xyz=list(p.getBasePositionAndOrientation(body_id)[0]) if body_id is not None else None
                home_state=p.getJointState(scene['robot'],scene['joints']['stage_z_move'])
                row={'case':case_name,'physical_truth':kind,'variant':variant,
                     'physical_body':'collision sphere + UV mesh texture',
                     'nominal_mass_kg':sports.SPECS[kind]['mass_kg'],
                     'pybullet_dynamics_mass_kg':body_mass,
                     'ball_radius_m':sports.SPECS[kind]['radius_m'],
                     'scene_body_id':body_id,'body_xyz_m_at_end':actual_xyz,
                     'load_samples_kg':gaterec.get('load_samples_kg'),
                     'load_event':load,'upper_image_captured':gaterec.get('upper_image_captured'),
                     'upper_image_file':f'upper_view_{case_name}.png' if captured['upper_view_sha256'] else None,
                     'upper_image_sha256':captured['upper_view_sha256'],
                     'camera_wall_ms':gaterec.get('top_capture_wall_ms'),
                     'ai_object_infer_wall_ms':gaterec.get('apple_id_infer_wall_ms'),
                     'predicted_object':prediction.get('predicted_object'),
                     'prediction_confidence':prediction.get('confidence'),
                     'apple_probability':(prediction.get('scores') or {}).get('apple'),
                     'prediction_status':prediction.get('status'),
                     'allow_descent':gaterec.get('allow_descent'),
                     'trial_status':report.get('status'),
                     'simulation_cycle_s':report.get('simulated_time_s'),
                     'classification':report.get('classification'),
                     'actual_stages':actual_stages,'stage_counts':{stage:trial_stages.count(stage) for stage in actual_stages},
                     'descent_executed':any(x not in SAFE_STAGES for x in trial_stages),
                     'robot_z_joint_m_at_end':float(home_state[0]),
                     'load_trigger_and_imaged':bool(physical_trigger),
                     'reject_before_descent':bool(fail_closed),
                     'classified_as_nonapple':bool(nonapple_type),
                     'correct_specific_ball_class':bool(strict_class),
                     'strict_negative_pass':bool(fail_closed and nonapple_type),
                     'time_wall_s':round(time.perf_counter()-start,3),
                     'photo_is_real_ball':False,'material_is_modelled':True,
                     'ai_label_from_truth_hardcoded':False,'mass_used_for_grade':False}
            except Exception as e:
                row={'case':case_name,'physical_truth':kind,'variant':variant,
                     'error':repr(e),'strict_negative_pass':False,
                     'actual_stages':sorted(set(trial_stages)),
                     'descent_executed':any(x not in SAFE_STAGES for x in trial_stages)}
            rows.append(row)
            (out/f'case_{case_name}.json').write_text(json.dumps(row,indent=2,ensure_ascii=False),encoding='utf8')
            print('NEGATIVE_CASE',case_name,'load',row.get('load_trigger_and_imaged'),
                  'predicted',row.get('predicted_object'),'descent',row.get('descent_executed'),
                  'pass',row.get('strict_negative_pass'),flush=True)
    finally:
        rp.spawn_textured_apple=unmodified_spawn
        rp._STAGE_CONTACT_OBSERVER=None
        rp.STAGE32_PREGRIP_GATE=None
        if p.isConnected():p.disconnect()
    (out/'worker_results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
    return 0 if len(rows)==len(CASES) and all(x['strict_negative_pass'] for x in rows) else 2


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default='stage34_negative_results')
    ap.add_argument('--train-only',action='store_true')
    ap.add_argument('--preflight',action='store_true')
    ap.add_argument('--internal',action='store_true')
    ap.add_argument('--engine')
    args=ap.parse_args()
    if args.internal:return worker(args.engine)
    if args.preflight:
        import test_stage34_offline
        return test_stage34_offline.check()
    if args.train_only:
        print(previous.prepare_models(ROOT/'models'))
        return 0
    if importlib.util.find_spec('pybullet') is None:
        print('PHYSICS_NOT_RUN: pybullet not installed here. Use Colab.',file=sys.stderr)
        return 3
    previous.prepare_models(ROOT/'models')
    out=Path(args.output).resolve()
    if out.exists():raise FileExistsError(f'No overwrite: {out}')
    eng=previous.clone_engine(out)
    manifest={'stage':34,'time_jst':now(),
       'engine_stage33_patched_sha256':sha(eng/'apple_stage13/src/run_physics.py'),
       'original_stage27_sha256':previous.prev.EXPECTED_SOURCE_SHA,
       'urdf_sha256':sha(eng/'apple_stage13/robot_v14_phase30.urdf'),
       'object_cases':[{'kind':k,'variant':v,**sports.SPECS[k]} for k,v in CASES],
       'real_pybullet_renders':True,'real_world_balls':False,
       'no_prior_gripper_code_changes':True,
       'mass_for_start_signal_only':True,
       'no_gripper_descent_expected_for_negative_cases':True}
    (out/'stage34_experiment_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    rc=subprocess.call([sys.executable,'-u',str(Path(__file__).resolve()),'--internal','--engine',str(eng)],cwd=out)
    rowsfile=out/'worker_results.json'
    rows=json.loads(rowsfile.read_text(encoding='utf8')) if rowsfile.is_file() else []
    keys=['case','physical_truth','variant','pybullet_dynamics_mass_kg','load_trigger_and_imaged',
         'predicted_object','prediction_confidence','apple_probability','allow_descent',
         'trial_status','descent_executed','camera_wall_ms','ai_object_infer_wall_ms',
         'reject_before_descent','classified_as_nonapple','strict_negative_pass']
    with (out/'results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        writer.writerows({key:r.get(key) for key in keys} for r in rows)
    summary={'time_jst':now(),'stage':34,'status':'PHYSICS_EXECUTED' if rows else 'PHYSICS_NOT_COMPLETED',
          'negative_cases_expected':len(CASES),'negative_cases_recorded':len(rows),
          'fully_passed':sum(bool(r.get('strict_negative_pass')) for r in rows),
          'correct_specific_ball_label':sum(bool(r.get('correct_specific_ball_class')) for r in rows),
          'false_gripper_descent_cases':[r.get('case') for r in rows if r.get('descent_executed')],
          'test_suite_success':rc==0 and len(rows)==len(CASES),
          'notes':'All negatives are simulated material/geometry. Positive apples tested separately via unchanged Stage33.',
          'model_is_pretrained_before_trials':True,'uses_weight_for_grading':False}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print('STAGE34_RESULT',json.dumps(summary,ensure_ascii=False),flush=True)
    return 0 if summary['test_suite_success'] else 2

if __name__=='__main__':raise SystemExit(main())
