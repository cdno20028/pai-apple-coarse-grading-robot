#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage33 (Stage32 camera-domain correction): load-triggered apple-ID gate -> unchanged Stage31 visual A/B/C/D -> Stage30 physics.

FAIL-CLOSED: no gripper descent occurs before the apple-vs-other decision.
The load signal is a PyBullet plate contact-force proxy, not a physical sensor.
Negative baseball/tennis checks are synthetic-camera offline tests, not Bullet trials.
"""
from __future__ import annotations
import argparse,csv,datetime,hashlib,importlib.util,json,os,subprocess,sys,time
from dataclasses import replace
from pathlib import Path
import joblib
import numpy as np
import object_gate_ai as obj
import visual_grade_ai as grade
import compare_stage30 as prev

ROOT=Path(__file__).resolve().parent
GRADES=grade.LABELS
SLOTS={'A':'A3','B':'B3','C':'C1','D':'D1'}


def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y/%m/%d %H:%M:%S JST')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_models(folder):
    """Train ONCE before trial runs; save both models and training wall times."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    files=(folder/'apple_identifier.joblib',folder/'abcd_visual_grade.joblib')
    if any(f.exists() for f in files):
        if not all(f.exists() for f in files):raise FileExistsError('Partial model store; do not silently overwrite')
        return {'reused_existing':True,'model_files':[x.name for x in files]}
    start=time.perf_counter();gate=obj.train_object_model();gate_s=time.perf_counter()-start
    start=time.perf_counter();grade_model=grade.train_visual_model();grade_s=time.perf_counter()-start
    joblib.dump(gate,files[0]);joblib.dump(grade_model,files[1])
    details={'date_jst':now(),'reused_existing':False,'apple_detector_train_wall_s':round(gate_s,4),
             'grade_classifier_train_wall_s':round(grade_s,4),
             'labels_gate':list(obj.OBJECT_TYPES),'labels_grade':list(GRADES),
             'trained_on':'synthetic UV fruit, simulated ball images; NOT real photographs',
             'model_sha256':{f.name:sha(f) for f in files}}
    (folder/'training_manifest.json').write_text(json.dumps(details,ensure_ascii=False,indent=2),encoding='utf8')
    return details


def patch_gate(source_text):
    anchor='    scale_mass=plate_norm/9.81 if plate_norm>0 else None\n'
    assert source_text.count(anchor)==1
    # Sensor is checked only after plate settling; no descent/scan is done on rejection.
    patch='''    _stage32_gate_cb=globals().get('STAGE32_PREGRIP_GATE')
    _stage32_gate_record=None
    if _stage32_gate_cb is not None:
        # 3 samples (initial + 2 x 1/240s) prevent transient load from
        # initiating descent. This costs 0.00833s of simulated time.
        _stage32_load_samples=[scale_mass]
        for _ in range(2):
            send(init);step_and_trace('stage32_load_debounce')
            _contact_force=sum(max(0,cp[9]) for cp in p.getContactPoints(apple,plate))
            _stage32_load_samples.append(_contact_force/9.81 if _contact_force>0 else None)
        _stage32_stable_load=(min(float(x) for x in _stage32_load_samples)
            if all(x is not None and float(x)>=.025 for x in _stage32_load_samples)
            else None)
        _stage32_gate_record=_stage32_gate_cb(p,robot,joints,_stage32_stable_load)
        _stage32_gate_record['load_samples_kg']=_stage32_load_samples
        if not _stage32_gate_record['allow_descent']:
            rejected={'status':'STAGE32_REJECTED_BEFORE_DESCENT',
                'success_physical_and_ai':False,'sort_pass':False,
                'stage32_gate':_stage32_gate_record,
                'classification':{'status':'NOT_STARTED','class':None},
                'simulated_time_s':n*DT,
                'simulation_stage_durations_s':{k:round(v*DT,5) for k,v in stage_steps.items()},
                'wall_clock_time_including_rendering_and_ai_s':round(time.perf_counter()-wall_started,4),
                'descent_executed':False,'gripper_closed':False,'uses_mass_for_grading':False}
            (loc/'trial.json').write_text(json.dumps(rejected,ensure_ascii=False,indent=2),encoding='utf8')
            if recorder:recorder.close(False,n*DT)
            return (rejected,scene) if return_scene else rejected
'''
    new=source_text.replace(anchor,anchor+patch,1)
    target="    report={'world_geometry':"
    assert new.count(target)==1
    new=new.replace(target,"    report={'stage32_gate':locals().get('_stage32_gate_record'),'world_geometry':",1)
    compile(new,'run_physics.py','exec')
    return new


def clone_engine(base):
    """Original Stage27 hashes unchanged; Stage30 safe home gate + Stage32 stop interlock."""
    engine=prev.source_clone('baseline_safe',Path(base))
    f=engine/'apple_stage13/src/run_physics.py'; before=f.read_bytes()
    s=patch_gate(before.decode('utf8'))
    f.write_text(s,encoding='utf8')
    metadata={'stage':'33','time_jst':now(),'original_stage27_run_physics_sha256':prev.EXPECTED_SOURCE_SHA,
              'stage30_pre_gate_sha256':hashlib.sha256(before).hexdigest(),
              'stage32_patched_sha256':sha(f),'urdf_sha256':sha(engine/'apple_stage13/robot_v14_phase30.urdf'),
              'interlock':'after mass threshold sensor, before scan_and_close',
              'mass_for_grade':False,'no_real_ball_physics_claim':True,
              'return_speed_gate':'position<30mm, speed<=0.1m/s for 10 simulation steps'}
    (Path(base)/'stage32_patch_manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf8')
    return engine


def camera_palm_only(p,robot,joints,side=144):
    """Uses same fixed palm camera geometry as original Stage27; captures ONE upper view."""
    position,rotation=p.getLinkState(robot,joints['palm_camera_fixed'],computeForwardKinematics=True)[4:6]
    eye=np.asarray(position);R=np.asarray(p.getMatrixFromQuaternion(rotation)).reshape(3,3)
    target=eye+R@np.array([0.,0.,-.30]);up=R@np.array([0.,1.,0.])
    vm=p.computeViewMatrix(eye.tolist(),target.tolist(),up.tolist())
    from mechanism_v09_reference import G
    proj=p.computeProjectionMatrixFOV(fov=G.camera_fov_deg,aspect=1,nearVal=.005,farVal=.8)
    _,_,rgba,_,_=p.getCameraImage(side,side,viewMatrix=vm,projectionMatrix=proj,renderer=p.ER_TINY_RENDERER)
    return np.asarray(rgba,dtype=np.uint8).reshape(side,side,4)[:,:,:3]




def modify_recording_overlay(engine):
    """Stage35 only: modify visualization and end-slate, never physics/control."""
    path=Path(engine)/'apple_stage13'/'src'/'video_capture.py'
    old=path.read_text(encoding='utf8')
    pairs=[
      ("'Synthetic colour/weight labels, NOT real fruit grading'", "'Synthetic apple; AI based on rendered visual features'"),
      ('for _ in range(20):self.writer.append_data(np.asarray(im))', 'for _ in range(0):self.writer.append_data(np.asarray(im))'),
      ("f'APPLE SORTING — PHYSICS TRIAL {self.label}'", "f'PyBullet PHYSICAL MOTION  |  Grade {self.label}'"),
      ("f'{self.timestamp} | PyBullet DIRECT, synthetic apple'", "f'{self.timestamp} | Actual PyBullet render / synthetic apple / independent episode'"),
    ]
    for orig,repl in pairs:
       if old.count(orig)!=1:raise ValueError('VideoCapture source mismatch: '+orig)
       old=old.replace(orig,repl,1)
    compile(old,str(path),'exec')
    path.write_text(old,encoding='utf8')

def worker(mode,engine):
    if importlib.util.find_spec('pybullet') is None:raise RuntimeError('PHYSICS_NOT_RUN: install pybullet in Colab')
    engine=Path(engine).resolve(); app=engine/'apple_stage13'
    sys.path[:0]=[str(engine),str(app),str(app/'src')]
    import pybullet as p
    import run_physics as rp
    from run_stage03 import plan
    from stage23_full_supervisor import FullGradeSupervisor
    model_dir=ROOT/'models'
    # Load ONCE at worker startup, BEFORE starting trial stopwatch.
    startup=time.perf_counter()
    identifier=joblib.load(model_dir/'apple_identifier.joblib')
    grade_model=joblib.load(model_dir/'abcd_visual_grade.joblib')
    load_time=(time.perf_counter()-startup)*1000
    timing={'training_during_trial':False,'model_load_wall_ms_once_per_worker':load_time}
    rp.train=lambda:grade_model
    def measure_grade(model,images,scale_mass_kg):
        start=time.perf_counter()
        result=grade.infer_visual(model,images,mode=mode)
        result['grade_infer_wall_ms']=round((time.perf_counter()-start)*1000,4)
        result['mass_ignored_for_grading']=True
        return result
    rp.infer=measure_grade
    def pregrip_gate(p,robot,joints,mass):
        start=time.perf_counter();event=obj.detect_placement(mass)
        if not event.triggered:
            return {'allow_descent':False,'load':event.__dict__,'upper_image_captured':False,
                    'top_capture_wall_ms':0.,'apple_id_infer_wall_ms':0.,
                    'gate_wall_ms':round((time.perf_counter()-start)*1000,4)}
        image_start=time.perf_counter()
        frame=camera_palm_only(p,robot,joints)
        camera_ms=(time.perf_counter()-image_start)*1000
        # Save the actual Bullet palm view separately for later false-reject audit.
        # Do not use the file name / known fruit label as classifier input.
        from PIL import Image
        Image.fromarray(frame).save(directory/f'upper_view_{i:03d}_{truth}.png')
        result=obj.identify_top_frame(identifier,frame)
        return {'allow_descent':bool(result['accepted']),'load':event.__dict__,
                'upper_image_captured':True,'top_capture_wall_ms':round(camera_ms,4),
                'apple_id_infer_wall_ms':round(result['infer_wall_ms'],4),
                'object':result,'gate_wall_ms':round((time.perf_counter()-start)*1000,4)}
    rp.STAGE32_PREGRIP_GATE=pregrip_gate
    rp.MOTION_SETTINGS['settled_gate_09']=replace(rp.MOTION_SETTINGS['settled_gate_09'],x_effective_rate=1.45)
    rp.STAGE23_AFTER_INFERENCE_HOOK=FullGradeSupervisor().process
    os.environ.update({'APPLE_NEST_ORDER':'nearest','STAGE20_V40D_LOWER_RAMP_S':'0.08',
                       'STAGE22_V42_RETURN_SCALE':'0.65'})
    directory=engine.parent;rows=[]
    if p.connect(p.DIRECT)<0:raise RuntimeError('PyBullet cannot connect')
    try:
        for i,truth in enumerate(GRADES*int(os.environ.get("STAGE35_VIDEO_CYCLES","1")),1):
            cfg0=next(cfg for _,g,cfg in plan('settled_gate_09') if g==truth)
            cfg=replace(cfg0,profile='nominal',replicate=3200+i,scan_duration_s=.32,torque_Nm=12.)
            try:
                report,sc=rp.run_trial(p,truth,directory/'episodes',record_video=True,cfg=cfg,
                                       persistent_scene=None,return_scene=True)
                gate=report.get('stage32_gate') or {}
                classification=report.get('classification') or {}
                phys_success=bool(report.get('success_physical_and_ai'))
                accepted=bool(gate.get('allow_descent'))
                if accepted:
                    pos,vel=p.getJointState(sc['robot'],sc['joints']['stage_x_move'])[:2]
                else:
                    pos=vel=None
                all_ok=(accepted and classification.get('class')==truth and
                        report.get('assigned_nest')==SLOTS[truth] and phys_success and
                        report.get('return_home_pass') and
                        bool((report.get('post_return_check') or {}).get('ok')) and
                        pos is not None and abs(pos)<.030 and abs(vel)<=.1)
                r={'mode':mode,'synthetic_apple':truth,'object_gate_accepted':accepted,
                   'gate_status':(gate.get('object') or {}).get('status'),
                   'load_sensor_kg_trigger':(gate.get('load') or {}).get('mass_kg'),
                   'top_capture_wall_ms':gate.get('top_capture_wall_ms'),
                   'apple_id_infer_wall_ms':gate.get('apple_id_infer_wall_ms'),
                   'grade_infer_wall_ms':classification.get('grade_infer_wall_ms'),
                   'predicted_grade':classification.get('class'),
                   'route':report.get('assigned_nest'),
                   'simulation_cycle_s':report.get('simulated_time_s'),
                   'home_x_m':pos,'home_x_velocity_m_s':vel,
                   'physical_and_ai_success':bool(all_ok),
                   'report_status':report.get('status'),
                   'uses_weight_for_grading':False,'apple_image_real':False,
                   'stage32_gate':gate}
            except Exception as exc:
                r={'mode':mode,'synthetic_apple':truth,'physical_and_ai_success':False,'error':repr(exc)}
            rows.append(r)
            (directory/f'case_{i:03d}_{truth}.json').write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf8')
            print('TRIAL',mode,truth,'detected',r.get('gate_status'),'grade',r.get('predicted_grade'),
                  'nest',r.get('route'),'success',r['physical_and_ai_success'],flush=True)
    finally:
        if p.isConnected():p.disconnect()
    (directory/'worker_results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
    (directory/'model_load_wall_ms.json').write_text(json.dumps(timing,indent=2),encoding='utf8')
    return 0 if len(rows)==4*int(os.environ.get('STAGE35_VIDEO_CYCLES','1')) and all(r['physical_and_ai_success'] for r in rows) else 2


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true')
    ap.add_argument('--train-only',action='store_true');ap.add_argument('--mode',choices=grade.MODES,default='normal')
    ap.add_argument('--output',default='stage33_results_normal');ap.add_argument('--internal',action='store_true')
    ap.add_argument('--engine');ap.add_argument('--cycles',type=int,default=1,help='number of A B C D rounds; separate physics episodes')
    args=ap.parse_args()
    if not (1<=args.cycles<=10):ap.error('cycles must be 1 to 10')
    os.environ['STAGE35_VIDEO_CYCLES']=str(args.cycles)
    if args.internal:return worker(args.mode,args.engine)
    if args.preflight:
        import test_stage32_offline
        test_stage32_offline.main();return 0
    prepare_models(ROOT/'models')
    if args.train_only:
        print('MODELS_PRETRAINED_ONCE',str(ROOT/'models'));return 0
    if importlib.util.find_spec('pybullet') is None:
        print('PHYSICS_NOT_RUN: pybullet is unavailable; use Colab.',file=sys.stderr);return 3
    output=Path(args.output).resolve()
    if output.exists():raise FileExistsError('No overwrite '+str(output))
    engine=clone_engine(output)
    modify_recording_overlay(engine)
    rc=subprocess.call([sys.executable,'-u',str(Path(__file__).resolve()),'--internal',
                        '--engine',str(engine),'--mode',args.mode],cwd=output)
    path=output/'worker_results.json'
    rows=json.loads(path.read_text(encoding='utf8')) if path.exists() else []
    keys=['mode','synthetic_apple','object_gate_accepted','gate_status','load_sensor_kg_trigger',
          'top_capture_wall_ms','apple_id_infer_wall_ms','grade_infer_wall_ms','predicted_grade',
          'route','simulation_cycle_s','physical_and_ai_success']
    with (output/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r.get(k) for k in keys} for r in rows)
    summary={'date_jst':now(),'status':'PHYSICS_EXECUTED' if rows else 'PHYSICS_NOT_COMPLETED',
             'grade_mode':args.mode,'count':len(rows),'successful':sum(r.get('physical_and_ai_success',False) for r in rows),
             'stage27_source_sha256':prev.EXPECTED_SOURCE_SHA,'load_switch_only':True,
             'uses_weight_for_grading':False,'nonapple_physical_tests_executed':False,
             'real_apples_or_balls_validated':False,'AI_wall_clock_is_not_simulated_time':True,
             'any_failure_or_unverified':bool(rc or len(rows)!=4)}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print('STAGE32_SUMMARY',json.dumps(summary,ensure_ascii=False),flush=True)
    return 0 if not rc and len(rows)==4*int(os.environ.get('STAGE35_VIDEO_CYCLES','1')) else 2

if __name__=='__main__':raise SystemExit(main())
