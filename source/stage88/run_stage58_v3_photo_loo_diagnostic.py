#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""松尾研 Physical AI 工程58｜移動中の指先連続撮影・上昇中の下部撮影。

Important:
 - The Stage51 work-cell source is not modified. A fresh COPY of its validated
   engine is used. Stage58 is a SEPARATE test, not a replacement for the
   independently verified uninterrupted 33.5-second work-cell film.
 - At least 3 different wrist scan instants for each of 3 finger cameras.
 - Palm camera used to decide apple/foreign object BEFORE scan/close.
 - Lower inspection camera in original run_physics.images_from_actual_links():
   positioned just above the inspection plate, looking UP.
 - During gravity lift, take lower-side frames at successive fruit heights.
 - The grade inference is actually executed by 50mm fruit displacement,
   using image pixels only. It is an OLD SYNTHETIC-TRAINED visual model,
   not an expert-validated real-apple defect detector.
 - These are one-real-photo-front + synthetic-back texture objects; camera
   coverage limitations remain.
 - The engine still commands full 100mm lift before horizontal motion.
 - Never use the filename's A/B/C/D as input to grading.
"""
from __future__ import annotations
import argparse, csv, datetime, hashlib, importlib.util, json, os, sys, traceback, zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
import joblib
from PIL import ImageFilter

JST = datetime.timezone(datetime.timedelta(hours=9))
CAMERAS=('指先1','指先2','指先3','掌','検査台下部')
SCAN_TARGET_TICKS=(16,34,52,70)
BOTTOM_HEIGHT_TARGETS=(.020,.034,.048)
FPS=20

def stamp():return datetime.datetime.now(JST).strftime('%Y/%m/%d %H:%M:%S JST')
def dump(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def font(size):
    for f in ('/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:return ImageFont.truetype(f,size)
        except OSError:pass
    return ImageFont.load_default()

def error_report(status, reason, data):
    return {'status':status,'reason':reason,'diagnostic':data,'physical_success_not_claimed':True}

def create_storyboard(items,kind,grade,events):
    """A visual CONTACT SHEET of actual frames at distinct measured timestamps.

    Presented as observation evidence, not claimed real-time film; no invented
    frames or timestamp values. Empty camera slots are visibly marked.
    """
    W,H=1280,720
    dst=Image.new('RGB',(W,H),(14,22,39));d=ImageDraw.Draw(dst)
    ttl=font(26);sm=font(17);ft=font(21)
    d.text((20,13),'PAI STAGE58 | MOTION-TIMED 5-CAMERA OBSERVATION',font=ttl,fill='white')
    d.text((20,54),f'PHOTO REF: {kind}  |  MID-LIFT GRADE: {grade or "HOLD"}',font=ft,fill=(184,226,246))
    # Four finger phases from first finger on each row, but three cameras in columns.
    for ci in range(3):
        for ti in range(4):
            x=20 + ci*246; y=108 + ti*124
            arr=items.get(('finger',ci,ti))
            if arr is not None:
                dst.paste(Image.fromarray(arr).resize((112,112)),(x,y))
            else:
                d.rectangle((x,y,x+112,y+112),outline=(160,70,70),width=2)
                d.text((x+9,y+45),'NO DATA',font=sm,fill=(244,144,144))
            if ti==0:d.text((x+119,y+5),f'F{ci+1}',font=ft,fill='white')
            if ci==2:
                times=[e['t_sim_s'] for e in events if e.get('phase')=='finger_approach' and e.get('camera')=='finger_3' and e.get('scan_sample')==ti+1]
                d.text((x+119,y+42),f'T{ti+1} / {times[0]:.2f}s' if times else f'T{ti+1}',font=sm,fill=(204,222,241))
    # Bottom progression and palm.
    xx=790
    d.text((xx,108),'Palm: before finger descent',font=sm,fill='#e2f1ff')
    a=items.get(('palm',0,0))
    if a is not None:dst.paste(Image.fromarray(a).resize((126,126)),(xx,136))
    d.text((xx,296),'Lower camera: rising fruit',font=sm,fill='#e2f1ff')
    for k,delta in enumerate((.020,.034,.048)):
        a=items.get(('bottom',0,k));x=xx+k*151;y=326
        if a is not None:dst.paste(Image.fromarray(a).resize((137,137)),(x,y))
        else:d.rectangle((x,y,x+137,y+137),outline='#c66b72',width=2)
        ev=[e for e in events if e.get('phase')=='underside_during_lift' and e.get('file')==f'under_plate_lift_{k+1:02d}.png']
        d.text((x,y+144),f'Z {ev[0]["apple_lift_delta_m"]*1000:.0f}mm' if ev else f'Z ~ {int(delta*1000)}mm',font=sm,fill='#cde6fd')
        if ev:d.text((x,y+164),f't={ev[0]["t_sim_s"]:.2f}s',font=sm,fill='#cde6fd')
    d.text((20,630),'Actual PyBullet camera frames sampled along finger motion and gravity lift.',font=sm,fill=(222,229,245))
    d.text((20,660),'Stage57 photo-front + synthetic-back; legacy synthetic grade AI (NOT orchard-validated).',font=sm,fill=(240,204,173))
    d.text((20,690),stamp()+' | diagnostic presentation, not real-time sensor frame rate',font=sm,fill=(180,204,222))
    return dst

EXTERNAL_TENNIS_GATE_FEATURES = [0.6705882549285889, 0.7372549176216125, 0.16470588743686676, 0.07549918442964554, 0.10365188121795654, 0.10193067044019699, 0.5686274766921997, 0.5843137502670288, 0.12941177189350128, 0.7411764860153198, 0.8156862854957581, 0.239215686917305, 0.06588855385780334, 0.8783885836601257, 0.03614457696676254, 0.0, 0.013554217293858528, 0.2881944477558136, 0.025332922115921974, -0.017254114151000977]


def prepare_photo_loo_gate(root, gate):
    """Diagnostic leave-one-APPLE-photo-out models, NOT farm-ready object recognition.

    The trial target is omitted from its own training. Selection by asset ID is
    an experimental cross-validation schedule, NOT a deployable on-line gate.
    Negative examples are procedural, plus separately checked actual Bullet
    tennis frame used ONLY for holdout rejection, not model training.
    """
    from sklearn.ensemble import RandomForestClassifier
    prior=root/'stage58_v2_timed_5camera_5cm_results'
    paths=sorted(prior.glob('case_*/palm_object_gate.png'))
    if len(paths)!=8:
        raise RuntimeError('Need 8 prior Stage58 V2 PALM frames for LOO calibration, got '+str(len(paths))+': '+str(prior))
    imgs=[np.asarray(Image.open(p).convert('RGB')) for p in paths]
    ids=[p.parent.name.split('_',2)[-1] for p in paths]
    if len(set(ids))!=8:raise RuntimeError('Duplicate photo identifiers in calibration')
    rng=np.random.default_rng(44)
    def augment(arr,j):
        a=np.asarray(arr,dtype=np.float32).copy()
        bg=np.array([162,167,174],dtype=np.float32)
        inside=(np.max(np.abs(a-bg),axis=2)>25)&(a[:,:,0]>a[:,:,2]*1.1)
        a[inside]*=float(rng.uniform(.81,1.21))
        if j%3==0:
            a=np.asarray(Image.fromarray(np.uint8(np.clip(a,0,255))).filter(ImageFilter.GaussianBlur(radius=.50)))
        return np.uint8(np.clip(a,0,255))
    neg_X=[];neg_y=[]
    for kind in ('tennis','baseball','red_ball'):
        for i in range(80):
            frame=(gate.simulated_ball_bullet_background(kind,1000+i) if i%2
                    else gate.mock_ball_frame(kind,900+i))
            feats=gate.object_features(frame)
            if feats is not None:neg_X.append(feats);neg_y.append(kind)
    # This 20-element feature is from a REAL PyBullet gate image of a synthetic
    # tennis ball from Stage54/55. It was not part of calibration training.
    tennis_feat=np.asarray(EXTERNAL_TENNIS_GATE_FEATURES,dtype=np.float32)
    models={};diagnostics=[]
    for held in range(len(imgs)):
        X=[];ys=[]
        for j,img in enumerate(imgs):
            if j==held:continue
            for k in range(45):
                feat=gate.object_features(augment(img,k))
                if feat is not None:X.append(feat);ys.append('apple')
        X+=neg_X;ys+=neg_y
        model=RandomForestClassifier(n_estimators=150,min_samples_leaf=2,
                random_state=13,class_weight='balanced',n_jobs=1)
        model.fit(np.asarray(X),ys)
        hold=gate.identify_top_frame(model,imgs[held],confidence_floor=.76)
        # Check real-Bullet-tennis feature directly; no fake RGB reconstruction.
        pp=model.predict_proba(tennis_feat[None,:])[0]
        scores={str(c):float(v) for c,v in zip(model.classes_,pp)}
        tennis_allow=bool(str(model.classes_[int(np.argmax(pp))])=='apple' and max(pp)>=.76)
        models[ids[held]]=model
        diagnostics.append({'held_out_asset_id':ids[held],
            'held_out_apple':hold,'external_bullet_tennis_scores':scores,
            'external_bullet_tennis_wrongly_accepted':tennis_allow,
            'training_apples_from_other_source_photos':7,
            'training_positive_frame_augmentations':7*45,
            'training_negative_mock_features':len(neg_y)})
        if tennis_allow:raise RuntimeError('LOO calibration rejected: known external tennis would pass: '+ids[held])
    return models,diagnostics


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--assets',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();root=a.root.resolve();assets=a.assets.resolve();out=a.output.resolve()
    if out.exists():raise FileExistsError('Results exist; do not overwrite: '+str(out))
    # Stage51.ensure_engine(out) creates `out/engine` itself. Never mkdir(out)
    # before calling it: Stage30's source_clone refuses any existing `out`.
    # Only the PARENT may be created here; each retry gets a fresh out path.
    out.parent.mkdir(parents=True,exist_ok=True)
    if not (assets/'manifest.json').is_file():
        raise FileNotFoundError('Stage57 prepared assets missing. Run Stage57 in THIS Colab runtime: '+str(assets))
    ameta=json.loads((assets/'manifest.json').read_text(encoding='utf-8'))
    for asset in ameta['assets']:
        fn=assets/asset['texture_path']
        if not fn.is_file():raise FileNotFoundError(str(fn))
    for req in ('run_stage51_postgate_scan_overlap_v2.py','run_stage35_motion.py','visual_grade_ai.py','object_gate_ai.py'):
        if not (root/req).is_file():raise FileNotFoundError('Existing Stage51 environment missing '+req)
    sys.path.insert(0,str(root))
    import visual_grade_ai as grade_ai
    import object_gate_ai as object_ai
    import run_stage35_motion as s35
    sspec=importlib.util.spec_from_file_location('stage51_parent_for_stage58',root/'run_stage51_postgate_scan_overlap_v2.py')
    S51=importlib.util.module_from_spec(sspec);sspec.loader.exec_module(S51)
    engine=S51.ensure_engine(out)
    app=engine/'apple_stage13'
    sys.path[:0]=[str(engine),str(app),str(app/'src')]
    import pybullet as p
    import run_physics as rp
    from run_stage03 import plan
    from mechanism_v09_reference import G
    from workbench_geometry import LAYOUT
    if p.connect(p.DIRECT)<0:raise RuntimeError('PyBullet DIRECT failed')
    obj_model=joblib.load(root/'models'/'apple_identifier.joblib')
    photo_gate_models,photo_gate_diagnostics=prepare_photo_loo_gate(root,object_ai)
    dump(out/'stage58_v3_loo_gate_calibration.json',photo_gate_diagnostics)
    grade_model=joblib.load(root/'models'/'abcd_visual_grade.joblib')
    # Stage51 motor settings, evaluated on a CLONED engine only.
    base=rp.MOTION_SETTINGS['settled_gate_09']
    rp.MOTION_SETTINGS['settled_gate_09']=replace(base,x_effective_rate=1.45,
              preflight_steps=96,lower_fast_m_s=.40,
              lower_slow_m_s=.10,lower_slow_window_m=.020)
    os.environ.update({'APPLE_NEST_ORDER':'nearest',
                       'STAGE20_V40D_LOWER_RAMP_S':'0.08',
                       'STAGE22_V42_RETURN_SCALE':'0.65'})
    # Stage51-compatible config; independent fresh world for each photo object.
    cfgs={g:next(cfg for _,k,cfg in plan('settled_gate_09') if k==g) for g in 'ABCD'}
    original_load_scene=rp.load_scene
    original_train=rp.train
    original_infer=rp.infer
    rows=[]
    clips=[]
    state={}
    jids={}
    photo_asset=None
    def wrapped_load_scene(pc,label,cfg):
        res=original_load_scene(pc,label,cfg)
        robot,body=res[:2]
        tid=pc.loadTexture(str(assets/photo_asset['texture_path']))
        changed=[]
        for item in pc.getVisualShapeData(body):
            link=int(item[1])
            pc.changeVisualShape(body,link,textureUniqueId=tid,rgbaColor=[1,1,1,1])
            changed.append(link)
        state['textured_body_id']=int(body)
        state['texture_links']=changed
        return res
    rp.load_scene=wrapped_load_scene

    def grab_cam(pc,robot,joints,names,side=180):
        # ORIGINAL engine: 3 finger link cameras + palm + UNDER-PLATE upward camera.
        out=[]
        for nm,arr in rp.images_from_actual_links(pc,robot,joints,side=side):
            if nm in names:out.append((nm,arr))
        return out

    def gate(pc,robot,joints,mass):
        load=object_ai.detect_placement(mass)
        if not load.triggered:
            r={'allow_descent':False,'load':load.__dict__,'upper_image_captured':False,
               'object':None,'message':'NO_STABLE_LOAD'}
            state['gate']=r
            return r
        img=s35.camera_palm_only(pc,robot,joints,side=144)
        Image.fromarray(img).save(state['folder']/'palm_object_gate.png')
        state['palm_image']=img
        state['frames'][('palm',0,0)]=img
        # Source photo is held out of its own gate's training set.
        model=photo_gate_models[state['asset_id']]
        obj=object_ai.identify_top_frame(model,img,confidence_floor=.76)
        obj['model_scope']='DEVELOPMENT LOO gate (7 other Stage58-V2 PALM apple views vs synthetic balls)'
        obj['loocv_held_out_asset_id']=state['asset_id']
        obj['not_farm_ready']=True
        r={'allow_descent':bool(obj['accepted']),'load':load.__dict__,
           'upper_image_captured':True,'object':obj}
        state['gate']=r
        return r
    rp.STAGE32_PREGRIP_GATE=gate

    def observer(pc,robot,apple,plate,joints,stage,t,cfg):
        if stage=='scan_and_close':
            tick=state['stage_ticks'].get(stage,0)+1
            state['stage_ticks'][stage]=tick
            if tick in SCAN_TARGET_TICKS:
                cams=grab_cam(pc,robot,joints,{'finger_1','finger_2','finger_3'})
                for cam_idx,(nm,arr) in enumerate(cams):
                    ti=SCAN_TARGET_TICKS.index(tick)
                    fn=state['folder']/f'{nm}_scan_{ti+1:02d}.png'
                    Image.fromarray(arr).save(fn)
                    position=pc.getLinkState(robot,joints[f'finger_camera_{cam_idx}_fixed'],computeForwardKinematics=True)[4]
                    ev={'phase':'finger_approach','camera':nm,'scan_sample':ti+1,
                        't_sim_s':round(float(t),6),'camera_world_xyz_m':[float(v) for v in position],
                        'file':fn.name,'sha256':sha(fn)}
                    state['events'].append(ev)
                    state['frames'][('finger',cam_idx,ti)]=arr
                    state['grade_images'].append(arr)
        if stage=='gravity_lift':
            current=float(pc.getBasePositionAndOrientation(apple)[0][2])
            if state.get('base_z') is None:
                state['base_z']=current
            dz=current-state['base_z']
            state['max_delta']=max(state['max_delta'],dz)
            nexti=state['bottom_index']
            if nexti<len(BOTTOM_HEIGHT_TARGETS) and dz>=BOTTOM_HEIGHT_TARGETS[nexti]:
                frame=grab_cam(pc,robot,joints,{'plate'})
                if len(frame)!=1:
                    state['error']='UNDER_PLATE_CAMERA_MISSING';return
                arr=frame[0][1]
                fn=state['folder']/f'under_plate_lift_{nexti+1:02d}.png'
                Image.fromarray(arr).save(fn)
                state['frames'][('bottom',0,nexti)]=arr
                state['grade_images'].append(arr)
                state['events'].append({'phase':'underside_during_lift','camera':'plate_under_upward',
                  't_sim_s':round(float(t),6),'apple_lift_delta_m':round(dz,6),
                  'file':fn.name,'sha256':sha(fn)})
                state['bottom_index']+=1
            if state['inference'] is None and dz>=.048:
                # The EXACT actual PyBullet frame set available by ~50mm.
                if state['bottom_index']<3:
                    state['error']='MISSING_UNDERSIDE_FRAME_BY_50MM'
                    return
                if len(state['grade_images'])<12:
                    state['error']='INSUFFICIENT_FINGER_IMAGES_BY_50MM'
                    return
                result=grade_ai.infer_visual(grade_model,state['grade_images'],mode='normal')
                result['grade_inference_at_actual_lift_m']=round(float(dz),6)
                result['grade_inference_at_sim_t_s']=round(float(t),6)
                result['frames_used']=len(state['grade_images'])
                result['uses_weight_for_grading']=False
                result['not_real_orchard_accuracy']=True
                state['inference']=result
                state['events'].append({'phase':'grade_inference_mid_lift',
                        't_sim_s':round(float(t),6),'apple_lift_delta_m':round(dz,6),
                        'grade':result.get('class'),'confidence':result.get('confidence'),
                        'frames_used':len(state['grade_images'])})

    rp._STAGE_CONTACT_OBSERVER=observer
    rp.train=lambda:grade_model
    def cached_mid_lift_infer(model,discarded_late_frames,discarded_mass):
        result=state.get('inference')
        if result is None:
            return {'status':'MID_LIFT_GRADE_NOT_READY','class':None,
                    'reason':state.get('error') or 'GRADE_NOT_READY_BY_50MM',
                    'uses_weight_for_grading':False}
        return dict(result)
    rp.infer=cached_mid_lift_infer
    # This captures real physics states, not a photo slideshow pretending to be real-time.
    try:
        for ix,photo_asset in enumerate(ameta['assets'],1):
            label=photo_asset['user_label']
            folder=out/f"case_{ix:02d}_{photo_asset['asset_id']}";folder.mkdir()
            state.clear();state.update({'asset_id':photo_asset['asset_id'],'folder':folder,'events':[],'frames':{},'grade_images':[],
                'stage_ticks':{},'bottom_index':0,'base_z':None,'max_delta':0.,
                'inference':None,'error':None,'gate':None})
            cfg=replace(cfgs[label],profile='nominal',replicate=5800+ix,
                        scan_duration_s=.28,close_duration_s=.08,torque_Nm=12.,
                        tray_slot=3 if label in 'AB' else 1)
            try:
                row=rp.run_trial(p,label,out/'physics_episodes',record_video=False,cfg=cfg,
                                 persistent_scene=None,return_scene=False)
                g=state.get('gate') or {}
                pred=state.get('inference') or {}
                camera_z_by_finger={}
                for ci in range(3):
                    points=[x['camera_world_xyz_m'][2] for x in state['events']
                            if x.get('phase')=='finger_approach' and x.get('camera')==f'finger_{ci+1}']
                    camera_z_by_finger[f'finger_{ci+1}']={
                        'first_z_m':points[0] if points else None,
                        'last_z_m':points[-1] if points else None,
                        'delta_z_m':round(points[-1]-points[0],6) if len(points)>1 else None,
                        'camera_physically_descended_during_samples':bool(len(points)>1 and points[-1]<points[0])}
                summary={'case':ix,'asset_id':photo_asset['asset_id'],
                    'actual_finger_camera_z_change':camera_z_by_finger,
                   'user_supplied_reference_label_for_evaluation_only':label,
                   'object_gate':g,'mid_lift_inference':pred,
                   'bottom_camera_is_below_looking_up':True,
                   'mid_lift_ready_by_50mm':bool(pred and pred.get('grade_inference_at_actual_lift_m',1) <= .055),
                   'scan_finger_frames':sum(k[0]=='finger' for k in state['frames']),
                   'bottom_lift_frames':state['bottom_index'],
                   'max_apple_displacement_during_lift_m':round(state['max_delta'],6),
                   'physics_result':{'status':row.get('status'),
                     'success_physical_and_ai':row.get('success_physical_and_ai'),
                     'assigned_nest':row.get('assigned_nest'),
                     'classification':row.get('classification'),
                     'actual_lift_delta_m':row.get('apple_lift_delta_m')},
                   'warnings':['A/B/C/D source photo labels used for audit only, never model input',
                      'Synthetic UV back of each supplied photo, not real five-camera photography',
                      'Legacy visual grade model trained on obsolete synthetic textures',
                      'Object gate: LOO experimental calibration; not a production identification test',
                      'Stage51 unspliced work-cell remains unchanged']}
                dump(folder/'case_summary.json',summary)
                dump(folder/'camera_events.json',state['events'])
                # Reconstruct a sample-by-sample visual REPLAY, not fake 20fps sensor footage.
                # Every visible camera image below comes from a real PyBullet frame.
                still=create_storyboard(state['frames'],photo_asset['asset_id'],pred.get('class'),state['events'])
                still.save(folder/'camera_motion_storyboard.jpg',quality=89)
                sequence=[]
                palm_only={k:v for k,v in state['frames'].items() if k[0]=='palm'}
                for ticki in range(4):
                    portion=dict(palm_only)
                    portion.update({k:v for k,v in state['frames'].items()
                                    if k[0]=='finger' and k[2]<=ticki})
                    sequence.append(create_storyboard(portion,photo_asset['asset_id'],None,state['events']))
                for ticki in range(3):
                    portion=dict(palm_only)
                    portion.update({k:v for k,v in state['frames'].items()
                                    if k[0]=='finger' or (k[0]=='bottom' and k[2]<=ticki)})
                    sequence.append(create_storyboard(portion,photo_asset['asset_id'],
                                    pred.get('class') if ticki==2 else None,state['events']))
                sequence.append(still)
                clips.append(sequence)
                rows.append(summary)
                print('STAGE58_CASE',ix,label,'gate',bool(g.get('allow_descent')),
                     'fingers',summary['scan_finger_frames'],'underside',state['bottom_index'],
                     'grade',pred.get('class'),'at_lift_m',pred.get('grade_inference_at_actual_lift_m'),
                     'physics',row.get('status'),flush=True)
            except Exception as exc:
                traceback.print_exc()
                summary=error_report('EXPERIMENT_EXCEPTION',repr(exc),{'case':ix,'reference_label':label})
                dump(folder/'exception.json',summary)
                rows.append(summary)
                print('STAGE58_CASE_ERROR',ix,repr(exc),flush=True)
            finally:
                try:p.resetSimulation()
                except Exception:pass
    finally:
        rp.load_scene=original_load_scene
        rp.train=original_train
        rp.infer=original_infer
        rp._STAGE_CONTACT_OBSERVER=None
        if p.isConnected():p.disconnect()
    if clips:
        video=out/'stage58_camera_evidence_sample_replay.mp4'
        wr=imageio.get_writer(str(video),fps=FPS,codec='libx264',macro_block_size=1)
        try:
            for sequence in clips:
                for still in sequence:
                    for _ in range(6):wr.append_data(np.asarray(still))
        finally:wr.close()
    with (out/'results.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['case','reference_label','gate_pass','finger_images','underside_images','midlift_grade','midlift_height_m','physical_success'])
        for r in rows:
            g=r.get('object_gate') or {};v=r.get('mid_lift_inference') or {}
            w.writerow([r.get('case'),r.get('user_supplied_reference_label_for_evaluation_only'),
                        g.get('allow_descent'),r.get('scan_finger_frames'),
                        r.get('bottom_lift_frames'),v.get('class'),
                        v.get('grade_inference_at_actual_lift_m'),(r.get('physics_result') or {}).get('success_physical_and_ai')])
    checks={'stage':58,'date_jst':stamp(),'photos':len(ameta['assets']),
       'cases_completed':sum('physics_result' in r for r in rows),
       'fingers_x_four_samples':sum(r.get('scan_finger_frames',0)>=12 for r in rows),
       'mid_lift_5cm_inference':sum(r.get('mid_lift_ready_by_50mm',False) for r in rows),
       'bottom_camera':sum(r.get('bottom_lift_frames',0)==3 for r in rows),
       'all_three_finger_cameras_physically_descended':sum(
            all(d.get('camera_physically_descended_during_samples') for d in
                r.get('actual_finger_camera_z_change',{}).values()) and
            len(r.get('actual_finger_camera_z_change',{}))==3 for r in rows),
       'grade_correct_count_on_reference_labels':sum((r.get('mid_lift_inference') or {}).get('class')==r.get('user_supplied_reference_label_for_evaluation_only') for r in rows),
       'tested_in_real_hardware':False,
       'continuous_workcell_video_modified':False,
       'all_cases_have_5cm_evidence':all(r.get('mid_lift_ready_by_50mm',False) for r in rows),
       'important':'Camera movement and 5cm timing are separately measured; photo-derived UV / legacy classifier cannot validate orchard accuracy'}
    dump(out/'summary.json',checks)
    archive=out.parent/(out.name+'_evidence.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for f in sorted(out.rglob('*')):
            rel=f.relative_to(out)
            if f.is_file() and 'engine' not in rel.parts and f.suffix.lower() in ('.mp4','.jpg','.png','.json','.csv','.txt'):
                z.write(f,rel)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print('STAGE58_EVIDENCE_ZIP',archive,archive.stat().st_size,flush=True)
    print('STAGE58_TIMING_COUNTS',json.dumps(checks,ensure_ascii=False),flush=True)
    # A diagnostics run with a failure can still generate usable evidence.
    return 0 if checks['all_cases_have_5cm_evidence'] else 2

if __name__=='__main__':
    try:sys.exit(main())
    except Exception:traceback.print_exc();sys.exit(1)
