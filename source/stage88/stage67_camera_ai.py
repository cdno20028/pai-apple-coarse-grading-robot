# -*- coding: utf-8 -*-
"""Stage62: stage61 five-camera observation in ONE persistent work cell.
Not orchard-validated. Known photos may recur in cases and gate uses development LOO.
2026/10/22 JST
"""
from pathlib import Path
import json,time,hashlib
import numpy as np
from PIL import Image
from stage_real17_coarse_camera_ai import fuse_cameras
from stage67_highview_gate import train_shared_stage67_high_gate
import run_stage61_live_sim_hud as S61

# Stage62 v3: same Palm FOV calibration for object gate and visual grading.
# Fixed geometric ROI in the *raw live Palm photograph*, selected from measured
# fruit diameter ratio (~65px Stage62 vs ~120px Stage61) without using truth labels.
# Cropping is applied to apples AND non-apples, before the identical AI gate.
# The full raw camera image remains saved and unaltered for independent audit.
PALM_GATE_CENTRE_CROP_RATIO=1.0

def normalize_palm_fov(frame):
    arr=np.asarray(frame,dtype=np.uint8)
    if arr.shape!=(144,144,3):raise RuntimeError('STAGE67_EXPECTED_RAW_PALM_144_RGB')
    return arr.copy(),{'method':'RAW_RGB_NO_CROP','ratio':1.0,
         'input_size_px':[144,144], 'output_size_px':[144,144],
         'source':'ACTUAL_PYBULLET_CAMERA_TWICE: HIGH_GATING_AND_LOW_GRADING',
         'warning':'EXACT_HIGH_CAMERA_TRAIN_FRAMES_NOT_INDEPENDENT'}

SCAN_TICKS=(16,34,52,70)
LOWER_LIFTS=(.020,.034,.048)
PHOTO_INDEX={1:0,2:1,3:3,4:7,5:2,6:0,7:0,8:0,10:4}
# Stage76 supplement only: explicit opt-in inherited by Stage70's spawned physics process.
# Changes source TEXTURE only; reference grades never feed inference.
import os as _s76_os
if _s76_os.environ.get('STAGE76_C_EXTRA_PHOTOS')=='1':
    PHOTO_INDEX.update({3:5,10:6})


class Stage67CameraAI:
    def __init__(self,root,results,assets,object_ai):
        self.root=Path(root);self.results=Path(results);self.assets=Path(assets)
        manifest=json.loads((self.assets/'manifest.json').read_text(encoding='utf8'))
        self.photos=manifest['assets']
        assert len(self.photos)==8
        self.gate_model,self.gate_diagnostics=train_shared_stage67_high_gate(self.root)
        (self.results/'stage67_highview_gate_development.json').write_text(json.dumps(self.gate_diagnostics,indent=2,ensure_ascii=False),encoding='utf8')
        self.material_ids={};self.history=[];self.current={};self.rec=None

    def photo_for_case(self,idx):
        j=PHOTO_INDEX.get(idx)
        return self.photos[j] if j is not None else None

    def texture_body(self,p,body,idx):
        photo=self.photo_for_case(idx)
        if photo is None:return
        if int(body) in self.material_ids:
            if self.material_ids[int(body)]!=idx:
                raise RuntimeError('PHYSICAL_BODY_TEXTURE_REASSIGNMENT')
            return
        path=self.assets/photo['texture_path']
        tex=p.loadTexture(str(path))
        for vis in p.getVisualShapeData(body):
            p.changeVisualShape(body,int(vis[1]),textureUniqueId=tex,rgbaColor=[1,1,1,1])
        self.material_ids[int(body)]=idx

    def start_case(self,idx,typ,truth,rec):
        photo=self.photo_for_case(idx)
        self.current={'case':idx,'kind':typ,'reference_label_for_audit_only':truth,
                      'photo_asset_id':photo['asset_id'] if photo else None,
                      'repeated_source_photo':idx in (6,7,8),
                      'events':[],'frames':{},'stage_ticks':{},'underside_count':0,
                      'base_lift_z':None,'inference':None,'object_gate':None,
                      'actual_finger_camera_z':{f'finger_{k}':[] for k in (1,2,3)},
                      'grade_infer_wall_ms':None, 'grade_palm_sha256':None}
        self.rec=rec;rec.stage62_camera_frames={};rec.stage62_grade=None
        # Stage71: a labelled REFERENCE image, not an inference input.
        # The original JPEG bytes are not carried in the Stage70 ZIP;
        # this is only the original-photograph-derived portion of its UV.
        rec.stage71_evidence=None;rec.stage71_source_preview=None
        # Stage72 audit annotation: user-provided reference label is DISPLAY ONLY.
        # Do not send filename or reference grade to any gate or fuse_cameras.
        rec.stage72_source_info=(dict(asset_id=photo['asset_id'],
            filename=photo['source_filename'],reference_grade=photo['user_label'],
            actual_ai_grade=None) if photo is not None else None)
        if photo is not None:
            uv=np.asarray(Image.open(self.assets/photo['texture_path']).convert('RGB'))
            rec.stage71_source_preview=uv[249:480,220:800].copy()

    def gate(self,pc,robot,joints,mass,object_ai,s35,old_identifier,live):
        e=object_ai.detect_placement(mass)
        if not e.triggered:
            result={'allow_descent':False,'upper_image_captured':False,'load':e.__dict__,'object':None}
            self.current['object_gate']=result
            return result
        img=s35.camera_palm_only(pc,robot,joints,side=144)
        # The Palm camera is 44mm farther from the apple in Stage62. Use the
        # same geometric field-of-view correction for both object gating AND
        # subsequent five-camera skin grading. The raw Palm frame is still
        # retained for the live HUD and independent audit. Stage62 v2 used
        # the raw image for grading, creating a spurious second damage camera
        # (B -> C) while the gate itself used a cropped image.
        gate_img,gate_fov=normalize_palm_fov(img)
        # Do NOT use high stand-by gate frame for grade; sample RAW palm at reference Z during scan.
        self.current['gate_palm_sha256']=hashlib.sha256(gate_img.tobytes()).hexdigest()
        self.rec.stage62_camera_frames['palm']=img
        path=self.results/'ai_camera'/f'{live["case"]}_stage62_palm.png'
        path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(img).save(path)
        photo=self.photo_for_case(self.current['case'])
        model=self.gate_model  # SAME model for all objects, no case-based routing
        normalized_path=self.results/'ai_camera'/f'{live["case"]}_stage62_palm_gate_roi.png'
        Image.fromarray(gate_img).save(normalized_path)
        obj=object_ai.identify_top_frame(model,gate_img,confidence_floor=.76)
        gate_fov['raw_sha256']=hashlib.sha256(img.tobytes()).hexdigest()
        gate_fov['normalized_sha256']=hashlib.sha256(gate_img.tobytes()).hexdigest()
        gate_fov['source_file']=str(path.relative_to(self.results))
        gate_fov['normalized_file']=str(normalized_path.relative_to(self.results))
        gate_fov['shared_for_gate_and_grade']=False
        gate_fov['grade_palm_frame_sha256']=None  # late RAW 5-camera grade scan; recorded at scan tick 70
        obj['fov_normalization']=gate_fov
        if photo is not None:
            obj['model_scope']='KNOWN HIGH-CAMERA IMAGES INCLUDED IN COMMON MODEL TRAINING; NOT ORCHARD VALIDATED'
        self.rec.pred=f"物体 {obj.get('predicted_object','?')} ({obj.get('confidence',0):.2f})"
        result={'allow_descent':bool(obj['accepted']),'load':e.__dict__,
                'upper_image_captured':True,'object':obj}
        self.current['object_gate']=result
        self.current['events'].append({'event':'STAGE62_PALM_GATE','allow_descent':result['allow_descent'],
             'palm_sha256':hashlib.sha256(img.tobytes()).hexdigest(),
             'model_scope':obj.get('model_scope','Stage51 synthetic trained reference')})
        return result

    def observe(self,p,robot,apple,plate,joints,stage,t,cfg,rp):
        row=self.current
        if row['kind']!='apple':return
        self.texture_body(p,apple,row['case'])
        if stage=='scan_and_close':
            tick=row['stage_ticks'].get(stage,0)+1
            row['stage_ticks'][stage]=tick
            if tick in SCAN_TICKS:
                samples={n:arr for n,arr in rp.images_from_actual_links(p,robot,joints,side=180)}
                ix=SCAN_TICKS.index(tick)
                for cam in range(3):
                    nm=f'finger_{cam+1}'
                    if nm not in samples:raise RuntimeError('MISSING_FINGER_'+nm)
                    a=samples[nm]
                    row['frames'][('finger',cam,ix)]=a
                    self.rec.stage62_camera_frames[nm]=a
                    z=float(p.getLinkState(robot,joints[f'finger_camera_{cam}_fixed'],computeForwardKinematics=True)[4][2])
                    row['actual_finger_camera_z'][nm].append(z)
                    row['events'].append({'event':'FINGER_DESCENT_FRAME','camera':nm,'sample':ix+1,
                        'sim_t_s':float(t),'actual_camera_z_m':z,'pixel_sha256':hashlib.sha256(a.tobytes()).hexdigest()})
                if tick==SCAN_TICKS[-1]:
                    grade_palm=__import__('run_stage35_motion').camera_palm_only(p,robot,joints,side=144)
                    row['frames'][('palm',0,0)]=grade_palm
                    self.rec.stage62_camera_frames['palm']=grade_palm
                    gz=float(p.getJointState(robot,joints['stage_z_move'])[0])
                    row['events'].append({'event':'STAGE67_RAW_PALM_GRADING_IMAGE_AT_SCAN_END',
                       'actual_stage_z_m':gz,'sample_tick':tick,'sim_t_s':float(t),
                       'pixel_sha256':hashlib.sha256(grade_palm.tobytes()).hexdigest(),
                       'not_a_crop':True})
                    row['grade_palm_sha256']=hashlib.sha256(grade_palm.tobytes()).hexdigest()
                    raw=self.results/'ai_camera'/f"{row['case']:02d}_stage67_grade_palm_raw.png"
                    raw.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(grade_palm).save(raw)

        if stage=='gravity_lift':
            z=float(p.getBasePositionAndOrientation(apple)[0][2])
            if row['base_lift_z'] is None:row['base_lift_z']=z
            delta=z-row['base_lift_z']
            nexti=row['underside_count']
            if nexti<3 and delta>=LOWER_LIFTS[nexti]:
                imgs={n:arr for n,arr in rp.images_from_actual_links(p,robot,joints,side=180)}
                if 'plate' not in imgs:raise RuntimeError('UNDERPLATE_UPWARD_CAMERA_MISSING')
                a=imgs['plate']
                row['frames'][('bottom',0,nexti)]=a
                self.rec.stage62_camera_frames['inspection_under']=a
                row['events'].append({'event':'UNDERPLATE_UPWARD_FRAME','sample':nexti+1,
                    'sim_t_s':float(t),'lift_delta_m':float(delta),'pixel_sha256':hashlib.sha256(a.tobytes()).hexdigest()})
                row['underside_count']+=1
            if row['inference'] is None and delta>=.048:
                if row['underside_count']!=3 or sum(k[0]=='finger' for k in row['frames'])!=12 or ('palm',0,0) not in row['frames']:
                    raise RuntimeError('STAGE62_FIVE_CAMERA_NOT_READY_BY_50MM')
                tic=time.perf_counter()
                infer=fuse_cameras(row['frames'])
                infer['grade_infer_wall_ms']=round((time.perf_counter()-tic)*1000,4)
                infer['grade_inference_at_actual_lift_m']=float(delta)
                infer['grade_inference_at_sim_t_s']=float(t)
                infer['status']=('ABSTAINED_CAMERA_DAMAGE_UNVERIFIED' if (
                    _s76_os.environ.get('STAGE90_DAMAGE_EVIDENCE')=='1' and
                    infer.get('stage90_C_not_verified')) else 'PREDICTED')
                infer['underlying_algorithm_status']='EXPERIMENTAL_NOT_VALIDATED'
                infer['confidence']=None  # heuristic has no calibrated posterior
                infer['scores']={}  # no false probabilities
                infer['uses_weight_for_grading']=False
                row['inference']=infer
                # Stage71: numbers are from ACTUAL camera frames and this
                # inference; no reference label, filename or preview is used.
                _pc=infer['per_camera']
                if self.rec.stage72_source_info is not None:
                    self.rec.stage72_source_info['actual_ai_grade']=infer['class']
                self.rec.stage71_evidence={
                    'grade':infer['class'],'reason':infer['decision_reason'],
                    'positive':list(infer['positive_distinct_cameras']),
                    'dark_pct':{n:100.*float(_pc[n]['largest_dark_patch_max'])
                                for n in ('finger_1','finger_2','finger_3','palm')},
                    'max_pct':100.*float(infer['max_localized_dark_patch_fraction']),
                    'photo_not_model_input':True,
                    'source_UV_derived_from_user_photo':True,
                    'unvalidated_provisional_thresholds':True,
                }
                self.rec.stage62_grade=infer['class']
                self.rec.pred=f"外観AI {infer['class']}級（未校正の試作）"
                row['events'].append({'event':'STAGE62_GRADE_DECISION','grade':infer['class'],
                      'sim_t_s':float(t),'lift_delta_m':float(delta),'grade_infer_wall_ms':infer['grade_infer_wall_ms']})

    def grade(self,model,images,mass):
        result=self.current.get('inference')
        if result is None:
            return {'status':'NOT_READY','class':None,'uses_weight_for_grading':False}
        return dict(result)

    def conclude(self,case_name,rp_result,rec):
        row=self.current
        row['physical_result']={'success':rp_result.get('success_physical_and_ai'),
          'status':rp_result.get('status'),'destination':rp_result.get('assigned_nest'),
          'classification':(rp_result.get('classification') or {}).get('class'),
          'return_home_pass':rp_result.get('return_home_pass')}
        for nm,z in row['actual_finger_camera_z'].items():
            row.setdefault('actual_descending_camera_pass',{})[nm]=bool(len(z)==4 and
                 all(b<a-.0002 for a,b in zip(z,z[1:])))
        row.pop('frames') # store hashes and metadata, not huge arrays
        target=self.results/'stage62_camera_cases'/f'{case_name}.json'
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(row,ensure_ascii=False,indent=2,default=str),encoding='utf8')
        self.history.append({'case':row['case'],'name':case_name,
             'grade':(row['inference'] or {}).get('class'),
             'reference_label':row['reference_label_for_audit_only'],
             'finger_frames':sum(e['event']=='FINGER_DESCENT_FRAME' for e in row['events']),
             'under_camera_frames':row['underside_count'],
             'descending':all(row.get('actual_descending_camera_pass',{}).values()) and
                 len(row.get('actual_descending_camera_pass',{}))==3,
             'destination':rp_result.get('assigned_nest'),
             'grade_palm_raw_captured':bool(row.get('grade_palm_sha256')),
             'physical_success':rp_result.get('success_physical_and_ai')})
        return self.history[-1]
