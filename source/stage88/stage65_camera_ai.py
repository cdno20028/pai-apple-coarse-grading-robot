# -*- coding: utf-8 -*-
"""Stage65: stage61 five-camera observation in ONE persistent work cell.
Not orchard-validated. Known photos may recur in cases and gate uses development LOO.
2026/10/22 04:58 JST
"""
from pathlib import Path
import json,time,hashlib
import numpy as np
from PIL import Image
from stage61_roughgrade import fuse_cameras
from stage65_common_gate import train_shared_stage65_gate
import run_stage61_live_sim_hud as S61

# Stage64: restore raw camera pixels; NO ROI scaling or crop. The palm camera
# physically approaches to exactly the original Stage61 target height.
PALM_GATE_CENTRE_CROP_RATIO=1.0

def normalize_palm_fov(frame):
    a=np.asarray(frame,dtype=np.uint8)
    if a.ndim!=3 or a.shape[2]!=3 or a.shape[0]!=a.shape[1]:
        raise ValueError('STAGE64_GATE_EXPECTS_SQUARE_RGB')
    side=a.shape[0]
    return a.copy(),{'method':'RAW_PALM_NO_CROP_AT_STAGE61_REFERENCE',
        'ratio':1.0,'input_size_px':[int(side),int(side)],
        'roi_xywh_px':[0,0,int(side),int(side)],
        'output_size_px':[int(side),int(side)],
        'geometry_approximation':'stage joint commanded to original Stage61 root_z; measured error logged',
        'warning':'NOT_ORCHARD_VALIDATED; MUST CHECK_PYBULLET_GATE_AND_GRADES'}

SCAN_TICKS=(16,34,52,70)  # old Stage63 scan ticks; replaced by stage64 open-finger approach Z thresholds
APPROACH_FRAME_Z_OFFSETS=(.036,.026,.016,.006)  # metres ABOVE Stage61 reference
LOWER_LIFTS=(.020,.034,.048)
PHOTO_INDEX={1:0,2:1,3:3,4:7,5:2,6:0,7:0,8:0,10:4}

class Stage65CameraAI:
    def __init__(self,root,results,assets,object_ai):
        self.root=Path(root);self.results=Path(results);self.assets=Path(assets)
        manifest=json.loads((self.assets/'manifest.json').read_text(encoding='utf8'))
        self.photos=manifest['assets']
        assert len(self.photos)==8
        self.gate_model,self.gate_diagnostics=train_shared_stage65_gate(self.root)
        (self.results/'stage65_shared_gate_development.json').write_text(json.dumps(self.gate_diagnostics,indent=2,ensure_ascii=False),encoding='utf8')
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
                      'grade_infer_wall_ms':None}
        self.rec=rec;rec.stage62_camera_frames={};rec.stage62_grade=None

    def gate(self,pc,robot,joints,mass,object_ai,s35,old_identifier,live):
        e=object_ai.detect_placement(mass)
        if not e.triggered:
            result={'allow_descent':False,'upper_image_captured':False,'load':e.__dict__,'object':None}
            self.current['object_gate']=result
            return result
        img=s35.camera_palm_only(pc,robot,joints,side=144)
        # S64: Stage61 reference gate position and raw 144x144 RGB pixels.
        # NO center crop. Both gate and external grade consume the SAME raw frame.
        gate_img,gate_fov=normalize_palm_fov(img)
        self.current['frames'][('palm',0,0)]=gate_img
        self.rec.stage62_camera_frames['palm']=img
        path=self.results/'ai_camera'/f'{live["case"]}_stage62_palm.png'
        path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(img).save(path)
        photo=self.photo_for_case(self.current['case'])
        model=self.gate_model  # Same model for every object/case; NO asset-ID routing
        normalized_path=self.results/'ai_camera'/f'{live["case"]}_stage62_palm_gate_roi.png'
        Image.fromarray(gate_img).save(normalized_path)
        obj=object_ai.identify_top_frame(model,gate_img,confidence_floor=.76)
        obj['stage64_actual_gate_joint_z']=float(pc.getJointState(robot,joints['stage_z_move'])[0])
        gate_fov['raw_sha256']=hashlib.sha256(img.tobytes()).hexdigest()
        gate_fov['normalized_sha256']=hashlib.sha256(gate_img.tobytes()).hexdigest()
        gate_fov['source_file']=str(path.relative_to(self.results))
        gate_fov['normalized_file']=str(normalized_path.relative_to(self.results))
        gate_fov['shared_for_gate_and_grade']=True
        gate_fov['grade_palm_frame_sha256']=hashlib.sha256(
            self.current['frames'][('palm',0,0)].tobytes()).hexdigest()
        obj['fov_normalization']=gate_fov
        if photo is not None:
            obj['model_scope']='ONE COMMON development model TRAINED on all 8 known photo-domain views; NO independent validation'
        self.rec.pred=f"物体 {obj.get('predicted_object','?')} ({obj.get('confidence',0):.2f})"
        result={'allow_descent':bool(obj['accepted']),'load':e.__dict__,
                'upper_image_captured':True,'object':obj}
        self.current['object_gate']=result
        self.current['events'].append({'event':'STAGE62_PALM_GATE','allow_descent':result['allow_descent'],
             'palm_sha256':hashlib.sha256(img.tobytes()).hexdigest(),
             'model_scope':obj.get('model_scope','Stage65 shared source-photo training')})
        return result

    def observe(self,p,robot,apple,plate,joints,stage,t,cfg,rp):
        row=self.current
        if row['kind']!='apple':return
        self.texture_body(p,apple,row['case'])
        # Stage64: three cameras observe during the existing monotone 44mm
        # OPEN-FINGER descent, before object-gate acceptance. This is passive
        # imaging only: NO closure/fruit contact is permitted until the gate.
        # The physical grasp trajectory after gate is the unmodified Stage51
        # wrist scan, at the original G.root_z Z endpoint.
        if stage=='stage64_open_finger_reference_approach':
            from mechanism_v09_reference import G
            tick=row['stage_ticks'].get(stage,0)+1
            row['stage_ticks'][stage]=tick
            actual_z=float(p.getJointState(robot,joints['stage_z_move'])[0])
            next_ix=int(row.get('approach_sample_index',0))
            if next_ix<4 and actual_z<=G.root_z+APPROACH_FRAME_Z_OFFSETS[next_ix]:
                shots={n:arr for n,arr in rp.images_from_actual_links(p,robot,joints,side=180)}
                for cam in range(3):
                    nm=f'finger_{cam+1}'
                    if nm not in shots:raise RuntimeError('STAGE64_MISSING_FINGER_'+nm)
                    a=shots[nm]
                    row['frames'][('finger',cam,next_ix)]=a
                    self.rec.stage62_camera_frames[nm]=a
                    pz=float(p.getLinkState(robot,joints[f'finger_camera_{cam}_fixed'],computeForwardKinematics=True)[4][2])
                    row['actual_finger_camera_z'][nm].append(pz)
                    row['events'].append({'event':'OPEN_FINGER_PRE_GATE_DESCENT_FRAME',
                         'camera':nm,'sample':next_ix+1,'sim_t_s':float(t),
                         'actual_camera_z_m':pz,'actual_stage_z_m':actual_z,
                         'pixel_sha256':hashlib.sha256(a.tobytes()).hexdigest(),
                         'contact_free_pre_gate':not bool(p.getContactPoints(robot,apple))})
                row['approach_sample_index']=next_ix+1
        if stage=='scan_and_close':
            row['stage_ticks'][stage]=row['stage_ticks'].get(stage,0)+1
            # no duplicated finger observations; the open-finger descent
            # already contains all twelve actual camera RGB frames.
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
                if row['underside_count']!=3 or sum(k[0]=='finger' for k in row['frames'])!=12:
                    raise RuntimeError('STAGE62_FIVE_CAMERA_NOT_READY_BY_50MM')
                tic=time.perf_counter()
                infer=fuse_cameras(row['frames'])
                infer['grade_infer_wall_ms']=round((time.perf_counter()-tic)*1000,4)
                infer['grade_inference_at_actual_lift_m']=float(delta)
                infer['grade_inference_at_sim_t_s']=float(t)
                infer['status']='PREDICTED'  # Stage23 supervisor protocol; accuracy NOT validated
                infer['underlying_algorithm_status']='EXPERIMENTAL_NOT_VALIDATED'
                infer['confidence']=None  # heuristic has no calibrated posterior
                infer['scores']={}  # no false probabilities
                infer['uses_weight_for_grading']=False
                row['inference']=infer
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
             'finger_frames':sum(e['event']=='OPEN_FINGER_PRE_GATE_DESCENT_FRAME' for e in row['events']),
             'under_camera_frames':row['underside_count'],
             'descending':all(row.get('actual_descending_camera_pass',{}).values()) and
                 len(row.get('actual_descending_camera_pass',{}))==3,
             'destination':rp_result.get('assigned_nest'),
             'physical_success':rp_result.get('success_physical_and_ai')})
        return self.history[-1]
