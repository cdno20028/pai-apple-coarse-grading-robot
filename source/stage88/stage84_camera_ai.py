# -*- coding:utf-8 -*-
"""Stage84: stage83 five real cameras + targeted side UV defect scenario."""
from stage68_camera_ai import Stage68CameraAI
from stage84_scar_surface import Stage84Surface
from stage83_pole_surface import Stage83Surface
from stage_direct_photo_surface import StageDirectPhotoSurface
import os
from PIL import Image

class Stage84CameraAI(Stage68CameraAI):
    def __init__(self,root,results,assets,object_ai):
        super().__init__(root,results,assets,object_ai)
        # The focused five-case small-scar demonstration is NOT the full-ten-case program.
        # In the default full run, use grade-matched head/tail and side UV as supplied.
        if os.environ.get('STAGE84_SMALL_SCAR_DEMO')=='1':
            self.repair=Stage84Surface(assets,results,self.root/'stage79_normal_pole_reference')
        else:
            self.repair=StageDirectPhotoSurface(assets,results)
        # STAGE99: only the tenth experimental fruit gets its own C-1 UV.
        # The ordinary 8-asset manifest and all 1-9 cases are left untouched.
        if os.environ.get('STAGE99_CASE10_C1_UV')=='1':
            from pathlib import Path
            import hashlib, json
            special=self.assets/'uv_photo_front'/'10_C1_uv.png'
            expected='22ab1a6c93555164e74f8b9694bc7792f2c08ce465c0e51d026460503f60f2fc'
            actual=hashlib.sha256(special.read_bytes()).hexdigest()
            if actual!=expected:raise RuntimeError('STAGE99_CASE10_UV_HASH_MISMATCH')
            self.repair._paths['10_C1']=special
            (self.results/'stage99_case10_uv_binding.json').write_text(json.dumps({
                'case':10,'asset_id':'10_C1','source_photo':'判定C-1.jfif',
                'source_photo_sha256':'24e3a20b4be3dde61626fe4449342c23f3c55960f65cb9839c26c5de00012d18',
                'actual_runtime_texture':str(special), 'sha256':actual,
                'case01_to_09_unchanged':True,'AI_grade_inference_changed':False,
                'physical_controller_changed':False,'camera_mount_changed':False,
                'effect':'input surface only; no C-grade routing override',
                'result':'PREPARED_NOT_PYBULLET_VERIFIED',
            },ensure_ascii=False,indent=2),encoding='utf-8')

    def photo_for_case(self,idx):
        photo=super().photo_for_case(idx)
        if idx!=10 or os.environ.get('STAGE99_CASE10_C1_UV')!='1':
            return photo
        # Audit/preview metadata matches the SAME texture loaded on the body.
        # Never send source filename, known grade or photo ID into fuse_cameras.
        if photo is None or photo['asset_id']!='05_C':
            raise RuntimeError('STAGE99_SOURCE_CASE10_MAPPING_CHANGED')
        selected=dict(photo)
        selected.update({
            'asset_id':'10_C1',
            'source_filename':'判定C-1.jfif',
            'direct_user_photo':'判定C-1.jfif',
            'direct_user_photo_sha256':'24e3a20b4be3dde61626fe4449342c23f3c55960f65cb9839c26c5de00012d18',
            'source_sha256':'22ab1a6c93555164e74f8b9694bc7792f2c08ce465c0e51d026460503f60f2fc',
            'direct_runtime_uv_sha256':'22ab1a6c93555164e74f8b9694bc7792f2c08ce465c0e51d026460503f60f2fc',
            'texture_path':'uv_photo_front/10_C1_uv.png',
            'user_label':'C',
        })
        return selected

    def start_case(self,idx,typ,truth,rec):
        super().start_case(idx,typ,truth,rec)
        if idx==5 and os.environ.get('STAGE84_SMALL_SCAR_DEMO')=='1':
            # Original source B label is retained for provenance. The NEW
            # synthetic scar scenario is C-labelled for audit/display ONLY.
            # Classifier receives ONLY real camera RGB frames, NEVER this label.
            info=rec.stage72_source_info
            info['original_user_label_before_edit']=info['reference_grade']
            # Grade label for the HUD and the audit is a CLASS TOKEN, not prose.
            # The additional explanation belongs in separate metadata.
            # Do not pass either into the camera-only classifier.
            info['reference_grade']='C'
            info['reference_grade_context']='人工小傷を追加した開発試験（原本はB級）'
            info['synthetic_small_scratch_demo']=True
            self.current['stage84_synthetic_damaged_source_B']=True
            self.current['stage84_inference_is_only_from_camera_pixels']=True

    def observe(self,p,robot,apple,plate,joints,stage,t,cfg,rp):
        super().observe(p,robot,apple,plate,joints,stage,t,cfg,rp)
        if os.environ.get('STAGE92_FIXED_MOUNT')=='1' and stage=='scan_and_close':
            row=self.current
            # Case numbers choose what evidence to SAVE, not which grade to predict.
            if row.get('case') in (2,3) and row.get('stage_ticks',{}).get(stage)==70:
                import hashlib, json
                from stage89_real_scar_ai import detect_dark_linear_wound
                root=self.results/'stage92_actual_camera';root.mkdir(exist_ok=True)
                hits=[]
                for idx in range(4):
                    rgb=row['frames'][('finger',1,idx)]
                    Image.fromarray(rgb).save(root/f'case{row["case"]:02d}_finger2_scan{idx+1}.png')
                    e=detect_dark_linear_wound(rgb)
                    hits.append({'capture':idx+1,'image_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),
                                 'image_only_dark_lines':e['hits']})
                (root/f'case{row["case"]:02d}_mount_witness.json').write_text(
                    json.dumps({'case_index_for_audit_only':row['case'],
                                'finger2_local_z_eye_mm':-15,'pitch_deg':20,
                                'from_live_stage62_camera_frames':True,
                                'not_input_to_ai':True,'frames':hits},
                                ensure_ascii=False,indent=2),encoding='utf8')
        if os.environ.get("STAGE91_MOUNT_GRID_DIAGNOSTIC")=="1" and stage=="scan_and_close":
            row=self.current
            if row.get("case") in (2,3) and row.get("stage_ticks",{}).get(stage)==70:
                from stage91_lens_probe import capture_mount_grid
                capture_mount_grid(p,robot,apple,joints,row["case"],self.results,row["frames"])
                row["stage91_mount_grid_captured"]=True
        if os.environ.get('STAGE90_DAMAGE_EVIDENCE')=='1':
            row=self.current
            if row['kind']=='apple' and stage=='scan_and_close':
                tick=row['stage_ticks'].get('scan_and_close',0)
                if 8 <= tick <= 76 and tick % 4 == 0 and tick not in (16,34,52,70):
                    # The SAME three real physical finger link cameras; no virtual
                    # camera, grade ID or source photo enters the camera capture.
                    rgb=dict(rp.images_from_actual_links(p,robot,joints,side=180))
                    for k in range(3):
                        nm=f'finger_{k+1}'
                        assert nm in rgb, 'STAGE90_MISSING_REAL_FINGER_CAMERA'
                        row['frames'][('dense',k,tick)]=rgb[nm]
                    row.setdefault('stage90_dense_tick_records',[]).append(int(tick))
            if row['kind']=='apple' and row.get('inference') is not None:
                s=row['inference'].get('stage90_severe_wound_evidence')
                if s is not None:
                    row['stage90_proof_verified']=bool(s['confirmed_severe_wound'])
                    if row['inference'].get('stage90_C_not_verified'):
                        self.rec.pred='外観AI 判定保留（深い傷の根拠不足）'
        if os.environ.get('STAGE89_REAL_WOUND_GATE')!='1':return
        row=self.current
        if row['kind']!='apple' or row.get('stage89_actual_capture_saved'):
            return
        if row.get('inference') is None:return
        import hashlib, json
        from stage89_real_scar_ai import detect_dark_linear_wound
        evidence=row['inference'].get('stage89_actual_camera_scar_evidence')
        assert evidence is not None,'STAGE89_WOUND_EVIDENCE_MISSING'
        out=self.results/'stage89_actual_scar_frames'
        out.mkdir(parents=True,exist_ok=True)
        hashes={}
        for tindex in (2,3):
            frame=row['frames'][('finger',1,tindex)]
            name=f"case{row['case']:02d}_finger2_scan{tindex+1}.png"
            Image.fromarray(frame).save(out/name)
            hashes[name]=hashlib.sha256(frame.tobytes()).hexdigest()
            if tindex==3:
                annotated=frame.copy()
                import cv2, numpy as np
                for hit in detect_dark_linear_wound(frame)['hits']:
                    x1,y1,x2,y2=hit['bbox_xyxy']
                    cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),1)
                Image.fromarray(annotated).save(out/f"case{row['case']:02d}_finger2_scar_annotated.png")
        (out/f"case{row['case']:02d}_scar_evidence.json").write_text(
            json.dumps({'case_for_audit_only':row['case'],'capture_hashes':hashes,
                        'camera_optics':'finger2 fixed pitched 20 degrees',
                        'inference_without_reference_label':evidence},
                       ensure_ascii=False,indent=2),encoding='utf8')
        row['stage89_actual_capture_saved']=True

    def conclude(self,case_name,rp_result,rec):
        if os.environ.get('STAGE90_DAMAGE_EVIDENCE')=='1' and self.current['kind']=='apple':
            import json
            from stage90_wound_evidence import assess
            row=self.current
            diag=assess(row['frames'])
            out=self.results/'stage90_real_camera_damage'
            out.mkdir(parents=True,exist_ok=True)
            audit={'case_for_post_run_verification_only':row['case'],
                   'predicted_class':(row.get('inference') or {}).get('class'),
                   'decision_reason':(row.get('inference') or {}).get('decision_reason'),
                   'dense_ticks':row.get('stage90_dense_tick_records',[]),
                   'severe_wound':diag}
            (out/f"{case_name}_wound_audit.json").write_text(
                json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
            # Store unmodified observation and a SEPARATE annotated witness.
            best=[r for r in diag['actual_camera_observations'] if r['hits']]
            if best:
                row0=max(best,key=lambda r:max(h['local_grayscale_contrast'] for h in r['hits']))
                fr=row0['frame'];ca=int(row0['camera'].split('_')[-1])-1
                src=(row['frames'].get(('dense',ca,int(fr.split('tick')[-1])))
                     if fr.startswith('dense_tick') else
                     row['frames'][('finger',ca,int(fr.split('scan')[-1])-1)])
                orig=src.copy(); annotated=src.copy()
                import cv2
                for hit in row0['hits']:
                    x1,y1,x2,y2=hit['bbox_xyxy']
                    cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),2)
                Image.fromarray(orig).save(out/f'{case_name}_witness_raw.png')
                Image.fromarray(annotated).save(out/f'{case_name}_witness_annotated.png')
        # STAGE99: preserve the raw, unmodified actual-link camera observations
        # for comparing B case05 with C-1 case10. Diagnostic only; NOT inference.
        if os.environ.get('STAGE99_CASE10_C1_UV')=='1' and self.current['case'] in (5,10):
            from pathlib import Path
            import json, hashlib
            sample_dir=self.results/'stage99_case05_case10_camera'
            sample_dir.mkdir(parents=True,exist_ok=True)
            record=[]
            for (mode,cam,tick),rgb in sorted(self.current['frames'].items()):
                if mode not in ('finger','dense'):continue
                name=f"case{self.current['case']:02d}_{mode}{cam+1}_tick{tick:03d}.png"
                Image.fromarray(rgb).save(sample_dir/name)
                record.append({'image':name,'image_bytes_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()})
            (sample_dir/f"case{self.current['case']:02d}_capture_manifest.json").write_text(
                json.dumps({'source':'actual camera frames, saved prior to inherited conclude',
                  'case':self.current['case'],'photo_asset_id':self.current.get('photo_asset_id'),
                  'frame_count':len(record),'frames':record},ensure_ascii=False,indent=2),encoding='utf-8')
        # Audit REAL camera inference crop (NOT the synthetic reference UV).
        row=self.current
        if row.get('case')==5:
            for (kind,j,t),frame in row['frames'].items():
                if kind=='finger' and t==3:
                    img_path=self.results/'ai_camera'/f'05_actual_finger_{j+1}_scan4.png'
                    img_path.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(frame).save(img_path)
        return super().conclude(case_name,rp_result,rec)
