# -*- coding: utf-8 -*-
"""Stage67 public variant: shared development apple/ball gate with photo calibration.
Train from known camera images plus the user's three original top-view photos (not grade labels);
do not select per case/photo ID at runtime. This is a non-independent known-photo demo,
NOT blind orchard validation. Reuses Stage61 features and synthetic ball baselines.
"""
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter
from sklearn.ensemble import RandomForestClassifier
import object_gate_ai as gate
from run_stage61_live_sim_hud import EXTERNAL_TENNIS_GATE_FEATURES


def train_shared_stage67_high_gate(root):
    root=Path(root)
    paths=sorted((root/'stage58_v2_timed_5camera_5cm_results').glob('case_*/palm_object_gate.png'))
    if len(paths)!=8: raise RuntimeError('STAGE65_NEED_EIGHT_KNOWN_REFERENCE_CAMERAS')
    images=[np.asarray(Image.open(q).convert('RGB')) for q in paths]
    high_paths=sorted((root/'stage62_v4_actual_palm_frames').glob('*.png'))
    if len(high_paths)!=4:raise RuntimeError('STAGE67_NEED_4_RECOVERED_WIDE_FIELD_CAMERA_FRAMES')
    high_images=[np.asarray(Image.open(q).convert('RGB')) for q in high_paths]
    # Public-only development calibration from an actual rendered PyBullet frame
    # of a procedural apple. Calibration is not blind validation; this shared
    # model never receives case grade or asset ID during inference.
    cal_root=root/'stage67_public_camera_calibration'
    cal_apple_paths=[cal_root/'base_capture_public_synthetic_apple.png']+sorted(cal_root.glob('apple_[ABCD]_*.png'))
    cal_negative_paths=sorted(cal_root.glob('tennis_*.png'))+sorted(cal_root.glob('baseball_*.png'))+sorted(cal_root.glob('red_ball_*.png'))
    if len(cal_apple_paths)!=9 or len(cal_negative_paths)!=12:
        raise RuntimeError('PUBLIC_RENDER_CALIBRATION_IMAGES_MISSING')
    cal_apple_images=[np.asarray(Image.open(q).convert('RGB')) for q in cal_apple_paths]
    # Failure-sample domain calibration: independently generated public artwork
    # changed the actual gate-camera distribution. No grade labels or case ID used at runtime.
    art_root=root/'stage67_public_art_gate_calibration'
    art_paths=sorted(art_root.glob('apple_*.png'))
    if len(art_paths)!=1: raise RuntimeError('PUBLIC_ART_OBSERVED_CAMERA_FRAME_MISSING')
    art_images=[np.asarray(Image.open(q).convert('RGB')) for q in art_paths]
    cal_negative_images=[(q.stem.rsplit('_',1)[0],np.asarray(Image.open(q).convert('RGB'))) for q in cal_negative_paths]
    source_ids=[q.parent.name.split('_',2)[-1] for q in paths]
    if len(set(source_ids))!=8:raise RuntimeError('STAGE65_DUPLICATE_PHOTO_IDS')
    rng=np.random.default_rng(44)
    X=[];y=[]
    for image in images+high_images+cal_apple_images+art_images:
        for k in range(45):
            a=np.asarray(image,dtype=np.float32).copy()
            bg=np.array([162,167,174],dtype=np.float32)
            inside=(np.max(np.abs(a-bg),axis=2)>25)&(a[:,:,0]>a[:,:,2]*1.1)
            a[inside]*=float(rng.uniform(.81,1.21))
            if k%3==0:
                a=np.asarray(Image.fromarray(np.uint8(np.clip(a,0,255))).filter(ImageFilter.GaussianBlur(radius=.50)))
            feat=gate.object_features(np.uint8(np.clip(a,0,255)))
            if feat is not None:X.append(feat);y.append('apple')
    # Development-only rendered-camera calibration for the supplied real-photo UV.
    # Samples were captured in prior FAILED/safe diagnostic runs of the same cases.
    # NON-INDEPENDENT; never use this score as evidence of unseen orchard accuracy.
    real_cal=root/'stage67_realphoto_camera_calibration'
    real_apple_paths=sorted(real_cal.glob('apple_*.png'))
    real_tennis_paths=sorted(real_cal.glob('tennis_*.png'))
    if len(real_apple_paths)!=8 or len(real_tennis_paths)!=1:
        raise RuntimeError('REALPHOTO_CALIBRATION_FRAMES_MISSING')
    for q in real_apple_paths:
        img=np.asarray(Image.open(q).convert('RGB'))
        for k in range(45):
            a=np.asarray(img,dtype=np.float32).copy()
            bg=np.array([162,167,174],dtype=np.float32)
            inside=(np.max(np.abs(a-bg),axis=2)>25)&(a[:,:,0]>a[:,:,2]*1.1)
            a[inside]*=float(rng.uniform(.81,1.21))
            if k%3==0:
                a=np.asarray(Image.fromarray(np.uint8(np.clip(a,0,255))).filter(ImageFilter.GaussianBlur(radius=.50)))
            feat=gate.object_features(np.uint8(np.clip(a,0,255)))
            if feat is not None:X.append(feat);y.append('apple')
    # REAL17_POLAR_GATECAL_V2: augment the shared gate with the user's three
    # original top-view photographs. They are source-photo positives, not
    # independent tests; this helps cover the actual polar-cap appearance.
    import io as _real17_io, zipfile as _real17_zip
    source_zip=root.parent/'REAL17_USER_SOURCE_17.zip'
    if not source_zip.is_file(): raise RuntimeError('REAL17_USER_SOURCE_ARCHIVE_MISSING')
    with _real17_zip.ZipFile(source_zip) as _zsrc:
        real17_top_names=sorted(n for n in _zsrc.namelist() if n.startswith('上方-') and n.endswith('.jfif'))
        if len(real17_top_names)!=3: raise RuntimeError('REAL17_EXPECT_THREE_TOP_CAP_SOURCE_PHOTOS')
        real17_top_augmented=0
        for _name in real17_top_names:
            _img=np.asarray(Image.open(_real17_io.BytesIO(_zsrc.read(_name))).convert('RGB').resize((144,144)),dtype=np.uint8)
            for _k in range(45):
                _a=np.asarray(_img,dtype=np.float32).copy()
                _a*=float(rng.uniform(.82,1.18))
                if _k%3==0:
                    _a=np.asarray(Image.fromarray(np.uint8(np.clip(_a,0,255))).filter(ImageFilter.GaussianBlur(radius=.45)),dtype=np.float32)
                _feat=gate.object_features(np.uint8(np.clip(_a,0,255)))
                if _feat is not None:
                    X.append(_feat); y.append('apple'); real17_top_augmented+=1
    positive_count=len(X)
    for kind in ('tennis','baseball','red_ball'):
        for i in range(80):
            frame=(gate.simulated_ball_bullet_background(kind,1000+i) if i%2
                   else gate.mock_ball_frame(kind,900+i))
            feat=gate.object_features(frame)
            if feat is not None:X.append(feat);y.append(kind)
    for kind,frame in cal_negative_images:
        for k in range(24):
            a=np.asarray(frame,dtype=np.float32).copy()
            if k%3==0:
                a=np.asarray(Image.fromarray(np.uint8(a)).filter(ImageFilter.GaussianBlur(radius=.45))).astype(np.float32)
            feat=gate.object_features(np.uint8(np.clip(a,0,255)))
            if feat is not None:X.append(feat);y.append(kind)
    # A real rendered tennis control provides camera-domain hard negatives.
    for q in real_tennis_paths:
        img=np.asarray(Image.open(q).convert('RGB'))
        for k in range(60):
            a=np.asarray(img,dtype=np.float32).copy()
            if k%3==0:
                a=np.asarray(Image.fromarray(np.uint8(a)).filter(ImageFilter.GaussianBlur(radius=.45))).astype(np.float32)
            if k%5==0:
                a*=float(rng.uniform(.91,1.09))
            feat=gate.object_features(np.uint8(np.clip(a,0,255)))
            if feat is not None:X.append(feat);y.append('tennis')
    model=RandomForestClassifier(n_estimators=150,min_samples_leaf=2,
                                 random_state=13,class_weight='balanced',n_jobs=1)
    model.fit(np.asarray(X),y)
    # All 8 are TRAINING-origin images; this is a domain-fit check, not held-out accuracy.
    developer_checks=[]
    for photo_id,frame in zip(source_ids,images):
        result=gate.identify_top_frame(model,frame,confidence_floor=.76)
        developer_checks.append({'photo_id':photo_id,'apple_score':result['scores'].get('apple'),
                                 'apple_accepted':result['accepted'],'held_out':False})
    observed_high_views=[]
    for fname,frame in zip(high_paths,high_images):
        r=gate.identify_top_frame(model,frame,confidence_floor=.76)
        observed_high_views.append({'image':fname.name,'apple_score':r['scores'].get('apple'),'accepted':r['accepted'],'IN_TRAINING':True})
    if not all(r['accepted'] for r in observed_high_views):raise RuntimeError('STAGE67_HIGH_VIEW_GATE_TRAIN_DOMAIN_FAIL')
    cal_apple_checks=[]
    for q,img in zip(cal_apple_paths,cal_apple_images):
        result=gate.identify_top_frame(model,img,confidence_floor=.76)
        cal_apple_checks.append({'image':q.name,'apple_score':result['scores'].get('apple',0),'accepted':result['accepted'],'IN_TRAINING':True})
    if not all(q['accepted'] for q in cal_apple_checks):raise RuntimeError('PUBLIC_RENDER_CALIBRATION_POSITIVE_FAIL')
    art_checks=[{'file':q.name,'result':gate.identify_top_frame(model,img,confidence_floor=.76),'training_origin':True} for q,img in zip(art_paths,art_images)]
    if not all(r['result']['accepted'] for r in art_checks):raise RuntimeError('PUBLIC_ART_CAM_GATE_DOMAIN_FAIL')
    cal_negative_checks=[]
    for typ,img in cal_negative_images:
        result=gate.identify_top_frame(model,img,confidence_floor=.76)
        cal_negative_checks.append({'type':typ,'accepted':result['accepted'],'prediction':result['predicted_object'],'IN_TRAINING':True})
    if any(q['accepted'] for q in cal_negative_checks):raise RuntimeError('PUBLIC_RENDER_CALIBRATION_NEGATIVE_FALSE_ACCEPT')
    tennis_feature=np.asarray(EXTERNAL_TENNIS_GATE_FEATURES,dtype=np.float32)
    pred=model.predict_proba(tennis_feature[None,:])[0]
    tennis_probs={str(k):float(v) for k,v in zip(model.classes_,pred)}
    if tennis_probs.get('apple',0)>=.76:raise RuntimeError('STAGE65_EXTERNAL_BULLET_TENNIS_FALSE_ACCEPT')
    # These artificial negatives were not used in training (different seeds).
    false_accepts=[]
    for kind in ('tennis','baseball','red_ball'):
        for i in range(240):
            frame=(gate.simulated_ball_bullet_background(kind,20000+i) if i%2
                   else gate.mock_ball_frame(kind,30000+i))
            answer=gate.identify_top_frame(model,frame,confidence_floor=.76)
            if answer['accepted']:
                false_accepts.append({'kind':kind,'index':i,'score':answer['scores'].get('apple')})
    if false_accepts:raise RuntimeError('STAGE65_SYNTHETIC_NEGATIVE_FALSE_ACCEPT:'+str(false_accepts[:3]))
    calibration_checks={q.name:gate.identify_top_frame(model,np.asarray(Image.open(q).convert('RGB')),confidence_floor=.76) for q in real_apple_paths+real_tennis_paths}
    if not all(calibration_checks[q.name]['accepted'] for q in real_apple_paths):
        raise RuntimeError('REALPHOTO_KNOWN_CAMERA_REPLAY_FAILED')
    if any(calibration_checks[q.name]['accepted'] for q in real_tennis_paths):
        raise RuntimeError('REALPHOTO_KNOWN_TENNIS_FALSE_ACCEPT')
    evidence={'real_photo_camera_calibration_nonindependent':True,
              'real_photo_camera_calibration_checks':calibration_checks,
              'real_photo_camera_calibration_origin':'FAILED_V7R2_PHYSICS_RUN',
              'model_scope':'PUBLIC_SYNTHETIC_SHARED_RF_WITH_PYBULLET_RENDER_CALIBRATION_NON_INDEPENDENT',
              'important':'Public render calibration includes case01 screenshot from previous failure. NEVER claim this is held-out, test accuracy, or orchard reliability.',
              'high_view_training_and_replay_checks':observed_high_views,
              'public_synthetic_render_calibration_positive':cal_apple_checks,
              'public_art_observed_camera_calibration':art_checks,
              'art_camera_calibration_is_NON_INDEPENDENT':True,
              'public_synthetic_render_calibration_negative':cal_negative_checks,
              'CALIBRATION_NOT_INDEPENDENT':'Calibration apple screenshot came from first failed run; cannot claim blind accuracy.',
              'test_camera_image_used_in_training':True,
              'no_grade_labels_used':True,'no_runtime_asset_id_model_selection':True,
              'confidence_floor':.76,'positive_training_augmentations':positive_count,
              'negative_training_examples':len(X)-positive_count,
              'known_training_photo_checks':developer_checks,
              'external_heldout_bullet_tennis_scores':tennis_probs,
              'independent_synthetic_negatives_checked':720,'synthetic_negative_false_accepts':len(false_accepts),
              'real17_polar_gatecal_v2':{
                  'top_cap_source_archive':'REAL17_USER_SOURCE_17.zip',
                  'top_cap_source_images':real17_top_names,
                  'top_cap_positive_augmentations':real17_top_augmented,
                  'grade_labels_used_for_gate':False,
                  'runtime_asset_id_model_selection':False,
                  'known_camera_calibration_is_nonindependent':True,
                  'current_polar_composite_camera_frames_in_training':[
                      'apple_01_apple_A_A3_stage62_palm.png',
                      'apple_02_apple_B_B3_stage62_palm.png'],
                  'legacy_direct_photo_camera_frame_in_training':[
                      'apple_04_apple_D_D1_stage62_palm.png'],
                  'remaining_camera_frames_are_legacy_calibration':True,
                  'limit':'Development calibration only; no blind accuracy or orchard reliability claim.'
              }}
    return model,evidence
