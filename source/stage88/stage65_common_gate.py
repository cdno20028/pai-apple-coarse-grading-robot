# -*- coding: utf-8 -*-
"""Stage65: one COMMON, explicitly non-independent synthetic workcell apple/ball gate.
Train from all EIGHT known Stage58 palm-reference images (not grade labels); do not
select per case/photo ID at runtime. The real worker demonstration is NOT blind
orchard validation. Reuses Stage61 features and synthetic ball baselines.
"""
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter
from sklearn.ensemble import RandomForestClassifier
import object_gate_ai as gate
from run_stage61_live_sim_hud import EXTERNAL_TENNIS_GATE_FEATURES


def train_shared_stage65_gate(root):
    root=Path(root)
    paths=sorted((root/'stage58_v2_timed_5camera_5cm_results').glob('case_*/palm_object_gate.png'))
    if len(paths)!=8: raise RuntimeError('STAGE65_NEED_EIGHT_KNOWN_REFERENCE_CAMERAS')
    images=[np.asarray(Image.open(q).convert('RGB')) for q in paths]
    source_ids=[q.parent.name.split('_',2)[-1] for q in paths]
    if len(set(source_ids))!=8:raise RuntimeError('STAGE65_DUPLICATE_PHOTO_IDS')
    rng=np.random.default_rng(44)
    X=[];y=[]
    for image in images:
        for k in range(45):
            a=np.asarray(image,dtype=np.float32).copy()
            bg=np.array([162,167,174],dtype=np.float32)
            inside=(np.max(np.abs(a-bg),axis=2)>25)&(a[:,:,0]>a[:,:,2]*1.1)
            a[inside]*=float(rng.uniform(.81,1.21))
            if k%3==0:
                a=np.asarray(Image.fromarray(np.uint8(np.clip(a,0,255))).filter(ImageFilter.GaussianBlur(radius=.50)))
            feat=gate.object_features(np.uint8(np.clip(a,0,255)))
            if feat is not None:X.append(feat);y.append('apple')
    positive_count=len(X)
    for kind in ('tennis','baseball','red_ball'):
        for i in range(80):
            frame=(gate.simulated_ball_bullet_background(kind,1000+i) if i%2
                   else gate.mock_ball_frame(kind,900+i))
            feat=gate.object_features(frame)
            if feat is not None:X.append(feat);y.append(kind)
    model=RandomForestClassifier(n_estimators=150,min_samples_leaf=2,
                                 random_state=13,class_weight='balanced',n_jobs=1)
    model.fit(np.asarray(X),y)
    # All 8 are TRAINING-origin images; this is a domain-fit check, not held-out accuracy.
    developer_checks=[]
    for photo_id,frame in zip(source_ids,images):
        result=gate.identify_top_frame(model,frame,confidence_floor=.76)
        developer_checks.append({'photo_id':photo_id,'apple_score':result['scores'].get('apple'),
                                 'apple_accepted':result['accepted'],'held_out':False})
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
    evidence={'model_scope':'ONE_SHARED_RANDOM_FOREST_TRAINED_ON_ALL_8_KNOWN_STAGE58_PALM_FRAMES',
              'important':'Known-photo development camera domain only. NO leave-one-out or orchard generalization claim.',
              'no_grade_labels_used':True,'no_runtime_asset_id_model_selection':True,
              'confidence_floor':.76,'positive_training_augmentations':positive_count,
              'negative_training_examples':len(X)-positive_count,
              'known_training_photo_checks':developer_checks,
              'external_heldout_bullet_tennis_scores':tennis_probs,
              'independent_synthetic_negatives_checked':720,'synthetic_negative_false_accepts':len(false_accepts)}
    return model,evidence
