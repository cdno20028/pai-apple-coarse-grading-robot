#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage31: SYNTHETIC visual-only apple-grading prototype.

Input: images captured from the 5 simulated cameras. Neither a scale reading nor
an apple class/texture identifier is supplied to prediction. The model is trained
ONLY from artificial UV textures inherited from Stage13, *not* orchard labels.
A/B/C/D are experimental appearance proxies, NOT professional defect grading.
C -> mock brown blemish; D -> green texture; shape is NOT validated because all
collision fruits are spherical. Three policies are PROVISIONAL, not farm criteria.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

LABELS=('A','B','C','D')
MODES=('strict','normal','lenient')
JAPANESE_MODES={'strict':'厳しめ','normal':'普通','lenient':'甘め'}
TEXTURE_DIR=Path(__file__).resolve().parent/'textures'
FEATURE_NAMES=('red','green','blue','red_over_green','green_over_red',
               'blue_over_red','red_pixel_share','green_pixel_share',
               'dark_pixel_share','within_fruit_dark_share','color_irregularity')

@dataclass(frozen=True)
class Policy:
    # Calibrated values must be replaced by expert-labeled orchard training.
    a_probability_min: float
    c_probability_min: float
    certainty_floor: float=.43

POLICIES={
 'strict': Policy(.70,.36),
 'normal': Policy(.52,.50),
 'lenient':Policy(.34,.70),
}


def _mask(rgb):
    # Inherited color mask from Stage27, with a neutral-background suppression.
    r,g,b=[rgb[...,i] for i in range(3)]
    red=(r>.13)&(r>g*1.20)&(r>b*1.60)&((r-g)>.04)
    green=(g>.19)&(g>r*1.06)&(g>b*1.40)
    return red|green


def frame_features(image):
    a=np.asarray(image)
    if a.ndim!=3 or a.shape[-1]<3:raise ValueError('RGB frame required')
    rgb=np.asarray(a[...,:3],np.float32)/255.
    mask=_mask(rgb)
    if mask.sum()<35:return None
    v=rgb[mask]
    med=np.median(v,axis=0)
    bright=np.mean(v,axis=1)
    red,green,blue=med
    feats=[red,green,blue,
      red/(green+.02),green/(red+.02),blue/(red+.02),
      float((v[:,0]>v[:,1]*1.32).mean()),
      float((v[:,1]>v[:,0]*1.05).mean()),
      float((bright<.24).mean()),
      float((bright<max(.10,float(np.median(bright))*.70)).mean()),
      float(np.mean(np.std(v,axis=0)))]
    return np.array(feats,dtype=float)


@lru_cache(maxsize=4)
def _cached_texture(label):
    # Exact same source pixels; cache only avoids re-reading 512x256 PNG per view.
    return np.array(Image.open(TEXTURE_DIR/f'apple_{label}.png').convert('RGB'),dtype=np.uint8)


def make_simulated_frame(label,seed,side=96):
    """Test/training surrogate image, not a Bullet camera frame or real photo.

    Uses UV texture patch at randomized yaw, illumination and white balance,
    over a neutral background. It does not create genuine shape defects.
    """
    rng=np.random.default_rng(seed)
    tex=_cached_texture(label)
    h,w=tex.shape[:2]
    yy,xx=np.mgrid[:side,:side]
    dx=float(rng.uniform(-4,4));dy=float(rng.uniform(-4,4))
    radius=float(rng.uniform(32,42))
    qx=(xx-side/2-dx)/radius;qy=(yy-side/2-dy)/radius
    shape=(qx*qx+qy*qy)<1
    # Random texture yaw and latitude: same texture has different views.
    yaw=float(rng.uniform(0,1));lat=float(rng.uniform(.27,.73))
    uvx=((qx*.31+yaw)%1*w).astype(int)%w
    uvy=np.clip(((qy*.23+lat)*h).astype(int),0,h-1)
    fruit=tex[uvy,uvx].astype(float)/255.
    gain=float(rng.uniform(.48,.84))
    wb=rng.uniform(.94,1.06,3)
    shading=np.clip(1.0-.24*(qx*qx+qy*qy),.72,1.0)
    fruit=np.clip(fruit*gain*wb*shading[...,None]+rng.normal(0,.008,fruit.shape),0,1)
    # Conservative off-white / gray backgrounds, similar to TinyRenderer images.
    back=np.stack([np.full((side,side),.86),np.full((side,side),.87),np.full((side,side),.88)],axis=-1)
    out=np.where(shape[...,None],fruit,back)
    return np.uint8(np.clip(out*255,0,255))


def aggregate_features(images):
    found=[frame_features(f) for f in images]
    found=[x for x in found if x is not None]
    if not found:return None,0
    return np.median(np.stack(found),axis=0),len(found)

@lru_cache(maxsize=1)
def train_visual_model():
    # Same exact assets as Stage13; no weight, label input at inference, or
    # old Stage27 evaluation observations included in fit.
    train_seed=20261020
    X=[];y=[]
    for j,label in enumerate(LABELS):
        for i in range(145):
            frames=[make_simulated_frame(label,train_seed+2000*j+i*7+k) for k in range(3)]
            vec,count=aggregate_features(frames)
            if count<2:continue
            X.append(vec);y.append(label)
    model=RandomForestClassifier(n_estimators=90,min_samples_leaf=2,
                                  class_weight='balanced',random_state=31,n_jobs=1)
    model.fit(X,y)
    return model


def decide_from_scores(scores,mode='normal'):
    """Provisional policy: D green and C blemish proxies have priority.

    It is a *routing* policy over classifier scores, not a learned expert policy.
    Ambiguous observations fail closed as a test interlock; in production such
    thresholds would need explicit validation with human grading labels.
    """
    if mode not in POLICIES:raise ValueError('mode must be strict/normal/lenient')
    p=POLICIES[mode]
    s={g:float(scores.get(g,0)) for g in LABELS}
    if max(s.values())<p.certainty_floor:
        return None,'LOW_CONFIDENCE'
    if s['D']>=.54 and s['D']==max(s.values()):return 'D','GREEN_PROXY'
    if s['C']>=p.c_probability_min and s['C']>=max(s['A'],s['B']):
        return 'C','BLEMISH_PROXY'
    if s['D']>=.42 and s['D']>s['A'] and s['D']>s['B']:
        return 'D','GREEN_PROXY'
    # Strict mode makes A eligibility harder; this is an engineered threshold.
    if s['A']>=p.a_probability_min and s['A']>=s['B']*.75:
        return 'A','RED_APPEARANCE_PROXY'
    if s['B']>=.21 or s['A']>=.21:
        return 'B','ORANGE_OR_AMBIGUOUS_PROXY'
    return None,'UNCERTAIN_FAIL_CLOSED'


def infer_visual(model,images,mode='normal',**kwargs):
    """Explicitly image-only. Any supplied mass is ignored (legacy adapter)."""
    feat,n=aggregate_features(images)
    if feat is None:
        return {'status':'NO_FRUIT_PIXELS','class':None,'confidence':None,'usable_frames':0,
                'mode':mode,'uses_weight_for_grading':False}
    probabilities=model.predict_proba(feat.reshape(1,-1))[0]
    scores={str(c):float(v) for c,v in zip(model.classes_,probabilities)}
    grade,reason=decide_from_scores(scores,mode)
    return {'status':'PREDICTED' if grade else reason,'class':grade,
            'confidence':float(max(scores.values())),'scores':scores,'usable_frames':n,
            'mode':mode,'reason':reason,'features':dict(zip(FEATURE_NAMES,feat.tolist())),
            'uses_weight_for_grading':False,
            'model_scope':'Synthetic UV texture appearance, not real fruit appearance grading',
            'shape_defects_validated':False,
            'expert_labeled_training_data':False}


def stage27_inference_adapter(model,images,scale_mass_kg):
    """Signature compatible with src/run_physics.py without touching motion code.

    Even a valid or invalid numeric scale value must not influence prediction.
    """
    import os
    mode=os.environ.get('APPLE_VISUAL_GRADE_MODE','normal')
    return infer_visual(model,images,mode=mode)


if __name__=='__main__':
    import json
    clf=train_visual_model()
    for mode in MODES:
        for grade in LABELS:
            imgs=[make_simulated_frame(grade,999000+i+1000*LABELS.index(grade)) for i in range(5)]
            r=infer_visual(clf,imgs,mode)
            print(json.dumps({'input':grade,'mode':mode,'predicted':r['class'],
                              'confidence':r['confidence'],'status':r['status']},ensure_ascii=False))
