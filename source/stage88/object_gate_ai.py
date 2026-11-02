#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage32: SYNTHETIC-only apple vs ball classifier and load-event interlock.

Two separate decisions: load changes trigger imaging; ONLY top-image AI permits
robot descent. Load value NEVER enters A/B/C/D grade classifier.
Baseball/tennis examples are procedural mock images, not real photographs.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import time
import numpy as np
from PIL import Image,ImageDraw
from sklearn.ensemble import RandomForestClassifier
import visual_grade_ai as grade

OBJECT_TYPES=('apple','baseball','tennis','red_ball')

@dataclass
class LoadEvent:
    triggered:bool
    mass_kg:float|None
    reason:str


def detect_placement(mass_kg, threshold_kg=.025):
    """Mass threshold is a switch, NOT object identity or grade prediction.

    Tennis ball (~0.057kg) and baseball (~0.145kg) should BOTH trigger imaging.
    A separately trained visual classifier must reject them.
    """
    if mass_kg is None or not np.isfinite(float(mass_kg)) or float(mass_kg)<threshold_kg:
        return LoadEvent(False,None if mass_kg is None else float(mass_kg),'NO_STABLE_LOAD')
    return LoadEvent(True,float(mass_kg),'LOAD_TRIGGER_CAPTURE')


def mock_ball_frame(kind,seed,side=96):
    """Synthetic white baseball, yellow/green tennis ball, red rubber ball.

    Test surrogate: not a camera measurement or evidence of real-object rejection.
    """
    if kind not in ('baseball','tennis','red_ball'):raise ValueError(kind)
    rng=np.random.default_rng(int(seed)); yy,xx=np.mgrid[:side,:side]
    dx=float(rng.uniform(-4,4));dy=float(rng.uniform(-4,4));radius=float(rng.uniform(32,42))
    u=(xx-side/2-dx)/radius;v=(yy-side/2-dy)/radius
    shape=(u*u+v*v)<1
    if kind=='baseball':col=np.array([.88,.84,.75]);noise=.025
    elif kind=='tennis':col=np.array([.69,.80,.16]);noise=.027
    else:col=np.array([.76,.15,.12]);noise=.007
    shade=np.clip(1-.19*(u*u+v*v),.76,1)
    fruit=np.clip(col[None,None,:]*float(rng.uniform(.65,.95))*shade[:,:,None]
                  +rng.normal(0,noise,(side,side,3)),0,1)
    theta=np.arctan2(v,u)
    if kind=='baseball':
        # Two curved red stitched seams: simplified fake stitching.
        stitch=(np.abs(u-.45*np.cos(v*3.0))<.023)|(np.abs(u+.45*np.cos(v*3.0))<.023)
        fruit[shape & stitch]=np.array([.62,.17,.17])
    elif kind=='tennis':
        seam=np.abs(u-.45*np.sin(v*2.5))<.035
        fruit[shape & seam]=np.array([.90,.90,.73])
    bg=np.stack([np.full((side,side),.86),np.full((side,side),.87),np.full((side,side),.88)],axis=-1)
    img=np.where(shape[:,:,None],fruit,bg)
    return np.asarray(np.round(np.clip(img,0,1)*255),dtype=np.uint8)


def object_features(frame):
    """Camera-domain neutral-background segmentation, with a central fruit ROI.

    Old Stage32 assumed a hard-coded RGB background (.86,.87,.88), while the
    actual Bullet/TinyRenderer camera produced (~.635,.655,.682). This made
    the entire background count as fruit and confused apples with tennis balls.
    Use corner pixels to estimate the actual scene background instead. Only
    the central silhouette is used, avoiding the blue gripper parts at edges.
    This is a simulator-specific segmentation method, not validated outdoors.
    """
    src=np.asarray(frame)
    if src.ndim!=3 or src.shape[2]<3:raise ValueError('RGB frame required')
    rgb=np.asarray(Image.fromarray(src[:,:,:3].astype('uint8')).resize((96,96)),np.float32)/255.
    yy,xx=np.mgrid[:96,:96]
    radius=np.hypot(xx-48,yy-48)
    # Four edge corners are expected to contain the empty inspection table.
    corners=np.concatenate((rgb[:8,:8].reshape(-1,3),rgb[:8,-8:].reshape(-1,3),
                            rgb[-8:,:8].reshape(-1,3),rgb[-8:,-8:].reshape(-1,3)))
    background=np.median(corners,axis=0)
    not_background=(np.max(np.abs(rgb-background),axis=2)>.10)
    blue_robot=(rgb[:,:,2]>rgb[:,:,0]*1.28)&(rgb[:,:,2]>rgb[:,:,1]*1.03)
    # Public-synthetic PyBullet domain: a grey inspection plate fills the camera
    # and must not be mistaken for the central fruit. Segment chromatic fruit
    # pixels; blue robot links remain excluded. This is NOT a universal detector.
    chroma=np.max(rgb,axis=2)-np.min(rgb,axis=2)
    fruit_color=(chroma>.14)&((rgb[:,:,0]>rgb[:,:,2]*1.12)|(rgb[:,:,1]>rgb[:,:,2]*1.20))
    mask=not_background & (radius<39) & ~blue_robot & fruit_color
    if mask.sum()<120:return None
    colors=rgb[mask]
    m=np.median(colors,axis=0)
    std=np.std(colors,axis=0)
    percent=np.percentile(colors,[10,90],axis=0).flatten()
    brightness=colors.mean(axis=1)
    d=np.absolute(rgb[:-1,1:]-rgb[1:,1:]).mean(axis=2)
    interior=mask[:-1,1:] & mask[1:,1:]
    edges=float(np.mean(d[interior])) if np.any(interior) else 0.
    # Apple calyx/depression proxy at top centre. Unlike a spherical toy ball,
    # actual Stage32 synthetic apples show a darker centre than the annulus.
    # A lack of this proxy must not by itself trigger a positive identification.
    light=rgb.mean(axis=2)
    central=light[radius<5]
    ring=light[(radius>=12)&(radius<22)]
    top_cavity_contrast=float(ring.mean()-central.mean())
    feats=[*m,*std,*percent,
      float((colors[:,0]>colors[:,1]*1.22).mean()),
      float((colors[:,1]>colors[:,0]*1.06).mean()),
      float((colors[:,2]>colors[:,0]*.78).mean()),
      float((brightness<.28).mean()),float((brightness>.70).mean()),
      float(mask.mean()),edges,top_cavity_contrast]
    return np.asarray(feats,dtype=np.float32)


def augment_apple_illumination(frame,seed):
    """Synthetic training only: broaden brightness of original apple textures.

    The four PyBullet screenshots are NOT added as training examples. The
    broader gain distribution was selected while inspecting those 4 frames;
    results on those same frames are therefore development diagnostics only.
    """
    rng=np.random.default_rng(int(seed))
    a=np.asarray(frame,dtype=np.float32).copy()
    yy,xx=np.mgrid[:a.shape[0],:a.shape[1]]
    radius=np.hypot(xx-a.shape[1]/2,yy-a.shape[0]/2)
    nominal_background=np.array([.86,.87,.88],dtype=np.float32)*255
    fruit=(radius<a.shape[0]*.45)&(np.max(np.abs(a-nominal_background),axis=2)>18)
    if int(seed)%4:
        gain=(float(rng.uniform(.6,.95)) if int(seed)%4==1
              else float(rng.uniform(.9,1.6)))
        a[fruit]*=gain
    return np.uint8(np.clip(a,0,255))


def make_object_frame(kind,seed):
    if kind=='apple':
        g=grade.LABELS[int(seed)%4]
        return grade.make_simulated_frame(g,int(seed)*29+1001)
    return mock_ball_frame(kind,int(seed)*31+23)

def simulated_apple_top_dimple(frame,seed):
    """Synthetic central calyx depression: geometric proxy, not real damage.

    Only the synthetic apple training images receive this augmentation. The
    actual four Stage32 rendered camera screenshots are never training input.
    """
    a=np.asarray(frame,dtype=np.float32).copy()
    h,w=a.shape[:2]
    yy,xx=np.mgrid[:h,:w]
    rng=np.random.default_rng(int(seed)+913)
    cx=w/2+float(rng.uniform(-2,2));cy=h/2+float(rng.uniform(-2,2))
    r=np.hypot(xx-cx,yy-cy)
    center=np.exp(-r*r/(2*6.7**2))
    ridge=np.exp(-r*r/(2*20.0**2))
    # Depth + slightly brighter shoulder replicates gross top depression.
    strength=float(rng.uniform(.18,.37))
    multiplier=1+.10*ridge-strength*center
    is_fruit=(r<h*.42)&(np.max(np.abs(a-np.array([.86,.87,.88])*255),axis=2)>20)
    a[is_fruit]*=multiplier[is_fruit,None]
    return np.uint8(np.clip(a,0,255))


def simulated_ball_bullet_background(kind,seed):
    """Training augmentation: synthetic ball on a darker virtual table.

    Matches only gross camera-background colour/framing. This is NOT a real
    PyBullet rendering of a tennis/baseball or a photographed ball.
    """
    src=mock_ball_frame(kind,seed,side=96)
    a=np.asarray(Image.fromarray(src).resize((144,144)),dtype=np.uint8).copy()
    old_background=np.array([.86,.87,.88])*255
    neutral=np.max(np.abs(a.astype(float)-old_background),axis=2)<18
    rng=np.random.default_rng(seed)
    col=np.array([162,167,174],dtype=np.int16)+rng.integers(-5,6,3)
    a[neutral]=np.uint8(np.clip(col,0,255))
    im=Image.fromarray(a)
    dr=ImageDraw.Draw(im)
    # Simulated peripheral blue gripper hardware, not a category marker.
    for pos in ((0,35,10,46),(133,37,143,47),(4,120,14,141),(130,120,140,141)):
        dr.rectangle(pos,fill=(16,82,154))
    return np.asarray(im,dtype=np.uint8)


@lru_cache(maxsize=1)
def train_object_model():
    X=[];ys=[]
    for kind in OBJECT_TYPES:
        for i in range(160):
            frame=(simulated_ball_bullet_background(kind,11000+i)
                   if kind!='apple' and i%2 else make_object_frame(kind,11000+i))
            if kind=='apple':
                frame=augment_apple_illumination(frame,11000+i)
                if i%2!=0:
                    frame=simulated_apple_top_dimple(frame,11000+i)
            vec=object_features(frame)
            assert vec is not None,(kind,i)
            X.append(vec);ys.append(kind)
    model=RandomForestClassifier(n_estimators=240,min_samples_leaf=2,
                                 class_weight='balanced',random_state=32,n_jobs=1)
    model.fit(np.asarray(X),ys)
    return model


def identify_top_frame(model,frame,confidence_floor=.76):
    start=time.perf_counter()
    feat=object_features(frame)
    if feat is None:
        return {'accepted':False,'predicted_object':None,'status':'TOP_IMAGE_NO_OBJECT',
                'confidence':0.,'infer_wall_ms':(time.perf_counter()-start)*1000}
    scores=model.predict_proba(feat[None,:])[0]
    probabilities={str(c):float(p) for c,p in zip(model.classes_,scores)}
    idx=int(np.argmax(scores));kind=str(model.classes_[idx]);conf=float(scores[idx])
    accepted=kind=='apple' and conf>=confidence_floor
    return {'accepted':bool(accepted),'predicted_object':kind,
            'status':'APPLE_CONFIRMED' if accepted else ('NOT_APPLE' if kind!='apple' else 'UNCERTAIN_HOLD'),
            'confidence':conf,'scores':probabilities,'confidence_floor':confidence_floor,
            'infer_wall_ms':(time.perf_counter()-start)*1000,
            'model_scope':'shared RF; synthetic negatives + real-photo source/camera calibration; non-independent development model'}


def gate_candidate(model,mass_kg,top_frame):
    t=time.perf_counter();event=detect_placement(mass_kg)
    if not event.triggered:
        return {'allow_descent':False,'trigger':event.__dict__,'object':None,
                'gate_wall_ms':(time.perf_counter()-t)*1000}
    obj=identify_top_frame(model,top_frame)
    return {'allow_descent':bool(obj['accepted']),'trigger':event.__dict__,
            'object':obj,'gate_wall_ms':(time.perf_counter()-t)*1000}

if __name__=='__main__':
    import json
    m=train_object_model()
    for kind in OBJECT_TYPES:
        r=gate_candidate(m,.14,make_object_frame(kind,5001))
        print(json.dumps({'test_kind':kind,'allow':r['allow_descent'],
                          'predicted':r['object']['predicted_object'],'ms':r['gate_wall_ms']},ensure_ascii=False))
