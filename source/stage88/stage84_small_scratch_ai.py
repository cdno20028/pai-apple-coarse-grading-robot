# -*- coding: utf-8 -*-
"""Experimental camera-only recognition of a SHORT LINEAR dark surface wound.
No fruit IDs, filenames, oracle classifications or scripted grades are consulted.
Normal stem/calyx darkness is NOT a validated real orchard negative dataset.
Only finger images are eligible; data from all five cameras remains logged.
"""
import cv2
import numpy as np
from stage82_cd_fusion import fuse_cameras as baseline_fuse


def detect_small_linear_scratch(rgb):
    a=np.asarray(rgb,dtype=np.uint8)[:,:,:3]
    if a.ndim!=3 or a.shape[:2][0]<96:return {'detected':False,'reason':'invalid_frame'}
    h,w=a.shape[:2]
    hsv=cv2.cvtColor(a,cv2.COLOR_RGB2HSV);H,S,V=cv2.split(hsv)
    # A scar lies within red apple skin, not in green stem/calyx anatomy.
    skin=(((H<=12)|(H>=170))&(S>=55)&(V>60)).astype(np.uint8)
    # Restrict to actual red skin; non-red stems / green collar excluded.
    skin=cv2.morphologyEx(skin,cv2.MORPH_CLOSE,np.ones((11,11),np.uint8))
    core=cv2.erode(skin,np.ones((5,5),np.uint8))
    # Local dark-line anomaly without 3x3-opening (would delete small wounds).
    val=V.astype(np.float32)
    local=cv2.GaussianBlur(val,(0,0),sigmaX=8)
    diff=local-val
    # Detect contiguous narrow lines surrounded by real, reddish skin.
    candidate=((diff>=26)&(V<155)&(core>0)).astype(np.uint8)
    # Subtract isolated single pixels, keep thin linked tracks.
    n,lab,st,_=cv2.connectedComponentsWithStats(candidate,8)
    found=[]
    for i in range(1,n):
        x,y,bw,bh,area=map(int,st[i]);
        if area<10 or area>1300 or min(bw,bh)<2:continue
        if x<4 or y<4 or x+bw>=w-4 or y+bh>=h-4:continue
        # Elongation and dark contrast, not single-point black dots.
        xy=np.column_stack(np.where(lab==i))[:,::-1].astype(np.float32)
        eig=np.linalg.eigvalsh(np.cov(xy.T))
        aspect=(eig[-1]+.5)/(eig[0]+.5)
        if aspect<8.0:continue
        # A ring of surrounding pixels should be healthy red fruit skin.
        ring=np.zeros((h,w),np.uint8)
        cv2.drawContours(ring,cv2.findContours((lab==i).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,255,-1)
        outer=cv2.dilate(ring,np.ones((13,13),np.uint8))>0
        inner=cv2.dilate(ring,np.ones((3,3),np.uint8))>0
        border=outer&~inner
        ring_frac=float(skin[border].mean()) if border.any() else 0.
        if ring_frac<.90:continue
        strength=float(diff[lab==i].mean())
        if strength<32:continue
        found.append({'xyxy':[x,y,x+bw,y+bh],'pixels':area,
                      'elongation_ratio':round(float(aspect),2),
                      'red_skin_ring_fraction':round(ring_frac,3),
                      'mean_contrast':round(strength,1)})
    return {'detected':bool(found),'candidate_count':len(found),
       'best':max(found,key=lambda d:d['mean_contrast'],default=None),
       'rule':'dark elongated 10+ pixels; edge-excluded; reddish surrounding skin',
       'calibrated_on_real_apples':False}


def fuse_cameras(frames):
    out=baseline_fuse(frames)
    diagnostic={}
    for finger in range(3):
        nm=f'finger_{finger+1}'
        tests=[detect_small_linear_scratch(img)
               for (kind,j,t),img in frames.items() if kind=='finger' and j==finger]
        diagnostic[nm]={'observations':len(tests),'positive':sum(t['detected'] for t in tests),
                        'per_frame':tests}
    # Require two separate timed frames from the SAME real finger camera.
    # Not counting one view's frames as distinct cameras.
    hits=[k for k,v in diagnostic.items() if v['positive']>=2]
    out['stage84_scratch_detection']={'per_finger':diagnostic,'positive_fingers':hits,
        'sample_confirmed_on_two_actual_frames':bool(hits),
        'uses_case_id_or_known_grade':False,'real_orchard_validation':False}
    if hits and out['class']!='D':
        out['class']='C';out['decision_reason']='C_SMALL_SCRATCH_PERSISTENT_ON_FINGER'
    return out
