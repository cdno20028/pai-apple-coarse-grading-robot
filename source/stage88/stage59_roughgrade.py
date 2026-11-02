#!/usr/bin/env python3
# coding: utf-8
"""Experimental pixel-only five-camera apple skin anomaly fusion.

Never uses user-provided grade/filename/fruit identity as a predictor. The
camera samples are grouped by the physical camera ID, not by number of frames.
The underside camera is evaluated and logged, but its photo-front/back seam
can be flagged as unreliable; no farm validated damage recognition is claimed.
"""
from __future__ import annotations
import cv2
import numpy as np

def one_view(rgb):
    """Unsupervised contrast/colour cue measurements, no grade-label input."""
    a=np.asarray(rgb,dtype=np.uint8)
    if a.ndim!=3 or a.shape[2]<3:return {'valid':False,'reason':'not_RGB'}
    a=a[:,:,:3]
    hsv=cv2.cvtColor(a,cv2.COLOR_RGB2HSV)
    H,S,V=cv2.split(hsv)
    sat=((S>48)&(V>43)).astype(np.uint8)
    n,lab,stats,_=cv2.connectedComponentsWithStats(sat,8)
    if n<2:return {'valid':False,'reason':'no_saturated_object'}
    # Select a connected component covering image middle; do not use grade.
    centre=lambda s: (s[0]<a.shape[1]*.67 and s[0]+s[2]>a.shape[1]*.33 and
                      s[1]<a.shape[0]*.67 and s[1]+s[3]>a.shape[0]*.33)
    ids=sorted(range(1,n),key=lambda j:int(stats[j,4]),reverse=True)
    ident=next((j for j in ids if stats[j,4]>100 and centre(stats[j])),ids[0])
    mask=(lab==ident).astype(np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((13,13),np.uint8))
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if cnts:cv2.drawContours(mask,[max(cnts,key=cv2.contourArea)],-1,1,-1)
    core=cv2.erode(mask,np.ones((9,9),np.uint8))>0
    if int(core.sum())<350:return {'valid':False,'reason':'too_small_for_local_skin_damage'}
    val=V.astype(float);smooth=cv2.GaussianBlur(val,(0,0),sigmaX=15)
    rel_dark=(smooth-val)/(smooth+1)
    lesion=((rel_dark>.25)&core&(val<170)).astype(np.uint8)
    lesion=cv2.morphologyEx(lesion,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    _,_,components,_=cv2.connectedComponentsWithStats(lesion,8)
    sizes=sorted((int(c[4]) for c in components[1:] if c[4]>=5),reverse=True)
    biggest=sizes[0]/core.sum() if sizes else 0.0
    yellow=((H>=12)&(H<=40)&(S>50)&core)
    rg=float(np.median((a[:,:,0][core].astype(float)+1)/(a[:,:,1][core].astype(float)+1)))
    return {'valid':True,'visible_pixels':int(core.sum()),
       'largest_dark_patch_fraction':round(float(biggest),5),
       'all_dark_patch_fraction':round(float(np.count_nonzero(lesion)/core.sum()),5),
       'yellow_green_pixel_fraction':round(float(yellow.sum()/core.sum()),5),
       'median_red_green_ratio':round(rg,4),'dark_components_px':sizes[:5]}

def fuse_cameras(frame_dict):
    """Use camera IDs only. 4 frames of one finger can NEVER count as 4 cameras.

    Input DIL experiment mapping: ('finger', 0..2, time_index), ('palm',0,0),
    ('bottom',0,0..2). No asset ID, reference label, or result folder is read.
    """
    camera_frames={f'finger_{k+1}':[] for k in range(3)}
    camera_frames.update({'palm':[],'inspection_under':[]})
    for (kind,idx,_tick),rgb in frame_dict.items():
        name=f'finger_{idx+1}' if kind=='finger' else 'palm' if kind=='palm' else 'inspection_under' if kind=='bottom' else None
        if name is not None:camera_frames[name].append(one_view(rgb))
    per={}
    for name,rows in camera_frames.items():
        valid=[r for r in rows if r.get('valid')]
        per[name]={'frames_seen':len(rows),'frames_valid':len(valid),
                   'largest_dark_patch_max':max((r['largest_dark_patch_fraction'] for r in valid),default=0.),
                   'yellow_green_max':max((r['yellow_green_pixel_fraction'] for r in valid),default=0.),
                   'median_rg_min':min((r['median_red_green_ratio'] for r in valid),default=None),
                   'frame_features':rows}
    # Synthetic backside of supplied single-view photos creates a conspicuous
    # vertical seam. Do not interpret that seam as a real-life rot/bruising.
    # Its measurements are published but NOT trusted for automatic wound count.
    per['inspection_under']['defect_valid_for_fusion']=False
    per['inspection_under']['reason']='PHOTO_UV_BACK_SEAM_QUALITY_GATE'
    # Finger-1/3 camera detects may include seam and arm shadow; require a
    # larger localized region than finger-2, not a second video frame.
    threshold={'finger_1':.030,'finger_2':.010,'finger_3':.030,'palm':.035}
    positives=[name for name,limit in threshold.items() if per[name]['largest_dark_patch_max']>=limit]
    max_patch=max((per[name]['largest_dark_patch_max'] for name in threshold),default=0.)
    # Unripe D colour: palm or middle finger must show a broad green-yellow
    # region rather than a small yellow patch surrounding an ordinary stem.
    palm_y=per['palm']['yellow_green_max']
    f2_y=per['finger_2']['yellow_green_max']
    immature=(palm_y>=.60 and f2_y>=.65)
    if immature:
        label='D';reason='majority_yellow_green_skin_on_palm_and_finger2'
    elif max_patch>=.070 or len(positives)>=2:
        label='C';reason='large_local_dark_patch_or_2_distinct_cameras'
    elif positives:
        label='B';reason='localized_dark_patch_one_camera'
    else:
        label='A';reason='no_local_dark_patch_above_diagnostic_threshold'
    return {'status':'EXPERIMENTAL_NOT_VALIDATED','class':label,
      'decision_reason':reason,'positive_distinct_cameras':positives,
      'distinct_cameras_with_damage':len(positives),
      'max_localized_dark_patch_fraction':round(float(max_patch),5),
      'd_immature_skin_detected':bool(immature),'per_camera':per,
      'only_images_used':True,'uses_weight_for_grading':False,
      'under_plate_photo_seam_excluded_from_defect_count':True,
      'no_defect_triangulation_or_real_farm_accuracy_claimed':True}
