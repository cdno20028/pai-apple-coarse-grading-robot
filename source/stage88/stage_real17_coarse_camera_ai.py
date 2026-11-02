# -*- coding:utf-8 -*-
"""Supervised nearest-prototype *known-photo* coarse grading, not blind validation.

A proof of concept: 3 real finger-camera RGB views, no case/asset/truth at inference.
The training set contains images from the same 9 known apple presentations in
an earlier PyBullet run. Do not use as evidence of field or unknown-apple accuracy.
"""
import os
from pathlib import Path
import cv2
import numpy as np
from stage84_small_scratch_ai import fuse_cameras as inherited_fuse

ROOT=Path(__file__).resolve().parent
MODEL=ROOT/'real17_coarse_camera_training'/'camera_rgb_prototypes.npz'

def embed_finger_images(images):
    if len(images)!=3:raise ValueError('THREE_REAL_FINGERS_REQUIRED')
    chunks=[]
    for image in images:
        x=np.asarray(image,dtype=np.uint8)
        if x.ndim!=3 or x.shape[2]!=3: raise ValueError('CAMERA_RGB_SHAPE_INVALID')
        lab=cv2.cvtColor(cv2.resize(x,(32,32),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2LAB)
        chunks.append(lab.astype(np.float32).ravel()/255.0)
    return np.concatenate(chunks)

def infer_known_photo_grade(frames):
    imgs=[]
    for camera in range(3):
        img=frames.get(('finger',camera,0))
        if img is None:return {'accepted':False,'reason':'MISSING_FIRST_SCAN_VIEW'}
        imgs.append(img)
    x=embed_finger_images(imgs)
    with np.load(MODEL,allow_pickle=False) as npz:
        protos=npz['prototypes']; labels=npz['labels'].astype(str)
    # Stage89: finger 2 has a fixed downward optical mount. The old
    # supervised camera prototypes contain a front-facing finger 2,
    # so compare only finger 1 and finger 3 when this mode is enabled.
    # This *known-photo* baseline is still not independent evidence.
    if os.environ.get('STAGE89_REAL_WOUND_GATE')=='1':
        keep=np.ones(x.shape,dtype=bool);keep[32*32*3:2*32*32*3]=False
        distances=np.sqrt(np.sum((protos[:,keep]-x[None,keep])**2,axis=1))
    else:
        distances=np.sqrt(np.sum((protos-x[None,:])**2,axis=1))
    first=int(np.argmin(distances)); best=float(distances[first]);grade=labels[first]
    different=distances[labels!=grade]
    other=float(np.min(different)) if different.size else float('inf')
    # Known-photo envelope: exact reruns are 0, different-grade prototypes >5.
    # This is a sample similarity gate, NOT calibrated grade confidence.
    accept=bool(best<=4.0 and (other-best)>=0.35)
    return {'accepted':accept,'class':str(grade) if accept else None,
            'nearest_prototype_distance':round(best,5),
            'nearest_other_grade_distance':round(other,5),
            'rule':'3 actual-link finger views, 32x32 LAB nearest supervised prototype',
            'training_photos_also_test_photos':True,
            'not_independent_orchard_validation':True,
            'reason':'KNOWN_REALPHOTO_CAMERA_MATCH' if accept else 'OUTSIDE_KNOWN_PHOTO_ENVELOPE'}

def fuse_cameras(frames):
    result=inherited_fuse(frames)
    if os.environ.get('REAL17_COARSE_CAMERA_MODEL')!='1':return result
    supervised=infer_known_photo_grade(frames)
    result['original_unsupervised_class_before_coarse_model']=result['class']
    result['original_unsupervised_decision_reason_before_coarse_model']=result.get('decision_reason')
    result['real17_supervised_known_photo_development']=supervised
    if supervised['accepted']:
        result['class']=supervised['class']
        result['decision_reason']='COARSE_SUPERVISED_KNOWN_PHOTO_MULTICAMERA'
    else:
        result['decision_reason']='COARSE_LOW_SIMILARITY_FALLBACK_'+str(result.get('decision_reason','UNKNOWN'))
    if os.environ.get('STAGE89_REAL_WOUND_GATE')=='1':
        from stage89_real_scar_ai import grade_scar_evidence
        scar=grade_scar_evidence(frames)
        result['stage89_actual_camera_scar_evidence']=scar
        if scar['visible_wound'] and os.environ.get('STAGE90_DAMAGE_EVIDENCE')!='1':
            result['class']='C'
            result['decision_reason']='C_ACTUAL_FINGER2_VISIBLE_DARK_LINEAR_WOUND'
            result['grade_reasons_do_not_use_known_photo_similarity']=True
        else:
            result['stage89_scar_rule_did_not_support_grade']=True
    if os.environ.get('STAGE90_DAMAGE_EVIDENCE')=='1':
        from stage90_wound_evidence import assess
        evidence=assess(frames)
        result['stage90_severe_wound_evidence']=evidence
        if evidence['confirmed_severe_wound']:
            result['class']='C'
            result['decision_reason']='STAGE90_MULTI_FRAME_SEVERE_WOUND_FROM_REAL_CAMERAS'
            result['grade_reasons_do_not_use_known_photo_similarity']=True
        elif result['class']=='C':
            # Do not present known-image C as verified damage recognition.
            result['stage90_C_not_verified']=True
            result['stage90_provisional_known_photo_class']='C'
            result['class']=None
            result['decision_reason']='UNVERIFIED_SEVERE_DAMAGE_HOLD_NO_C_ROUTING'
    return result
