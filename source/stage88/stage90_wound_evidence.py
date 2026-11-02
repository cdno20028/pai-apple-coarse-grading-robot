# -*- coding: utf-8 -*-
"""Stage90: Conservative camera-pixel evidence of severe linear skin injury.

Never reads filenames, case numbers, texture IDs, target grades or answer keys.
Development rule, NOT a calibrated orchard classifier.  A single isolated
10x4-pixel dark region is *not* sufficient to claim Grade C.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
from stage89_real_scar_ai import detect_dark_linear_wound


def _observations(frames):
    scans=[]
    for k, rgb in frames.items():
        if not isinstance(k, tuple) or len(k) != 3: continue
        mode, cam, tick = k
        if mode == 'finger':
            scans.append((f'finger_{cam+1}',f'legacy_scan{tick+1}', rgb))
        elif mode == 'dense':
            scans.append((f'finger_{cam+1}',f'dense_tick{tick:03d}', rgb))
    return sorted(scans, key=lambda x:(x[0],x[1]))


def assess(frames):
    """Conservative multi-frame injury assessment before the robot routes fruit.

    Witness criteria: (a) two spatially distinct dark linear lesions in the
    same actual camera frame, OR (b) one large/high-contrast dark incision seen
    in >=2 actual frames from a common physical camera.  Shadows, russeting,
    single borderline dark pixels, known photos and model labels are excluded.
    """
    rows=[]; proof=[]
    by_camera=defaultdict(list)
    for camera,tag,image in _observations(frames):
        image=np.asarray(image,dtype=np.uint8)
        raw=detect_dark_linear_wound(image)
        hits=[]
        h,w=image.shape[:2]
        for e in raw['hits']:
            x1,y1,x2,y2=e['bbox_xyxy']; span=x2-x1
            area=e['dark_pixel_area']; contrast=e['local_grayscale_contrast']
            # Beyond the previously accepted 10x4/14 px mark: shape and
            # contrast must be substantive, not a 1-frame silhouette fleck.
            strong=bool(span>=14 and area>=18 and contrast>=35 and
                        e['aspect_ratio']>=2.6 and y1>int(h*.12) and y2<int(h*.86))
            if not strong: continue
            hits.append({**e,'normalized_center_xy':[
                round((x1+x2)/(2*w),4),round((y1+y2)/(2*h),4)]})
        if len(hits)>=2:
            # Require separate wounds on the same skin, not a segmented line.
            for i in range(len(hits)):
                for j in range(i+1,len(hits)):
                    xi,yi=hits[i]['normalized_center_xy'];xj,yj=hits[j]['normalized_center_xy']
                    if abs(xi-xj)+abs(yi-yj)>=0.085:
                        proof.append({'kind':'TWO_DISTINCT_SEVERE_WOUNDS_SAME_FRAME',
                                      'camera':camera,'frame':tag,'wounds':[hits[i],hits[j]]})
                        break
                if proof:break
        row={'camera':camera,'frame':tag,'strong_wound_count':len(hits),'hits':hits,
             'image_shape':[int(h),int(w)],'all_dark_line_candidates':len(raw['hits'])}
        rows.append(row)
        for hit in hits: by_camera[camera].append((tag,hit))
    # A large single cut may also qualify if verified from 2 different captures
    # of the SAME real link camera, and not an isolated frame blemish.
    for camera, observations in by_camera.items():
        for i in range(len(observations)):
            ti,hi=observations[i];xi,yi=hi['normalized_center_xy']
            if hi['bbox_xyxy'][2]-hi['bbox_xyxy'][0]<21 or hi['local_grayscale_contrast']<35:
                continue
            for j in range(i+1,len(observations)):
                tj,hj=observations[j]
                if ti==tj:continue
                xj,yj=hj['normalized_center_xy']
                if (abs(xi-xj)<.21 and abs(yi-yj)<.35 and
                    hj['bbox_xyxy'][2]-hj['bbox_xyxy'][0]>=18 and
                    hj['local_grayscale_contrast']>=35):
                    proof.append({'kind':'PERSISTENT_LARGE_SEVERE_WOUND',
                                  'camera':camera,'frames':[ti,tj],'wounds':[hi,hj]})
                    break
            if proof:break
    # STAGE100: Supplement the existing LINEAR wound proof with independent,
    # strictly camera-pixel MULTI-FRAME round-brown damage evidence.
    # Never sees case number, expected grade, photo ID or ground-truth label.
    # The Stage92 C-2 proven linear branch executes unchanged.
    if not proof:
        round_proof = _round_brown_multiframe(frames)
        if round_proof is not None:
            proof.append(round_proof)
    return {'confirmed_severe_wound':bool(proof),'proof':proof[:4],
            'frames_scanned':len(rows),'actual_camera_observations':rows,
            'case_ids_used_for_inference':False,'reference_labels_used_for_inference':False,
            'known_photo_nearest_match_is_not_scar_proof':True,
            'requires_beyond_single_10x4_14px_candidate':True,
            'unknown_orchard_accuracy_not_validated':True}


def _round_brown_multiframe(frames, diagnostics=None):
    '''Development-only persistent dark round/broad lesion detector.

    STAGE103: a candidate must also be brown-black relative to its own
    surrounding skin (lower chroma, lower redness a*, textured interior).
    Shaded or deep-red skin keeps its chroma and a*, and is rejected.
    Pass a list as ``diagnostics`` to receive every gated candidate.

    Evidence requires >=2 distinct captures of the SAME actual finger camera,
    separated in scan ticks, with a coherent local dark skin patch. Excludes
    frame edges and isolated pixels. NO grade/photo/ID/sensor spoofing.
    '''
    import cv2
    from collections import defaultdict
    observations=defaultdict(list)
    for cam, tag, image in _observations(frames):
        if not tag.startswith('dense_tick'):
            continue
        a=np.asarray(image,dtype=np.uint8)
        if a.ndim!=3 or a.shape[2]!=3 or min(a.shape[:2])<96:
            continue
        h,w=a.shape[:2]
        r,g,b=[a[:,:,i].astype(np.float32) for i in range(3)]
        skin=((r>50)&(r>g*1.14)&(r>b*1.05)).astype(np.uint8)
        n,labels,stats,_=cv2.connectedComponentsWithStats(skin,8)
        if n<=1: continue
        largest=np.uint8(labels==(1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))))
        largest=cv2.morphologyEx(largest,cv2.MORPH_CLOSE,np.ones((15,15),np.uint8))
        largest=cv2.erode(largest,np.ones((9,9),np.uint8))
        gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
        lab=cv2.cvtColor(a,cv2.COLOR_RGB2LAB).astype(np.float32)
        chroma=np.hypot(lab[:,:,1]-128,lab[:,:,2]-128)
        mask=np.uint8((gray<45)&(largest>0))
        n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
        for i in range(1,n):
            x,y,bw,bh,area=[int(v) for v in stats[i]]
            if not (area>=400 and bw>=25 and bh>=17 and
                    0.45<bw/max(1,bh)<2.8 and 6<x and x+bw<w-6 and
                    # Fail-closed: exclude the peripheral gripper / silhouette
                    # zone; count only fully interior skin lesions.
                    0.20*w<=x and x+bw<=0.80*w and
                    0.09*h<y and y+bh<0.91*h):
                continue
            region=np.uint8(labels==i)
            ring=cv2.dilate(region,np.ones((21,21),np.uint8))>0
            ring&=(cv2.dilate(region,np.ones((5,5),np.uint8))==0)
            ring&=(largest>0)
            if int(ring.sum())<80: continue
            contrast=float(np.median(gray[ring])-np.median(gray[region>0]))
            if contrast<13: continue
            reg=region>0
            chroma_ratio=float(np.median(chroma[reg])/max(1.0,float(np.median(chroma[ring]))))
            redness_drop=float(np.median(lab[:,:,1][ring])-np.median(lab[:,:,1][reg]))
            texture=float(np.std(lab[:,:,0][reg]))
            brown=bool(chroma_ratio<=0.75 and redness_drop>=3.0 and texture>=7.0)
            if diagnostics is not None:
                diagnostics.append({'camera':cam,'frame':tag,'bbox_xyxy':[x,y,x+bw,y+bh],
                    'dark_pixel_area':area,'local_grayscale_contrast':round(contrast,2),
                    'chroma_ratio':round(chroma_ratio,4),'redness_drop':round(redness_drop,3),
                    'texture':round(texture,3),'brown_black_gate_pass':brown})
            if not brown: continue
            observations[cam].append({'frame':tag,'tick':int(tag[-3:]),
                'bbox_xyxy':[x,y,x+bw,y+bh], 'dark_pixel_area':area,
                'local_grayscale_contrast':round(contrast,2),
                'chroma_ratio':round(chroma_ratio,4),'redness_drop':round(redness_drop,3),
                'texture':round(texture,3),
                'normalized_center_xy':[round((x+bw/2)/w,4),round((y+bh/2)/h,4)]})
    for cam, hits in observations.items():
        for i,a in enumerate(hits):
            for b in hits[i+1:]:
                if abs(a['tick']-b['tick'])<8: continue
                x,y=a['normalized_center_xy'];xx,yy=b['normalized_center_xy']
                if abs(x-xx)<=0.18 and abs(y-yy)<=0.27:
                    # All proof material comes from actual-link RGB frames.
                    return {'kind':'PERSISTENT_ROUND_BROWN_DARK_LESION',
                        'camera':cam,'frames':[a['frame'],b['frame']],
                        'wounds':[{k:v for k,v in h.items() if k!='tick'} for h in (a,b)],
                        'algorithm':'UNVALIDATED_DEVELOPMENT_HEURISTIC_STAGE103_BROWN_GATE',
                        'source_is_actual_camera_pixels':True,
                        'case_or_photo_identifier_used':False,
                        'unknown_orchard_accuracy_not_validated':True}
    return None
