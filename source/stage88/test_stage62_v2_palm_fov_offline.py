# -*- coding: utf-8 -*-
"""Stage62 v2: OFFLINE regression preflight, not an independent field accuracy claim."""
from pathlib import Path
from PIL import Image
import sys, json, numpy as np
import object_gate_ai as oa
import run_stage61_live_sim_hud as s61
from stage62_camera_ai import normalize_palm_fov
ROOT=Path(__file__).resolve().parent

def main():
    models,diag=s61.prepare_photo_loo_gate(ROOT,oa)
    assert len(models)==8 and all(i['held_out_apple']['accepted'] for i in diag)
    raw=np.asarray(Image.open(ROOT/'diagnostic_stage62_case1_before_patch.png').convert('RGB'))
    normalized,fov=normalize_palm_fov(raw)
    a=oa.identify_top_frame(models['01_A'],normalized,confidence_floor=.76)
    b=oa.identify_top_frame(models['01_A'],raw,confidence_floor=.76)
    assert not b['accepted'] and a['accepted'],(b,a)
    # Domain-alignment regression: scale existing 8 Stage61 images down to
    # the measured Stage62 camera crop, then normalize them back with fixed ROI.
    cases=[]
    for p in sorted((ROOT/'stage58_v2_timed_5camera_5cm_results').glob('case_*/palm_object_gate.png')):
        im=np.asarray(Image.open(p).convert('RGB'))
        bg=np.median(np.concatenate((im[:10,:10].reshape(-1,3),im[-10:,-10:].reshape(-1,3))),axis=0).astype('uint8')
        k=round(im.shape[0]*fov['ratio']);start=(im.shape[0]-k)//2
        small=np.asarray(Image.fromarray(im).resize((k,k),Image.Resampling.BILINEAR))
        far=np.full_like(im,bg);far[start:start+k,start:start+k]=small
        pred=oa.identify_top_frame(models[p.parent.name.split('_',2)[-1]],normalize_palm_fov(far)[0],.76)
        cases.append({'case':p.parent.name,'accepted':pred['accepted'],'apple_score':pred['scores']['apple']})
    assert len(cases)==8 and sum(x['accepted'] for x in cases)>=7,cases
    # An illustrative downscaled D photo can still be rejected. Keep that warning visible.
    calibration_residual_risk=[x for x in cases if not x['accepted']]
    # Synthetic negatives: 3 classes x 30 images x 8 developmental LOO models
    # All using the exact same camera correction as the positive cases.
    false_accept=[];n=0
    for kind in ('baseball','tennis','red_ball'):
        for i in range(30):
            frame=(oa.simulated_ball_bullet_background(kind,1000+i) if i%2
                else oa.mock_ball_frame(kind,900+i,side=144))
            feat_img=normalize_palm_fov(frame)[0]
            for label,model in models.items():
                r=oa.identify_top_frame(model,feat_img,.76);n+=1
                if r['accepted']:false_accept.append((kind,i,label))
    assert n==720 and not false_accept,false_accept[:20]
    print('STAGE62_V2_OFFLINE_GATE_PREFLIGHT_OK',json.dumps({
        'real_failed_case_01_v1_scores':b['scores'],
        'real_failed_case_01_after_fix_scores':a['scores'],
        'stage61_photo_domain_rescale_8':cases,'downscaled_photo_false_rejections':calibration_residual_risk,
        'synthetic_negative_checks':n,'false_accepts':len(false_accept),
        'fov':fov, 'disclaimer':'THIS IS DEVELOPMENT DATA, not blind orchard validation'},
        ensure_ascii=False),flush=True)
    return 0
if __name__=='__main__':sys.exit(main())
