#!/usr/bin/env python3
"""Stage33: 4 development screenshots (NOT independent validation) and synthetic controls."""
from pathlib import Path
import json,statistics,time
import numpy as np
from PIL import Image,ImageDraw
import object_gate_ai as obj
import visual_grade_ai as grade
import run_stage33_physics as run
HERE=Path(__file__).resolve().parent

def bullet_style_ball(kind,seed):
    """A synthetic negative, not a real PyBullet image or ball photograph."""
    base=Image.fromarray(obj.mock_ball_frame(kind,seed,side=96)).resize((144,144))
    data=np.array(base)
    # Match rendered grey table background, not the original 220-grey.
    back=np.array([.86,.87,.88])*255
    neutral=np.max(np.abs(data.astype(float)-back),axis=2)<18
    data[neutral]=[162,167,174]
    im=Image.fromarray(data)
    d=ImageDraw.Draw(im)
    # Blue robot arm shapes outside center do not determine the label.
    for pos in ((0,35,10,46),(133,37,143,47),(4,120,14,141),(130,120,140,141)):
        d.rectangle(pos,fill=(16,82,154))
    return np.asarray(im,dtype=np.uint8)

def main():
    t=time.perf_counter();model=obj.train_object_model();train_ms=(time.perf_counter()-t)*1000
    checks=[]
    screenshots=[]
    false_rejects=[]
    for grade_label in 'ABCD':
        path=HERE/'diagnostic_stage32_camera'/f'upper_view_{grade_label}.png'
        frame=np.asarray(Image.open(path).convert('RGB'))
        result=obj.identify_top_frame(model,frame)
        assert result['accepted'],(grade_label,result)
        screenshots.append({'actual_stage32_synthetic_fruit':grade_label,
                            'predicted_object':result['predicted_object'],
                            'apple_probability':result['scores']['apple'],
                            'accepted':result['accepted'],'infer_wall_ms':result['infer_wall_ms']})
    for kind in obj.OBJECT_TYPES:
        for i in range(50):
            # Held-out randomization from same generator / textures.
            image=obj.make_object_frame(kind,42000+i)
            pred=obj.identify_top_frame(model,image)
            if kind=='apple' and not pred['accepted']:
                false_rejects.append({'seed':42000+i,'confidence':pred['confidence']})
            if kind!='apple':
                assert not pred['accepted'],(kind,i,pred)
            checks.append({'type':kind,'domain':'synthetic_light_background','accepted':pred['accepted']})
    for kind in ('baseball','tennis','red_ball'):
        for i in range(50):
            image=bullet_style_ball(kind,15000+i)
            pred=obj.identify_top_frame(model,image)
            assert not pred['accepted'],(kind,i,pred)
            checks.append({'type':kind,'domain':'synthetic_bullet_style_background','accepted':pred['accepted']})
    assert len(false_rejects)<=5,('too many synthetic apple false rejects',false_rejects)
    for mass in (.057,.145,.18,.3):
        assert obj.detect_placement(mass).triggered
    assert not obj.detect_placement(.00).triggered
    # The legacy physical motion source remains unchanged.
    import zipfile,hashlib
    with zipfile.ZipFile(HERE/'stage27_original_96files.zip') as z:
        original=z.read('apple_stage13/src/run_physics.py')
    assert hashlib.sha256(original).hexdigest()==run.prev.EXPECTED_SOURCE_SHA
    # Interlock stays before scan/close; no permissive threshold change.
    assert 'STAGE32_PREGRIP_GATE' in run.patch_gate(original.decode('utf8').replace(run.prev.OLD_GATE,run.prev.NEW_GATE,1))
    report={'timestamp_jst':run.now(),'stage':'33','result':'OFFLINE_TEST_PASS',
            'dev_camera_screenshots':screenshots,'dev_camera_acceptance':sum(r['accepted'] for r in screenshots),
            'synthetic_gate_trials':len(checks),
            'synthetic_apple_accepted':sum(r['accepted'] and r['type']=='apple' for r in checks),
            'synthetic_apple_false_rejected':false_rejects,
            'synthetic_nonapple_rejected':sum(not r['accepted'] and r['type']!='apple' for r in checks),
            'object_training_wall_ms':round(train_ms,3),'mass_for_grading':False,
            'confidence_floor':.76,'camera_images_in_training':False,
            'images_used_for_design_iteration':True,
            'physical_trial_run_now':False,
            'real_ball_camera_negative_controls':False,
            'independent_real_world_accuracy_confirmed':False}
    (HERE/'OFFLINE_STAGE33_RESULTS.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print('STAGE33_OFFLINE_CHECK',json.dumps(report,ensure_ascii=False,indent=2))
    return report
if __name__=='__main__':main()
