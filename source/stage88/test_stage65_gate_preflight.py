# -*- coding: utf-8 -*-
from pathlib import Path
import json
import numpy as np
from PIL import Image
from stage65_common_gate import train_shared_stage65_gate
import object_gate_ai as ai

root=Path(__file__).resolve().parent
model,record=train_shared_stage65_gate(root)
case=np.asarray(Image.open(root/'stage65_regression/stage64_actual_case1_A.png').convert('RGB'))
result=ai.identify_top_frame(model,case,confidence_floor=.76)
record['stage64_real_failure_frame_regression']={
 'original_stage64_apple_score':0.5879428237200239,
 'common_stage65_score':result['scores']['apple'],
 'passes_threshold':result['accepted'],
 'photo_was_known_to_training_source_collection':True,
 'real_apple_hardware_test':False}
assert result['accepted'],'STAGE65_STAGE64_REAL_FRAME_STILL_REJECTED'
assert all(x['apple_accepted'] for x in record['known_training_photo_checks'])
record['grade_not_prevalidated']=True
record['physical_sim_not_run_in_this_preflight']=True
record['limitations']='Development regression only; this is NOT orchard validation or independent held-out test.'
(root/'STAGE65_OFFLINE_PREFLIGHT.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
print('STAGE65_OFFLINE_PREFLIGHT_PASS',json.dumps({'actual_case1_apple_score':result['scores']['apple'],
 'synthetic_negative_false_accepts':record['synthetic_negative_false_accepts'],
 'known_photo_train_check_count':len(record['known_training_photo_checks']),
 'caution':'Known photo assets WERE used in training. Not independent.'},ensure_ascii=False))
