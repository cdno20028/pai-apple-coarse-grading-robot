# -*- coding: utf-8 -*-
"""Offline regression on four previously recorded REAL PyBullet Palm frames.
Case IDs are used solely to select held-out developmental models and check results;
no case ID, truth label or expected grade is passed to the runtime pixel transform.
Not a new simulation, an orchard validation, or a safety certification.
2026/10/22 JST
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import object_gate_ai as gate
import run_stage61_live_sim_hud as s61
from stage62_camera_ai import normalize_palm_fov
ROOT=Path(__file__).parent
models,diagnostics=s61.prepare_photo_loo_gate(ROOT,gate)
rows=[]
for name,photoid in (
  ('01_apple_A_A3','01_A'),('02_apple_B_B3','02_B'),
  ('03_apple_C_C1','04_C'),('04_apple_D_D1','08_D')):
    img=np.asarray(Image.open(ROOT/'stage62_v4_actual_palm_frames'/f'{name}.png').convert('RGB'))
    corr,settings=normalize_palm_fov(img)
    pred=gate.identify_top_frame(models[photoid],corr,confidence_floor=.76)
    rows.append({'case':name,'development_model_id':photoid,'accepted':pred['accepted'],
                 'predicted':pred['predicted_object'],
                 'apple_score':pred.get('scores',{}).get('apple'),
                 'crop_ratio':settings['ratio']})
assert len(rows)==4 and all(row['accepted'] and row['predicted']=='apple' for row in rows),rows
assert all(row['crop_ratio']==.54 for row in rows)
# Negative regression already run by test_stage62_v2_palm_fov_offline.py
print('STAGE62_V4_ACTUAL_CAMERA_REPLAY',json.dumps(rows,ensure_ascii=False),flush=True)
print('STAGE62_V4_ACTUAL_CAMERA_REPLAY_OK: 4 DEVELOPMENT FRAMES; NOT INDEPENDENT TEST',flush=True)
