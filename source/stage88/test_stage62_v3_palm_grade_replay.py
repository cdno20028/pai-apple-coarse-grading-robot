#!/usr/bin/env python3
# Offline replay of actual Stage62-v2 Palm images + recorded three finger-camera
# feature maxima. NO PyBullet execution, NO unknown-fruit validation.
from pathlib import Path
import json, numpy as np
from PIL import Image
from stage62_camera_ai import normalize_palm_fov
from stage61_roughgrade import one_view
root=Path(__file__).parent/'stage62_v3_regression_inputs'
limits={'finger_1':.030,'finger_2':.010,'finger_3':.030,'palm':.035}
for name,expected in [('01_apple_A_A3','A'),('02_apple_B_B3','B')]:
    base=root/('palm_'+name)
    rgb=np.asarray(Image.open(base.with_suffix('.png')).convert('RGB'))
    record=json.loads(base.with_suffix('.json').read_text(encoding='utf8'))
    corrected,info=normalize_palm_fov(rgb)
    raw=one_view(rgb);fixed=one_view(corrected)
    assert raw['valid'] and fixed['valid']
    assert abs(raw['largest_dark_patch_fraction']-record['recorded_palm_feature_max'])<.00002
    fingers=record['finger_camera_patch_max']
    def decision(palm_fraction):
        readings={**fingers,'palm':palm_fraction}
        positives=[k for k,v in readings.items() if v>=limits[k]]
        if max(readings.values())>=.070 or len(positives)>=2:return 'C'
        if positives:return 'B'
        return 'A'
    grade_before=decision(raw['largest_dark_patch_fraction'])
    grade_after=decision(fixed['largest_dark_patch_fraction'])
    assert grade_before==record['recorded_v2_class']
    assert grade_after==expected, (name,grade_after,readings)
    assert info['ratio']==.54
    print('STAGE62_V3_REPLAY',name,'old',grade_before,'fixed',grade_after,
          'raw_palm',raw['largest_dark_patch_fraction'],
          'corrected_palm',fixed['largest_dark_patch_fraction'],flush=True)
print('STAGE62_V3_REPLAY_OK: 2/2 DEVELOPMENT REPLAY; NOT PHYSICAL CONTINUOUS TEST')
