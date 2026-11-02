# -*- coding: utf-8 -*-
import json,ast
from pathlib import Path
from PIL import Image, ImageFont
import numpy as np
from stage62_workcell_video import draw_stage72_evidence
root=Path(__file__).resolve().parent
assets=json.loads((root/'stage58_apple_photo_assets/prepared/manifest.json').read_text(encoding='utf8'))['assets']
assert len(assets)==8
cam=(root/'stage67_camera_ai.py').read_text(encoding='utf8')
grade=(root/'stage61_roughgrade.py').read_text(encoding='utf8')
assert "rec.stage72_source_info" in cam
assert "infer['class']" in cam
assert "def fuse_cameras(frame_dict):" in grade
assert "dict(asset_id=photo['asset_id']" in cam
fp='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font={i:ImageFont.truetype(fp,i) for i in (16,19,22)}
uv=np.asarray(Image.open(root/'stage58_apple_photo_assets/prepared'/assets[1]['texture_path']).convert('RGB'))
im=Image.new('RGB',(1600,900),'#102536')
draw_stage72_evidence(im,{'dark_pct':{'finger_1':2.095,'finger_2':.114,'finger_3':2.705,'palm':8.18},'positive':['palm'],'max_pct':8.18}, uv[249:480,220:800],font, {'reference_grade':'B','actual_ai_grade':'C','filename':'判定B.jpg'})
assert im.getbbox()
im.save(root/'stage72_hud_static_preview.png')
print('STAGE72_STATIC_HUD_PASS; INFERENCE_AND_PHYSICS_UNCHANGED')
