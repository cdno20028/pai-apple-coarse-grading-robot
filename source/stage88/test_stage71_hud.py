# -*- coding: utf-8 -*-
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageFont
from stage62_workcell_video import draw_stage71_evidence
r=Path(__file__).resolve().parent
fonts={i:ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',i) for i in [16,19,22]}
# Japanese glyph coverage differs locally; only checks renderer runs and dimensions.
preview=np.asarray(Image.open(r/'stage58_apple_photo_assets/prepared/uv_photo_front/04_C_uv.png').convert('RGB'))[249:480,220:800]
evidence={'grade':'C','reason':'large_local_dark_patch_or_2_distinct_cameras',
  'positive':['finger_1','finger_2','finger_3'],
  'dark_pct':{'finger_1':4.07,'finger_2':4.97,'finger_3':4.85,'palm':0.0},'max_pct':4.97}
out=draw_stage71_evidence(Image.new('RGB',(1600,900),'white'),evidence,preview,fonts)
p=r/'stage71_hud_static_preview.png';out.save(p)
assert p.stat().st_size>10000
assert out.size==(1600,900)
print('STAGE71_HUD_STATIC_OK',p,p.stat().st_size)
