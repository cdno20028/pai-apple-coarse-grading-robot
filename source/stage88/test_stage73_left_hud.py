# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
from PIL import Image,ImageFont,ImageDraw
from stage62_workcell_video import draw_stage72_evidence,STAGE73_EVIDENCE_PANEL,STAGE73_GRADE_PANEL
base=Path(__file__).resolve().parent
fp='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fonts={i:ImageFont.truetype(fp,i) for i in (16,19,22)}
assert STAGE73_EVIDENCE_PANEL[2]<280
assert STAGE73_GRADE_PANEL[2]<280
im=Image.new('RGB',(1600,900),'white')
uv=np.full((231,580,3),110,dtype=np.uint8)
draw_stage72_evidence(im,{'dark_pct':{'finger_1':2.095,'finger_2':.114,'finger_3':2.705,'palm':8.18},'positive':['palm'],'max_pct':8.18},uv,fonts,{'reference_grade':'B','actual_ai_grade':'C','filename':'判定B.jpg'})
# This panel is limited to x<=273; no painted pixels in the worker-C scene.
assert im.crop((290,100,1035,810)).getbbox() is not None
assert np.asarray(im)[100:810,290:1035].min()==255
im.save(base/'stage73_hud_left_static_preview.png')
print('STAGE73_HUD_LEFT_NO_WORKER_OCCLUSION_STATIC_PASS')
