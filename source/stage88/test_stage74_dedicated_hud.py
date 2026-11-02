# -*- coding: utf-8 -*-
# 工程74 / 2026/10/22 JST
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import stage62_workcell_video as mod
root=Path(__file__).resolve().parent
fp='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fonts={i:ImageFont.truetype(fp,i) for i in [16,19,22,30,100]}
# Static geometric proof: HUD (x<=273) is distinct from 1600-px full scene (x>=320).
assert mod.STAGE73_EVIDENCE_PANEL[2] < 320
assert mod.STAGE73_GRADE_PANEL[2] < 320
raw=Image.new('RGB',(1600,900),'white')
d=ImageDraw.Draw(raw)
# Simulate worker C at LEFT END of world: it previously overlapped HUD.
d.rectangle((230,300,390,690),fill='#1a8c48')
full=Image.new('RGB',(1920,1080),'#081c2d')
full.paste(raw,(320,90))
mod.draw_stage72_evidence(full,None,None,fonts,None)
# Source actor pixels are intact, not masked by evidence drawing.
a=np.asarray(full)
assert np.all(a[390:780,550:710,:]==np.array([26,140,72],dtype=np.uint8))
# Camera panel stays unchanged in full-resolution original position +320,+90.
assert np.all(a[190:290,1700:1750,:]==255)
full.save(root/'stage74_sidebar_static_preview.png')
print('STAGE74_DEDICATED_GUTTER_PREFLIGHT_PASS: 1920x1080; original 1600x900 scene unscaled and unoccluded')
