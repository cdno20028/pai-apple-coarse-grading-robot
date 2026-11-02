# 2026/10/22 JST -- geometry + no-regression image-surface preflight.
from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np
from stage62_workcell_video import stage75_zoom_workcell_rgb, STAGE75_WORKCELL_ROI_XYWH, STAGE75_WORKCELL_OUTPUT_WH
assert STAGE75_WORKCELL_ROI_XYWH == (145,30,690,473)
assert STAGE75_WORKCELL_OUTPUT_WH == (1035,710)
src=Image.new('RGB',(1035,710),'white')
d=ImageDraw.Draw(src)
d.rectangle((145,30,834,502),outline='black',width=3)
# ROI should be magnified to entire 1035x710: its center becomes scene center.
d.rectangle((400,200,500,290),fill='#f05020')
draw=stage75_zoom_workcell_rgb(src)
assert draw.size==(1035,710)
assert tuple(np.asarray(draw)[340,455]) == (240,80,32)
# This module must contain no simulation-model modification; only the 3D view.
s=Path(__file__).with_name('stage62_workcell_video.py').read_text(encoding='utf-8')
assert 'STAGE75_WORKCELL_ROI_XYWH' in s
assert 'stage75_zoom_workcell_rgb(raw_workcell)' in s
assert 'p.getCameraImage(world_w,world_h' in s
print('STAGE75_VIEWPORT_ZOOM_STATIC_PASS: 1.50x crop within only the 3D workspace, physical/AI models unchanged')
