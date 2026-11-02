# coding: utf-8
"""Stage84 hotfix: labelled C synthetic scenario never crashes narrow monitoring HUD.
This test runs without PyBullet. Inference and robot control are not modified.
"""
import ast, pathlib
from PIL import Image,ImageFont
from stage62_workcell_video import draw_stage72_evidence
root=pathlib.Path(__file__).resolve().parent
src=(root/'stage84_camera_ai.py').read_text(encoding='utf8')
assert "info['reference_grade']='C'" in src
assert "info['reference_grade_context']='人工小傷" in src
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
assert pathlib.Path(font).exists(),'NotoSansCJK required by video HUD'
fonts={n:ImageFont.truetype(font,n) for n in (16,19,22)}
canvas=Image.new('RGB',(1920,1080),'#081c2d')
evidence={'dark_pct':{k:0.6 for k in ('finger_1','finger_2','finger_3','palm')},
          'positive':['finger_2'],'max_pct':9.5}
info={'actual_ai_grade':'C','reference_grade':'C',
      'reference_grade_context':'人工小傷を追加した開発試験（原本はB級）',
      'filename':'判定B-2.jpg','synthetic_small_scratch_demo':True}
draw_stage72_evidence(canvas,evidence,None,fonts,info)
# Other development/metadata strings may also be long; all must be nonfatal.
info['reference_grade']='C（人工小傷を追加した試験）'
draw_stage72_evidence(canvas,evidence,None,fonts,info)
print('STAGE84_HUD_LABEL_HOTFIX_PASS; C is class token, metadata is separate; long text never crashes')
