#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild GATE8 C-grade public synthetic assets from the included generated source image.

This updates appearance assets only. It does not change AI thresholds, safety logic,
robot control, case order, bin-full behavior, human-recovery logic, or physics settings.
"""
from pathlib import Path
import json, hashlib, math
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'public_ai_apple_grade/sources/C_grade_one_larger_scar_reference.png'
SIDE=ROOT/'public_ai_apple_grade/multiview/C_side_AI.png'
EXAMPLE=ROOT/'public_ai_apple_grade/grade_C_AI_example.png'
NOW='2026/10/27 08:04 JST'  # release revision timestamp; outputs are deterministic
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

# Keep the approved 1254x1254 source intact. A normalized 512x400 side view is
# generated for the existing rendering path; no text is overlaid on model input.
source=Image.open(SOURCE).convert('RGB')
assert source.size==(1254,1254), source.size
source.resize((512,400),Image.Resampling.LANCZOS).save(SIDE,optimize=True)
source.save(EXAMPLE,optimize=True)
new_img=Image.open(SIDE).convert('RGB')
new=np.asarray(new_img,dtype=np.uint8)

# Same projection convention used by GATE7; only C-labelled UV inputs are changed.
assets=ROOT/'stage88/stage58_apple_photo_assets/prepared'
manifest_path=assets/'manifest.json'
manifest=json.loads(manifest_path.read_text('utf8'))
Y,X=np.meshgrid(np.arange(512,dtype=np.float32),np.arange(1024,dtype=np.float32),indexing='ij')
phi=(X+.5)/1024*2*math.pi-math.pi
lat=(Y+.5)/512*math.pi-math.pi/2
front=255+215*np.sin(phi)*np.cos(lat)
back=255+65*np.sin(phi)
b=np.clip((math.pi*.52-np.abs(phi))/(math.pi*.07),0,1); b=b*b*(3-2*b)
uvx=(front*b+back*(1-b)).astype(np.float32)
uvy=(200+165*np.sin(lat)).astype(np.float32)
changed=[]
for i,item in enumerate(manifest['assets']):
    if item['user_label']!='C': continue
    aid=item['asset_id']
    arr=cv2.remap(new,uvx,uvy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    arr=np.uint8(np.clip(arr.astype(np.int16)+(i%3-1)*3,0,255))
    # Retain the existing cap/seam cues used by the GATE7 asset convention.
    arr[251:259,1008:1023]=[79,49,31]
    arr[265:273,1008:1023]=[62,43,29]
    target=assets/'uv_photo_front'/f'{aid}_uv.png'
    Image.fromarray(arr).save(target,optimize=True)
    item['source_sha256']=sha(target)
    item['generator_source_view_sha256']=sha(SIDE)
    changed.append(target.relative_to(ROOT).as_posix())
assert len(changed)==4, changed
manifest['grade_C_gate8_revision_jst']=NOW
manifest['grade_C_gate8_revision']='One moderately larger irregular scar plus multiple small surface blemishes; source is synthetic generated art; worker A still removes grossly obvious damage.'
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

# The current synthetic visual grader trains from stage88/textures/apple_*.png.
# Keep that training input aligned with the same C UV appearance.
texture=ROOT/'stage88/textures/apple_C.png'
Image.open(assets/'uv_photo_front/04_C_uv.png').convert('RGB').resize((512,256),Image.Resampling.LANCZOS).save(texture,optimize=True)

# Regenerate the multiview review sheet with the release date, keeping A/B/D/head/tail unchanged.
V=ROOT/'public_ai_apple_grade/multiview'
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if not Path(font).exists(): font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
F=ImageFont.truetype(font,25); S=ImageFont.truetype(font,18)
board=Image.new('RGB',(1510,1240),'#f7f9fb'); d=ImageDraw.Draw(board)
d.text((25,14),'Physical AI / A-D 等級の公開用画像（上・横・下）',font=F,fill='#223347')
d.text((25,50),NOW+'  |  C級：小傷複数＋やや大きい傷1か所 / 合成素材',font=S,fill='#4a5f71')
for i,g in enumerate('ABCD'):
    x0=32+i*370
    d.rectangle((x0,100,x0+343,144),fill=['#e9effa','#daf0e1','#e9e2fa','#f8edd7'][i])
    d.text((x0+12,109),f'{g}  '+{'A':'無傷','B':'浅い小傷1か所','C':'小傷複数＋傷1か所','D':'未熟色'}[g],font=S,fill='#263950')
    for j,view in enumerate(('head','side','tail')):
        im=ImageOps.fit(Image.open(V/f'{g}_{view}_AI.png').convert('RGB'),(340,330),method=Image.Resampling.LANCZOS)
        y0=156+j*339; board.paste(im,(x0,y0))
        d.text((x0+12,y0+294),['上（果梗側）','横（側面）','下（がく側）'][j],font=S,fill='#fff',stroke_width=2,stroke_fill='#192633')
d.text((25,1195),'AI生成合成画像。実りんごの選果精度や今回の物理実行成功を証明するものではない。',font=S,fill='#583d34')
board_path=ROOT/'public_ai_apple_grade/PAI_A_D_12VIEW_REVIEW.png'
board.save(board_path,optimize=True)

# Before/after visual audit image, with date.
old=Image.open(ROOT/'public_ai_apple_grade/multiview/C_side_AI.png').convert('RGB')
# The previous image is in the GATE7 zip; this is a separately stored source path if available.
previous=ROOT/'audit/GATE7_C_side_before.png'
if previous.exists(): old=Image.open(previous).convert('RGB')
comp=Image.new('RGB',(1170,535),'#f8fafc'); dc=ImageDraw.Draw(comp)
dc.text((20,12),f'GATE8 C級外観変更（{NOW}）',font=F,fill='#23394d')
dc.text((20,54),'小傷を複数残し、1か所だけ少し大きい傷を追加。明らかな大傷を除外する作業員Aの工程は維持。',font=S,fill='#465b70')
for idx,(label,im) in enumerate([('GATE7：小傷2か所',old),('GATE8：小傷複数＋やや大きい傷1か所',new)]):
    x0=20+idx*570; dc.text((x0,91),label,font=S,fill='#264057')
    comp.paste(ImageOps.fit(Image.fromarray(im) if isinstance(im,np.ndarray) else im,(512,400),method=Image.Resampling.LANCZOS),(x0,123))
comp.save(ROOT/'public_ai_apple_grade/PAI_GATE8_C_APPEARANCE_REVIEW.png',optimize=True)

# Update integrity lists for the active public image/model assets.
active_path=ROOT/'public_ai_apple_grade/active_asset_hashes.json'
active=json.loads(active_path.read_text('utf8'))
entries={e['file']:e for e in active['files']}
for rel in [
 'public_ai_apple_grade/grade_C_AI_example.png',
 'public_ai_apple_grade/multiview/C_side_AI.png',
 'public_ai_apple_grade/PAI_A_D_12VIEW_REVIEW.png',
 'public_ai_apple_grade/PAI_GATE8_C_APPEARANCE_REVIEW.png',
 'public_ai_apple_grade/sources/C_grade_one_larger_scar_reference.png',
 'stage88/textures/apple_C.png',
 *changed]:
    p=ROOT/rel
    entries[rel]={'file':rel,'sha256':sha(p)}
for rel,e in entries.items():
    p=ROOT/rel
    if p.exists(): e['sha256']=sha(p)
active['files']=sorted(entries.values(),key=lambda e:e['file'])
active['revision']='GATE8_C_ONE_LARGER_SCAR_PLUS_MULTIPLE_MINOR_BLEMISHES'
active['generated_jst']=NOW
active_path.write_text(json.dumps(active,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

prov_path=ROOT/'public_ai_apple_grade/asset_provenance.json'
prov=json.loads(prov_path.read_text('utf8'))
for e in prov.get('multiview_12',[]): e['sha256']=sha(ROOT/e['path'])
for e in prov.get('physics_asset_mapping',[]):
    uv=assets/'uv_photo_front'/f"{e['id']}_uv.png"
    if uv.exists(): e['final_uv_sha256']=sha(uv)
prov.update({
 'GATE8_revision_jst':NOW,
 'GATE8_C_source_image':'public_ai_apple_grade/sources/C_grade_one_larger_scar_reference.png',
 'GATE8_C_source_sha256':sha(SOURCE),
 'GATE8_C_grade_example_sha256':sha(EXAMPLE),
 'GATE8_C_side_view_sha256':sha(SIDE),
 'GATE8_C_active_uvs':{p:sha(ROOT/p) for p in changed},
 'GATE8_C_training_texture_sha256':sha(texture),
 'GATE8_C_visual_description':'multiple small blemishes plus one moderately larger irregular scar',
 'GATE8_C_generated_art_not_real_orchard_evidence':True,
 'GATE8_runtime_physics_verified':False,
 'GATE8_control_code_changed':False,
})
prov_path.write_text(json.dumps(prov,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('GATE8_ASSETS_REBUILT',json.dumps({'side':str(SIDE),'uvs':changed,'texture':str(texture),'source_sha256':sha(SOURCE)},ensure_ascii=False))
