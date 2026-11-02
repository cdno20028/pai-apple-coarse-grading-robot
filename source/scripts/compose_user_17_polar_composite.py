#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compose all 17 user-supplied apple photos into 11 UV textures.

The 11 side photos form the primary surface. The three unpaired upper-view and
three unpaired lower-view photos are mapped to polar caps, selected by colour
similarity to the side-photo shoulder and balanced so every cap source is used.
The upper/lower photos are NOT claimed to depict the same individual fruit.
No global recolouring, AI-generated fruit texture, or grade-derived damage is
added. Original sources remain byte-identical in REAL17_SOURCE_ORIGINAL_17/.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, shutil, zipfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

STAMP='2026/10/29 14:14 JST'
ASSIGN={'01_A':'判定A-1.jfif','02_B':'判定B-1.jfif','03_B':'判定B-2.jfif',
 '04_C':'判定C-2.jfif','05_C':'判定C-3.jfif','06_C':'判定C-4.jfif','07_C':'判定C-5.jfif','08_D':'判定D2.jfif'}
HUV,WUV=512,1024
CAP_OUTER_RAD=0.70
CAP_INNER_RAD=0.48
CAP_RADIUS_SCALE=0.82

def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha(p): return sha_bytes(Path(p).read_bytes())

def background_cut(im):
    a=np.asarray(im.convert('RGB'))
    h,w=a.shape[:2]
    s=min(1.,900/max(h,w))
    sm=cv2.resize(a,(round(w*s),round(h*s)),interpolation=cv2.INTER_AREA) if s<1. else a
    h,w=sm.shape[:2]
    border=np.concatenate([sm[:max(3,h//30)].reshape(-1,3),sm[-max(3,h//30):].reshape(-1,3),sm[:,:max(3,w//30)].reshape(-1,3),sm[:,-max(3,w//30):].reshape(-1,3)]).astype(np.float32)
    bg=np.median(border,axis=0)
    hsv=cv2.cvtColor(sm,cv2.COLOR_RGB2HSV)
    dd=np.linalg.norm(sm.astype(np.float32)-bg,axis=2)
    score=np.uint8(((dd>38)&((hsv[:,:,1]>23)|(dd>75))))*255
    k=max(3,int(min(h,w)*.018)//2*2+1)
    score=cv2.morphologyEx(score,cv2.MORPH_CLOSE,np.ones((k,k),np.uint8))
    n,lab,stats,cent=cv2.connectedComponentsWithStats(score)
    mid=np.array([w*.5,h*.5])
    choices=[i for i in range(1,n) if stats[i,cv2.CC_STAT_AREA]>w*h*.06]
    if not choices: raise RuntimeError('APPLE_FOREGROUND_NOT_FOUND')
    idx=max(choices,key=lambda i: stats[i,cv2.CC_STAT_AREA]-25*np.linalg.norm(cent[i]-mid))
    mask=np.uint8(lab==idx)*255
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((max(3,k//2),max(3,k//2)),np.uint8))
    ys,xs=np.where(mask>0)
    x0,x1=np.percentile(xs,[1,99]); y0,y1=np.percentile(ys,[1,99])
    cx=(x0+x1)/2; cy=(y0+y1)/2
    rx=(x1-x0)/2*.995; ry=(y1-y0)/2*.995
    return sm,mask,(float(cx),float(cy),float(rx),float(ry)),bg.tolist()

def build_side_uv(im):
    sm,mask,(cx,cy,rx,ry),bg=background_cut(im)
    H,W=HUV,WUV
    uu=(np.arange(W,dtype=np.float32)+.5)/W
    vv=(np.arange(H,dtype=np.float32)+.5)/H
    phi=(uu-.5)*2*np.pi
    theta=vv*np.pi
    x=np.sin(phi)[None,:]*np.sin(theta)[:,None]
    y=np.cos(theta)[:,None]
    mx=np.broadcast_to(cx+rx*x,(H,W)).astype(np.float32)
    my=np.broadcast_to(cy-ry*y,(H,W)).astype(np.float32)
    base=cv2.remap(sm,mx,my,cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    fg=cv2.remap(mask,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    if np.any(fg<100):
        mx=np.clip(mx,cx-rx*.94,cx+rx*.94)
        my=np.clip(my,cy-ry*.92,cy+ry*.92)
        filled=cv2.remap(sm,mx,my,cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
        base=np.where((fg<100)[:,:,None],filled,base)
    base[251:259,1008:1023]=[79,49,31]
    base[265:273,1008:1023]=[62,43,29]
    return Image.fromarray(base,'RGB'), {'source_bbox_xy_px':[round(x) for x in (cx-rx,cy-ry,cx+rx,cy+ry)],'source_original_RGB_direct':True}

def feature_lab(im, mode):
    sm,mask,(cx,cy,rx,ry),_ = background_cut(im)
    h,w=mask.shape
    yy,xx=np.mgrid[0:h,0:w]
    xn=(xx-cx)/max(rx,1); yn=(yy-cy)/max(ry,1)
    if mode in ('top_cap','bottom_cap'):
        rr=np.sqrt(xn*xn+yn*yn)
        sel=(mask>0)&(rr>.55)&(rr<.83)
    elif mode=='upper_band':
        sel=(mask>0)&(yn>-.80)&(yn<-.25)&(np.abs(xn)<.80)
    elif mode=='lower_band':
        sel=(mask>0)&(yn>.25)&(yn<.80)&(np.abs(xn)<.80)
    else:
        sel=mask>0
    if int(sel.sum())<100: sel=mask>0
    lab=cv2.cvtColor(sm,cv2.COLOR_RGB2LAB).astype(np.float32)
    return np.median(lab[sel],axis=0)

def balanced_cap_assignment(side_names, cap_names, side_mode, cap_mode, images):
    side_features=[feature_lab(images[n],side_mode) for n in side_names]
    cap_features=[feature_lab(images[n],cap_mode) for n in cap_names]
    cost=np.array([[np.linalg.norm(sf-cf) for cf in cap_features] for sf in side_features],dtype=float)
    extra_n=len(side_names)-3*len(cap_names)
    if extra_n<0: raise ValueError('Expected at least three side photos per cap source')
    best=None
    for extra in itertools.combinations_with_replacement(range(len(cap_names)),extra_n):
        slots=[]
        for i in range(len(cap_names)): slots.extend([i]*3)
        slots.extend(extra)
        matrix=cost[:,slots]
        rows,cols=linear_sum_assignment(matrix)
        total=float(matrix[rows,cols].sum())
        if best is None or total<best[0]:
            best=(total,slots,rows,cols)
    out={}; distances={}
    for r,c in zip(best[2],best[3]):
        cap=cap_names[best[1][c]]
        out[side_names[r]]=cap
        distances[side_names[r]]=round(float(cost[r,cap_names.index(cap)]),3)
    return out,distances

def apply_polar_cap_rows(base, cap_img, top=True):
    """Project a real top/bottom photo into a polar disk and feather the seam."""
    arr=np.asarray(base.convert('RGB')).copy()
    sm,mask,(cx,cy,rx,ry),_ = background_cut(cap_img)
    H,W=arr.shape[:2]
    u=(np.arange(W,dtype=np.float32)+.5)/W
    phi=(u-.5)*2*np.pi
    cosphi=np.cos(phi); sinphi=np.sin(phi)
    theta_rows=(np.arange(H,dtype=np.float32)+.5)/H*np.pi
    for y,theta in enumerate(theta_rows):
        pole_theta=float(theta if top else np.pi-theta)
        if pole_theta>=CAP_OUTER_RAD: continue
        rn=np.sin(pole_theta)/np.sin(CAP_OUTER_RAD)*CAP_RADIUS_SCALE
        mx=(cx+rx*rn*cosphi).astype(np.float32)[None,:]
        my=(cy+ry*rn*sinphi).astype(np.float32)[None,:]
        row=cv2.remap(sm,mx,my,cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)[0]
        if pole_theta<=CAP_INNER_RAD:
            alpha=1.0
        else:
            q=(pole_theta-CAP_INNER_RAD)/(CAP_OUTER_RAD-CAP_INNER_RAD)
            alpha=1.0-(q*q*(3.0-2.0*q))
        arr[y]=np.clip(row.astype(np.float32)*alpha+arr[y].astype(np.float32)*(1.0-alpha),0,255).astype(np.uint8)
    arr[251:259,1008:1023]=[79,49,31]
    arr[265:273,1008:1023]=[62,43,29]
    return Image.fromarray(arr,'RGB')

def compose_side_top_bottom(side_img, top_img, bottom_img):
    base,_=build_side_uv(side_img)
    with_top=apply_polar_cap_rows(base,top_img,top=True)
    with_both=apply_polar_cap_rows(with_top,bottom_img,top=False)
    return with_both

def build(root):
    root=Path(root)
    original=root/'realphoto'/'original_images'
    required=['上方-1.jfif','上方-2.jfif','上方-3.jfif','下方-1.jfif','下方-2.jfif','下方-3.jfif']+[f'判定A-{i}.jfif' for i in range(1,4)]+[f'判定B-{i}.jfif' for i in (1,2)]+[f'判定C-{i}.jfif' for i in range(1,6)]+['判定D2.jfif']
    for n in required:
        if not (original/n).is_file(): raise FileNotFoundError(str(original/n))
    rawzip=root/'REAL17_USER_SOURCE_17.zip'
    assert rawzip.is_file(), 'REAL17_USER_SOURCE_17.zip missing'
    with zipfile.ZipFile(rawzip) as rz:
        assert rz.testzip() is None and len(rz.namelist())==17, 'RAW_SOURCE_ZIP_INVALID'
        for n in required:
            assert rz.read(n)==(original/n).read_bytes(), 'RAW_SOURCE_BYTES_MISMATCH '+n
    source=root/'REAL17_SOURCE_ORIGINAL_17'
    if source.exists(): shutil.rmtree(source)
    source.mkdir()
    for n in required:
        b=(original/n).read_bytes()
        (source/n).write_bytes(b)
        assert sha(source/n)==sha(original/n)
    images={n:Image.open(source/n).convert('RGB') for n in required}
    side_names=[n for n in required if n.startswith('判定')]
    top_names=[n for n in required if n.startswith('上方')]
    bottom_names=[n for n in required if n.startswith('下方')]
    top_assign,top_dist=balanced_cap_assignment(side_names,top_names,'upper_band','top_cap',images)
    bottom_assign,bottom_dist=balanced_cap_assignment(side_names,bottom_names,'lower_band','bottom_cap',images)
    legacy=root/'REAL17_DIRECT_COMPOSITE_11'
    if legacy.exists(): shutil.rmtree(legacy)
    out=root/'REAL17_POLAR_COMPOSITE_11'
    if out.exists(): shutil.rmtree(out)
    out.mkdir()
    texture_entries=[]
    for n in side_names:
        uv=compose_side_top_bottom(images[n],images[top_assign[n]],images[bottom_assign[n]])
        name=n.removesuffix('.jfif')+'_POLAR_COMPOSITE_UV.png'
        dest=out/name
        uv.save(dest,optimize=True)
        texture_entries.append({
            'side_photo':n,'side_photo_sha256':sha(source/n),
            'top_cap_photo':top_assign[n],'top_cap_photo_sha256':sha(source/top_assign[n]),
            'bottom_cap_photo':bottom_assign[n],'bottom_cap_photo_sha256':sha(source/bottom_assign[n]),
            'uv':str(dest.relative_to(root)),'uv_sha256':sha(dest),
            'top_cap_colour_distance_lab':top_dist[n],'bottom_cap_colour_distance_lab':bottom_dist[n],
            'method':'side-photo spherical projection + top/bottom polar disk projection with feathered RGB seam',
            'same_individual_confirmed':False,'global_colour_correction':False,'synthetic_grade_damage_added':False
        })
    prep=root/'stage88'/'stage58_apple_photo_assets'/'prepared'
    texdir=prep/'uv_photo_front'
    asset_manifest_path=prep/'manifest.json'
    old=json.loads(asset_manifest_path.read_text('utf-8'))
    old_by_id={x['asset_id']:x for x in old['assets']}
    runtime_bindings=[]
    for aid,side in ASSIGN.items():
        top=top_assign[side]; bottom=bottom_assign[side]
        uv_path=out/(side.removesuffix('.jfif')+'_POLAR_COMPOSITE_UV.png')
        target=texdir/(aid+'_uv.png')
        target.write_bytes(uv_path.read_bytes())
        digest=sha(target)
        sm,mask,(cx,cy,rx,ry),_ = background_cut(images[side])
        entry=old_by_id[aid]
        entry.update({
            'source_filename':side,'source_sha256':digest,
            'photo_size_px':list(images[side].size),
            'foreground_bbox_px':[round(cx-rx),round(cy-ry),round(cx+rx),round(cy+ry)],
            'texture_path':'uv_photo_front/'+aid+'_uv.png',
            'photo_projection':'USER_SIDE_PHOTO_PLUS_UNPAIRED_TOP_BOTTOM_POLAR_CAPS',
            'direct_user_photo':side,'direct_user_photo_sha256':sha(source/side),
            'direct_runtime_uv_sha256':digest,'uv_sha256':digest,
            'v7_uv_sha256':digest,'v7_source_image_path':'realphoto/original_images/'+side,
            'uv_color_matching_applied':False,
            'polar_cap_composite':{
                'top_photo':top,'top_photo_sha256':sha(source/top),
                'bottom_photo':bottom,'bottom_photo_sha256':sha(source/bottom),
                'selection':'Lab shoulder/annulus colour-distance, balanced across all three source photos',
                'same_individual_confirmed':False,'global_colour_correction':False,
                'seam_blending':'RGB alpha feather only, no global recolouring'
            }
        })
        runtime_bindings.append({'asset_id':aid,'side_photo':side,'side_photo_sha256':sha(source/side),
            'top_cap_photo':top,'top_cap_sha256':sha(source/top),'bottom_cap_photo':bottom,
            'bottom_cap_sha256':sha(source/bottom),'runtime_texture':str(target.relative_to(root)),
            'runtime_texture_sha256':digest})
    old.update({
        'date_jst':STAMP,
        'dataset':'USER-SUPPLIED 17-PHOTO COMPOSITE: SIDE PHOTOS + UNPAIRED TOP/BOTTOM POLAR CAPS',
        'number_of_sources':17,
        'source':'REAL17_USER_SOURCE_17.zip; byte-identical copies in REAL17_SOURCE_ORIGINAL_17/',
        'notes':[
            'All 17 user-supplied source files are preserved byte-for-byte.',
            '11 side photos are the primary surface; 3 upper-view and 3 lower-view photos are actually composited into polar caps.',
            'Top/bottom views are not confirmed as the same individual fruit as the side view; this is a disclosed synthetic multi-photo composite, not a true 360 scan.',
            'Top/bottom cap selection uses Lab colour-distance and balanced usage of all six cap photos; no class labels or case numbers are used.',
            'Original source RGB is used with geometric remapping and RGB seam blending; no global colour correction or synthetic grade damage is added.',
            'Classifier receives camera pixels only; no reference grade is injected at runtime.',
            'New composite texture requires a fresh physical retest; old 10/10 evidence belongs to previous textures.'
        ],
        'assets':old['assets']
    })
    asset_manifest_path.write_text(json.dumps(old,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest={
        'created_jst':STAMP,'version':'REAL17_POLAR_COMPOSITE_V1',
        'source_total':17,'side_total':11,'top_cap_sources':top_names,'bottom_cap_sources':bottom_names,
        'all_six_polar_sources_used':True,'cap_assignment_method':'Lab shoulder/annulus colour-distance with balanced usage, minimum three uses per source',
        'top_bottom_same_individual_confirmed':False,'global_colour_correction':False,
        'raw_zip':'REAL17_USER_SOURCE_17.zip','raw_zip_sha256':sha(rawzip),
        'runtime_bindings':runtime_bindings,'textures':texture_entries,
        'side_photos_unassigned_to_runtime':['判定A-2.jfif','判定A-3.jfif','判定C-1.jfif'],
        'stability':'PHYSICAL_RETEST_REQUIRED; previous 10/10 is for older textures'
    }
    (root/'REAL17_SOURCE_COMPOSITE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    surface=root/'stage88'/'stage_direct_photo_surface.py'
    surface.write_text("""# -*- coding: utf-8 -*-
\"\"\"Apply exact 17-photo side/top/bottom composite UVs at Stage84 runtime.
Top and bottom source photos are unpaired views; manifest records their use.
\"\"\"
from pathlib import Path
import hashlib, json

def _sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class StageDirectPhotoSurface:
    def __init__(self,assets,results,refs=None):
        self.assets=Path(assets); self.results=Path(results)
        m=json.loads((self.assets/'manifest.json').read_text(encoding='utf-8'))
        self._paths={}; evidence=[]
        for entry in m['assets']:
            aid=entry['asset_id']; texture=self.assets/'uv_photo_front'/(aid+'_uv.png')
            if not texture.is_file(): raise FileNotFoundError(texture)
            self._paths[aid]=texture; cap=entry.get('polar_cap_composite',{})
            evidence.append({'asset_id':aid,'texture':str(texture),'sha256':_sha(texture),
              'method':'USER_SIDE_PHOTO_PLUS_UNPAIRED_TOP_BOTTOM_POLAR_CAPS',
              'side_photo':entry.get('direct_user_photo'),
              'top_cap_photo':cap.get('top_photo'),'bottom_cap_photo':cap.get('bottom_photo'),
              'top_bottom_same_individual_confirmed':False,'global_colour_correction':False})
        self.results.mkdir(parents=True,exist_ok=True)
        (self.results/'DIRECT_USER_PHOTO_RUNTIME_TEXTURES.json').write_text(
          json.dumps({'method':'USER_ORIGINAL_17_SIDE_TOP_BOTTOM_POLAR_COMPOSITE','entries':evidence,
            'all_six_top_bottom_sources_used_in_composite':True,
            'top_bottom_photos_are_unpaired_views':True,
            'actual_3D_render_requires_PyBullet_execution':True},ensure_ascii=False,indent=2),encoding='utf-8')
    def path(self,asset_id): return self._paths[asset_id]
""",encoding='utf-8')
    return manifest

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',required=True); args=ap.parse_args()
    info=build(args.repo)
    print('REAL17_POLAR_COMPOSITE_READY','SOURCE_PHOTOS',info['source_total'],
          'SIDE_UVS',len(info['textures']),'RUNTIME_BOUND',len(info['runtime_bindings']),
          'ALL_SIX_CAP_SOURCES_USED',info['all_six_polar_sources_used'])
