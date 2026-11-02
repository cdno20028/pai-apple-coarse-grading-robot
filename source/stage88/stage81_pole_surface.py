# -*- coding:utf-8 -*-
"""Stage81 visibly natural head/tail CAP (synthetic composite from six examples).
Original eight grade photos unchanged. Reference photos are unpaired examples;
this is a DEVELOPMENT-ONLY visual 3D composite, not an actual full apple scan.
"""
import hashlib,json, math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from stage68_surface import Stage68Surface

JST_NOTE='2026/10/23 JST'
CENTERS={
 'head-1.jpg':(250,238),'head-2.jpg':(239,204),'head-3.jpg':(239,228),
 'tail-1.jpg':(211,256),'tail-2.jpg':(262,230),'tail-3.jpg':(229,217)}
REFERENCE_RADIUS={'head-1.jpg':188,'head-2.jpg':181,'head-3.jpg':187,
                  'tail-1.jpg':171,'tail-2.jpg':184,'tail-3.jpg':188}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def compose_poles(base,head,tail,head_name,tail_name):
    """Equirectangular pole photo mapping, smooth fade to preserved side UV.
    Only frontal <=0.83 radians from each pole is synthetic-composited.
    It does not claim to recover unknown, real fruit skin on the far side.
    """
    src=np.asarray(base.convert('RGB'),dtype=np.uint8)
    if src.shape!=(512,1024,3):raise ValueError(src.shape)
    H,W=src.shape[:2]
    out=src.astype(np.float32).copy()
    destmeta={}
    cols=np.arange(W,dtype=np.float32)
    angles=2*np.pi*(cols+0.5)/W
    for label,img,name,top in (('head',head,head_name,True),('tail',tail,tail_name,False)):
        ref=np.asarray(img.convert('RGB'),dtype=np.uint8)
        cx,cy=CENTERS[name];radius=REFERENCE_RADIUS[name]
        # Center coordinate is aligned to observed stem/calyx; never assume centered fruit.
        yrows=np.arange(0,193) if top else np.arange(H-193,H)
        theta=(yrows+0.5)*np.pi/H if top else (H-0.5-yrows)*np.pi/H
        rr=np.sin(theta)*radius
        # Photo position is never artificially modified to "fix" classification.
        mapx=(cx+rr[:,None]*np.cos(angles)[None,:]).astype(np.float32)
        mapy=(cy+rr[:,None]*np.sin(angles)[None,:]).astype(np.float32)
        samp=cv2.remap(ref,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101).astype(np.float32)
        # Global luminance & chroma difference at fade ring reduced, photo detail retained.
        compare=np.mean(out[yrows[120:138]],axis=(0,1))
        photo_compare=np.mean(samp[120:138],axis=(0,1))
        shift=np.clip((compare-photo_compare)*.42,-24,24)
        samp=np.clip(samp+shift,0,255)
        alpha=np.ones_like(theta,dtype=np.float32)
        fade=theta>=.78
        alpha[fade]=np.clip((1.17-theta[fade])/(1.17-.78),0.,1.)
        alpha=alpha*alpha*(3-2*alpha)
        out[yrows]=out[yrows]*(1-alpha[:,None,None])+samp*alpha[:,None,None]
        destmeta[label]={'reference':name,'reference_sha256':None,'pole_center_px':[cx,cy],
                         'polar_influence_max_rad':1.17,'fully_photo_based_inner_rad':0.78,
                         'source_photo_used_as_texture_illustration':True}
    # tiny brown/sepals palette squares for connected visual mesh parts only.
    # This reserved texture region is not part of the original eight source photos.
    out[251:259,1008:1023]=[79,49,31]
    out[265:273,1008:1023]=[62,43,29]
    return Image.fromarray(np.uint8(np.clip(np.rint(out),0,255)),'RGB'),destmeta

class Stage81Surface(Stage68Surface):
    def __init__(self,assets,output,refs):
        super().__init__(assets,output)
        self.refs=Path(refs)
        self.out81=Path(output)/'stage81_head_tail_textures';self.out81.mkdir(parents=True,exist_ok=True)
        self.stage81_items={}
        base_manifest=json.loads((Path(assets)/'manifest.json').read_text(encoding='utf8'))
        for i,rec in enumerate(base_manifest['assets']):
            aid=rec['asset_id']
            # Grade-independent deterministic reference cycling; NOT an actual head/tail pair.
            hn=f'head-{i%3+1}.jpg';tn=f'tail-{(i+1)%3+1}.jpg'
            for n in (hn,tn):
                if not (self.refs/n).is_file():raise FileNotFoundError(self.refs/n)
            im,meta=compose_poles(Image.open(super().path(aid)),Image.open(self.refs/hn),Image.open(self.refs/tn),hn,tn)
            outfile=self.out81/(aid+'_head_tail_visual_composite.png')
            im.save(outfile,compress_level=6)
            recmeta={'asset_id':aid,'original_user_uv_sha256':sha(Path(assets)/rec['texture_path']),
                     'repaired_pole_baseline_sha256':sha(super().path(aid)),
                     'synthetic_head_reference':hn,'synthetic_head_sha256':sha(self.refs/hn),
                     'synthetic_tail_reference':tn,'synthetic_tail_sha256':sha(self.refs/tn),
                     'render_texture_sha256':sha(outfile),'texture_file':str(outfile.relative_to(Path(output))),
                     'source_photos_unchanged':True,'reference_grade_not_input_to_mapping':True,
                     'two_reference_photos_are_NOT_same_fruit':True,
                     'composite_is_AI_camera_visible_and_needs_grading_revalidation':True,**meta}
            self.stage81_items[aid]=recmeta
        record={'stage':81,'created_jst':JST_NOTE,'actual_three_dimensional_reconstruction':False,
                'photo_based_synthetic_cap_composite':True,'grades_not_derived_from_reference_selection':True,
                'not_orchard_validation':True,'items':list(self.stage81_items.values())}
        (Path(output)/'stage81_pole_visual_manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')

    def path(self,asset_id):
        return self.out81/(asset_id+'_head_tail_visual_composite.png')
