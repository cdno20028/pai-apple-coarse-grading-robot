# -*- coding: utf-8 -*-
"""Stage83: photograph-driven NATURAL apple pole texture, with per-fruit colour matching.

Source photos: 8 fruit photos and 6 unpaired NORMAL head/tail examples.
No pairing, 360-degree reconstruction, real damage learning or field validation claimed.
Use reference high-resolution morphology and detail, but map reference skin hue to
that particular fruit's original mid-body skin via LAB colour adaptation.
Unlike Stage82, visibly retains anatomy and photographic texture in a 0.60rad
polar cap; unlike Stage81, does not import an unrelated RED skin into a GREEN fruit.
No orchard labels or reference grades are read during mapping.
"""
from pathlib import Path
from PIL import Image
import hashlib, json, math
import numpy as np, cv2
from stage68_surface import Stage68Surface
from stage81_pole_surface import CENTERS, REFERENCE_RADIUS

CAP_FULL_RAD=0.31
CAP_END_RAD=0.60  # about 34.4 degrees; NOT the former stage81 67 degrees

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def smoothstep(x):
    x=np.clip(x,0.0,1.0)
    return x*x*(3.0-2.0*x)

def compose_color_matched_poles(original,head,tail,head_name,tail_name):
    """Photo texture for pole anatomy with normalized, individual-fruit skin tone.

    The stem/calyx brown darkness is retained; the surrounding fruit skin is
    LAB-chroma-matched to genuine apple colour extracted from that same base UV.
    Outside +/-0.60rad, the Stage68 source UV is pixel-identical.
    """
    base=np.asarray(original.convert('RGB'),np.uint8)
    H,W=base.shape[:2]
    if (H,W)!=(512,1024): raise ValueError('Unsupported equirect UV '+str(base.shape))
    result=base.astype(np.float32).copy()
    # Note: y=275..318 holds photo-derived source skin after Stage68.
    # Robust skin colour target is based on this same fruit, NEVER on grade label.
    skin=cv2.cvtColor(base[276:320],cv2.COLOR_RGB2LAB).astype(np.float32)
    target_skin=np.median(skin.reshape(-1,3),axis=0)
    angles=2*np.pi*(np.arange(W,dtype=np.float32)+.5)/W
    meta={}
    for end,reference,name,top in [('head',head,head_name,True),('tail',tail,tail_name,False)]:
        ref=np.asarray(reference.convert('RGB'),np.uint8)
        cx,cy=CENTERS.get(name,(256,256));rad=REFERENCE_RADIUS.get(name,208)
        nrows=int(np.ceil(CAP_END_RAD*H/math.pi))+2
        rows=np.arange(nrows,dtype=np.int32) if top else np.arange(H-nrows,H,dtype=np.int32)
        theta=((rows+.5) if top else (H-rows-.5))*math.pi/H
        r=(np.sin(theta)*rad).astype(np.float32)
        mx=(cx+r[:,None]*np.cos(angles)[None,:]).astype(np.float32)
        my=(cy+r[:,None]*np.sin(angles)[None,:]).astype(np.float32)
        photo=cv2.remap(ref,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        photo_lab=cv2.cvtColor(photo,cv2.COLOR_RGB2LAB).astype(np.float32)
        # Skin ring excludes central normal anatomical opening.
        ring=(theta>.37)&(theta<.51)
        if not np.any(ring): raise RuntimeError('reference skin ring absent')
        ref_skin=np.median(photo_lab[ring].reshape(-1,3),axis=0)
        adjusted=photo_lab+(target_skin-ref_skin).reshape(1,1,3)
        # Keep genuine low-luminance stem/calyx anatomy instead of flat filling.
        # Global LAB translation can lighten dark openings; constrain L only.
        center=(theta<.17)
        adjusted[center,:,0]=np.minimum(adjusted[center,:,0],photo_lab[center,:,0]+10.0)
        # Preserve the identity of the individual skin hue, not the reference hue:
        # the reference's local (a,b) colour streak variation is moderated.
        # The dark central opening itself is kept neutral brown via a small core.
        adjusted[:,:,1:3]=target_skin[1:3].reshape(1,1,2)+.55*(photo_lab[:,:,1:3]-ref_skin[1:3].reshape(1,1,2))
        core_strength=np.clip((.105-theta)/.055,0,1).astype(np.float32)
        adjusted[:,:,1:3]=(1-core_strength[:,None,None])*adjusted[:,:,1:3]+core_strength[:,None,None]*np.minimum(photo_lab[:,:,1:3]+(target_skin[1:3]-ref_skin[1:3]).reshape(1,1,2)*.20,240)
        adjusted=np.uint8(np.clip(np.rint(adjusted),0,255))
        recolored=cv2.cvtColor(adjusted,cv2.COLOR_LAB2RGB).astype(np.float32)
        alpha=np.ones_like(theta,dtype=np.float32)
        outer=theta>CAP_FULL_RAD
        alpha[outer]=smoothstep((CAP_END_RAD-theta[outer])/(CAP_END_RAD-CAP_FULL_RAD))
        alpha[theta>=CAP_END_RAD]=0.0
        result[rows]=result[rows]*(1-alpha[:,None,None])+recolored*alpha[:,None,None]
        meta[end]={'reference_photo':name,'reference_sha256':None,
                   'unpaired_example_not_same_fruit':True,
                   'reference_detail_and_anatomy_used':True,
                   'skin_tone_source':'ORIGINAL_INDIVIDUAL_FRUIT_UV_MIDBODY',
                   'reference_colour_transplanted_without_matching':False,
                   'mapped_full_deg':round(math.degrees(CAP_FULL_RAD),2),
                   'mapped_fade_to_deg':round(math.degrees(CAP_END_RAD),2),
                   'original_apple_midbody_lab':np.rint(target_skin).astype(int).tolist(),
                   'reference_skin_lab':np.rint(ref_skin).astype(int).tolist()}
    new=np.uint8(np.clip(np.rint(result),0,255))
    # 3D geometry uses reserved palette swatches; not user photographic regions.
    new[251:259,1008:1023]=[79,49,31]
    new[265:273,1008:1023]=[62,43,29]
    keep=int(math.ceil(CAP_END_RAD*H/math.pi))+2
    assert np.array_equal(base[keep:H-keep,:1007],new[keep:H-keep,:1007]),'MIDBODY_TEXTURE_CHANGED'
    return Image.fromarray(new,'RGB'),meta

class Stage83Surface(Stage68Surface):
    def __init__(self,assets,output,refs):
        super().__init__(assets,output)
        self.refs=Path(refs);self.out83=Path(output)/'stage83_photo_normalized_poles';self.out83.mkdir(parents=True,exist_ok=True)
        dat=json.loads((Path(assets)/'manifest.json').read_text(encoding='utf8'))
        self.items=[]
        for i,entry in enumerate(dat['assets']):
            aid=entry['asset_id']
            hn=f'head-{aid}.jpg' if (self.refs/f'head-{aid}.jpg').exists() else f'head-{i%3+1}.jpg'
            tn=f'tail-{aid}.jpg' if (self.refs/f'tail-{aid}.jpg').exists() else f'tail-{(i+1)%3+1}.jpg'
            for n in (hn,tn):
                if not (self.refs/n).is_file():raise FileNotFoundError(self.refs/n)
            orig=Path(super().path(aid))
            composite,detail=compose_color_matched_poles(Image.open(orig),Image.open(self.refs/hn),Image.open(self.refs/tn),hn,tn)
            dest=self.path(aid);composite.save(dest,compress_level=6)
            self.items.append({'asset_id':aid,'original_fruit_sha256':entry['source_sha256'],
                 'stage68_skin_sha256':sha(orig),'stage83_visual_sha256':sha(dest),
                 'reference_head_sha256':sha(self.refs/hn),'reference_tail_sha256':sha(self.refs/tn),
                 'head_tail_geometry_stage81':True,'grade_is_not_input_to_texture':True,'detail':detail})
        (Path(output)/'stage83_pole_photo_visual_manifest.json').write_text(json.dumps({
             'stage':83,'note_jst':'2026/10/23 JST','source_photos_untouched':True,
             'normal_head_tail_examples_are_not_same_fruit_as_graded_apple':True,
             'NOT_A_REAL_3D_SCAN':True,'collision_shape_remains_sphere':True,
             'grade_model_retrained':False,'orchard_accuracy_verified':False,
             'items':self.items},ensure_ascii=False,indent=2),encoding='utf8')
    def path(self,asset_id):return self.out83/(asset_id+'_PHOTO_HUE_MATCHED_head_tail.png')
