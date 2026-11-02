# -*- coding: utf-8 -*-
"""Stage68 development visualization: remove artificial UV top-pole split.

Never edits the user photographs nor the source UV. Blends ONLY the synthesized
polar cap of an already synthetic, single-photo-derived 3D UV texture. This is
not evidence of a genuine underside, wound-free fruit, or orchard accuracy.
"""
import hashlib, json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def repair_top_pole(original):
    rgb=np.asarray(original.convert('RGB'),dtype=np.uint8)
    h,w=rgb.shape[:2]
    if (w,h)!=(1024,512):
        raise ValueError(f'Expected synthetic UV equirectangular 1024x512, got {w}x{h}')
    # Synthesized TOP pole only. The front skin below v=0.49 is BYTE-IDENTICAL
    # to the source UV; the supplied real photograph is never changed.
    # The donor is a gently defocused skin-colour signal from a region BELOW
    # the pole and ABOVE the valid lower front body, reused across all grades.
    donor=cv2.GaussianBlur(rgb[270:327].astype(np.float32),(0,0),sigmaX=22,sigmaY=12)
    palette=np.mean(donor,axis=0)  # natural lateral colour variation across original texture
    palette=cv2.GaussianBlur(palette[None,:,:],(0,0),sigmaX=19,sigmaY=0)[0]
    target=rgb.astype(np.float32).copy()
    top_cut=173
    finish=249
    for y in range(finish):
        # full replacement of synthetic pole up to 173px, smooth crossfade below
        if y<=top_cut:
            alpha=1.0
        else:
            u=(finish-y)/(finish-top_cut)
            alpha=u*u*(3-2*u)
        target[y,:,:]=(1-alpha)*target[y,:,:]+alpha*palette
    target=np.uint8(np.clip(np.rint(target),0,255))
    # Original UV and ALL real-photo-derived body pixels y>=249 untouched.
    assert np.array_equal(target[finish:],rgb[finish:])
    return Image.fromarray(target,'RGB'),{'top_cap_replaced_rows':top_cut+1,
             'crossfade_rows':[top_cut+1,finish-1],
             'untouched_uv_rows':[finish,h-1],
             'supplied_photo_changed':False,
             'is_real_full_surface_scan':False,
             'draws_grade_from_label':False,
             'synthetic_top_pole_replacement_not_wound_removal':True}


class Stage68Surface:
    def __init__(self,assets,output):
        self.assets=Path(assets); self.output=Path(output)
        self.out_tex=self.output/'stage68_uv_review';self.out_tex.mkdir(parents=True,exist_ok=True)
        self.entries={}
        manifest=json.loads((self.assets/'manifest.json').read_text(encoding='utf-8'))
        for item in manifest['assets']:
            p=self.assets/item['texture_path']
            new,record=repair_top_pole(Image.open(p))
            dest=self.out_tex/(item['asset_id']+'_synthetic_pole_repair.png')
            new.save(dest,compress_level=6)
            self.entries[item['asset_id']]={'source':str(p.relative_to(self.assets)),
                 'source_sha256':digest(p),'output':str(dest.relative_to(self.output)),
                 'output_sha256':digest(dest),**record}
        (self.output/'stage68_uv_repair_manifest.json').write_text(
            json.dumps({'stage':68,'applies_to_all_8_independent_of_reference_grade':True,
             'source_untouched':True,'limitation':'POLAR TEXTURE IS SYNTHETIC; NOT REAL UNKNOWN SKIN; AI classification must be reverified',
             'assets':self.entries},indent=2,ensure_ascii=False),encoding='utf8')

    def path(self,asset_id):
        return self.out_tex/(asset_id+'_synthetic_pole_repair.png')
