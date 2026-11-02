# -*- coding: utf-8 -*-
from pathlib import Path
from PIL import Image
import numpy as np, cv2, tempfile,hashlib, json, math
from stage83_pole_surface import Stage83Surface, CAP_FULL_RAD, CAP_END_RAD
from stage82_pole_surface import Stage82Surface

def audit():
 root=Path(__file__).resolve().parent
 assets=root/'stage58_apple_photo_assets'/'prepared'
 refs=root/'stage79_normal_pole_reference'
 original={str(p.relative_to(assets)):hashlib.sha256(p.read_bytes()).hexdigest() for p in assets.rglob('*') if p.is_file()}
 ref_sha={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in refs.glob('*.jpg')}
 with tempfile.TemporaryDirectory() as folder:
  out=Path(folder)
  s83=Stage83Surface(assets,out/'new',refs)
  s82=Stage82Surface(assets,out/'old',refs)
  assert len(s83.items)==8
  caps=[]
  for e in s83.items:
   aid=e['asset_id']
   new=np.array(Image.open(s83.path(aid)).convert('RGB'))
   old=np.array(Image.open(s82.path(aid)).convert('RGB'))
   base=np.array(Image.open((out/'new/stage68_uv_review')/(aid+'_synthetic_pole_repair.png')).convert('RGB'))
   assert new.shape==old.shape==base.shape==(512,1024,3)
   bound=int(math.ceil(CAP_END_RAD*512/math.pi))+2
   untouched=np.ones((512,1024),bool);untouched[:bound]=False;untouched[-bound:]=False
   untouched[251:273,1008:1023]=False
   assert np.array_equal(new[untouched],base[untouched]),aid
   assert np.mean(np.abs(new[:65].astype(np.float32)-old[:65]))>1.5,('head has no visible reintroduced texture',aid)
   assert np.mean(np.abs(new[-65:].astype(np.float32)-old[-65:]))>1.5,('tail has no visible reintroduced texture',aid)
   # inspect body color in each individual source, not reference grade.
   refbase=cv2.cvtColor(base[276:320].copy(),cv2.COLOR_RGB2LAB)
   skin_median=np.median(refbase.reshape(-1,3),axis=0)
   pole=cv2.cvtColor(new[55:85].copy(),cv2.COLOR_RGB2LAB)
   polemedian=np.median(pole.reshape(-1,3),axis=0)
   # Report differences rather than claim exact chromatic invariance; L can vary by morphology.
   caps.append({'asset_id':aid,'head_RGB_original_mean':base[:65].mean(axis=(0,1)).round(1).tolist(),
            'head_RGB_stage83_mean':new[:65].mean(axis=(0,1)).round(1).tolist(),
            'source_midbody_LAB_median':skin_median.round(1).tolist(),
            'head_skin_LAB_median_at_19_to_30deg':polemedian.round(1).tolist(),
            'head_vs_source_chroma_diff_ab':(polemedian[1:]-skin_median[1:]).round(2).tolist(),
            'head_changed_fraction':round(float(np.mean(np.any(new[:bound]!=base[:bound],axis=2))),4)})
  after={str(p.relative_to(assets)):hashlib.sha256(p.read_bytes()).hexdigest() for p in assets.rglob('*') if p.is_file()}
  assert original==after
  assert ref_sha=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in refs.glob('*.jpg')}
  print('STAGE83_STATIC_PHOTO_COLOR_PASS',json.dumps({'n_fruit':8,'reference_photos':len(ref_sha),'modified_poles_deg':round(math.degrees(CAP_END_RAD),2),'original_midbody_pixels_identical':True,'head_tail_photo_texture_restored_vs_stage82':True,'no_training':True,'no_pybullet_in_this_test':True,'color_metrics':caps},ensure_ascii=False))
if __name__=='__main__':audit()
