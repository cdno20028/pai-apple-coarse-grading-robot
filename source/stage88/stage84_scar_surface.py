# -*- coding: utf-8 -*-
"""Stage84 focused demonstration: repair one directionally misplaced top photo in side UV.

One requested 03_B test material becomes a SCENARIO-SPECIFIC synthetic damaged
sample; the user's original JPEG and all other seven fruit UVs remain untouched.
Its class is NOT passed to the vision classifier. Preserve Stage83 photoreal
head/tail caps (top/bottom <=34.4 degrees) and Stage80 physical control.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import cv2
from PIL import Image
from stage83_pole_surface import Stage83Surface

TARGET='03_B'

def build_small_side_scar(im):
    base=np.asarray(im.convert('RGB'),dtype=np.uint8)
    h,w=base.shape[:2];assert (w,h)==(1024,512)
    # Side-only: top/head and underside/tail photographic anatomy NEVER edited.
    # Old stem-on-side was from a wrongly front-hemisphere-mapped top-view photo.
    yy,xx=np.mgrid[:h,:w].astype(np.float32)
    # Only modify source front-body part, not the original input photograph.
    # An elliptical feather makes the repair visually continuous with existing skin.
    distance=((xx-504.)/293.)**2+((yy-248.)/154.)**2
    alpha=np.clip((1.13-distance)/.44,0,1)
    alpha=alpha*alpha*(3-2*alpha)
    # Absolute polar guards preserve head and bottom PHOTO caps pixel-for-pixel.
    alpha[(yy<103.)|(yy>407.)]=0
    # Use same-fruit B skin outside the misplaced top-camera-photo region.
    # Sample side-donor stripes (not a photograph of a different apple).
    arr=base.astype(np.float32)
    left=cv2.GaussianBlur(arr[:,146:206,:],(0,0),sigmaX=12).mean(axis=1)
    right=cv2.GaussianBlur(arr[:,816:876,:],(0,0),sigmaX=12).mean(axis=1)
    mix=np.clip((xx-190.)/650.,0,1)[...,None]
    tone=left[:,None,:]*(1-mix)+right[:,None,:]*mix
    # Tile very low-amplitude same-fruit skin grain, not the discarded stem colors.
    donor=arr[:,110:216,:]
    residual=donor-cv2.GaussianBlur(donor,(0,0),sigmaX=4,sigmaY=4)
    rep=np.tile(residual,(1,10,1))[:,:w,:]
    skin=np.clip(tone+rep*.8,0,255)
    out=arr*(1-alpha[...,None])+skin*alpha[...,None]

    # STAGE85: natural asymmetric SMALL scar pattern adapted from the user's
    # target appearance mockup. The mockup is only a procedural APPEARANCE
    # REFERENCE, never passed to the classifier or represented as a camera frame.
    # Preserve fruit skin colour; transfer local intensity/detail only.
    reference_path=Path(__file__).with_name('stage85_scar_appearance_REFERENCE_NOT_AI_EVIDENCE.png')
    reference=np.asarray(Image.open(reference_path).convert('RGB'),np.uint8)
    # Tight patch: the defect is at most 36 UV pixels long in the middle.
    src=cv2.resize(reference,(69,46),interpolation=cv2.INTER_AREA)
    src_lab=cv2.cvtColor(src,cv2.COLOR_RGB2LAB).astype(np.float32)
    # Bright fruit-skin interpolation at patch periphery, leaves only scar
    # local luminance+microtexture. Bright skin/photo glare not transplanted.
    src_L=src_lab[:,:,0]
    lowfreq=cv2.GaussianBlur(src_L,(0,0),sigmaX=13,sigmaY=10)
    detail=np.minimum(src_L-lowfreq,3.5)
    # Constrain the synthetic lesion to the central narrow, elliptical area.
    ph,pw=detail.shape
    gy,gx=np.mgrid[:ph,:pw].astype(np.float32)
    ellipse=((gx-35)/25.5)**2+((gy-24)/11.0)**2
    edge=np.clip((1.2-ellipse)/.44,0,1)
    edge=edge*edge*(3-2*edge)
    # The chosen picture contains a natural dark irregular scratch, not a
    # straight computer-painted line. Attenuate/scale to retain A/B context.
    detail=np.clip(detail*1.85,-118.,3.5)
    scar_dark=(-np.minimum(detail,0.))*edge
    y0,x0=229,470
    patch0=out[y0:y0+ph,x0:x0+pw].copy()
    # Match actual individual apple's base skin; never paste unrelated red hue.
    lab=cv2.cvtColor(np.uint8(np.clip(patch0,0,255)),cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:,:,0]=np.clip(lab[:,:,0]-scar_dark,0,255)
    # Darker tissue is muted brown and the surrounding skin stays the original.
    lab[:,:,1]=np.clip(lab[:,:,1]-scar_dark*.055,0,255)
    lab[:,:,2]=np.clip(lab[:,:,2]-scar_dark*.04,0,255)
    out[y0:y0+ph,x0:x0+pw]=cv2.cvtColor(np.uint8(np.rint(lab)),cv2.COLOR_LAB2RGB).astype(np.float32)
    out=np.uint8(np.clip(np.rint(out),0,255))
    assert np.array_equal(out[:103],base[:103]),'HEAD_PHOTo_ALTERED'
    assert np.array_equal(out[408:],base[408:]),'TAIL_PHOTO_ALTERED'
    assert np.array_equal(out[:, :150],base[:, :150]),'NON_TARGET_SIDE_CHANGED'
    return Image.fromarray(out,'RGB'),{'synthetic_wound':True,'approx_synthetic_wound_UV_px':[36,4],
      'source_top_view_removed_from_side_region':True,
      'edit_region_uv':'elliptical around (504,248), cap guard y=103..407',
      'original_apple_file_unchanged':True,'different_original_asset_labels_not_input_to_grade':True,
      'natural_scar_appearance_from_separate_mockup':True,'mockup_used_as_appearance_reference_ONLY':True,'visual_is_simulated_not_a_real_damage_annotation':True}

class Stage84Surface(Stage83Surface):
    def __init__(self,assets,output,refs):
        super().__init__(assets,output,refs)
        path=self.path(TARGET)
        pre=path.read_bytes()
        im,meta=build_small_side_scar(Image.open(path))
        im.save(path,compress_level=6)
        self.audit={'stage':85,'target_asset':TARGET,'scenario':'camera-side subtle organic irregular scar C candidate',
          'parent_stage83_visual_sha256':hashlib.sha256(pre).hexdigest(),
          'patched_uv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'source_jpegs_all_untouched':True,
          'other_seven_fruits_all_untouched':True,
          'head_tail_cap_pixel_identical':True,
          'classifier_uses_case_truth_or_asset_id':False,
          'actual_camera_and_physics_unverified_until_colab':True,**meta}
        Path(output,'stage84_small_scar_synthetic_manifest.json').write_text(
           json.dumps(self.audit,indent=2,ensure_ascii=False),encoding='utf8')
