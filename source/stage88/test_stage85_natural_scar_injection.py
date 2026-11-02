#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact-change natural scar check. Static textures only; no physics claims."""
import pathlib,sys,tempfile,json
import numpy as np,cv2
from PIL import Image
from stage83_pole_surface import Stage83Surface
from stage84_scar_surface import Stage84Surface
from stage84_small_scratch_ai import detect_small_linear_scratch
P=pathlib.Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='stage85_audit_') as t:
 old=Stage83Surface(P/'stage58_apple_photo_assets'/'prepared',pathlib.Path(t)/'base',P/'stage79_normal_pole_reference')
 new=Stage84Surface(P/'stage58_apple_photo_assets'/'prepared',pathlib.Path(t)/'new',P/'stage79_normal_pole_reference')
 for fruit in ('01_A','02_B','04_C','05_C','06_C','07_C','08_D'):
  assert old.path(fruit).read_bytes()==new.path(fruit).read_bytes()
 o=np.asarray(Image.open(old.path('03_B')).convert('RGB'))
 n=np.asarray(Image.open(new.path('03_B')).convert('RGB'))
 assert np.array_equal(o[:103],n[:103]) and np.array_equal(o[408:],n[408:])
 assert np.array_equal(o[:, :150],n[:, :150])
 x,y,X,Y=444,204,566,306
 old_img=cv2.resize(o[y:Y,x:X],(180,180),interpolation=cv2.INTER_AREA)
 new_img=cv2.resize(n[y:Y,x:X],(180,180),interpolation=cv2.INTER_AREA)
 old_detect=detect_small_linear_scratch(old_img)
 new_detect=detect_small_linear_scratch(new_img)
 assert not old_detect['detected'],old_detect
 assert new_detect['detected'],new_detect
 assert new_detect['best']['elongation_ratio']>=8
 assert np.count_nonzero(np.any(n!=o,axis=2))<260000
 print('STAGE85_NATURAL_SCAR_IN_UV_STATIC_PASS',json.dumps({
  'scratch_region_old_detected':False,'scratch_region_new_detected':True,
  'scar_candidate':new_detect['best'],'seven_other_apple_textures_unchanged':True,
  'head_tail_caps_unchanged':True,'not_a_real_camera_frame':True,
  'actual_pybullet_has_not_been_executed_by_this_test':True},ensure_ascii=False))
