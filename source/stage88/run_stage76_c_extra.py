#!/usr/bin/env python3
# coding: utf-8
"""Supplemental physical validation: use previously untried C3 and C4 source textures.
Separate 10-case one-world run, NOT a claim of 12 sequential physical cases.
The reference label remains C and is never passed into the grade predictor.
"""
import runpy,os
os.environ['STAGE76_C_EXTRA_PHOTOS']='1'
import stage67_camera_ai as cam
cam.PHOTO_INDEX[3]=5  # C-3 from provided photo set
cam.PHOTO_INDEX[10]=6 # C-4 from provided photo set
print('STAGE76_C_EXTRA_PHOTOS: case03 index5 C-3, case10 index6 C-4; separate one-world physics run',flush=True)
runpy.run_module('run_stage70_continuous',run_name='__main__')
