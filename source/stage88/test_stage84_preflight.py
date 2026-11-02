#!/usr/bin/env python3
# coding: utf-8
"""Static audit only: cannot certify PyBullet camera or output classification."""
import json, tempfile, pathlib, sys, hashlib
import numpy as np,cv2
from PIL import Image
from stage84_scar_surface import Stage84Surface
from stage83_pole_surface import Stage83Surface
from stage84_small_scratch_ai import detect_small_linear_scratch, fuse_cameras
P=pathlib.Path(__file__).resolve().parent
A=P/'stage58_apple_photo_assets'/'prepared'
with tempfile.TemporaryDirectory(prefix='s84_') as tmp:
    parent=Stage83Surface(A,pathlib.Path(tmp)/'parent',P/'stage79_normal_pole_reference')
    new=Stage84Surface(A,pathlib.Path(tmp)/'new',P/'stage79_normal_pole_reference')
    for aid in ('01_A','02_B','04_C','05_C','06_C','07_C','08_D'):
        assert pathlib.Path(parent.path(aid)).read_bytes()==pathlib.Path(new.path(aid)).read_bytes(),aid
    before=np.asarray(Image.open(parent.path('03_B')).convert('RGB'))
    after=np.asarray(Image.open(new.path('03_B')).convert('RGB'))
    assert np.array_equal(before[:103],after[:103])
    assert np.array_equal(before[408:],after[408:])
    assert np.array_equal(before[:, :150],after[:, :150])
    assert int(np.sum(before!=after))>1000
    # Tiny scratch detectable at several plausible simulated pixel scales.
    patch=after[58:458,244:744]
    raw=cv2.resize(patch,(180,180),interpolation=cv2.INTER_AREA)
    base=cv2.resize(before[58:458,244:744],(180,180),interpolation=cv2.INTER_AREA)
    detect=detect_small_linear_scratch(raw)
    old=detect_small_linear_scratch(base)
    assert detect['detected'],detect
    assert not old['detected'],old
    # Two temporal observations of the same finger corroborate the small
    # crack without being counted as 2 distinct cameras.
    frames={('finger',1,0):raw,('finger',1,1):raw,
            ('finger',0,0):np.zeros_like(raw),('finger',2,0):np.zeros_like(raw),
            ('palm',0,0):np.zeros((144,144,3),np.uint8)}
    result=fuse_cameras(frames)
    assert result['class']=='C',result
    assert result['stage84_scratch_detection']['positive_fingers']==['finger_2']
    desc={'original_photo_count':8,'other_seven_render_UVs_byte_identical':True,
     'head_tail_photo_pixels_identical':True,
     'source_jpgs_not_modified':True,
     'new_synthetic_small_scratch_detected_on_static_simulated_crop':True,
     'same_crop_before_edit_no_scratch':True,
     'scenario_grade_not_fed_to_detection':True,
     'actual_pybullet_C_result_NOT_tested':True,
     'actual_finger2_camera_placement_NOT_tested':True,
     'static_candidate':detect['best']}
    print('STAGE84_STATIC_PASS',json.dumps(desc,ensure_ascii=False))
