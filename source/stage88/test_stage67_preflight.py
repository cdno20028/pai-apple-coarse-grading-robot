# -*- coding: utf-8 -*-
from pathlib import Path
import tempfile, json
import run_stage67_continuous as run
from stage62_engine_patch import apply_stage62_engine_patch
from stage66_grasp_guard import apply_stage66_grasp_guard
from stage67_highview_gate import train_shared_stage67_high_gate
from stage67_camera_ai import Stage67CameraAI, normalize_palm_fov
import numpy as np

def main():
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)
        engine=run.s35.clone_engine(p/'engine')
        run._stage49_patch_motion_engine(engine,p)
        run._stage51_patch_scan_overlap(engine,p)
        a=apply_stage62_engine_patch(engine,p)
        b=apply_stage66_grasp_guard(engine,p)
        assert b['pre_patch_sha256']==a['patched_sha256']
        code=(engine/'apple_stage13/src/run_physics.py').read_text()
        assert '_s59_up_step' not in code
        assert "send(_s62_command);step_and_trace('scan_and_close')" in code
        assert "if _s66_dz>.003 and not _s66_contact_ready" in code
        assert 'STAGE32_REJECTED_BEFORE_DESCENT' in code
        assert 'STAGE23_AFTER_INFERENCE_HOOK' in code
        assert 'STAGE64_OPEN_APPROACH_POSE_NOT_REACHED' not in code
        print('STAGE67_STATIC_PHYSICAL_PATCH_PASS',json.dumps({'stage62_motion':a,'stage66_grasp_witness':b},ensure_ascii=False))
    model, e=train_shared_stage67_high_gate(Path(__file__).resolve().parent)
    assert e['test_camera_image_used_in_training'] and len(e['high_view_training_and_replay_checks'])==4
    assert e['real17_polar_gatecal_v2']['top_cap_source_images'] and len(e['real17_polar_gatecal_v2']['top_cap_source_images'])==3
    assert e['real17_polar_gatecal_v2']['top_cap_positive_augmentations']>=135
    assert len(e['real17_polar_gatecal_v2']['current_polar_composite_camera_frames_in_training'])==2
    assert e['synthetic_negative_false_accepts']==0
    assert len(e['public_art_observed_camera_calibration'])==1
    assert e['public_art_observed_camera_calibration'][0]['result']['accepted']
    print('PUBLIC_ART_CAMERA_DOMAIN_CALIBRATION_NON_INDEPENDENT_PASS', e['public_art_observed_camera_calibration'][0]['result']['confidence'])
    img=np.ones((144,144,3),dtype=np.uint8)*130
    raw,meta=normalize_palm_fov(img)
    assert meta['ratio']==1.0 and np.array_equal(img,raw)
    src=Path(__file__).with_name('stage67_camera_ai.py').read_text()
    assert "if tick==SCAN_TICKS[-1]" in src and "('palm',0,0)]=grade_palm" in src
    assert 'train_shared_stage67_high_gate' in src
    print('STAGE67_HIGH_VIEW_DEVELOPMENT_MODEL_PASS',json.dumps(e,ensure_ascii=False))
    print('STAGE67_CAMERA_RAW_AND_PROVEN_SCAN_STATIC_PASS')
    print('STAGE67_PYBULLET_FULL_CONTINUOUS_PHYSICS_NOT_EXECUTED')
if __name__=='__main__':main()
