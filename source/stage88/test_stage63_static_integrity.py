# -*- coding: utf-8 -*-
"""Stage63 static regressions: no physical success claims before Colab PyBullet."""
from pathlib import Path
import hashlib, json, tempfile, zipfile
import numpy as np
from stage63_camera_ai import normalize_palm_fov
from stage63_engine_patch import apply_stage63_engine_patch
import run_stage63_continuous as s63

def main():
    a=np.arange(144*144*3,dtype=np.uint8).reshape(144,144,3)
    b,m=normalize_palm_fov(a)
    assert np.array_equal(a,b) and hashlib.sha256(a.tobytes()).hexdigest()==hashlib.sha256(b.tobytes()).hexdigest()
    assert m['ratio']==1. and m['method']=='RAW_PALM_NO_CROP_STAGE61_NEAR_REFERENCE'
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)
        # Exercise Stage51 cloned engine chain with its baseline archive.
        eng=s63.s35.clone_engine(p/'fresh_engine')
        s63._stage49_patch_motion_engine(eng,p)
        s63._stage51_patch_scan_overlap(eng,p)
        rec=apply_stage63_engine_patch(eng,p)
        full=(eng/'apple_stage13/src/run_physics.py').read_text(encoding='utf8')
        assert rec['stage59_bob_steps']==0 and '_s59_up_step' not in full
        assert 'stage63_open_finger_reference_approach' in full
        assert full.index("_s63_gate_z=G.root_z+.006")<full.index("_stage32_gate_cb=globals().get")
        assert "'stage_z_move']=G.root_z+.006*(1.-smootherstep" in full
        assert 'STAGE32_REJECTED_BEFORE_DESCENT' in full
        assert 'gravity_lift' in full and 'STAGE23_AFTER_INFERENCE_HOOK' in full
        print('STAGE63_STATIC_PHYSICS_PATCH_OK',json.dumps(rec,ensure_ascii=False))
    print('STAGE63_CAMERA_RAW_PIXEL_SHA_IDENTICAL',True)
    print('STAGE63_PHYSICS_NOT_EXECUTED_IN_THIS_OFFLINE_TEST')
if __name__=='__main__':main()
