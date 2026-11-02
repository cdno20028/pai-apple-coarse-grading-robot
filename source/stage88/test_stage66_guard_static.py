# -*- coding: utf-8 -*-
import tempfile,json
from pathlib import Path
import run_stage66_continuous as s66
from stage64_engine_patch import apply_stage64_engine_patch
from stage66_grasp_guard import apply_stage66_grasp_guard

def main():
    with tempfile.TemporaryDirectory() as d:
        base=Path(d)
        eng=s66.s35.clone_engine(base/'engine')
        s66._stage49_patch_motion_engine(eng,base)
        s66._stage51_patch_scan_overlap(eng,base)
        p64=apply_stage64_engine_patch(eng,base)
        p66=apply_stage66_grasp_guard(eng,base)
        text=(eng/'apple_stage13/src/run_physics.py').read_text()
        assert p66['pre_patch_sha256']==p64['patched_sha256']
        assert "and _s66_contact_ready and result.get('class')==label" in text
        assert "if _s66_dz>.003 and not _s66_contact_ready:" in text
        assert "contact_pad_count_before_lift':padcount" in text
        assert '_s59_up_step' not in text
        assert 'STAGE32_REJECTED_BEFORE_DESCENT' in text
        assert 'STAGE23_AFTER_INFERENCE_HOOK' in text
        assert 'STAGE66_ABORT_NO_2_LINK_GRIP_BEFORE_FRUIT_DEPARTURE' in text
        compile(text,'run_physics.py','exec')
        print('STAGE66_STATIC_GRASP_PATCH_OK',json.dumps(p66,ensure_ascii=False))
    print('STAGE66_PYBULLET_CONTINUOUS_PHYSICS_NOT_EXECUTED_BY_PREFLIGHT')

if __name__=='__main__':main()
