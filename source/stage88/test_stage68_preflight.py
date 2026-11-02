# -*- coding: utf-8 -*-
"""Pure local tests; physics simulator absence must NOT be disguised as pass."""
from pathlib import Path
import tempfile, json, math
import numpy as np
from PIL import Image
from stage68_surface import Stage68Surface
from stage68_motion_patch import apply_stage68_motion_patch
import run_stage68_continuous as run

def herm(v0,v1,d0,d1,dt,u):
 return ((2*u**3-3*u*u+1)*v0+(u**3-2*u*u+u)*dt*d0+(-2*u**3+3*u*u)*v1+(u**3-u*u)*dt*d1)
def wrist_z(t):
 scan=.28;close=.08
 tt=[0.,scan*.5,scan,scan+close]
 zz=[.078,(.078+.097)/2,.097,.097]
 dz=[0.,(.097-.078)/scan,0.,0.]
 t=max(0.,min(t,scan+close))
 i=0 if t<=tt[1] else 1 if t<=tt[2] else 2
 return herm(zz[i],zz[i+1],dz[i],dz[i+1],tt[i+1]-tt[i],(t-tt[i])/(tt[i+1]-tt[i]))

def main():
 root=Path(__file__).resolve().parent
 with tempfile.TemporaryDirectory() as tmp:
    tmp=Path(tmp)
    tex=Stage68Surface(root/'stage58_apple_photo_assets/prepared',tmp)
    assert len(tex.entries)==8
    for item in tex.entries.values():
        src=np.asarray(Image.open(root/'stage58_apple_photo_assets/prepared'/item['source']).convert('RGB'))
        after=np.asarray(Image.open(tmp/item['output']).convert('RGB'))
        assert np.array_equal(src[249:],after[249:]) and not np.array_equal(src[:174],after[:174])
    # Reproduce the exact patch order without running physics. The full
    # ensure_engine() deliberately requires PyBullet and is tested in Colab.
    from stage62_engine_patch import apply_stage62_engine_patch
    from stage66_grasp_guard import apply_stage66_grasp_guard
    from stage68_motion_patch import apply_stage68_motion_patch
    manifest_dir=tmp/'engine_outputs'
    eng=run.s35.clone_engine(manifest_dir)
    run._stage49_patch_motion_engine(eng,manifest_dir)
    run._stage51_patch_scan_overlap(eng,manifest_dir)
    apply_stage62_engine_patch(eng,manifest_dir)
    apply_stage66_grasp_guard(eng,manifest_dir)
    apply_stage68_motion_patch(eng,manifest_dir)
    patch=json.loads((tmp/'engine_outputs'/'stage68_motion_patch_manifest.json').read_text())
    code=(eng/'apple_stage13/src/run_physics.py').read_text(encoding='utf8')
    assert '_s68_cam_target_z' in code and 'STAGE32_REJECTED_BEFORE_DESCENT' in code
    assert 'if _s66_dz>.003 and not _s66_contact_ready' in code
    assert '_s59_up_step' not in code
    assert patch['requested_camera_total_descent_m']==.025
    values=[];zroots=[]
    for t in np.linspace(0.,.28,280):
        u=t/.28;w=wrist_z(float(t))
        target=.078+.044-.025*u
        rootz=.175+target-w
        values.append(rootz+w)
        zroots.append(rootz)
    values=np.array(values);zroots=np.array(zroots)
    assert np.max(np.diff(values))<0 and abs((values[0]-values[-1])-.025)<1e-7
    assert abs(zroots[0]-(.175+.044))<1e-7 and abs(zroots[-1]-.175)<1e-7
    # No cheating: offline check does not include Bullet motor lags or camera link FK.
    print('STAGE68_STATIC_PREFLIGHT_PASS',json.dumps({'source_uv_count':8,'original_uv_rows_249_to_end_untouched':True,
       'motion_patch':patch,'target_camera_descent_m':round(values[0]-values[-1],6),
       'no_real_pybullet_test':True},ensure_ascii=False))
if __name__=='__main__':main()
