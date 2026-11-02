# -*- coding:utf-8 -*-
"""Geometry/colour provenance and Stage76/81 camera replay smoke-test.
Recorded known cases are NOT blind test data. PyBullet must be run separately.
"""
from pathlib import Path
import json,hashlib,io,zipfile,tempfile,sys
import numpy as np
from PIL import Image
from stage82_pole_surface import Stage82Surface,OUTER_ANGLE_RAD
from stage82_cd_fusion import fuse_cameras
from stage61_roughgrade import decide_from_camera_metrics

def audit():
    root=Path(__file__).resolve().parent
    assets=root/'stage58_apple_photo_assets'/'prepared'
    original={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in assets.rglob('*') if p.is_file()}
    refs=root/'stage79_normal_pole_reference'
    with tempfile.TemporaryDirectory() as tmp:
        surf=Stage82Surface(assets,Path(tmp),refs)
        assert len(surf.items)==8
        base_dir=Path(tmp)/'stage68_uv_review'
        max_side_diff=0
        for item in surf.items:
            aid=item['asset_id']
            old=np.asarray(Image.open(base_dir/(aid+'_synthetic_pole_repair.png')).convert('RGB'))
            new=np.asarray(Image.open(surf.path(aid)).convert('RGB'))
            side=np.ones((512,1024),bool);side[:46]=False;side[-46:]=False;side[251:273,1008:]=False
            assert np.array_equal(new[side],old[side])
            assert np.max(abs(new.astype(np.int16)-old.astype(np.int16)))<=max(200,1) # swatch reserved
            max_side_diff=max(max_side_diff,int(np.abs(new[side].astype(int)-old[side].astype(int)).max(initial=0)))
        current={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in assets.rglob('*') if p.is_file()}
        assert original==current
    # Compare branch ordering using supplied DEVELOPMENT recorded features.
    fake=lambda dark,green: {'largest_dark_patch_max':dark,'yellow_green_max':green}
    baseline=[
        ('A',[.00645,.00109,.01883],[.01087,.01087,.01087],.00784),
        ('B',[.02095,.00114,.02705],[.01,.01,.01],.00784),
        ('C',[.03934,.04967,.04853],[.01,.01,.01],.00784),
        ('D',[.0103,.01202,.0373],[.76958,.99856,.70387],.86053)]
    from stage82_cd_fusion import inherited_fuse
    for label,d,g,palm in baseline:
        per={f'finger_{i+1}':fake(d[i],g[i]) for i in range(3)}
        per['palm']=fake(0.,palm);per['inspection_under']=fake(0.,0.)
        # separate decision unit: calls returned features, not guessed labels
        from stage82_cd_fusion import cd_from_features
        got=cd_from_features(per,decide_from_camera_metrics(per)['class'])['class']
        assert got==label,(label,got)
    assert max_side_diff==0
    print('STAGE82_POLE_COLOR_AND_CD_PREFLIGHT_PASS',json.dumps({
        'eight_source_photos_unchanged':True,'side_UV_byte_identical':True,
        'pole_influence_deg':round(OUTER_ANGLE_RAD*180/np.pi,2),
        'recorded_dev_A_B_C_D_branch_smoke_test':True,
        'no_orchard_validation':True,'pybullet_NOT_executed_by_this_test':True}))
if __name__=='__main__':audit()
