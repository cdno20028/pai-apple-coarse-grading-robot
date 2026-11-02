#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage34 STATIC & OFFLINE ONLY; explicitly does not simulate balls in Bullet."""
from pathlib import Path
import hashlib,json,zipfile
import numpy as np
from PIL import Image
import compare_stage30 as baseline
import ball_scene_assets as balls
import run_stage33_physics as stage33
import run_stage34_physics as stage34

ROOT=Path(__file__).resolve().parent

def check():
    data=balls.static_checks()
    assert len(stage34.CASES)==6
    for k in balls.KINDS:
        assert balls.SPECS[k]['mass_kg']>.025, k
    for k,v in stage34.CASES:
        path=ROOT/'sports_ball_assets'/f'{k}_variant{v}.png'
        a=np.asarray(Image.open(path))
        assert a.shape==(256,512,3)
    # The body creation interface is mocked, not the perception or physics test.
    class FakePybullet:
        GEOM_SPHERE=2;GEOM_MESH=5
        def __init__(self):self.log=[]
        def createCollisionShape(self,*args,**kwargs):self.log.append(('collision',args,kwargs));return 11
        def createVisualShape(self,*args,**kwargs):self.log.append(('visual',args,kwargs));return 22
        def getQuaternionFromEuler(self,x):return (0,0,0,1)
        def createMultiBody(self,**kwargs):self.log.append(('body',(),kwargs));return 33
        def loadTexture(self,x):self.log.append(('texture',(),{'filename':x}));return 44
        def changeVisualShape(self,*args,**kwargs):self.log.append(('attach_texture',args,kwargs))
        def changeDynamics(self,*args,**kwargs):self.log.append(('dynamics',args,kwargs))
    for k in balls.KINDS:
        fake=FakePybullet()
        bod=balls.spawn_ball(fake,k,0,[0,0,.05],ROOT/'sports_ball_assets')
        assert bod==33
        c=[x for x in fake.log if x[0]=='collision'][0]
        assert abs(c[2]['radius']-balls.SPECS[k]['radius_m'])<1e-12
        b=[x for x in fake.log if x[0]=='body'][0]
        assert abs(b[2]['baseMass']-balls.SPECS[k]['mass_kg'])<1e-12
        assert any(x[0]=='texture' for x in fake.log)
    with zipfile.ZipFile(ROOT/'stage27_original_96files.zip') as z:
        assert z.testzip() is None
        assert hashlib.sha256(z.read('apple_stage13/src/run_physics.py')).hexdigest()==baseline.EXPECTED_SOURCE_SHA
        assert hashlib.sha256(z.read('apple_stage13/robot_v14_phase30.urdf')).hexdigest()==baseline.EXPECTED_URDF_SHA
        patched=stage33.patch_gate(baseline.NEW_GATE) if False else None
    assert stage34.SAFE_STAGES=={'station_preflight','wait_for_worker','scale_settle','stage32_load_debounce'}
    result={'stage':34,'time_jst':stage34.now(),'result':'OFFLINE_STATIC_PASS',
            'sports_models':data,'simulated_ball_physics_run':False,
            'baseline_stage27_source_verified':True,
            'all_masses_above_trigger_g':True,
            'camera_negative_test_run':False,
            'test_count':6}
    (ROOT/'OFFLINE_STATIC_STAGE34.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(check())
