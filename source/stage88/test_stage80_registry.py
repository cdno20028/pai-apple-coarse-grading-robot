# -*- coding: utf-8 -*-
"""Offline regression for Stage80. Does not claim real PyBullet pass."""
import ast
from stage80_physical_registry import patched_text,OLD,NEW

def verify_report(report,reference,actual,prior=None):
    assert prior is None or prior != actual
    code=NEW.split('        # Stage80:')[1].split('        # success_physical')[0]
    code='        # Stage80:'+code
    # Prepare exactly the source fragment and execute it as a standalone
    # function; do not copy a manually rewritten boolean for testing.
    fn='def tested(scene,report,dest,apple,padcount,result,label):\n'+code+'\n'
    m={}
    exec(compile(fn,'<stage80-regression>','exec'),m)
    scene={'tray_floors':{'A2':1},'placed_fruits':{}}
    m['tested'](scene,report,'A2',42,3,{'class':actual},reference)
    return scene,report

def valid_report():
    return {'lift_pass':True,'sort_pass':True,'return_home_pass':True,
        'placement':{'verified_support':True,'settled_on_floor':True},
        'post_return_check':{'ok':True},'no_drop_release':True,
        'success_physical_and_ai':False}

if __name__=='__main__':
    s='def mock():\n'+OLD+'\n'
    assert patched_text(s).count('stage80_physical_registry_verified')==1
    r=valid_report()
    scene,r=verify_report(r,'B','A')
    assert scene['placed_fruits']=={'A2':42}
    assert r['stage80_physical_registry_verified'] is True
    assert r['stage80_grade_agreed_with_reference'] is False
    assert r['success_physical_and_ai'] is False
    for prop in ['lift_pass','sort_pass','return_home_pass','no_drop_release']:
        r=valid_report();r[prop]=False
        scene,r=verify_report(r,'B','A')
        assert scene['placed_fruits']=={},prop
    for group,prop in [('placement','verified_support'),('placement','settled_on_floor'),('post_return_check','ok')]:
        r=valid_report();r[group][prop]=False
        scene,r=verify_report(r,'B','A')
        assert scene['placed_fruits']=={},(group,prop)
    r=valid_report()
    scene,r=verify_report(r,'B','B')
    assert r['stage80_grade_agreed_with_reference'] is True
    print('STAGE80_REGISTRY_STATIC_PASS physical body persisted after mismatch, strict label result unchanged, unsafe placements rejected')
