"""Strict offline geometry and pipeline preflight for v1.3.
Never asserts success of unexecuted rigid-body physics.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from xml.etree import ElementTree as ET
import json,sys,importlib.util,unittest
import numpy as np
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'src'))
from mechanism_v09_reference import G,full_trajectory,derived_state
from learned_classifier import train,infer,APPEARANCES


def synthetic_ai_benchmark(samples=30,seed=123):
    """Out-of-training-set artificial RGB examples; not real fruit performance."""
    rng=np.random.default_rng(seed);model=train()
    truth=[];pred=[];no_image=0
    for y,(color,mass) in APPEARANCES.items():
        for _ in range(samples):
            gain=rng.uniform(.62,1.04)
            rgb=np.clip(np.array(color)*gain+rng.normal(0,.012,3),0,1)
            img=np.zeros((72,72,3),dtype=np.uint8)
            # centered apple-colored disc on a white physical-image-like canvas
            img[:]=[245,246,248]
            yy,xx=np.mgrid[:72,:72];mask=(xx-36)**2+(yy-36)**2<23**2
            img[mask]=np.uint8(np.clip(rgb*255,0,255))
            ans=infer(model,[img],mass+rng.normal(0,.008))
            truth.append(y);pred.append(ans['class']);no_image+=(ans['class'] is None)
    labels=list(APPEARANCES)
    cm={a:{b:sum(1 for t,p in zip(truth,pred) if t==a and p==b) for b in labels} for a in labels}
    return {'samples_per_class':samples,'confusion_matrix':cm,'accuracy':round(sum(t==p for t,p in zip(truth,pred))/len(truth),4),
            'missing_image_count':no_image,'note':'Synthetic RGB + mass benchmark only: does not test actual bruises, quality, camera lighting or field apples.'}


def check():
    timestamp=datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y/%m/%d %H:%M JST')
    urdf=ET.parse(HERE/'robot_v13.urdf').getroot()
    joint_names=[j.get('name') for j in urdf.findall('joint')]
    expected=[f'{kind}_{i}' for i in range(3) for kind in ['shoulder','elbow']]
    expected += [f'wrist_{i}_joint' for i in range(3)]
    expected += [f'finger_camera_{i}_fixed' for i in range(3)]
    expected += ['stage_x_move','stage_y_move','stage_z_move','palm_camera_fixed']
    missing=sorted(set(expected)-set(joint_names))
    ts,q=full_trajectory();velocity=np.gradient(q,ts,axis=0);acc=np.gradient(velocity,ts,axis=0)
    contact_pre=[derived_state(float(t))['pad_apple_clearance_mm'] for t in np.linspace(0,.52,101)]
    test_suite=unittest.defaultTestLoader.discover(str(HERE/'tests'))
    with open(HERE/'output'/'offline_test_results.txt','w',encoding='utf8') as f:
        runner=unittest.TextTestRunner(stream=f,verbosity=2)
        results=runner.run(test_suite)
    pybullet_ready=importlib.util.find_spec('pybullet') is not None
    report={
        'generated_at_jst':timestamp,
        'physics_engine_pybullet_available':pybullet_ready,
        'pybullet_physics_executed_by_preflight':False,
        'robot_fingers':3,'finger_revolute_joints':9,'camera_total':5,
        'fixed_camera_and_pad_per_wrist':True,'urdf_missing_joints':missing,
        'joint_angles_rad_start':q[0].tolist(),'joint_angles_rad_end':q[-1].tolist(),
        'max_joint_velocities_rad_per_s':np.abs(velocity).max(axis=0).round(5).tolist(),
        'max_joint_accelerations_rad_per_s2':np.abs(acc).max(axis=0).round(4).tolist(),
        'min_precontact_pad_clearance_mm':round(float(min(contact_pre)),4),
        'offline_tests_passed':results.testsRun-len(results.errors)-len(results.failures),
        'offline_tests_total':results.testsRun,
        'offline_tests_all_pass':results.wasSuccessful(),
        'synthetic_ai_holdout_test':synthetic_ai_benchmark(),
        'physics_grasp_verified':False,
        'physics_lift_verified':False,
        'physics_bin_sort_verified':False,
        'three_second_cycle_verified':False,
        'warnings':['Offline tests do not substitute for PyBullet contact/gravity/motor simulation.',
                    'Synthetic AI class labels encode artificial RGB and mass, not commercial apple grading.',
                    'Success must be judged from output/physics/acceptance.json only after a real engine run.']
    }
    report['offline_preflight_pass']=bool(results.wasSuccessful() and not missing)
    path=HERE/'output'/'v12_preflight.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report

if __name__=='__main__':
    result=check()
    if not result['offline_preflight_pass']:raise SystemExit(1)
