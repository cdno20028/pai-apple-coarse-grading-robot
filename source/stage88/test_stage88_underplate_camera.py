# -*- coding: utf-8 -*-
"""Stage88 offline: geometry, timing, no control mutations, 3-panel mock render.
This is NOT an actual PyBullet run or AI inference test.
"""
from pathlib import Path
import sys, tempfile, types, json, ast
import numpy as np
ROOT=Path(__file__).resolve().parent
cam_source=(ROOT/'stage87_grip_camera.py').read_text(encoding='utf8')
run_source=(ROOT/'run_stage70_continuous.py').read_text(encoding='utf8')
ast.parse(cam_source);ast.parse(run_source)
assert cam_source.count('getCameraImage(')==2
assert cam_source.count('stepSimulation(')==0
assert cam_source.count('resetBasePositionAndOrientation(')==0
assert cam_source.count('setJointMotorControl')==0
assert 'stage88' not in run_source.lower(), 'Physics/AI runner should remain unchanged'
assert "camera_fov_deg" in cam_source and "relative_world_origin_z_m" in cam_source
assert "aspect=1,nearVal=.005,farVal=.8" in cam_source
assert "cameraTargetPosition=target" in cam_source
assert "'under':self._writer" in cam_source and "'three':self._writer" in cam_source
assert "self._render_underplate(apple)" in cam_source
assert "self._three_panel_frame(case_id,stage,pa,pb,under,under_pixels)" in cam_source
assert "grip_rec=Stage87ActualGripCamera(p,result_dir)" in run_source
assert "grip_rec.on_tick(live['case_number'],stage,robot,apple,joints)" in run_source
class MWriter:
    def __init__(self): self.shapes=[];self.closed=False
    def append_data(self,a):self.shapes.append(np.asarray(a).shape)
    def close(self): self.closed=True
class MockBullet:
    ER_TINY_RENDERER=0
    def __init__(self):self.cam_params=[];self.z=1.255
    def getBasePositionAndOrientation(self,apple):return ((0.,0.,self.z),(0,0,0,1))
    def computeViewMatrix(self,*args,**kwargs):return [0]*16
    def computeProjectionMatrixFOV(self,*args,**kwargs):return [0]*16
    def getCameraImage(self,w,h,**kwargs):
        self.cam_params.append((w,h,kwargs))
        img=np.full((h,w,4),123,dtype=np.uint8);img[...,3]=255
        seg=np.full((h,w),7,dtype=np.int32)
        return w,h,img,np.zeros((h,w)),seg
    def getContactPoints(self,*args):return [(0,)]
    def getJointState(self,*args):return (0.111,0,0,0)
sys.modules['run_physics']=types.SimpleNamespace(
    G=types.SimpleNamespace(plate_z=1.1,camera_fov_deg=61.),
    LAYOUT=types.SimpleNamespace(relative_world_origin_z_m=.08))
from stage87_grip_camera import Stage87ActualGripCamera
with tempfile.TemporaryDirectory() as d:
    p=MockBullet(); recorder=Stage87ActualGripCamera(p,d,sample_every_ticks=1)
    writers=[]
    def fake_writer(path,fps):
        w=MWriter();writers.append(w);return w
    recorder._writer=fake_writer
    for phase in ('scan_and_close','gravity_lift'):
        p.z+=.05
        recorder.on_tick(5,phase,1,7,{'stage_z_move':5})
    recorder.close()
    assert len(writers)==5 and all(len(w.shapes)==2 and w.closed for w in writers)
    assert writers[-1].shapes==[(900,1920,3)]*2,writers[-1].shapes
    assert writers[-2].shapes==[(640,640,3)]*2
    under=[x for x in p.cam_params if x[:2]==(640,640)]
    assert len(under)==2 and len(p.cam_params)==6
    assert all(x[2].get('renderer')==p.ER_TINY_RENDERER for x in under)
    j=json.loads((Path(d)/'stage87_actual_grip'/'camera_capture_audit.json').read_text())
    case=j['cases'][0]
    assert case['underplate_camera_is_existing_fifth_ai_camera']
    assert case['all_three_views_generated_in_same_simulation_observer_callback']
    assert case['underplate_camera_eye_xyz_m']==[0.,0.,1.1805]
    assert case['underplate_frames_with_visible_apple_id']==2
    assert case['captured_frames']==2
    assert all(rec['underplate_camera_fruit_pixels']>0 for rec in recorder.records[5])
print('STAGE88_UNDERPLATE_MOCK_TEST_PASS; only camera-rendering logic tested, NO real PyBullet result claimed')
