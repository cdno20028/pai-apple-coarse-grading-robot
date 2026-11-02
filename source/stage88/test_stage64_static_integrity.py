# -*- coding: utf-8 -*-
"""Stage64 code-only regression. No claim of passed PyBullet physics."""
import json,hashlib,tempfile,sys
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from stage64_camera_ai import normalize_palm_fov,APPROACH_FRAME_Z_OFFSETS,Stage64CameraAI
from stage64_engine_patch import apply_stage64_engine_patch
import run_stage64_continuous as s64

def main():
    arr=np.arange(144*144*3,dtype=np.uint8).reshape(144,144,3)
    norm,info=normalize_palm_fov(arr)
    assert np.array_equal(norm,arr) and info['ratio']==1.0
    assert info['method']=='RAW_PALM_NO_CROP_AT_STAGE61_REFERENCE'
    assert list(APPROACH_FRAME_Z_OFFSETS)==[.036,.026,.016,.006]
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)
        eng=s64.s35.clone_engine(p/'fresh_engine')
        s64._stage49_patch_motion_engine(eng,p)
        s64._stage51_patch_scan_overlap(eng,p)
        manifest=apply_stage64_engine_patch(eng,p)
        text=(eng/'apple_stage13/src/run_physics.py').read_text(encoding='utf8')
        assert manifest['stage']==64
        assert manifest['gate_offset_from_stage61_m']==0.0
        assert manifest['stage59_bob_steps']==0
        assert manifest['reference_camera_height_exactly_equal']
        assert not manifest['object_gate_before_finger_scan']
        assert manifest['object_gate_before_grip']
        assert manifest['open_finger_passive_imaging_before_gate']
        assert '_s59_up_step' not in text
        assert text.index('stage64_open_finger_reference_approach') < text.index("_stage32_gate_cb=globals().get")
        assert '_s64_gate_z=G.root_z\n' in text
        assert "send(grip_goals(t,cfg));step_and_trace('scan_and_close')" in text
        assert 'stage_z_move\']=G.root_z+.006' not in text
        assert 'STAGE64_PRE_GATE_ROBOT_FRUIT_CONTACT' in text
        assert 'gravity_lift' in text and 'STAGE32_REJECTED_BEFORE_DESCENT' in text
        assert '_STAGE_CONTACT_OBSERVER' in text
        # Three real sensor channels are exercised here with a fake geometry
        # adapter; only camera event accounting is checked, not real physics.
        sys.path.insert(0,str(eng/'apple_stage13/src'))
        from mechanism_v09_reference import G
        import stage64_camera_ai as camera_module
        c=Stage64CameraAI.__new__(Stage64CameraAI)
        c.current={'kind':'apple','case':1,'stage_ticks':{},'frames':{},
                   'events':[], 'actual_finger_camera_z':{f'finger_{i}':[] for i in (1,2,3)}}
        c.rec=SimpleNamespace(stage62_camera_frames={})
        c.texture_body=lambda *args:None
        class Sensor:
            z=G.root_z+.044
            def getJointState(self,robot,j):return (self.z,0.)
            def getLinkState(self,robot,j,computeForwardKinematics=True):
                d=[None]*6;d[4]=[0.,0.,1.+self.z];return d
            def getContactPoints(self,*args):return []
        sensor=Sensor()
        rp=SimpleNamespace(images_from_actual_links=lambda p,robot,joints,side=180:
               [(f'finger_{k}',np.full((side,side,3),k,dtype=np.uint8)) for k in (1,2,3)])
        joints={'stage_z_move':0,**{f'finger_camera_{k}_fixed':k for k in (0,1,2)}}
        for i,z in enumerate((.214,.209,.199,.189,.179,.175)):
            sensor.z=z
            c.observe(sensor,0,1,2,joints,'stage64_open_finger_reference_approach',i/240.,None,rp)
        assert c.current['approach_sample_index']==4
        assert len(c.current['frames'])==12
        assert sum(x['event']=='OPEN_FINGER_PRE_GATE_DESCENT_FRAME' for x in c.current['events'])==12
        for k in (1,2,3):
            v=c.current['actual_finger_camera_z'][f'finger_{k}']
            assert len(v)==4 and all(b<a for a,b in zip(v,v[1:]))
        print('STAGE64_PASSIVE_DESCENT_12_CAMERA_EVENTS_MOCK_PASS')
        print('STAGE64_STATIC_ENGINE_PASS',json.dumps(manifest,ensure_ascii=False))
    print('STAGE64_RAW_GATE_PIXEL_IDENTITY_PASS')
    print('STAGE64_STATIC_PHOTO_TIMING_PASS: pre-gate open-finger 12 shots; acceptance required before any finger closure')
    print('STAGE64_REAL_PYBULLET_NOT_EXECUTED')
if __name__=='__main__':main()
