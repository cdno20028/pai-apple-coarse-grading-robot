"""Tests for artifact plumbing; not a simulated grasp or physics test."""
from pathlib import Path
import importlib.util, sys, unittest, tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import run_physics
from video_capture import VideoCapture

class FakeCameraBackend:
    ER_TINY_RENDERER=10
    def computeViewMatrix(self,**kwargs):return [1.] * 16
    def computeProjectionMatrixFOV(self,**kwargs):return [1.] * 16
    def getCameraImage(self,w,h,**kwargs):
        rgba=np.zeros((h,w,4),dtype=np.uint8)
        rgba[:,:,:]=[210,215,223,255]
        return w,h,rgba,np.zeros((h,w)),np.zeros((h,w),dtype=int)

class PipelineTests(unittest.TestCase):
    def test_runner_imports_without_pybullet(self):
        self.assertTrue(hasattr(run_physics,'run_trial'))
    def test_recorder_can_export_mp4(self):
        if importlib.util.find_spec('imageio_ffmpeg') is None:self.skipTest('missing ffmpeg')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'recording_smoke.mp4'
            v=VideoCapture(FakeCameraBackend(),path,'TEST',width=320,height=240)
            v.add_frame(.05,'camera_scan');v.close(False,.10)
            self.assertTrue(path.is_file() and path.stat().st_size>1200)
    def test_urdf_does_not_have_six_gripping_fingers(self):
        from xml.etree import ElementTree as ET
        root=ET.parse(ROOT/'robot_v13.urdf').getroot()
        names=[j.get('name') for j in root.findall('joint')]
        self.assertEqual(len([n for n in names if n.startswith('shoulder_')]),3)
        self.assertEqual(len([n for n in names if n.startswith('finger_camera_')]),3)

if __name__=='__main__':unittest.main()
