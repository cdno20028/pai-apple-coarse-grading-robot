import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_physics import TrialConfig, grip_goals, wrist_target_with_close_duration, build_trial_plan,DT
from mechanism_v09_reference import commanded_wrist, G
import numpy as np
class V20Tests(unittest.TestCase):
 def test_legacy_equivalence(self):
  c=TrialConfig(close_duration_s=.08)
  for t in np.linspace(0,.60,121):
   self.assertTrue(np.allclose(wrist_target_with_close_duration(t,c),commanded_wrist(t),atol=1e-12))
 def test_scan_unchanged(self):
  for t in np.linspace(0,.52,21):
   targets=[wrist_target_with_close_duration(t,TrialConfig(close_duration_s=x)) for x in (.08,.12,.16)]
   self.assertTrue(np.allclose(targets[0],targets[1],atol=1e-12))
   self.assertTrue(np.allclose(targets[0],targets[2],atol=1e-12))
 def test_ends_same_pose(self):
  for d in (.08,.12,.16):
   c=TrialConfig(close_duration_s=d)
   self.assertTrue(np.allclose(wrist_target_with_close_duration(.52+d,c),commanded_wrist(.60),atol=1e-12))
 def test_finger_end_joint_pose_same(self):
  qs=[grip_goals(.52+x,TrialConfig(close_duration_s=x,grip_relief_mm=1.5)) for x in (.08,.12,.16)]
  for k in qs[0]:
   self.assertAlmostEqual(qs[0][k],qs[1][k],10)
   self.assertAlmostEqual(qs[0][k],qs[2][k],10)
 def test_3d_path_same_reparameterized(self):
  for u in np.linspace(0,1,20):
   arr=[wrist_target_with_close_duration(.52+u*d,TrialConfig(close_duration_s=d)) for d in (.08,.12,.16)]
   self.assertTrue(np.allclose(arr[0],arr[1],atol=1e-12))
   self.assertTrue(np.allclose(arr[0],arr[2],atol=1e-12))
 def test_unique_36_trial_plan(self):
  plan=build_trial_plan('ABC',[12.],['stress_mix'],[.75,1.5],[.7],['force_feedback_v2'],6,[2,3],[.08,.12,.16])
  self.assertEqual(len(plan),36)
  self.assertEqual(len({label+'_'+cfg.slug for label,cfg in plan}),36)
 def test_slug_changed_by_duration(self):
  self.assertNotEqual(TrialConfig(close_duration_s=.08).slug,TrialConfig(close_duration_s=.12).slug)
 def test_nominal_6_plan(self):
  plan=build_trial_plan('ABC',[12.],['nominal'],[.75,1.5],[.7],['force_feedback_v2'],1,[0],[.16])
  self.assertEqual(len(plan),6)
if __name__=='__main__':unittest.main()
