"""Physics-free v2.7 regression tests: only checks geometric post-return verdict."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_physics import post_return_placement_check, MOTION_SETTINGS, TrialConfig, motion_settings
class TestAfterHome(unittest.TestCase):
 def test_clean_placement(self):
  self.assertEqual(post_return_placement_check((.190,0,1.245),(.190,0),True,.01)[0],True)
 def test_old_c1_drift_fails(self):
  a=post_return_placement_check((.0733,0,1.2547),(.19,0),True,.0)
  self.assertFalse(a[0]);self.assertEqual(a[1],'POST_RETURN_POSITION_DRIFT')
 def test_old_c2_drift_fails(self):
  self.assertFalse(post_return_placement_check((.1448,0,1.245),(.305,0),True,.0)[0])
 def test_old_d1_drift_fails(self):
  self.assertFalse(post_return_placement_check((.5,0,1.2307),(.58,0),True,.0)[0])
 def test_lost_support_rejected(self):
  self.assertFalse(post_return_placement_check((.19,0,1.245),(.19,0),False,.0)[0])
 def test_motion_rejected(self):
  self.assertFalse(post_return_placement_check((.19,0,1.245),(.19,0),True,.12)[0])
 def test_guarded_max_stage_vel(self):
  m=motion_settings(TrialConfig(motion_profile='guarded_fast'))
  self.assertLessEqual(m.x_max_velocity,1.5)
  self.assertAlmostEqual(m.lower_slow_m_s,.10)
if __name__=='__main__': unittest.main()
