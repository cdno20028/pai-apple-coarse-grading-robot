"""Deterministic checks for Stage09; no PyBullet engine is invoked."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_physics import (post_return_placement_check,MotionSettings,
                         motion_settings,TrialConfig,MOTION_SETTINGS,staged_lower_target)
from run_stage09_continuous import summarize

class TestStage09(unittest.TestCase):
 def test_historical_B1_must_fail_stricter_criterion(self):
  ok,reason,err=post_return_placement_check((-.394009,-.046338,1.245),(-.42,0),True,.01)
  self.assertFalse(ok);self.assertEqual(reason,'POST_RETURN_POSITION_DRIFT')
  self.assertAlmostEqual(err,.05313,places=4)
 def test_nest_center_19mm_pass_21mm_fails(self):
  self.assertTrue(post_return_placement_check((.019,0,1),(0,0),True,0)[0])
  self.assertFalse(post_return_placement_check((.021,0,1),(0,0),True,0)[0])
 def test_contact_and_velocity_required(self):
  self.assertFalse(post_return_placement_check((0,0,1),(0,0),False,0)[0])
  self.assertFalse(post_return_placement_check((0,0,1),(0,0),True,.11)[0])
 def test_profiles_keep_100mm_and_actual_sensor_gates(self):
  base=motion_settings(TrialConfig(motion_profile='guarded_fast'))
  precise=motion_settings(TrialConfig(motion_profile='precision_fast'))
  quick=motion_settings(TrialConfig(motion_profile='quick_gate'))
  self.assertEqual(precise.lift_steps,base.lift_steps)
  self.assertEqual(quick.lift_steps,base.lift_steps)
  self.assertEqual(precise.feedback_steps,base.feedback_steps)
  self.assertEqual(precise.lower_fast_m_s,base.lower_fast_m_s)
  self.assertEqual(precise.lower_slow_m_s,base.lower_slow_m_s)
  self.assertEqual(precise.scale_settle_steps,base.scale_settle_steps)
  self.assertEqual(precise.release_steps,50)
  self.assertEqual(quick.release_steps,base.release_steps)
  self.assertLess(precise.preflight_steps,base.preflight_steps)
 def test_summary_requires_all_10(self):
  rows=[{'success_physical_and_ai':True,'all_retained_fruits_ok':True,
         'simulated_time_s':2.7,'diagnostics':{},'placement':{},
         'post_return_check':{'target_xy_error_m':.005}} for _ in range(9)]
  final=[{'ok':True} for _ in range(9)]
  s=summarize('precision_fast',rows,final,[])
  self.assertFalse(s['mean_3s_met']);self.assertEqual(s['within_20mm_n'],9)
 def test_strict_summary_rejects_one_failure(self):
  rows=[{'success_physical_and_ai':True,'all_retained_fruits_ok':True,
         'simulated_time_s':2.7,'diagnostics':{},'placement':{},
         'post_return_check':{'target_xy_error_m':.005}} for _ in range(10)]
  rows[1]=dict(rows[1], success_physical_and_ai=False)
  final=[{'ok':True} for _ in range(10)]
  self.assertFalse(summarize('precision_fast',rows,final,[])['mean_3s_met'])

if __name__=='__main__': unittest.main()
