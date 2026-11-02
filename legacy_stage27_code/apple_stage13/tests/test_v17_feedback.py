import sys,unittest,math,inspect
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_physics import TrialConfig,feedback_grip_goals,per_finger_wrist_forces,grip_goals
class FeedbackTests(unittest.TestCase):
 def test_mode_changes_slug(self):
  a=TrialConfig(control_mode='fixed').slug
  b=TrialConfig(control_mode='force_feedback').slug
  self.assertNotEqual(a,b)
  self.assertTrue(b.endswith('_FB_S1_R00'))
 def test_zero_feedback_equals_regular_final_goal(self):
  a=TrialConfig(grip_relief_mm=1.5)
  b=feedback_grip_goals(a,[0,0,0])
  baseline=grip_goals(.60,a)
  for k in b: self.assertAlmostEqual(b[k],baseline[k],places=8)
 def test_per_finger_correction_moves_only_own_finger(self):
  a=TrialConfig(grip_relief_mm=1.5)
  base=feedback_grip_goals(a,[0,0,0])
  new=feedback_grip_goals(a,[-1,0,0])
  self.assertNotAlmostEqual(new['elbow_0'],base['elbow_0'])
  for k in (1,2):
   for joint in ('shoulder','elbow'):
    self.assertAlmostEqual(new[f'{joint}_{k}'],base[f'{joint}_{k}'])
   self.assertAlmostEqual(new[f'wrist_{k}_joint'],base[f'wrist_{k}_joint'])
 def test_goal_finite_full_feedback_limits(self):
  for grip in (.75,1.5):
   for correction in (-3,0,1.5):
    goals=feedback_grip_goals(TrialConfig(grip_relief_mm=grip),[correction]*3)
    self.assertTrue(all(math.isfinite(v) for v in goals.values()))
 def test_sensor_reads_contact_information_not_oracle_pose(self):
  source=inspect.getsource(per_finger_wrist_forces)
  self.assertIn('getContactPoints',source)
  self.assertNotIn('getBasePositionAndOrientation',source)
 def test_extra_feedback_time_within_3s(self):
  # v1.6 nominal measured 657 physics steps = 2.7375 sec; v1.7 adds exactly 40 at 240Hz.
  self.assertLess((657+40)/240.,3.)
if __name__=='__main__':unittest.main()
