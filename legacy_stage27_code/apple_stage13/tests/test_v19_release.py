import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from run_physics import TrialConfig,feedback_step,feedback_grip_goals,grip_goals,release_joint_goals,build_trial_plan

class V19Tests(unittest.TestCase):
 def test_distinct_names(self):
  self.assertNotEqual(TrialConfig(control_mode="force_feedback").slug,TrialConfig(control_mode="force_feedback_v2").slug)
 def test_weak_contact_closes_faster(self):
  self.assertEqual(feedback_step(0.,0.,"force_feedback"),-.25)
  self.assertEqual(feedback_step(0.,0.,"force_feedback_v2"),-.5)
 def test_v2_bounds(self):
  c=0.
  for _ in range(60):c=feedback_step(0.,c,"force_feedback_v2")
  self.assertEqual(c,-5.0)
  for _ in range(80):c=feedback_step(20.,c,"force_feedback_v2")
  self.assertEqual(c,1.5)
 def test_release_start_no_jump(self):
  cfg=TrialConfig(control_mode="force_feedback_v2",grip_relief_mm=1.5)
  corrections=[1.5,-1.5,0.]
  expected=feedback_grip_goals(cfg,corrections);got=release_joint_goals(cfg,corrections,0)
  for k in got:self.assertAlmostEqual(got[k],expected[k],12)
 def test_release_end_open(self):
  cfg=TrialConfig(control_mode="force_feedback_v2",grip_relief_mm=.75)
  got=release_joint_goals(cfg,[-5.,0.,1.5],1.); expected=grip_goals(0,cfg)
  for k in got:self.assertAlmostEqual(got[k],expected[k],12)
 def test_release_continuous(self):
  cfg=TrialConfig(control_mode="force_feedback_v2",grip_relief_mm=1.5)
  corr=[1.5,-5.,.75];last=release_joint_goals(cfg,corr,0)
  for i in range(1,49):
   curr=release_joint_goals(cfg,corr,i/48)
   self.assertLess(max(abs(curr[j]-last[j]) for j in curr),.2)
   last=curr
 def test_test_plan_counts(self):
  stress=build_trial_plan("ABC",[12.],["stress_mix"],[.75,1.5],[.7],["force_feedback_v2"],6,[2,3,5])
  self.assertEqual(len(stress),18)
  self.assertEqual({cfg.replicate for _,cfg in stress},{2,3,5})
  nominal=build_trial_plan("ABC",[12.],["nominal"],[.75,1.5],[.7],["force_feedback_v2"],1)
  self.assertEqual(len(nominal),6)

if __name__=="__main__":unittest.main()
