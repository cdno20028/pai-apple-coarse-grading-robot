import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_physics import TrialConfig,PROFILES,grip_goals
class TestRobustness(unittest.TestCase):
 def test_profile_available(self):self.assertIn('stress_mix',PROFILES)
 def test_profile_repeatable(self):
  a=TrialConfig(profile='stress_mix',replicate=3)
  self.assertEqual(a.stress_parameters(),a.stress_parameters())
 def test_two_repeats_differ(self):
  a=TrialConfig(profile='stress_mix',replicate=0).stress_parameters()
  b=TrialConfig(profile='stress_mix',replicate=1).stress_parameters()
  self.assertNotEqual(a,b)
 def test_range(self):
  for i in range(100):
   c=TrialConfig(profile='stress_mix',replicate=i)
   x,y=c.offset_xy
   self.assertLessEqual((x*x+y*y)**.5,.0030000001)
   self.assertTrue(.9<=c.mass_scale<=1.1)
   self.assertTrue(.6<=c.apple_friction<=1)
   self.assertTrue(.038<=c.apple_radius_m<=.042)
 def test_paired_inputs(self):
  a=TrialConfig(profile='stress_mix',replicate=4,grip_relief_mm=.75)
  b=TrialConfig(profile='stress_mix',replicate=4,grip_relief_mm=1.50)
  self.assertEqual(a.offset_xy,b.offset_xy)
  self.assertEqual(a.mass_scale,b.mass_scale)
  self.assertEqual(a.apple_friction,b.apple_friction)
  self.assertEqual(a.apple_radius_m,b.apple_radius_m)
 def test_relief_kinematics_changes(self):
  a=grip_goals(.60,TrialConfig(grip_relief_mm=.75))
  b=grip_goals(.60,TrialConfig(grip_relief_mm=1.50))
  self.assertNotEqual(a['elbow_0'],b['elbow_0'])
if __name__=='__main__':unittest.main()
