"""Offline checks for v1.4 (NO PyBullet physics success claimed)."""
import sys, tempfile, json, csv, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from run_physics import TrialConfig,aggregate_rows,write_report,flatten_trial

class BenchmarkTests(unittest.TestCase):
 def test_default_geometry_and_torque(self):
  x=TrialConfig()
  self.assertEqual(x.apple_radius_m,.04)
  self.assertEqual(x.apple_friction,1.0)
  self.assertEqual(x.offset_xy,(0.,0.))
  self.assertEqual(x.torque_Nm,12)
 def test_control_profiles_are_explicit(self):
  self.assertAlmostEqual(TrialConfig(profile='heavy10').mass_scale,1.1)
  self.assertAlmostEqual(TrialConfig(profile='low_mu').apple_friction,.60)
  self.assertAlmostEqual(TrialConfig(profile='small2').apple_radius_m,.038)
  x,y=TrialConfig(profile='shift3',replicate=2).offset_xy
  self.assertAlmostEqual(x*x+y*y,.003**2,delta=1.e-10)
 def test_shift_changes_between_replicates(self):
  self.assertNotEqual(TrialConfig(profile='shift3',replicate=1).offset_xy,
                      TrialConfig(profile='shift3',replicate=2).offset_xy)
 def test_no_results_is_not_claimed(self):
  with tempfile.TemporaryDirectory() as d:
   self.assertFalse(Path(d,'acceptance.json').exists())
   a=write_report(Path(d),[])
   self.assertEqual(a['trial_count'],0)
   self.assertFalse(a['real_pybullet_was_executed'])
 def test_aggregates_failures_as_failures(self):
  rows=[{'label_synthetic_ground_truth':'A','experiment':{'profile':'nominal','replicate':0,'motor_joint_torque_limit_Nm':8},
         'success_physical_and_ai':True,'meets_3s_simulated_clock':True,'diagnostics':{'max_pad_normal_force_N':5}},
        {'label_synthetic_ground_truth':'B','experiment':{'profile':'nominal','replicate':0,'motor_joint_torque_limit_Nm':8},
         'success_physical_and_ai':False,'meets_3s_simulated_clock':False,'diagnostics':{'max_pad_normal_force_N':40}}]
  a=aggregate_rows(rows)[0]
  self.assertEqual(a['n'],2)
  self.assertAlmostEqual(a['success_rate'],.5)
  self.assertEqual(a['peak_normal_force_max_N'],40)
  with tempfile.TemporaryDirectory() as d:
   write_report(Path(d),rows)
   self.assertTrue(Path(d,'benchmark_summary.csv').exists())
 def test_errors_included_in_denominator(self):
  rows=[{'experiment':{'profile':'nominal','replicate':0,'motor_joint_torque_limit_Nm':5},'status':'ERROR','error':'test', 'success_physical_and_ai':False}]
  a=aggregate_rows(rows)[0]
  self.assertEqual(a['success_rate'],0)
  self.assertIsNone(a['peak_normal_force_max_N'])
  self.assertEqual(flatten_trial(rows[0])['error'],'test')

if __name__=='__main__':unittest.main()
