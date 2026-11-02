"""Offline tests: NOT physics success tests."""
import sys,unittest,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from run_physics import TrialConfig,grip_goals,aggregate_rows,flatten_trial,DT
from mechanism_v09_reference import G,fk,q_at
import numpy as np

class V15Tests(unittest.TestCase):
 def test_nominal_trajectory_unchanged(self):
  for t in [0.,.1,.3,.52,.55,.6]:
   d=grip_goals(t,TrialConfig(grip_relief_mm=0))
   np.testing.assert_allclose([d['shoulder_0'],d['elbow_0'],d['wrist_0_joint']],q_at(t),atol=1e-12)
 def test_grip_relief_increases_final_wrist_radius(self):
  a=grip_goals(.60,TrialConfig(grip_relief_mm=0))
  b=grip_goals(.60,TrialConfig(grip_relief_mm=1.5))
  qa=[a['shoulder_0'],a['elbow_0'],a['wrist_0_joint']]
  qb=[b['shoulder_0'],b['elbow_0'],b['wrist_0_joint']]
  ra=fk(qa)[2,0];rb=fk(qb)[2,0]
  self.assertAlmostEqual(rb-ra,.0015,places=7)
  self.assertGreater(rb,ra)
 def test_relief_applies_only_after_scan(self):
  a=grip_goals(.3,TrialConfig(grip_relief_mm=0))
  b=grip_goals(.3,TrialConfig(grip_relief_mm=1.5))
  self.assertEqual(a,b)
 def test_separate_aggregation_conditions(self):
  rows=[]
  for gr in [0,.75]:
   for mt in [.7,.9]:
    rows.append({'label_synthetic_ground_truth':'A','experiment':{'motor_joint_torque_limit_Nm':12,'profile':'nominal','grip_relief_mm':gr,'transport_duration_target_s':mt},'success_physical_and_ai':False,'meets_3s_simulated_clock':False})
  self.assertEqual(len(aggregate_rows(rows)),4)
 def test_duration_schedule(self):
  self.assertEqual(round(.70/DT),168)
  self.assertEqual(round(.90/DT),216)
  self.assertEqual(round(1.10/DT),264)
 def test_csv_keeps_new_variables(self):
  row=flatten_trial({'experiment':{'motor_joint_torque_limit_Nm':12,'profile':'nominal','grip_relief_mm':.75,'transport_duration_target_s':.9}})
  self.assertEqual(row['grip_relief_mm'],.75)
  self.assertEqual(row['transport_target_s'],.9)
if __name__=='__main__': unittest.main()
