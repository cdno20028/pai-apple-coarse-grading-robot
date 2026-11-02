from __future__ import annotations
from unittest import TestCase, mock
from pathlib import Path
import tempfile,sys,json,types
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import run_physics
class FailureRecordTests(TestCase):
 def test_error_A_still_writes_results_B_C(self):
  fake=types.SimpleNamespace(DIRECT=0,GUI=1,connect=lambda mode:0,setRealTimeSimulation=lambda x:None,disconnect=lambda:None)
  def simulate(p,label,out,record_video=False,cfg=None):
   if label=='A':raise RuntimeError('intentional contact failure test')
   return {'label_synthetic_ground_truth':label,'experiment':{'profile':cfg.profile,'replicate':cfg.replicate,'motor_joint_torque_limit_Nm':cfg.torque_Nm},'success_physical_and_ai':False,'meets_3s_simulated_clock':False,'status':'FAKE_NO_PHYSICS'}
  with tempfile.TemporaryDirectory() as folder:
   with mock.patch.dict(sys.modules,{'pybullet':fake}),mock.patch.object(run_physics,'HERE',Path(folder)),mock.patch.object(run_physics,'run_trial',side_effect=simulate),mock.patch.object(sys,'argv',['run_physics.py','--labels','A','B','C','--torques','12','--profiles','nominal','--grip-reliefs-mm','0','--transport-times','0.70','--output','output/physics_v14']):
    with self.assertRaises(SystemExit):run_physics.main()
   result=json.loads((Path(folder)/'output/physics_v14/acceptance.json').read_text())
   self.assertEqual(result['trial_count'],3)
   self.assertEqual(result['error_count'],1)
   self.assertEqual(result['overall_success_n'],0)
   self.assertTrue((Path(folder)/'output/physics_v14/A_nominal_T12_G0mm_M0.70s_S1_R00/error.json').exists())
