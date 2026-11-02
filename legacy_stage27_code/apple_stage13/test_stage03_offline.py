from pathlib import Path
import unittest,sys,numpy as np
root=Path(__file__).resolve().parent;sys.path.insert(0,str(root/'src'))
from tray_control import TRAY_NAMES,station_ready,placement_allowed
from learned_classifier import APPEARANCES,train,infer,rgb_features
from run_physics import TrialConfig
from run_stage03 import plan
class Stage03OfflineTests(unittest.TestCase):
 def test_12_single_row_nests(self):
  self.assertEqual(len(TRAY_NAMES),12)
  self.assertEqual(TRAY_NAMES[0:3],('A1','A2','A3'))
  self.assertEqual(TRAY_NAMES[-3:],('D1','D2','D3'))
 def test_empty_tray_interlock(self):
  state={k:False for k in TRAY_NAMES};self.assertTrue(station_ready(state))
  state['B2']=True;self.assertFalse(station_ready(state));self.assertFalse(placement_allowed('B2',state))
 def test_four_class_simulated_rgb(self):
  model=train()
  for label,(rgb,mass) in APPEARANCES.items():
   a=np.full((80,80,3),5,dtype=np.uint8);a[10:65,10:65]=np.uint8(np.array(rgb)*255)
   self.assertEqual(infer(model,[a],mass)['class'],label)
 def test_ten_distinct_episodes(self):
  rows=plan();self.assertEqual(len(rows),10)
  self.assertEqual({x[1] for x in rows},set('ABCD'))
  for i,lab,cfg in rows:self.assertTrue(1<=cfg.tray_slot<=3)
  self.assertEqual(len({cfg.slug for _,_,cfg in rows}),10)
 def test_stage03_plan_four_grades(self):
  self.assertEqual(set(APPEARANCES),set('ABCD'))
if __name__=='__main__':unittest.main()
