"""Logical safety scenarios, NOT tests of a certified human safety subsystem."""
import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from workcell_interlocks import InterlockController,run_four_scenarios

class TestInterlocks(unittest.TestCase):
 def test_normal(self):
  c=InterlockController();self.assertEqual(c.state,'WAIT_INPUT');self.assertFalse(c.motion_permitted)
  c.place_on_scale();c.classify('A');self.assertEqual(c.destination,'A1')
  c.grasp_done();self.assertTrue(c.motion_permitted);c.release_into_nest()
  self.assertTrue(c.occupied['A1']);self.assertEqual(c.state,'WAIT_INPUT')
 def test_wait_before_grasp_if_all_full(self):
  c=InterlockController()
  for i in (1,2,3):c.external_nest_occupied(f'D{i}')
  c.place_on_scale();c.classify('D');self.assertEqual(c.state,'WAIT_FULL')
  self.assertFalse(c.motion_permitted);self.assertTrue(c.apple_on_scale)
  c.worker_removes('D3');self.assertEqual(c.state,'GRASP');self.assertEqual(c.destination,'D3')
 def test_hold_after_unexpected_all_nests_filled(self):
  c=InterlockController();c.place_on_scale();c.classify('B');c.grasp_done()
  for n in ('B1','B2','B3'):c.external_nest_occupied(n)
  self.assertEqual(c.state,'HOLD_FULL');self.assertFalse(c.motion_permitted)
  c.worker_removes('B2');self.assertEqual(c.state,'MOVE');c.release_into_nest()
  self.assertTrue(c.occupied['B2'])
 def test_human_stop_no_auto_restart(self):
  c=InterlockController();c.place_on_scale();c.classify('C');c.grasp_done()
  c.enter_human_zone();self.assertEqual(c.state,'STOP_HUMAN')
  with self.assertRaises(RuntimeError):c.acknowledge_reset()
  with self.assertRaises(RuntimeError):c.release_into_nest()
  c.leave_human_zone();self.assertEqual(c.state,'STOP_HUMAN')
  c.acknowledge_reset();self.assertEqual(c.state,'MOVE')
 def test_no_input(self):
  c=InterlockController();self.assertEqual(c.state,'WAIT_INPUT')
  self.assertFalse(c.motion_permitted)
 def test_no_ground_truth_grade_access(self):
  c=InterlockController();c.place_on_scale()
  with self.assertRaises(ValueError):c.classify('ground_truth')
 def test_four_scenarios(self):
  x=run_four_scenarios();self.assertEqual(len(x),4)
  for key,v in x.items():
   self.assertEqual(v['last'],'WAIT_INPUT',key)
   self.assertTrue(any(e['event']=='RELEASED' for e in v['trace']))
if __name__=='__main__':unittest.main()
