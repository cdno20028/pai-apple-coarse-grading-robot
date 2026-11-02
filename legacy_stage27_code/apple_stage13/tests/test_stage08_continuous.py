"""Stage08: logic-only safeguards; never substituted for a Bullet physics run."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tray_control import next_free_nest, TRAY_NAMES
from run_stage03 import plan
from run_stage08_continuous import summarize
from run_physics import MOTION_SETTINGS, TrialConfig, motion_settings, inspect_retained_fruits

class TestStage08(unittest.TestCase):
    def test_one_grade_three_slots_only(self):
        occupied={n:False for n in TRAY_NAMES}
        self.assertEqual(next_free_nest('A',occupied),'A1')
        occupied['A1']=True
        self.assertEqual(next_free_nest('A',occupied),'A2')
        occupied['A2']=True
        self.assertEqual(next_free_nest('A',occupied),'A3')
        occupied['A3']=True
        self.assertIsNone(next_free_nest('A',occupied))
        self.assertEqual(next_free_nest('B',occupied),'B1')
    def test_missing_sensors_fails_closed(self):
        with self.assertRaises(ValueError):next_free_nest('A',{'A1':False})
    def test_grade_invalid(self):
        occupied={n:False for n in TRAY_NAMES}
        for grade in (None,'E','AB','a',42):
            self.assertIsNone(next_free_nest(grade,occupied))
    def test_plan_10_no_nest_reuse(self):
        cases=plan('guarded_fast')
        nests=[f'{label}{cfg.tray_slot}' for _,label,cfg in cases]
        self.assertEqual(len(cases),10)
        self.assertEqual(len(nests),len(set(nests)))
        self.assertEqual(sorted(set(x[0] for x in nests)),list('ABCD'))
    def test_gentle_profile_stays_100mm(self):
        gentle=motion_settings(TrialConfig(motion_profile='guarded_gentle'))
        fast=motion_settings(TrialConfig(motion_profile='guarded_fast'))
        self.assertEqual(gentle.lift_steps,fast.lift_steps)
        self.assertGreaterEqual(gentle.lower_slow_window_m,fast.lower_slow_window_m)
        self.assertLessEqual(gentle.lower_fast_m_s,fast.lower_fast_m_s)
        self.assertGreater(gentle.release_steps,fast.release_steps)
    def test_aggregate_not_claimed_without_10(self):
        one={'success_physical_and_ai':True,'all_retained_fruits_ok':True,
             'simulated_time_s':2.1,'diagnostics':{},'placement':{}}
        self.assertFalse(summarize('guarded_fast',[one],[],[])['mean_3s_met'])
        self.assertEqual(summarize('guarded_fast',[one],[],[])['retained_final_count'],0)
    def test_retained_verifier_detects_drift(self):
        class BulletStub:
            def getBasePositionAndOrientation(self, body):
                return ((.2 if body==100 else .5,0,1.24),(0,0,0,1))
            def getBaseVelocity(self,body):return ((0,0,0),(0,0,0))
            def getContactPoints(self,body,floor):return [('support',)]
        scene={'placed_fruits':{'A1':100,'A2':101},
               'tray_floors':{'A1':201,'A2':202},
               'trays':{'A1':(.2,0),'A2':(.3,0)}}
        audit=inspect_retained_fruits(BulletStub(),scene)
        self.assertTrue(audit[0]['ok'])
        self.assertFalse(audit[1]['ok'])
    def test_score_rejects_bad_prior_fruit(self):
        rows=[{'success_physical_and_ai':True,'all_retained_fruits_ok':True,
               'simulated_time_s':2.5,'diagnostics':{},'placement':{}}]*10
        final=[{'ok':True}]*9+[{'ok':False}]
        self.assertFalse(summarize('guarded_fast',rows,final,[])['mean_3s_met'])

if __name__=='__main__':unittest.main()
