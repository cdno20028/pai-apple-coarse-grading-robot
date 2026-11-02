import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_physics import build_trial_plan

class CoveragePlanTests(unittest.TestCase):
    def plan(self,profiles,repeats,rep_ids=None):
        return build_trial_plan(['A','B','C'],[12.],profiles,[.75,1.5],[.7],['force_feedback'],repeats,rep_ids)
    def test_stress_36_runs_all_replicates(self):
        plan=self.plan(['stress_mix'],6)
        self.assertEqual(len(plan),36)
        for label in 'ABC':
            for grip in (.75,1.5):
                ids=[c.replicate for l,c in plan if l==label and c.grip_relief_mm==grip]
                self.assertEqual(ids,list(range(6)))
    def test_nominal_6_runs(self):
        plan=self.plan(['nominal'],1)
        self.assertEqual(len(plan),6)
    def test_only_problem_replicates(self):
        plan=self.plan(['stress_mix'],6,[3,5])
        self.assertEqual(len(plan),12)
        self.assertEqual({c.replicate for l,c in plan},{3,5})
    def test_unique_slugs(self):
        plan=self.plan(['stress_mix','nominal'],6)
        self.assertEqual(len({f'{l}_{c.slug}' for l,c in plan}),len(plan))
    def test_duplicate_rep_ids_rejected(self):
        with self.assertRaises(ValueError):self.plan(['stress_mix'],6,[3,3])
    def test_empty_rep_ids_rejected(self):
        with self.assertRaises(ValueError):self.plan(['stress_mix'],6,[])
    def test_invalid_negative_replicate_rejected(self):
        with self.assertRaises(ValueError):self.plan(['stress_mix'],6,[-1])
if __name__=='__main__':unittest.main()
