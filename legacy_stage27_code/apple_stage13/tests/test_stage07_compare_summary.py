"""Regression: stage03_summary.json contains flat post_return_* fields."""
import unittest, tempfile, subprocess, sys, json
from pathlib import Path

class Stage07AggregateRegression(unittest.TestCase):
 def test_flat_fields_counted_and_incomplete_fails(self):
  script=Path(__file__).resolve().parents[1]/'run_stage07_compare.py'
  with tempfile.TemporaryDirectory() as t:
   d=Path(t)
   sh=d/'run_stage07_compare.py';sh.write_text(script.read_text())
   for prof in ('fast_trial','guarded_fast'):
    directory=d/'output'/f'stage07_{prof}';directory.mkdir(parents=True)
    rows=[dict(success=True,post_return_checked=True,post_return_ok=True,
               post_return_support=True,return_home_pass=True,simulated_time_s=3.0)]
    if prof=='guarded_fast':rows.append(dict(rows[0],post_return_support=False))
    (directory/'stage03_summary.json').write_text(json.dumps({'results':rows}))
   subprocess.run([sys.executable,str(sh)],cwd=d,check=True,capture_output=True,text=True)
   comparison=json.loads((d/'output/stage07_compare/comparison.json').read_text())
   self.assertEqual(comparison[1]['post_return_pass_n'],1)
   self.assertEqual(comparison[2]['post_return_pass_n'],1)
   self.assertEqual(comparison[2]['executed'],2)

if __name__=='__main__':unittest.main()
