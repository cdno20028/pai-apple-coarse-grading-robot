#!/usr/bin/env python3
"""Verifies provided v3 Colab result logs, NOT new physics or a ten-case safety pass."""
from pathlib import Path
import json, hashlib, sys
root=Path(__file__).resolve().parent.parent
base=root/'evidence'/'public_v3_colab_20261026'
summary=json.loads((base/'summary.json').read_text())
cases=json.loads((base/'cases.json').read_text())
cam=json.loads((base/'stage62_five_camera_summary.json').read_text())
legacy=json.loads((base/'stage85_five_case_historical_coverage_diagnostic.json').read_text())
assert len(cases)==len(cam)==summary['n_episodes']==5
assert summary['one_continuous_pybullet_world'] is True
assert summary['one_unspliced_recording'] is True
assert all(x['case']==y['name'] for x,y in zip(cases,cam))
checks=[(r['case'],bool(r['pass'] and r['success_physical_and_ai']
 and r['stage62_five_camera']['physical_success']
 and r['classification']['class']==r['truth_label']
 and r['destination']==r['expected_destination']
 and r['stage62_five_camera']['finger_frames']==12
 and r['stage62_five_camera']['under_camera_frames']==3)) for r in cases]
for n,ok in checks:print(('PASS' if ok else 'FAIL'),n)
assert len(legacy['not_applicable_full_ten_case_checks'])>0
assert summary['all_pass'] is False
focus_pass=all(ok for n,ok in checks)
print('FOCUSED_FIVE_CASE_PASS',focus_pass,'(',sum(ok for n,ok in checks),'/5 )')
print('LEGACY_TEN_CASE_FULL_SCENARIO_PASS',summary['all_pass'])
print('NONAPPLE_AND_FULL_BIN_RETEST_PERFORMED',False)
print('INSPECTED_ORIGINAL_COLAB_EVIDENCE_NOT_NEW_PHYSICS')
sys.exit(0 if focus_pass else 2)
