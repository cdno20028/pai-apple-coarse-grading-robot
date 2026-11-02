# -*- coding: utf-8 -*-
"""Stage76 recorded-case feature replay. NOT independent prediction performance."""
from pathlib import Path
import json,csv
from stage61_roughgrade import decide_from_camera_metrics
ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'stage76_recorded_camera_features.json').read_text(encoding='utf8'))
assert len(rows)==9
out=[]
for row in rows:
    decision=decide_from_camera_metrics(row['per_camera'])
    assert decision['class']==row['reference_grade'], (row['case'],decision)
    out.append({'case':row['case'],'reference_for_AUDIT_ONLY':row['reference_grade'],
                'old_grade':row['old_grade'],'new_grade':decision['class'],
                'tip_strong':','.join(decision['positive_distinct_cameras']),
                'tip_moderate':','.join(decision['moderate_distinct_cameras']),
                'palm_dark_untrusted_pct':round(100*row['per_camera']['palm']['largest_dark_patch_max'],3),
                'reason':decision['decision_reason']})
assert next(x for x in out if x['case'].startswith('02_'))['new_grade']=='B'
assert sum(x['old_grade']!=x['reference_for_AUDIT_ONLY'] for x in out)==1
with (ROOT/'stage76_recorded_case_replay.csv').open('w',encoding='utf8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=out[0].keys());w.writeheader();w.writerows(out)
print('STAGE76_RECORDED_FEATURE_REPLAY',len(out),'of',len(out),'MATCH',flush=True)
print('WARNING: all development cases / repeated A fruit / no blind orchard test',flush=True)
