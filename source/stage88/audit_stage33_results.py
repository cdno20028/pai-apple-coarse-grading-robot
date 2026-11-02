#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit physical logs only if Colab actually saved them; never invent success."""
import argparse,csv,datetime,json
from pathlib import Path

def audit(root):
    root=Path(root); f=root/'worker_results.json'
    rows=json.loads(f.read_text(encoding='utf8')) if f.exists() else []
    faults=[]
    grades=['A','B','C','D']
    for ix,grade in enumerate(grades):
        if ix>=len(rows):faults.append({'case':grade,'error':'MISSING_CASE'});continue
        r=rows[ix]
        if r.get('synthetic_apple')!=grade:faults.append({'case':grade,'error':'WRONG_CASE_ORDER'})
        for key in ('object_gate_accepted','gate_status','predicted_grade','route','physical_and_ai_success'):
            if key not in r:faults.append({'case':grade,'error':'MISSING_'+key})
        if not r.get('object_gate_accepted'):
            faults.append({'case':grade,'error':'GATE_REJECTED_OR_NOT_MEASURED','observed':r.get('gate_status')})
        if r.get('predicted_grade')!=grade:faults.append({'case':grade,'error':'GRADE_MISMATCH'})
        if r.get('route')!={'A':'A3','B':'B3','C':'C1','D':'D1'}[grade]:
            faults.append({'case':grade,'error':'ROUTE_MISMATCH'})
        if not r.get('physical_and_ai_success'):faults.append({'case':grade,'error':'PHYSICAL_OR_AI_FAIL'})
    for r in rows:
        for name in ('top_capture_wall_ms','apple_id_infer_wall_ms','grade_infer_wall_ms'):
            if r.get(name) is None:faults.append({'case':r.get('synthetic_apple'),'error':'UNMEASURED_'+name})
    complete=len(rows)==4 and not faults
    summary={'date_jst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
             'physical_cases_received':len(rows),'expected':4,'all_pass':complete,
             'faults':faults,
             'wall_clock_latency_measured_with_actual_pybullet_camera':complete,
             'real_apples_or_baseballs_or_tennis_balls_tested':False,
             'do_not_infer_3s_real_cycle':True}
    (root/'stage33_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print('STAGE33_AUDIT',json.dumps(summary,ensure_ascii=False));return complete

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args()
    raise SystemExit(0 if audit(a.folder) else 2)
