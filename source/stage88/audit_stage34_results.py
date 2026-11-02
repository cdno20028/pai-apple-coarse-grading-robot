#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit staged PyBullet negative cases by cross-checking raw episode trial.json.
Never infer success solely from summary/worker result flags.
"""
from pathlib import Path
from PIL import Image
import sys,json,hashlib
import ball_scene_assets as balls

def audit(negative:Path,positive:Path|None=None):
    neg=Path(negative);cases=[];errors=[]
    summaryfile=neg/'summary.json'
    summary=json.loads(summaryfile.read_text(encoding='utf8')) if summaryfile.exists() else {}
    resultfile=neg/'worker_results.json'
    rows=json.loads(resultfile.read_text(encoding='utf8')) if resultfile.exists() else []
    target={(kind,variant) for kind in balls.KINDS for variant in (0,1)}
    seen=set()
    for row in rows:
        kind,var=row.get('physical_truth'),row.get('variant');key=(kind,var)
        name=row.get('case','?');reasons=[]
        if key in seen:reasons.append('DUPLICATE_CASE')
        if key not in target:reasons.append('UNKNOWN_CASE')
        seen.add(key)
        path=neg/f'case_{name}.json'
        if not path.is_file() or json.loads(path.read_text(encoding='utf8'))!=row:reasons.append('CASE_DETAIL_MISMATCH')
        if kind in balls.SPECS:
            mass=balls.SPECS[kind]['mass_kg']
            if abs(float(row.get('pybullet_dynamics_mass_kg') or 0)-mass)>1e-5:reasons.append('NOT_REAL_MASS_BODY')
        sample=row.get('load_samples_kg') or []
        if len(sample)!=3 or not all(x is not None and float(x)>=.025 for x in sample):reasons.append('LOAD_NOT_CONFIRMED')
        if not row.get('upper_image_captured'):reasons.append('NO_CAPTURE')
        imgname=row.get('upper_image_file')
        if not imgname or not (neg/imgname).is_file():reasons.append('NO_IMAGE')
        else:
            img=(neg/imgname)
            try:
                im=Image.open(img);im.verify()
                with Image.open(img) as im2:
                    if im2.size!=(144,144):reasons.append('BAD_CAPTURE_SHAPE')
            except Exception:reasons.append('BAD_IMAGE')
            if hashlib.sha256(img.read_bytes()).hexdigest()!=row.get('upper_image_sha256'):reasons.append('IMAGE_HASH_MISMATCH')
        if row.get('allow_descent') is not False:reasons.append('FALSE_ACCEPT_OR_NO_GATE')
        if row.get('descent_executed') is not False:reasons.append('DESCENT_EXECUTED')
        if row.get('trial_status')!='STAGE32_REJECTED_BEFORE_DESCENT':reasons.append('WRONG_TRIAL_STATUS')
        if row.get('classification')!={'status':'NOT_STARTED','class':None}:reasons.append('GRADE_STARTED')
        if row.get('predicted_object') in (None,'apple'):reasons.append('AI_NOT_CLASSIFIED_NONAPPLE')
        stages=set(row.get('actual_stages') or [])
        if stages-{'station_preflight','wait_for_worker','scale_settle','stage32_load_debounce'}:reasons.append('UNSAFE_STAGE_EXECUTED')
        trial_files=list((neg/'episodes'/name).rglob('trial.json'))
        if len(trial_files)!=1:reasons.append('MISSING_RAW_TRIAL')
        else:
            report=json.loads(trial_files[0].read_text(encoding='utf8'))
            if report.get('status')!=row.get('trial_status'):reasons.append('RAW_STATUS_MISMATCH')
            gate=report.get('stage32_gate') or {}
            pred=(gate.get('object') or {}).get('predicted_object')
            if pred!=row.get('predicted_object'):reasons.append('RAW_PREDICTION_MISMATCH')
            if not gate.get('load',{}).get('triggered'):reasons.append('RAW_LOAD_MISSING')
            if gate.get('allow_descent') is not False:reasons.append('RAW_FALSE_ACCEPT')
        if reasons:errors.append({'case':name,'errors':reasons})
        cases.append({'case':name,'predicted':row.get('predicted_object'),'descent':row.get('descent_executed'),
                      'load_samples_kg':sample,'confidence':row.get('prediction_confidence'),
                      'pass':not reasons,'errors':reasons})
    if seen!=target:errors.append({'coverage':'WRONG_6_CASES','missing':list(map(str,target-seen))})
    if len(rows)!=6:errors.append({'count':'WRONG_CASE_COUNT','count_seen':len(rows)})
    positives={'executed':False,'count':0,'passed':0}
    if positive and Path(positive).exists():
        pos=Path(positive);pf=pos/'worker_results.json'
        if pf.exists():
            parr=json.loads(pf.read_text(encoding='utf8'))
            positives={'executed':True,'count':len(parr),'passed':0}
            expected={'A':'A3','B':'B3','C':'C1','D':'D1'}
            for a in parr:
                name=a.get('synthetic_apple')
                if (name in expected and a.get('predicted_grade')==name and a.get('route')==expected[name]
                    and a.get('object_gate_accepted') and a.get('physical_and_ai_success')
                    and abs(a.get('home_x_m',99))<.030 and abs(a.get('home_x_velocity_m_s',99))<=.1):
                    positives['passed']+=1
                else:errors.append({'positive':name,'error':'POSITIVE_TEST_FAILED'})
            if len(parr)!=4:errors.append({'positive_count':'NOT_FOUR'})
    report={'stage':34,'negative':{'cases':cases,'count':len(rows),'passed':sum(x['pass'] for x in cases)},
             'positive':positives,'audit_errors':errors,
             'passed_all_negatives_and_positives':not errors and positives['passed']==4}
    (neg/'audit_stage34.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'negative_passed':report['negative']['passed'],'negative_count':len(rows),
     'positive_passed':positives['passed'],'positive_count':positives['count'],
     'error_count':len(errors),'errors':errors[:5]},ensure_ascii=False,indent=2))
    return 0 if not errors and positives['passed']==4 else 2

if __name__=='__main__':
    raise SystemExit(audit(Path(sys.argv[1]),Path(sys.argv[2]) if len(sys.argv)>2 else None))
