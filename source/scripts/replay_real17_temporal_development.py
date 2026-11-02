#!/usr/bin/env python3
# 2026/10/29 JST: DEVELOPMENT AUDIT ONLY. Truth labels are read here only
# to score predictions, NEVER at runtime in the robot classification path.
import sys, json, zipfile, pathlib, csv
ROOT=pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'stage88'))
from stage82_cd_fusion import cd_from_features

def cases():
    for path in sorted((ROOT/'audit/REAL17_KNOWN_CASE_REPORTS_DEVELOPMENT_ONLY').glob('*.json')):
        yield 'real17_known',json.loads(path.read_text()),path.name
    z=zipfile.ZipFile(ROOT/'evidence/gate8_full10_verified_20261027/PAI_GATE8_FULL10_RESULT_original.zip')
    for name in sorted(z.namelist()):
        if name.startswith('case_reports/') and name.endswith('.json'):
            yield 'gate8_original_known',json.loads(z.read(name)),pathlib.Path(name).name

def run():
    rows=[]
    for group,j,filename in cases():
        truth=j['truth_label']
        if truth not in ('A','B','C','D'):continue
        recorded=j['classification'];per=recorded['per_camera']
        stage76=recorded['stage82_decision']['reference_stage76_grade_for_counterfactual']
        proposal=cd_from_features(per,stage76)
        stage84=recorded.get('stage84_scratch_detection',{}).get('positive_fingers',[])
        grade= 'C' if stage84 and proposal['class']!='D' else proposal['class']
        rows.append({'source':group,'case':filename,'known_reference':truth,'old_grade':recorded['class'],'candidate_grade':grade,'stage82_grade':proposal['class'],'stage84_scratch':','.join(stage84),'matches_known_reference':grade==truth})
    assert len(rows)==18,('expected 18 grades',len(rows))
    print('DEVELOPMENT_REPLAY:',sum(r['matches_known_reference'] for r in rows),'/',len(rows))
    for r in rows:print(json.dumps(r,ensure_ascii=False))
    assert all(r['matches_known_reference'] for r in rows),'regression in saved known features'
    return rows
if __name__=='__main__': run()
