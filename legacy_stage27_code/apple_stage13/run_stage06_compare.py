#!/usr/bin/env python3
"""Compare real PyBullet result JSON against observed stage05 reference.
No extrapolated times may be substituted for unavailable trials.
"""
from pathlib import Path
import sys,json,csv,statistics,argparse
from datetime import datetime
from zoneinfo import ZoneInfo
HERE=Path(__file__).resolve().parent

def summarize(out,profile,baseline):
    f=out/'stage03_summary.json'
    if not f.is_file(): return {'motion_profile':profile,'status':'NO_RESULTS','success':False}
    doc=json.loads(f.read_text())
    rows=doc.get('results',[])
    times=[x.get('simulated_time_s') for x in rows if x.get('simulated_time_s') is not None]
    success_n=sum(x.get('success') is True for x in rows)
    expected=10
    complete=len(rows)==expected and len(times)==expected
    # success_lift/sort/home already combined in each result['success']
    acc={'motion_profile':profile,'executed':len(rows),'success_n':success_n,
         'complete_10_trials':complete, 'mean_s':round(statistics.mean(times),3) if times else None,
         'min_s':round(min(times),3) if times else None,
         'max_s':round(max(times),3) if times else None,
         'mean_goal_3s':complete and success_n==expected and statistics.mean(times)<=3 if times else False,
         'all_within_4s':complete and success_n==expected and all(t<=4 for t in times),
         'baseline_mean_s':baseline['baseline_mean_s'],
         'physics_engine_executed':doc.get('engine_executed') is True,
         'world_reset_per_trial':True,
         'grade_model':'synthetic colour/weight (NOT real scratch recognition)',
         'timestamp_jst':datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y/%m/%d %H:%M JST')}
    acc['qualifies_physics_performance_goal']=bool(acc['mean_goal_3s'] and acc['physics_engine_executed'])
    return acc

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='output/stage06_compare')
    args=ap.parse_args()
    baseline=json.loads((HERE/'baseline_v25_reference.json').read_text())
    out=HERE/args.output;out.mkdir(parents=True,exist_ok=True)
    acc=[{'motion_profile':'v2.5 actual supplied reference','executed':10,
          'success_n':10,'mean_s':baseline['baseline_mean_s'],'min_s':baseline['baseline_min_s'],
          'max_s':baseline['baseline_max_s'],'mean_goal_3s':False,'physics_engine_executed':True,
          'world_reset_per_trial':True,'comparison_only':True,'qualifies_physics_performance_goal':False}]
    for profile in ('balanced','fast_trial'):
        acc.append(summarize(HERE/f'output/stage06_{profile}',profile,baseline))
    report={'timestamp_jst':datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y/%m/%d %H:%M JST'),
         'target':'2 seconds typical minimum / 4 seconds typical maximum / mean <=3 seconds, inclusive of home return',
         'baseline_reference_source':'user uploaded apple_stage05_outputs (1).zip; raw PyBullet v2.5 trials',
         'no_3_second_success_claim_without_execution':True,
         'checks':'A candidate is accepted only if all 10 trials finish, physical and classifier results all pass, and measured mean <=3.0s. Contact forces must also be reviewed separately; no fruit-damage proof.',
         'results':acc}
    (out/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    cols=['motion_profile','executed','success_n','min_s','mean_s','max_s','mean_goal_3s','qualifies_physics_performance_goal']
    with (out/'comparison.csv').open('w',newline='',encoding='utf8') as f:
      w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows({k:d.get(k) for k in cols} for d in acc)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
