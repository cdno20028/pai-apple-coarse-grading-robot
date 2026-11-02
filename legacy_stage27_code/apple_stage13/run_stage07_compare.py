#!/usr/bin/env python3
"""Compare v2.7 physics runs; require after-return placement check. Historical v2.6 shown for context only."""
import json,statistics,csv,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE/'output'
old={'profile':'v2.6 fast_trial (old success gate)','executed':10,'post_return_pass_n':7,'mean_s':3.3095833333333333,'min_s':2.7708333333333335,'max_s':3.995833333333333,'goal_met':False,'note':'historical audit; 7/10 within 75mm final X/Y; final support not verified'}
rows=[old]
for name in ('fast_trial','guarded_fast'):
 p=ROOT/f'stage07_{name}'/'stage03_summary.json'
 if not p.is_file():
  rows.append({'profile':name,'executed':0,'post_return_pass_n':0,'goal_met':False,'note':'NOT_RUN'});continue
 d=json.loads(p.read_text());r=d.get('results',[])
 ok=sum(all(x.get(k) is True for k in ('success','post_return_checked','post_return_ok','post_return_support','return_home_pass')) for x in r)
 ts=[x['simulated_time_s'] for x in r if x.get('simulated_time_s') is not None]
 rows.append({'profile':name,'executed':len(r),'post_return_pass_n':ok,
   'mean_s':round(statistics.mean(ts),4) if ts else None,
   'min_s':round(min(ts),4) if ts else None,'max_s':round(max(ts),4) if ts else None,
   'goal_met':len(r)==10 and ok==10 and len(ts)==10 and statistics.mean(ts)<=3 if ts else False,
   'note':'PyBullet physical after-return support and position verification; force peaks must be reviewed separately'})
out=ROOT/'stage07_compare';out.mkdir(parents=True,exist_ok=True)
(out/'comparison.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
with (out/'comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['profile','executed','post_return_pass_n','mean_s','min_s','max_s','goal_met','note']);w.writeheader();w.writerows(rows)
print(json.dumps(rows,ensure_ascii=False,indent=2))
