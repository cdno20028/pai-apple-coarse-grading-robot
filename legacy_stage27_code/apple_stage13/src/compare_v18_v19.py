"""Compare v1.9 paired trials with v1.8 actual observed CSV, without filling missing results."""
from pathlib import Path
import argparse,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--new',default='output/physics_v19_targeted/all_trials.csv');p.add_argument('--prior',default='v18_stress_observed.csv');p.add_argument('--out',default='output/v18_v19_paired.csv');args=p.parse_args()
old=pd.read_csv(ROOT/args.prior);new=pd.read_csv(ROOT/args.new)
keys=['label','grip_relief_mm','replicate']
assert not old.duplicated(keys).any() and not new.duplicated(keys).any()
res=old.merge(new,on=keys,suffixes=('_v18','_v19'),validate='one_to_one')
if len(res)!=len(new):raise SystemExit('ERROR: unpaired trial or duplicate')
out=ROOT/args.out;out.parent.mkdir(parents=True,exist_ok=True);res.to_csv(out,index=False)
for g,fr in res.groupby('grip_relief_mm'):
    print('grip relief',g,'v18 success',sum(fr.success_v18),'v19 success',sum(fr.success_v19),'/',len(fr))
    print('v18 peak max',fr.max_pad_normal_force_N_v18.max(),'v19 peak max',fr.max_pad_normal_force_N_v19.max())
stage_path=ROOT/'v18_targeted_stage_baseline.csv'
if stage_path.exists():
    stage=pd.read_csv(stage_path)
    paired=stage.merge(new,on=keys,validate='one_to_one')
    if len(paired)!=len(new):raise SystemExit('Missing stage-specific comparison')
    for rep in [2,3,5]:
        r=paired[paired.replicate==rep]
        print(f'R{rep:02d} v18 success {sum(r.success_v18)}/{len(r)}, v19 {sum(r.success)}/{len(r)}')
        print('  release peak v18',r.peak_release_v18_N.max(),'v19',r.peak_release_N.max())
    paired.to_csv(ROOT/'output/v18_v19_stages.csv',index=False)
print('CHANGED:',res.loc[res.success_v18!=res.success_v19,keys+['success_v18','success_v19']].to_string(index=False))
print('SAVED',out)
