"""Compare executed v1.8 feedback with paired v1.6 fixed-grip stress trials.
Usage: python src/compare_v16_v18.py --new output/physics_v18_stress/all_trials.csv
This script does not run physics; it checks experiment IDs and computes comparisons.
"""
import argparse
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--new',default='output/physics_v18_stress/all_trials.csv')
    p.add_argument('--old',default='v16_stress_observed.csv')
    p.add_argument('--out',default='output/physics_v18_stress')
    args=p.parse_args()
    newfile=HERE/args.new
    if not newfile.is_file():
        print('NO V1.8 PHYSICS RESULTS; do not claim measured improvement')
        return 2
    old=pd.read_csv(HERE/args.old)
    new=pd.read_csv(newfile)
    keys=['label','torque_limit_Nm','grip_relief_mm','transport_target_s','profile','replicate']
    old=old[old.profile.eq('stress_mix')]
    new=new[new.profile.eq('stress_mix') & new.control_mode.eq('force_feedback')]
    obs=old.merge(new,on=keys,how='outer',indicator=True,suffixes=('_v16','_v18'),validate='one_to_one')
    out=HERE/args.out;out.mkdir(parents=True,exist_ok=True)
    obs.to_csv(out/'paired_v16_vs_v18.csv',index=False)
    overlap=obs[obs['_merge'].eq('both')]
    complete=(len(new)==36 and len(obs)==36 and obs['_merge'].eq('both').all() and
              set(new.replicate)==set(range(6)))
    lines=['# v1.8 paired physics comparison','',
        f'- v1.6 recorded stress trials: {len(old)}',
        f'- v1.8 recorded stress trials: {len(new)}',
        f'- Matched condition pairs: {len(overlap)}',
        f'- COMPLETE (36 expected): **{complete}**','']
    if not complete: lines.append('**INCOMPLETE: Do not interpret this as the complete stress test.**')
    if len(overlap):
        for grip,frame in overlap.groupby('grip_relief_mm'):
            a=int(frame.success_v16.astype(str).str.lower().eq('true').sum())
            b=int(frame.success_v18.astype(str).str.lower().eq('true').sum())
            f16=frame.max_pad_normal_force_N_v16.max()
            f18=frame.max_pad_normal_force_N_v18.max()
            lines.append(f'- relief {grip} mm: v1.6 {a}/{len(frame)} successful, v1.8 {b}/{len(frame)}; max wrist normal sum {f16:.2f} N → {f18:.2f} N')
    lines.extend(['','Notes: identical deterministic stress conditions are paired by label/profile/replicate/grip. ',
                  'Normal force is aggregated wrist-link contact, not fruit pressure; physical real-fruit damage not simulated.'])
    (out/'paired_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines))
    return 0 if complete else 3
if __name__=='__main__':
    raise SystemExit(main())
