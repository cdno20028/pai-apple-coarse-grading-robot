#!/usr/bin/env python3
"""Stage11: 10 fruits processed in ONE persistent PyBullet world per profile.

All placed fruits remain physical rigid bodies. Explicit worker A input is
represented by creation of one *new* apple on the scale after previous home.
No worker C removal during the 10-fruit run, no fruit is ever teleported.
Per-fruit validation + revalidation of previously placed fruits is mandatory.
"""
from __future__ import annotations
import argparse,csv,json,statistics,traceback,sys,shutil
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'src'))
from run_physics import run_trial, DT, LAYOUT
from run_stage03 import plan, merge_videos


def summarize(profile,rows,world_final,worker_events):
    measured=[float(r['simulated_time_s']) for r in rows if r.get('simulated_time_s') is not None]
    success=[bool(r.get('success_physical_and_ai')) and bool(r.get('all_retained_fruits_ok')) for r in rows]
    final=[bool(v['ok']) for v in world_final]
    return {
      'timestamp_jst':datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y/%m/%d %H:%M JST'),
      'experiment':'one persistent PyBullet physics world, each fruit retained after placement',
      'independent_trial':False,'world_resets_per_profile':1,
      'physical_worker_C_modelled':False,'worker_A_input':'new fruit spawned on input scale (not animated)',
      'motion_profile':profile,'planned':10,'executed':len(rows),
      'per_episode_success_n':sum(success),'retained_final_count':len(world_final),
      'all_retained_final_ok':bool(len(world_final)==10 and all(final)),
      'mean_s':round(statistics.mean(measured),4) if measured else None,
      'strict_position_tolerance_m':.020,
      'within_20mm_n':sum(bool((r.get('post_return_check') or {}).get('target_xy_error_m',1e9)<.020) for r in rows),
      'max_placement_error_m':round(max((float((r.get('post_return_check') or {}).get('target_xy_error_m') or 0.) for r in rows),default=0.),5),
      'min_s':round(min(measured),4) if measured else None,
      'max_s':round(max(measured),4) if measured else None,
      'total_simulated_s':round(sum(measured),4),
      'within_2_to_4_n':sum(2<=v<=4 for v in measured),
      'mean_3s_met':bool(len(rows)==10 and all(success) and len(world_final)==10
                         and all(final) and statistics.mean(measured)<=3) if measured else False,
      'cycle_time_includes_return':True,'lift_command_m':.10,
      'grade_labels':{'A':'贈答用','B':'売店用','C':'傷果','D':'青味果・ジュース用'},
      'max_wrist_link_contact_N':max((float((r.get('diagnostics') or {}).get('max_wrist_contact_normal_force_N') or 0) for r in rows),default=0.),
      'max_support_contact_N':max((float((r.get('placement') or {}).get('max_floor_contact_N') or 0) for r in rows),default=0.),
      'max_robot_previous_fruit_contact_N':max((float((r.get('diagnostics') or {}).get('max_retained_fruit_robot_contact_N') or 0) for r in rows),default=0.),
      'nest_pitch_m':LAYOUT.nest_pitch_m,
      'fully_inside_nest_n':sum(bool((r.get('post_return_check') or {}).get('full_footprint_inside_nest')) for r in rows),
      'previous_contact_events':[{'episode':i,'first':(r.get('diagnostics') or {}).get('first_previous_fruit_contact'), 'peak':(r.get('diagnostics') or {}).get('max_previous_fruit_contact_detail')} for i,r in enumerate(rows,1) if (r.get('diagnostics') or {}).get('first_previous_fruit_contact')],
      'world_final_retained':world_final,'worker_events':worker_events,
      'episodes':[{ 'index':i,'expected':r.get('label_synthetic_ground_truth'),
                    'predicted':(r.get('classification') or {}).get('class'),
                    'nest':r.get('assigned_nest'),
                    'time_s':r.get('simulated_time_s'),
                    'success':r.get('success_physical_and_ai'),
                    'all_retained_ok':r.get('all_retained_fruits_ok'),
                    'placed_count':r.get('placed_fruit_count'),
                    'wrist_force_N':(r.get('diagnostics') or {}).get('max_wrist_contact_normal_force_N'),
                    'floor_force_N':(r.get('placement') or {}).get('max_floor_contact_N'),
                    'collision_with_previous_N':(r.get('diagnostics') or {}).get('max_retained_fruit_robot_contact_N'),
                    'placement_status':(r.get('placement') or {}).get('status'),
                    'post_return_error_m':(r.get('post_return_check') or {}).get('target_xy_error_m'),
                    'footprint_fully_inside':(r.get('post_return_check') or {}).get('full_footprint_inside_nest'),
                    'first_collision_nest':((r.get('diagnostics') or {}).get('first_previous_fruit_contact') or {}).get('nest'),
                    'first_collision_stage':((r.get('diagnostics') or {}).get('first_previous_fruit_contact') or {}).get('stage'),
                    'first_collision_robot_link':((r.get('diagnostics') or {}).get('first_previous_fruit_contact') or {}).get('robot_link_name'),
                    'error':r.get('error')}
                   for i,r in enumerate(rows,1)],
      'qualification':'Synthetic RGB+mass classifier. Fruit is rigid sphere: bruising, safety, real fruit vision, human handling not verified. Old provisional bench 3D geometry.'
    }


def run_profile(p,profile,output,video_count=0,limit=10):
    from run_physics import inspect_retained_fruits
    output.mkdir(parents=True,exist_ok=True)
    tasks=plan(profile)[:limit]
    rows=[];scene=None;clips=[];worker_events=[]
    for i,label,cfg in tasks:
        prev_count=len(scene['placed_fruits']) if scene else 0
        worker_events.append({'episode':i,'event':'WORKER_A_NEW_FRUIT_ON_SCALE',
                              'existing_fruit_count':prev_count,
                              'detail':'new fruit body; previous fruits retained; virtual worker'})
        print(f'CONTINUOUS {profile}: fruit {i}/{len(tasks)}, retained={prev_count}',flush=True)
        try:
            row,scene=run_trial(p,label,output,record_video=(i<=video_count),
                                cfg=cfg,persistent_scene=scene,return_scene=True)
        except Exception as e:
            row={'label_synthetic_ground_truth':label,'status':'ERROR',
                 'error':repr(e),'traceback':traceback.format_exc(),
                 'success_physical_and_ai':False,
                 'simulated_time_s':None,'all_retained_fruits_ok':False}
            (output/f'episode_{i:02d}_error.json').write_text(json.dumps(row,indent=2,ensure_ascii=False))
            print('ERROR',repr(e),flush=True)
        rows.append(row)
        clips.append(output/f'{label}_{cfg.slug}'/'physics_trial.mp4')
        if not row.get('success_physical_and_ai') or not row.get('all_retained_fruits_ok'):
            print('PAUSED/FAILED: failed physical fruit, retain scene for inspection; no further fruit inserted',flush=True)
            break
    final=inspect_retained_fruits(p,scene) if scene else []
    summary=summarize(profile,rows,final,worker_events)
    (output/'continuous_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,default=float),encoding='utf-8')
    if summary['episodes']:
        with (output/'continuous_episodes.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(summary['episodes'][0]));w.writeheader();w.writerows(summary['episodes'])
    if video_count:
        summary['video']=merge_videos(clips,output/'continuous_demo.mp4')
    (output/'continuous_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,default=float),encoding='utf-8')
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--profiles',nargs='+',choices=['settled_gate_09','settled_gate_12'],default=['settled_gate_09'])
    parser.add_argument('--output',default='output/stage10')
    parser.add_argument('--video-count',type=int,default=10)
    parser.add_argument('--limit',type=int,default=10)
    args=parser.parse_args()
    if not 1<=args.limit<=10 or args.video_count<0: parser.error('limit=1..10, video-count>=0')
    try:import pybullet as p
    except ImportError as e: raise SystemExit(f'PyBullet not available ({e}); run Colab notebook')
    base=HERE/args.output;base.mkdir(parents=True,exist_ok=True)
    summaries=[]
    client=p.connect(p.DIRECT)
    if client<0:raise RuntimeError('PyBullet initialization failed')
    try:
        for profile in args.profiles:
            # load_scene resets world ONLY once per selected profile.
            summary=run_profile(p,profile,base/profile,
                video_count=(args.video_count if profile==args.profiles[0] else 0),limit=args.limit)
            summaries.append(summary)
            print('FINISHED',profile,summary['executed'],summary['mean_s'],
                  summary['per_episode_success_n'],summary['retained_final_count'],flush=True)
    finally:p.disconnect()
    (base/'comparison.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2,default=float))
    fields=['motion_profile','executed','per_episode_success_n','retained_final_count',
            'all_retained_final_ok','mean_s','min_s','max_s','within_2_to_4_n',
            'mean_3s_met','within_20mm_n','max_placement_error_m','max_wrist_link_contact_N','max_support_contact_N',
            'max_robot_previous_fruit_contact_N','nest_pitch_m','fully_inside_nest_n']
    with (base/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:s.get(k) for k in fields} for s in summaries])
    print('SUMMARY:',json.dumps([{k:s.get(k) for k in fields} for s in summaries],ensure_ascii=False,indent=2),flush=True)
    if any(s['per_episode_success_n']!=args.limit or s['retained_final_count']!=args.limit or not s['all_retained_final_ok'] for s in summaries):
        raise SystemExit('Incomplete/failed continuous physics run under strict 20mm rule; see output audit')

if __name__=='__main__':main()
