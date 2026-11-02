#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage03 10 physical *episodes*, using the v2.2 PyBullet 3R gripper.

IMPORTANT: run_trial resets the PyBullet world for each apple. The output is
NOT a 10-apple continuous workcell simulation with retained fruits, worker C,
or continuous duty-cycle proof. No physical/AI success is assumed.
"""
from __future__ import annotations
import os,sys,json,csv,subprocess,shutil,traceback,argparse
from dataclasses import replace
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'src'))
from run_physics import TrialConfig,run_trial
from learned_classifier import LABELS
from workbench_geometry import LAYOUT

# Ten apples: at least two of each grade; each assigned to one of 3 nest slots.
# Variation includes shifted fruits and different friction/mass profiles.
CASES=[('A','nominal'),('B','nominal'),('C','nominal'),('D','nominal'),
       ('B','shift3'),('C','low_mu'),('A','shift3'),('D','heavy10'),
       ('B','nominal'),('A','nominal')]

def plan(motion_profile='baseline'):
    n={grade:0 for grade in LABELS}
    tasks=[]
    for idx,(label,profile) in enumerate(CASES,1):
        slot=(n[label]%3)+1;n[label]+=1
        tasks.append((idx,label,TrialConfig(
            profile=profile,replicate=idx,grip_relief_mm=1.5,transport_s=.70,
            control_mode='force_feedback_v2',close_duration_s=.12,tray_slot=slot,
            motion_profile=motion_profile)))
    return tasks

def aggregate(rows,output):
    output.mkdir(parents=True,exist_ok=True)
    summary={'timestamp_jst':datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y/%m/%d %H:%M JST'),
      'experiment':'10 independent PyBullet physics trials with a fresh world per fruit, NOT a continuous ten-fruit line',
      'planned':len(CASES),'executed':len(rows),
      'success_n':sum(bool(x.get('success_physical_and_ai')) for x in rows),
      'failed_n':sum(not bool(x.get('success_physical_and_ai')) for x in rows),
      'grade_labels':{'A':'贈答用','B':'売店用','C':'傷果','D':'青味果=ジュース用'},
      'time_s':[round(float(x.get('simulated_time_s')),3) if x.get('simulated_time_s') is not None else None for x in rows],
      'time_includes_home_return':True,'farthest_nest_x_abs_m':LAYOUT.farthest_nest_from_scale_m,'real_dimensions_applied':True,
      'caveats':['synthetic RGB+mass labels only, not actual scab/bruise texture',
                 'scene reset between individual physics trials; not an uninterrupted continuous run',
                 'no human C physically modeled; occupied slot reuse is not yet studied',
                 'fruit modeled as rigid sphere; no damage/bruising proof',
                 'three-second cycle and hardware safety not claimed'],
      'engine_executed':True,'motion_profile':rows[0].get('motion_profile') if rows else None,
      'target_mean_s':3.0,'target_typical_min_s':2.0,'target_typical_max_s':4.0,'results':[]}
    for i,row in enumerate(rows,1):
        summary['results'].append({'trial':i,'expected':row.get('label_synthetic_ground_truth'),
          'predicted':(row.get('classification') or {}).get('class'),
          'nest':row.get('assigned_nest'),
          'success':bool(row.get('success_physical_and_ai')),
          'placement_status':(row.get('placement') or {}).get('status'),
          'return_home_pass':row.get('return_home_pass'),
          'post_return_checked':(row.get('post_return_check') or {}).get('checked_after_home_return'),
          'post_return_ok':(row.get('post_return_check') or {}).get('ok'),
          'post_return_xy_error_m':(row.get('post_return_check') or {}).get('target_xy_error_m'),
          'post_return_support':(row.get('post_return_check') or {}).get('on_assigned_support'),
          'simulated_time_s':row.get('simulated_time_s'),
          'meets_3s_goal':row.get('meets_3s_simulated_clock'),
          'meets_2_to_4_target':row.get('meets_2_to_4_target'),
          'motion_profile':row.get('motion_profile'),
          'wrist_link_normal_force_N':(row.get('diagnostics') or {}).get('max_wrist_contact_normal_force_N'),
          'floor_peak_N':(row.get('placement') or {}).get('max_floor_contact_N'),
          'landing_speed_m_s':(row.get('placement') or {}).get('impact_vertical_speed_m_s'),
          'apple_lift_delta_m':row.get('apple_lift_delta_m'),
          'error':row.get('error')})
    (output/'stage03_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    with (output/'stage03_summary.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary['results'][0]));w.writeheader();w.writerows(summary['results'])
    return summary

def merge_videos(files,out):
    files=[f for f in files if f.exists()]
    if not files:return {'status':'NO_TRIAL_VIDEO_RECORDED'}
    if not shutil.which('ffmpeg'):return {'status':'FFMPEG_MISSING','files':[str(x) for x in files]}
    # Re-encode for robust concat across independent episodes.
    listing=out.parent/'_video_list.txt'
    listing.write_text(''.join("file '"+str(path.absolute()).replace("'","'\\''")+"'\n" for path in files),encoding='utf-8')
    cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(listing),'-pix_fmt','yuv420p','-c:v','libx264','-crf','23',str(out)]
    res=subprocess.run(cmd,capture_output=True,text=True)
    listing.unlink(missing_ok=True)
    return {'status':'CREATED' if res.returncode==0 else 'MERGE_FAILED','path':str(out),'detail':res.stderr[-1400:]}

def main():
    ap=argparse.ArgumentParser(description='Run 10 independent PyBullet trials, four synthetic grades and 12 possible nests')
    ap.add_argument('--output',default='output/stage03')
    ap.add_argument('--video-count',type=int,default=10,help='record first N episodes for MP4; 0 disables video')
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--motion-profile',choices=['baseline','balanced','fast_trial','guarded_fast'],default='baseline')
    ap.add_argument('--limit',type=int,default=10,help='1..10 for quick smoke run')
    args=ap.parse_args()
    out=HERE/args.output;out.mkdir(parents=True,exist_ok=True)
    tasks=plan(args.motion_profile)[:args.limit]
    dimensional_design={
      'timestamp_jst':datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y/%m/%d %H:%M JST'),
      'as_supplied_bench_m':{'length':LAYOUT.bench_length_m,'depth':LAYOUT.bench_depth_m,
             'adjustable_height_range':[.58,.85],'bench_mass_kg':10.8,'rated_load_kg':120},
      'as_supplied_crate_m':{'short_side_x':LAYOUT.crate_x_m,'long_side_y':LAYOUT.crate_y_m,
             'height':LAYOUT.crate_height_m},
      'design_scale_width_m':LAYOUT.scale_width_m,
      'four_crate_centres_x_m':LAYOUT.crate_centres_x_m,
      'twelve_elevated_nest_centres_x_m':LAYOUT.nest_centres_x_m,
      'farthest_nest_x_abs_m':LAYOUT.farthest_nest_from_scale_m,
      'packing_crates_are_NOT_robot_destinations':True,
      'provisional_clearance_nests_above_crate_rim_m':LAYOUT.nest_top_above_crate_rim_m,
      'provisional_tabletop_height_m':LAYOUT.bench_top_z_m,
      'not_verified':['physical ergonomics at 1.205m nest top','bench strength under moving robot',
                      '3s goal','fruit bruising','human C safety','real fruit image quality grading']}
    (out/'dimensioned_layout.json').write_text(json.dumps(dimensional_design,ensure_ascii=False,indent=2),encoding='utf8')
    if not (1<=args.limit<=10 and args.video_count>=0):ap.error('limit must be 1..10, video-count >=0')
    manifest=[{'trial':i,'label':label,'nest':f'{label}{cfg.tray_slot}',
       'profile':cfg.profile,'slug':cfg.slug,'record_video':i<=args.video_count} for i,label,cfg in tasks]
    (out/'trial_plan.json').write_text(json.dumps({'planned_episodes':manifest,
       'physical_world_reset_per_apple':True,'farthest_nest_x_abs_m':LAYOUT.farthest_nest_from_scale_m},ensure_ascii=False,indent=2),encoding='utf8')
    if args.dry_run:
        print('REAL-DIMENSION LAYOUT:',json.dumps(dimensional_design,ensure_ascii=False))
        print(json.dumps(manifest,indent=2,ensure_ascii=False));return
    try:import pybullet as p
    except ImportError as e:
        (out/'execution_status.json').write_text(json.dumps({'engine_executed':False,
        'status':'PYBULLET_NOT_INSTALLED','message':str(e)},ensure_ascii=False,indent=2),encoding='utf8')
        raise SystemExit('PyBullet unavailable; run the Colab notebook instead')
    client=p.connect(p.DIRECT)
    if client<0:raise RuntimeError('failed PyBullet connection')
    rows=[];clips=[]
    try:
        for idx,label,cfg in tasks:
            print(f'PHYSICS {idx}/{len(tasks)} {label} -> {label}{cfg.tray_slot}',flush=True)
            try:
                row=run_trial(p,label,out,record_video=(idx<=args.video_count),cfg=cfg)
                mp4=out/f'{label}_{cfg.slug}'/'physics_trial.mp4'
                if mp4.exists():clips.append(mp4)
            except Exception as e:
                row={'label_synthetic_ground_truth':label,'assigned_nest':f'{label}{cfg.tray_slot}',
                    'status':'ERROR','error':repr(e),'traceback':traceback.format_exc(),
                    'success_physical_and_ai':False}
                (out/f'trial_{idx:02}_error.json').write_text(json.dumps(row,indent=2,ensure_ascii=False),encoding='utf8')
                print('TRIAL FAILED',repr(e),flush=True)
            rows.append(row)
            sm=aggregate(rows,out)
            print('PROGRESS:',sm['success_n'],'/',sm['executed'],'success',flush=True)
    finally:p.disconnect()
    merged=merge_videos(clips,out/'stage03_pybullet_demo.mp4')
    summary=aggregate(rows,out)
    summary['video']=merged
    (out/'stage03_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf8')
    print('FINAL',json.dumps({'completed':summary['executed'],'success':summary['success_n'],'video':merged},ensure_ascii=False),flush=True)
    if summary['success_n']!=len(tasks):raise SystemExit(1)

if __name__=='__main__':main()
