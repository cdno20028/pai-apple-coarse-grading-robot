#!/usr/bin/env python3
"""Stage13 physics: original continuous rig + collision bowls + UV apples.

The historical strict 20mm verdict is preserved as a separate metric.
A provisional 30mm *supported and settled* bowl-capture criterion is reported
for comparison, not claimed as proof that fruit fits physically in soft foam.
"""
import argparse,json,sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'src'))
from cushion_texture_assets import ensure_assets,CUP_DEPTH_M,CUP_DIAMETER_M
from run_stage12_clearance import run_profile
from workbench_geometry import LAYOUT


def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--output',default='output/stage13_cushion_texture')
    pa.add_argument('--limit',type=int,default=10)
    pa.add_argument('--video-count',type=int,default=10)
    a=pa.parse_args()
    if not 1<=a.limit<=10 or not 0<=a.video_count<=10:pa.error('limit and video-count range 1..10 / 0..10')
    ensure_assets()
    try:import pybullet as p
    except ImportError as e:raise SystemExit('PyBullet unavailable locally; execute Colab notebook') from e
    output=HERE/a.output;output.mkdir(parents=True,exist_ok=True)
    cid=p.connect(p.DIRECT)
    if cid<0:raise RuntimeError('PyBullet DIRECT failed')
    try:
        summary=run_profile(p,'settled_gate_09',output,video_count=a.video_count,limit=a.limit)
    finally:p.disconnect()
    metrics={
        'timestamp_jst':datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y/%m/%d %H:%M JST'),
        'version':'stage13_v3.3',
        'geometry':'12 static concave cushion trimeshes',
        'cushion_depth_m':CUP_DEPTH_M,'cup_diameter_m':CUP_DIAMETER_M,
        'crates':4,'fruit_layers_per_crate':3,
        'nest_pitch_m':LAYOUT.nest_pitch_m,'nest_count':len(LAYOUT.nest_centres_x_m),
        'apple_collision':'sphere radius about 40mm',
        'apple_visual':'UV texture OBJ, PNG by grade A/B/C/D',
        'body_softness':'NOT simulated, static mesh only',
        'planned_fruits':a.limit,
        'executed_fruits':summary['executed'],
        'historical_strict_20mm_success_n':summary['per_episode_success_n'],
        'cup_capture_30mm_retained_n':sum(x['cup_capture_provisional'] for x in summary['world_final_retained']),
        'max_previous_fruit_contact_N':summary['max_robot_previous_fruit_contact_N'],
        'mean_processing_time_s':summary['mean_s'],
        'all_after_return_in_cup':bool(len(summary['world_final_retained'])==a.limit and
                    all(x['cup_capture_provisional'] for x in summary['world_final_retained'])),
        'qualification':'30mm is a provisional contact+velocity+position heuristic, not physical proof of stable foam cradle or fruit safety. Previous rig reference height still 200mm above crate rim.'
    }
    (output/'stage13_cup_metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf-8')
    print('STAGE13_METRICS',json.dumps(metrics,ensure_ascii=False,indent=2),flush=True)
    # Real physical failure remains nonzero; notebook always packages logs.
    if summary['per_episode_success_n'] != a.limit or not metrics['all_after_return_in_cup']:
        raise SystemExit('Needs physical tuning; inspect video and diagnostics')

if __name__=='__main__':main()
