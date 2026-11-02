#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage27 v4.7: measured reference 0.65, variation in grade order / tray vacancies.

2026/10/18 JST. All Stage25 robot, camera, AI, 12 nest meshes and safe hold/release
physics remain UNCHANGED.  3 independently runnable scenarios (8,8,12 trials).
Each trial is a separate real PyBullet simulation step sequence, in one persistent
scene per scenario.  Worker-C events delete an already placed fruit BETWEEN
trials; this is scripted, not a human animation.  Worker handling time excluded.
No mp4 is generated; outcomes are textual + JSON + ZIP.
"""
from __future__ import annotations
import os, sys, json, hashlib, zipfile, traceback, statistics, argparse
from pathlib import Path
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
BASE = HERE / 'apple_stage13'
if not (BASE / 'src' / 'run_physics.py').is_file():
    raise RuntimeError('工程25のソースがありません。試験せず停止: ' + str(BASE))
sys.path[:0] = [str(HERE), str(BASE), str(BASE / 'src')]

from stage27_plan import SCENARIOS, NEAREST, plan_expected, validate
import pybullet as p
import run_physics as rp
import run_stage25_v45_centerfirst as inherited
from stage23_full_supervisor import FullGradeSupervisor
from tray_control import tray_occupied, next_free_nest
from run_stage03 import plan

JST = lambda: datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y/%m/%d %H:%M JST')
ROOT = BASE / 'output' / 'stage27_v47_nearest_stability'
SOURCE_REQUIRED = ('stage27_plan.py', 'stage27_v47_stability.py')


def occupied(scene):
    return {n: bool(tray_occupied(p, floor, scene['robot']))
            for n,floor in scene['tray_floors'].items()}


def _save_json_atomic(path, obj):
    tmp=path.with_suffix(path.suffix + '.writing')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def clear_existing(scene, slot, supervisor):
    """Scripted worker C removal of only an already placed physical apple."""
    before=occupied(scene)
    if slot not in scene['placed_fruits'] or not before[slot]:
        raise AssertionError('回収指定先が実際に在荷していない: ' + slot)
    old=int(scene['placed_fruits'][slot])
    p.removeBody(old)
    del scene['placed_fruits'][slot]
    for _ in range(12):p.stepSimulation()
    after=occupied(scene)
    if after[slot]:raise AssertionError('回収後の荷重検出が空になっていない: ' + slot)
    for other,was in before.items():
        if other!=slot and was and not after[other]:
            raise AssertionError('ほかの受け台の在荷が消えた: ' + other)
    evt={'event':'WORKER_C_SCRIPTED_CLEAR', 'grade':slot[0], 'nest':slot,
         'body_removed':old,'before':before,'after':after,
         'human_motion_simulated':False,'human_handling_seconds_included':False}
    supervisor.events.append({'sequence':len(supervisor.events)+1,**evt})
    return evt


def _percentile_nearest_rank(values, fraction):
    values=sorted(values)
    return values[max(0,min(len(values)-1, int(__import__('math').ceil(len(values)*fraction))-1))]


def summarise(scenario, records, clears, failure=None):
    ok = len(records)==len(scenario.grades) and all(r['success'] for r in records)
    good=[r for r in records if r.get('success') and isinstance(r.get('cycle_s'), (int,float))]
    times=[r['cycle_s'] for r in good]
    by_grade={g:[r['cycle_s'] for r in good if r['grade']==g] for g in 'ABCD'}
    return {
        'stage':'27-v4.7','jst':JST(),'scenario':scenario.name,
        'description':scenario.description, 'return_scale':0.65,
        'n_planned':len(scenario.grades),'n_completed':len(records),
        'n_success':len(good),'all_success':bool(ok),
        'n_scripted_clears':len(clears),'scripted_clears':clears,
        'mean_cycle_s':statistics.mean(times) if times else None,
        'median_cycle_s':statistics.median(times) if times else None,
        'max_cycle_s':max(times) if times else None,
        'p95_nearest_rank_cycle_s':_percentile_nearest_rank(times, .95) if times else None,
        'n_over_3s':sum(t>3.0 for t in times),
        'n_at_or_below_3s':sum(t<=3.0 for t in times),
        'grades':{g:{'n':len(vals), 'mean_cycle_s':statistics.mean(vals) if vals else None,
                      'max_cycle_s':max(vals) if vals else None} for g,vals in by_grade.items()},
        'failure_or_error': failure,'records':records,
        'notes':['Synthetic-trained A/B/C/D classifier, not quality claims for real fruit.',
                 'Measured simulator cycle time excludes worker-C packing and idle time.',
                 'Scripted removal occurs between separate physical trials, no human character simulated.',
                 'Consecutive trials reuse the same PyBullet scene within each scenario.',
                 'Safeguards, robot geometry, original physics and 12 nests not changed.',
                 'These are controlled scenarios, not statistical assurance of orchard 3s throughput.']
    }


def _archive(dst, tag, *, summary_only=False):
    name='apple_stage27_v4.7_' + tag + '_results.zip'
    target=BASE/name
    if target.is_file():
        with zipfile.ZipFile(target) as z:
            if z.testzip() is not None: raise RuntimeError('既存ZIP破損・上書き不可: '+str(target))
        print('ZIP_ALREADY_EXISTS',target,'BYTES',target.stat().st_size,flush=True)
        return target
    tmp=target.with_suffix('.zip.partial')
    if tmp.exists():raise RuntimeError('未完成ZIPを検出・上書き不可: '+str(tmp))
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(dst.rglob('*')):
            if f.is_file() and not f.name.endswith('.writing'):
                z.write(f,arcname=str(f.relative_to(BASE)))
        for name in SOURCE_REQUIRED:
            z.write(HERE/name,arcname='CODE_USED/'+name)
        z.writestr('PROVENANCE.txt',JST()+'\nStage27 v4.7 BASELINE 0.65. '
                   'Scripted worker C; no animated worker; no timing of human action.\n')
    with zipfile.ZipFile(tmp) as z:
        bad=z.testzip()
        if bad:raise RuntimeError('ZIP破損: '+bad)
    tmp.rename(target)
    print('ZIP_EXISTS',target.is_file(),'ZIP_BYTES',target.stat().st_size,
          'ZIP_SHA256',hashlib.sha256(target.read_bytes()).hexdigest(),
          'ZIP_PATH',target,flush=True)
    return target


def run_scenario(name):
    scenario=SCENARIOS[name]
    expected=plan_expected(scenario)
    dst=ROOT/name
    final=dst/'summary.json'
    archive=BASE/('apple_stage27_v4.7_'+name+'_results.zip')
    if final.is_file():
        summary=json.loads(final.read_text(encoding='utf8'))
        print('既存の試行結果あり。再実験しません:',name,'完了数',summary['n_completed'],flush=True)
        _archive(dst,name)
        return summary
    if dst.exists():
        print('中断した試行フォルダを検出。上書きも再実験もしません:',dst,flush=True)
        print('残存結果を保管します。続きの物理試験は別ランタイムの設計が必要です。',flush=True)
        _archive(dst,name)
        return None
    if archive.is_file():
        _archive(dst,name)
        print('ZIPがあるため物理実験をスキップ',flush=True)
        return None
    inherited.verify()       # geometry, SHA physics source, 4-grade full-stop control
    validate()
    dst.mkdir(parents=True)
    prev_policy=rp.MOTION_SETTINGS['settled_gate_09']
    env_keys=('APPLE_NEST_ORDER','STAGE20_V40D_LOWER_RAMP_S','STAGE22_V42_RETURN_SCALE')
    old_env={k:os.environ.get(k) for k in env_keys}
    had_hook=hasattr(rp,'STAGE23_AFTER_INFERENCE_HOOK')
    old_hook=getattr(rp,'STAGE23_AFTER_INFERENCE_HOOK',None)
    cfg0=next(cfg for _,grade,cfg in plan('settled_gate_09') if grade=='A')
    supervisor=FullGradeSupervisor()
    records=[];clears=[];scene=None;error=None
    try:
        os.environ.update({'APPLE_NEST_ORDER':'nearest','STAGE20_V40D_LOWER_RAMP_S':'0.08',
                           'STAGE22_V42_RETURN_SCALE':'0.65'})
        rp.MOTION_SETTINGS['settled_gate_09']=replace(prev_policy,x_effective_rate=1.45)
        rp.STAGE23_AFTER_INFERENCE_HOOK=supervisor.process
        assert p.connect(p.DIRECT)>=0
        for ep in expected:
            index=ep['episode'];grade=ep['grade']
            print(f'TRIAL {name} {index}/{len(expected)} grade={grade}',flush=True)
            if ep['clear_slots']:
                assert scene is not None
                for slot in ep['clear_slots']:
                    removal=clear_existing(scene,slot,supervisor)
                    clears.append({'before_episode':index,**removal})
                    print('WORKER_C_SCRIPTED_CLEAR',slot,'SENSOR_FREE',not removal['after'][slot],flush=True)
            state=(occupied(scene) if scene else
                   {f'{g}{j}':False for g in 'ABCD' for j in (1,2,3)})
            if state!=ep['occupied_before']:
                raise AssertionError(f'事前荷重センサー不一致 ep{index}: '
                                     f'{[n for n in state if state[n]!=ep["occupied_before"][n]]}')
            slot=next_free_nest(grade,state)
            if slot!=ep['expected_slot']:
                raise AssertionError(f'受け台の選択が計画と不一致 ep{index}: {slot}')
            cfg=replace(cfg0,profile='nominal',replicate=2700 + index,
                        scan_duration_s=.32,torque_Nm=12.)
            start=len(supervisor.events)
            row,scene=rp.run_trial(p,grade,dst/'episodes',record_video=False,cfg=cfg,
                                   persistent_scene=scene,return_scene=True)
            events=supervisor.events[start:]
            actual=occupied(scene)
            successful=(row.get('classification',{}).get('class')==grade
                       and row.get('assigned_nest')==slot
                       and bool(row.get('success_physical_and_ai'))
                       and bool(row.get('all_retained_fruits_ok'))
                       and bool(row.get('return_home_pass'))
                       and bool((row.get('post_return_check') or {}).get('ok'))
                       and actual==ep['occupied_after']
                       and not any(e.get('event')=='THREE_FULL_STOP' for e in events))
            cycle=row.get('simulated_time_s')
            rec={'episode':index,'grade':grade,'expected_slot':slot,
                 'assigned_nest':row.get('assigned_nest'),
                 'predicted_grade':row.get('classification',{}).get('class'),
                 'cycle_s':cycle,'success':bool(successful),
                 'simulation_stage_durations_s':row.get('simulation_stage_durations_s'),
                 'occupied_before':state,'occupied_after':actual,
                 'expected_occupied_after':ep['occupied_after'],
                 'events':[e.get('event') for e in events],
                 'return_home_pass':row.get('return_home_pass'),
                 'all_retained_fruits_ok':row.get('all_retained_fruits_ok'),
                 'post_return_check':row.get('post_return_check')}
            records.append(rec)
            _save_json_atomic(dst/'progress.json',summarise(scenario,records,clears))
            print('RESULT',index,grade,slot,'SUCCESS',successful,'CYCLE_S',cycle,
                  'OVER_3S',bool(cycle is not None and cycle>3.0),flush=True)
            if not successful:
                error='PHYSICAL_OR_AI_FAIL_EP_'+str(index)
                print('STOP_FAIL_SAFELY',error,flush=True)
                break
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        (dst/'error_traceback.txt').write_text(traceback.format_exc(),encoding='utf8')
        print('ERROR_STAGE27',error,flush=True)
    finally:
        if p.isConnected():p.disconnect()
        rp.MOTION_SETTINGS['settled_gate_09']=prev_policy
        for k,old in old_env.items():
            if old is None:os.environ.pop(k,None)
            else:os.environ[k]=old
        if had_hook:rp.STAGE23_AFTER_INFERENCE_HOOK=old_hook
        elif hasattr(rp,'STAGE23_AFTER_INFERENCE_HOOK'):
            del rp.STAGE23_AFTER_INFERENCE_HOOK
    summary=summarise(scenario,records,clears,error)
    _save_json_atomic(final,summary)
    print('=== STAGE27 SUMMARY ===',name,flush=True)
    print('N',summary['n_success'],'/',summary['n_planned'],'ALL_SUCCESS',summary['all_success'],
          'MEAN_S',summary['mean_cycle_s'],'MEDIAN_S',summary['median_cycle_s'],
          'MAX_S',summary['max_cycle_s'],'P95_S',summary['p95_nearest_rank_cycle_s'],
          'OVER_3S',summary['n_over_3s'],'CLEAR_EVENTS',summary['n_scripted_clears'],flush=True)
    _archive(dst,name)
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scenario', choices=tuple(SCENARIOS),default='rapid_pickup_8')
    ap.add_argument('--list',action='store_true')
    opt=ap.parse_args()
    if opt.list:
        for name,sc in SCENARIOS.items():
            print(name,len(sc.grades),'episodes',sc.description)
        return 0
    s=run_scenario(opt.scenario)
    return 0 if s is not None and s['all_success'] else 1

if __name__=='__main__':
    sys.exit(main())
