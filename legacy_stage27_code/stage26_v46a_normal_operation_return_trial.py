#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage26 v4.6a: measured baseline vs ONE FACTOR return-command timing.

Evidence: Stage25 v4.5a (A2/A3 middle clearance) and Stage24 v4.4.
This does NOT change any original physical geometry, robot control source,
AI model, 3 fingers, five cameras, 12 tray nests, lifting/release safety gates.

Four synthetic apples A B C D, each grade on the nearest freely available nest.
Worker C's immediate removal between trials is SCRIPTED, not a human animation.
Two matched conditions differ ONLY in STAGE22_V42_RETURN_SCALE: .65 -> .55.
Pilot study; NOT 3 seconds of human-inclusive orchard throughput.
"""
from __future__ import annotations
import os, sys, json, traceback, zipfile, statistics, hashlib
from pathlib import Path
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

WORK = Path(__file__).resolve().parent
BASE = WORK / 'apple_stage13'
if not (BASE / 'src' / 'run_physics.py').is_file():
    raise FileNotFoundError('工程25のソースがありません: ' + str(BASE) + '\n過去の試行結果を消したり再実行したりせず、ソース位置を確認してください。')
sys.path[:0] = [str(WORK), str(BASE), str(BASE / 'src')]
import pybullet as p
import run_stage25_v45_centerfirst as previous
import run_physics as rp
from run_stage03 import plan
from tray_control import tray_occupied, next_free_nest
from stage23_full_supervisor import FullGradeSupervisor

NOW = lambda: datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y/%m/%d %H:%M JST')
SEQUENCE = ['A', 'B', 'C', 'D']
EXPECTED = ['A3', 'B3', 'C1', 'D1']
SETTINGS = [('reference_065', 0.65), ('return_only_055', 0.55)]
RUN_BASE = BASE / 'output' / 'stage26_v46a_return_one_factor'
ARCHIVE = BASE / 'apple_stage26_v4.6a_results.zip'


def occupied(scene):
    return {name: bool(tray_occupied(p, floor, scene['robot']))
            for name, floor in scene['tray_floors'].items()}


def worker_c_scripted_clear(scene, expected_slot):
    """Delete only the *already placed* apple; prove all other sensors stable.

    This is a scripted handoff between physics episodes, NOT simulated human
    motion and not part of reported robot cycle time.
    """
    before = occupied(scene)
    assert before[expected_slot] and expected_slot in scene['placed_fruits'], \
        '回収するはずの果実がありません: ' + expected_slot
    id_ = scene['placed_fruits'][expected_slot]
    p.removeBody(id_)
    del scene['placed_fruits'][expected_slot]
    for _ in range(12):
        p.stepSimulation()
    after = occupied(scene)
    assert not after[expected_slot], '荷重センサーの空き確認失敗: ' + expected_slot
    for k, v in before.items():
        if k != expected_slot and v:
            assert after[k], 'ほかの受け台の在荷が変化: ' + k
    return {'event': 'WORKER_C_SCRIPTED_CLEAR', 'cleared': expected_slot,
            'sensor_before': before, 'sensor_after': after,
            'not_human_animation': True}


def make_zip():
    """Always include partial logs if an experiment fails, never overwrite files."""
    if ARCHIVE.exists():
        print('既存の結果ZIPがあります。上書きしません:', ARCHIVE, flush=True)
        return
    tmp = ARCHIVE.with_suffix('.zip.partial')
    if tmp.exists():
        raise RuntimeError('以前の一時ZIPが残っています。上書きしません: ' + str(tmp))
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(RUN_BASE.rglob('*')):
            if f.is_file():
                z.write(f, arcname=str(f.relative_to(BASE)))
        z.write(__file__, arcname='CODE_USED/stage26_v46a_normal_operation_return_trial.py')
        z.writestr('PROVENANCE.txt', NOW() + '\nStage26 v4.6a matched one-factor return timing; '
                   'synthetic A/B/C/D; scripted worker C removal between episodes; '
                   'camera and physical motion original; no orchard throughput claim.\n')
    with zipfile.ZipFile(tmp) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError('ZIP integrity failure: ' + bad)
    tmp.rename(ARCHIVE)
    print('ZIP EXISTS:', ARCHIVE.is_file(), 'ZIP_BYTES:', ARCHIVE.stat().st_size,
          'ZIP_SHA256:', hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(), flush=True)
    print('ZIP_PATH:', ARCHIVE, flush=True)


def run_variant(title, return_scale):
    dst = RUN_BASE / title
    sf = dst / 'summary.json'
    if sf.is_file():
        obj = json.loads(sf.read_text(encoding='utf-8'))
        print('既存の確定結果を再利用（再実験しません）:', title, flush=True)
        return obj
    if dst.exists():
        raise RuntimeError('途中の試験フォルダが残っています。上書きせず停止: ' + str(dst))
    dst.mkdir(parents=True)
    old_policy = rp.MOTION_SETTINGS['settled_gate_09']
    old_order = os.environ.get('APPLE_NEST_ORDER')
    old_ramp = os.environ.get('STAGE20_V40D_LOWER_RAMP_S')
    old_return = os.environ.get('STAGE22_V42_RETURN_SCALE')
    had_hook = hasattr(rp, 'STAGE23_AFTER_INFERENCE_HOOK')
    old_hook = getattr(rp, 'STAGE23_AFTER_INFERENCE_HOOK', None)
    cfg0 = next(cfg for _, g, cfg in plan('settled_gate_09') if g == 'A')
    records, clears, err = [], [], None
    scene, previous_slot = None, None
    supervisor = FullGradeSupervisor()
    try:
        os.environ['APPLE_NEST_ORDER'] = 'nearest'
        os.environ['STAGE20_V40D_LOWER_RAMP_S'] = '0.08'
        os.environ['STAGE22_V42_RETURN_SCALE'] = str(return_scale)
        rp.MOTION_SETTINGS['settled_gate_09'] = replace(old_policy, x_effective_rate=1.45)
        rp.STAGE23_AFTER_INFERENCE_HOOK = supervisor.process
        connection = p.connect(p.DIRECT)
        if connection < 0:
            raise RuntimeError('PyBullet DIRECT 接続失敗')
        for i, (grade, slot) in enumerate(zip(SEQUENCE, EXPECTED), start=1):
            if scene is not None:
                clear = worker_c_scripted_clear(scene, previous_slot)
                clears.append(clear)
                print('WORKER_C_CLEAR', clear['cleared'], 'sensor free True', flush=True)
            cfg = replace(cfg0, profile='nominal', replicate=600 + i,
                          scan_duration_s=0.32, torque_Nm=12.0)
            assert next_free_nest(grade, (occupied(scene) if scene else
                    {f'{g}{j}': False for g in 'ABCD' for j in (1, 2, 3)})) == slot
            print(f'TRIAL {title} {i}/4 expected {grade}->{slot}', flush=True)
            ev_before = len(supervisor.events)
            row, scene = rp.run_trial(p, grade, dst / 'episodes', record_video=False,
                                     cfg=cfg, persistent_scene=scene, return_scene=True)
            evts = supervisor.events[ev_before:]
            measured = row.get('simulated_time_s')
            good = (row.get('classification', {}).get('class') == grade
                    and row.get('assigned_nest') == slot
                    and bool(row.get('success_physical_and_ai'))
                    and bool(row.get('all_retained_fruits_ok'))
                    and bool(row.get('return_home_pass'))
                    and bool((row.get('post_return_check') or {}).get('ok'))
                    and not any(e.get('event') == 'THREE_FULL_STOP' for e in evts))
            record = {'episode': i, 'grade': grade, 'expected_slot': slot,
                      'actual_slot': row.get('assigned_nest'),
                      'success': bool(good), 'cycle_s': measured,
                      'stages_s': row.get('simulation_stage_durations_s'),
                      'motor_return_home_x_m': row.get('stage_x_home_final_m'),
                      'post_return_verified': bool((row.get('post_return_check') or {}).get('ok')),
                      'events': [e.get('event') for e in evts]}
            records.append(record)
            previous_slot = slot
            (dst / 'progress.json').write_text(json.dumps(
                {'jst': NOW(), 'reference_return_scale': return_scale,
                 'records': records, 'clears': clears}, ensure_ascii=False, indent=2), encoding='utf8')
            print('RESULT', title, grade, slot, good, f'{measured:.6f}s', flush=True)
            if not good:
                print('FAILED physical gate; stop this variant', flush=True)
                break
    except Exception as e:
        err = repr(e)
        (dst / 'error_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        print('ERROR', title, err, flush=True)
    finally:
        if p.isConnected():
            p.disconnect()
        rp.MOTION_SETTINGS['settled_gate_09'] = old_policy
        for k, old in [('APPLE_NEST_ORDER', old_order),
                       ('STAGE20_V40D_LOWER_RAMP_S', old_ramp),
                       ('STAGE22_V42_RETURN_SCALE', old_return)]:
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old
        if had_hook:
            rp.STAGE23_AFTER_INFERENCE_HOOK = old_hook
        elif hasattr(rp, 'STAGE23_AFTER_INFERENCE_HOOK'):
            del rp.STAGE23_AFTER_INFERENCE_HOOK
    success = len(records) == 4 and all(x['success'] for x in records)
    summary = {'jst': NOW(), 'stage': '26-v4.6a', 'condition': title,
               'changed_parameter_only': 'STAGE22_V42_RETURN_SCALE',
               'return_scale': return_scale, 'sequence': SEQUENCE,
               'expected_destinations': EXPECTED, 'records': records,
               'worker_c_scripted_clears': clears,
               'all_success': success,
               'mean_cycle_s': statistics.mean([x['cycle_s'] for x in records]) if success else None,
               'failed_or_exception': err,
               'interpretation': 'Synthetic mixed-grade nearest free nest, 4 separate physical episodes in persistent scene; '
                                 'scripted fruit removal between episodes, not a physically modeled human; '
                                 'worker time excluded from per-apple cycle.'}
    sf.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
    return summary


def main():
    previous.verify()  # Sha-guard for original physical source, 12 physical nest geometry, AI control.
    if ARCHIVE.is_file():
        with zipfile.ZipFile(ARCHIVE) as z:
            assert z.testzip() is None, '既存ZIPが破損。上書きせず停止'
        print('結果ZIPはすでに存在。物理試験を再実行しません:', ARCHIVE, flush=True)
        return
    overall = []
    for title, value in SETTINGS:
        result = run_variant(title, value)
        overall.append(result)
        if not result['all_success']:
            print('物理検証で不成立。次の速度条件は実行しません。', flush=True)
            break
    combined = {'jst': NOW(), 'stage': '26-v4.6a', 'conditions': overall,
                'pilot': True, 'baseline_stage25_A_only_mean_s': 3.3341666666666665,
                'baseline_comparable_only_to_same_stage26_cases': True,
                'worker_C_measured_time_included': False,
                'further_trials_required': True}
    RUN_BASE.mkdir(parents=True, exist_ok=True)
    (RUN_BASE / 'comparison_summary.json').write_text(json.dumps(
        combined, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n=== STAGE26 TEST SUMMARY ===', flush=True)
    for c in overall:
        print(c['condition'], 'PASS', c['all_success'],
              'MEAN_CYCLE_S', c['mean_cycle_s'], 'N', len(c['records']), flush=True)
    if len(overall) == 2 and all(c['all_success'] for c in overall):
        base_mean, cand_mean = (c['mean_cycle_s'] for c in overall)
        print('MATCHED_DELTA_CANDIDATE_MINUS_REFERENCE_S', round(cand_mean - base_mean, 6), flush=True)
        print('GOAL <= 3.0S', cand_mean <= 3.0, 'REFERENCE_GOAL', base_mean <= 3.0, flush=True)
    make_zip()
    print('PyBullet全試験後の検証ログをChatGPTに貼れば、ZIPが転送できなくても評価可能です。', flush=True)


if __name__ == '__main__':
    main()
