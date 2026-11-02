# -*- coding: utf-8 -*-
"""Stage80: keep physical-world accounting independent of reference-label agreement.

This fixes Stage79's case02 physically supported A2 fruit not being registered
solely because its AI grade differed from the reference B. No grade, motion,
collision, sensor or worker controller rules are changed. Fail-closed physical
placement checks precede registration.
"""
from pathlib import Path
import hashlib,json,datetime

OLD = """        if report['success_physical_and_ai'] and dest is not None:
            scene['placed_fruits'][dest]=apple"""
NEW = """        # Stage80: keep the *real* Bullet workcell occupancy distinct from
        # agreement with the known photo/reference label. The model may make a
        # wrong AI decision, but a physically placed and verified apple exists.
        _s80_physically_placed=bool(
            dest is not None and dest in scene['tray_floors'] and
            report['lift_pass'] and report['sort_pass'] and
            report['return_home_pass'] and padcount >= 2 and
            report['placement'].get('verified_support') and
            report['placement'].get('settled_on_floor') and
            (report.get('post_return_check') or {}).get('ok') and
            report.get('no_drop_release'))
        report['stage80_physical_registry_verified']=_s80_physically_placed
        report['stage80_grade_agreed_with_reference']=bool(
            result.get('class')==label)
        if _s80_physically_placed:
            if dest in scene['placed_fruits'] and scene['placed_fruits'][dest]!=apple:
                raise RuntimeError('STAGE80_NEST_BODY_DUPLICATE:'+str(dest))
            scene['placed_fruits'][dest]=apple
        # success_physical_and_ai remains the original STRICT result; never
        # relabel or re-route a fruit to conceal an AI grade disagreement."""

def patched_text(source):
    if source.count(OLD)!=1:
        raise RuntimeError('STAGE80_REGISTRY_ANCHOR_MISMATCH')
    txt=source.replace(OLD,NEW,1)
    compile(txt,'run_physics.py','exec')
    assert txt.count('stage80_physical_registry_verified')==1
    return txt

def apply_stage80_physical_registry(engine,result_dir):
    engine=Path(engine);result_dir=Path(result_dir)
    target=engine/'apple_stage13'/'src'/'run_physics.py'
    original=target.read_text(encoding='utf8')
    patched=patched_text(original)
    target.write_text(patched,encoding='utf8')
    def sha(t):return hashlib.sha256(t.encode('utf8')).hexdigest()
    record={'stage':80,'note':'physical scene accounting only, not AI or motion',
       'before_sha256':sha(original),'after_sha256':sha(patched),
       'registers_only_if_lift_sort_home_support_settle_postreturn_confirmed':True,
       'strict_grade_reference_agreement_still_required_for_success':True,
       'worker_C_supervisor_safety_conditions_unchanged':True,
       'run_in_PyBullet_not_yet_verified':True}
    (result_dir/'stage80_registry_patch_manifest.json').write_text(
       json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
    return record
