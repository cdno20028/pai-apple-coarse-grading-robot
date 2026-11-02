# -*- coding: utf-8 -*-
'''Stage66: correct mismatched pre-lift contact and post-lift physical witness.

Retain the Stage51 grasp mechanism, trajectory, pressure diagnostics, lift/placement
safety interlocks, stage64 camera, Stage65 shared model.  Accept newly engaging
contacts only when at least two wrist links contact FOR CONSECUTIVE physics ticks
while the apple remains essentially stationary on the scale. Never treat
successful transport or the expected grade as proof of a good initial grip.
'''
import hashlib, json, datetime
from pathlib import Path


def apply_stage66_grasp_guard(engine, out):
    path=Path(engine)/'apple_stage13'/'src'/'run_physics.py'
    old=path.read_text(encoding='utf-8')
    anchor='    padcount,force,_=contacts(p,robot,apple,joints)\n    z0=p.getBasePositionAndOrientation(apple)[0][2]\n'
    assert old.count(anchor)==1, 'STAGE66_PRELIFT_CONTACT_ANCHOR_MISMATCH'
    replace='''    padcount,force,_=contacts(p,robot,apple,joints)
    # Stage66: preserve unmodified PRE-LIFT padcount as a literal observation.
    # Some successful Stage65 trials engage a two-link grip during the initial
    # stationary-fruit segment of the commanded lift. Measure this, do NOT fake
    # pre-lift contact and do NOT ignore missing initial contact.
    _s66_contact_ready=bool(padcount>=2 and force>=1.0)
    _s66_evidence_source='BEFORE_LIFT' if _s66_contact_ready else None
    _s66_contact_streak=0
    _s66_evidence=[]
    _s66_early_verified_sample=None
    z0=p.getBasePositionAndOrientation(apple)[0][2]
'''
    cur=old.replace(anchor,replace,1)
    anchor2="        send(cmd);step_and_trace('gravity_lift')\n"
    assert cur.count(anchor2)==1, 'STAGE66_GRAVITY_LIFT_ANCHOR_MISMATCH'
    insert='''        send(cmd);step_and_trace('gravity_lift')
        # A contact is credible only if measured BEFORE the fruit starts
        # lifting off its input support, NOT after it is already airborne.
        _s66_dz=float(p.getBasePositionAndOrientation(apple)[0][2]-z0)
        _s66_links,_s66_normal,_=contacts(p,robot,apple,joints)
        if not _s66_contact_ready and _s66_dz<=.002:
            _s66_contact_streak=(_s66_contact_streak+1 if
                 _s66_links>=2 and _s66_normal>=1.0 else 0)
            if _s66_contact_streak>=2:
                _s66_contact_ready=True
                _s66_evidence_source='LIFT_ENTRY_BEFORE_FRUIT_MOVEMENT'
                _s66_early_verified_sample={'step':i,'delta_z_m':_s66_dz,
                    'distinct_wrist_links':_s66_links,'wrist_normal_sum_N':_s66_normal,
                    'consecutive_ticks':_s66_contact_streak}
        if i<24:
            _s66_evidence.append({'step':i,'fruit_delta_z_m':round(_s66_dz,7),
                 'wrist_link_contacts':_s66_links,
                 'normal_sum_N':round(_s66_normal,4)})
        if _s66_dz>.003 and not _s66_contact_ready:
            (loc/'stage66_grasp_guard_failure.json').write_text(json.dumps({
                'reason':'CONTACT_NOT_ESTABLISHED_BEFORE_FRUIT_MOVEMENT',
                'prelift_contacts':padcount,'prelift_wrist_force_N':force,
                'fruit_delta_z_m':_s66_dz,'first_24_ticks':_s66_evidence},
                indent=2,ensure_ascii=False),encoding='utf-8')
            raise RuntimeError('STAGE66_ABORT_NO_2_LINK_GRIP_BEFORE_FRUIT_DEPARTURE')
'''
    cur=cur.replace(anchor2,insert,1)
    anchor3="    lift_pass=(z1-z0)>.05 and diagnostics['lift_multi_contact_samples']>=int(diagnostics['lift_samples']*.30)\n"
    assert cur.count(anchor3)==1, 'STAGE66_LIFT_WITNESS_ANCHOR_MISMATCH'
    cur=cur.replace(anchor3,"    lift_pass=(z1-z0)>.05 and diagnostics['lift_multi_contact_samples']>=int(diagnostics['lift_samples']*.30) and _s66_contact_ready\n",1)
    anchor4="       'contact_pad_count_before_lift':padcount,'normal_force_N_before_lift':force,\n"
    assert cur.count(anchor4)==1,'STAGE66_REPORT_WITNESS_ANCHOR_MISMATCH'
    cur=cur.replace(anchor4,anchor4+"       'stage66_grasp_contact_witness':{'pre_lift_padcount_observed':padcount,\n         'pre_lift_force_N_observed':force,'contact_ready_before_fruit_departure':_s66_contact_ready,\n         'contact_origin':_s66_evidence_source,'first_24_lift_ticks':_s66_evidence,\n         'early_witness':_s66_early_verified_sample,\n         'require_2_consecutive_ticks_and_2_links_before_2mm_lift':True},\n",1)
    anchor5="and padcount>=2 and result.get('class')==label),"
    assert cur.count(anchor5)==1,'STAGE66_SUCCESS_CONDITION_MISMATCH'
    cur=cur.replace(anchor5,"and _s66_contact_ready and result.get('class')==label),",1)
    # No other patch alters scene assignment, grade inference, destination,
    # occupancy, held-fruit collisions, or stops/worker replay.
    assert cur.count('STAGE32_REJECTED_BEFORE_DESCENT')==1
    assert cur.count('STAGE23_AFTER_INFERENCE_HOOK')>=1
    compile(cur,str(path),'exec')
    path.write_text(cur,encoding='utf-8')
    hh=lambda s:hashlib.sha256(s.encode()).hexdigest()
    manifest={'stage':66,'date_jst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
       'pre_patch_sha256':hh(old),'post_patch_sha256':hh(cur),'change':'grip witness and fail-closed early lift guard only',
       'prelift_padcount_is_preserved_verbatim':True,'guard_requires_two_real_wrist_links':True,
       'guard_requires_two_consecutive_ticks':True,'initial_2mm_contact_witness_and_abort_after_3mm_without_witness':True,
       'stage64_motion_unchanged':True,'stage65_object_gate_unchanged':True,
       'grade_dest_physical_control_unchanged':True,'simulation_test_unperformed':True}
    (Path(out)/'stage66_grasp_patch_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    return manifest
