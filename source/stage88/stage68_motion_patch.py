# -*- coding: utf-8 -*-
"""Stage68: couple Z motion with wrist IK to request a monotone camera approach.

The combined target finger CAMERA height follows a linear 25mm descent through
0..scan_duration, instead of two independent interpolation profiles for
stage_z and wrist_z. Real link motion is separately audited in every tick.
"""
from pathlib import Path
import hashlib,json,datetime
UTC9=datetime.timezone(datetime.timedelta(hours=9))

def apply_stage68_motion_patch(engine,out):
    path=Path(engine)/'apple_stage13/src/run_physics.py'
    original=path.read_text(encoding='utf8')
    old="        _s62_command['stage_z_move']=G.root_z+.044*(1.-smootherstep(_s62_fraction))"
    new=(
        "        # Stage68: wrist z rises 19mm during scan; compensate in root Z.\n"
        "        # The physical *requested camera-tip height* therefore descends\n"
        "        # 25mm without a target-position reversal.\n"
        "        _s68_wrist_z=wrist_target_with_close_duration(t,cfg)[1]\n"
        "        _s68_cam_target_z=G.scan_start_z+.044-.025*_s62_fraction\n"
        "        _s62_command['stage_z_move']=(G.root_z+_s68_cam_target_z-_s68_wrist_z)"
    )
    if original.count(old)!=1:
        raise RuntimeError('STAGE68_EXACT_ENGINE_ANCHOR_MISMATCH')
    updated=original.replace(old,new,1)
    if 'stage59_up_step' in updated or '_s59_up_step' in updated:
        raise RuntimeError('STAGE59_BOB_REINTRODUCED')
    for guard in ('STAGE32_REJECTED_BEFORE_DESCENT','_s66_contact_ready',
                  'gravity_lift','return_home_pass','STAGE23_AFTER_INFERENCE_HOOK'):
        if guard not in updated:raise RuntimeError('MISSING_SAFETY_BLOCK '+guard)
    compile(updated,str(path),'exec')
    path.write_text(updated,encoding='utf8')
    result={'stage':68,'time_jst':datetime.datetime.now(UTC9).strftime('%Y/%m/%d %H:%M:%S JST'),
        'before_sha256':hashlib.sha256(original.encode()).hexdigest(),
        'after_sha256':hashlib.sha256(updated.encode()).hexdigest(),
        'scan_start_root_offset_m':.044,'scan_end_root_offset_m':0.,
        'wrist_local_height_start_m':.078,'wrist_local_height_end_m':.097,
        'requested_camera_total_descent_m':.025,
        'motor_root_z_target_linked_to_wrist_z':True,
        'actual_motion_must_be_measured':True,
        'scanning_camera_convergence_untested_in_PyBullet':True,
        'no_motion_in_untouched_sources':True}
    (Path(out)/'stage68_motion_patch_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    return result
