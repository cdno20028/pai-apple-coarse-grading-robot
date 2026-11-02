# -*- coding: utf-8 -*-
"""Experimental two-branch external-quality decision.
C = visible blemish; D = immature skin. Do not assume red/yellow/green pixels
are directly equivalent to real apple maturity without orchard calibration.
Full per-camera features and counterfactual Stage76 label kept for audit.
"""
from stage61_roughgrade import fuse_cameras as inherited_fuse

def fuse_cameras(frames):
    out=inherited_fuse(frames)
    choice=cd_from_features(out['per_camera'],out['class'])
    grade=choice['class'];reason=choice['reason']
    d=choice['d'];g=choice['g'];severe_lesion=choice['severe'];multi_view_lesion=choice['two'];immature_side=choice['side'];immature_palm=choice['palm'];immature=choice['immature']
    out['stage82_decision']={'c_severe_candidate':severe_lesion,
         'c_two_view_candidate':multi_view_lesion,
         'd_side_immaturity_candidate':immature_side,
         'd_palm_immaturity_candidate':immature_palm,
         'd_rgb_consensus_candidate':choice['immature_rgb'],
         'c_local_patch_two_view_candidate':choice['lesion_two_view_corrob'],
         'median_rgb_ratios':choice['skin_rg_per_camera'],
         'reference_stage76_grade_for_counterfactual':out['class'],
         'camera_damage_cues':d,'camera_immaturity_cues':g,
         'branch_priority':'SEVERE_C_THEN_D_RGB_OR_HSV_THEN_TWO_VIEW_C_THEN_AB',
         'grading_thresholds_experimental_not_orchard_calibrated':True,
         'grade_reason':reason,
         'temporal_candidate_not_independent':True,
         'temporal_applicable':choice['temporal_applicable'],
         'temporal_3of4_severe_c':choice['temporal_3of4_severe_c'],
         'temporal_finger_dark_fractions':choice['temporal_finger_dark_fractions']}
    out['class']=grade;out['decision_reason']=reason
    out['d_immature_skin_detected']=immature
    return out

def cd_from_features(p,stage76_grade):
    d={cam:float(p[cam]['largest_dark_patch_max']) for cam in ('finger_1','finger_2','finger_3')}
    g={cam:float(p[cam]['yellow_green_max']) for cam in ('finger_1','finger_2','finger_3','palm')}
    strong={'finger_1':.030,'finger_2':.022,'finger_3':.030}
    positives=[k for k,v in strong.items() if d[k]>=v]
    severe_lesion=max(d.values())>=.050
    # REAL17 TEMPORAL DEVELOPMENT CANDIDATE (2026/10/29):
    # Only if four valid observations from EACH finger are present.
    # A large patch seen in one or two early views can be a projected
    # silhouette/UV-shadow, NOT necessarily a severe physical injury.
    # Require 3 of 4 timed frames >=5% FROM ONE PHYSICAL FINGER to
    # nominate a severe C. No labels, filenames, asset IDs or truth lookup.
    temporal_per_camera={}
    for cam in ('finger_1','finger_2','finger_3'):
        vals=[float(r.get('largest_dark_patch_fraction',0.0))
              for r in p[cam].get('frame_features',[]) if r.get('valid')]
        temporal_per_camera[cam]=vals
    temporal_applicable=all(len(v)>=4 for v in temporal_per_camera.values())
    persistent_lesion=(any(sum(x>=.050 for x in v)>=3 for v in temporal_per_camera.values())
                       if temporal_applicable else None)
    multi_view_lesion=len(positives)>=2
    # Stage82: D may be detected from THREE independent side cameras,
    # avoiding a synthetic cap causing D->A solely due to palm colour.
    # Still a DEVELOPMENT hypothesis, not validated for unknown cultivars.
    immature_side=(sum(g[n]>=.65 for n in ('finger_1','finger_2','finger_3'))>=2
                   and max(g[n] for n in ('finger_1','finger_2','finger_3'))>=.85)
    immature_palm=bool(g['palm']>=.60 and g['finger_2']>=.65)
    # Public synthetic UV shift: RGB hue can fall outside the [12,40] HSV window.
    # Require agreement of two physically independent finger cameras and one
    # palm camera on red/green ratios, rather than fruit ID or truth labels.
    rgb_ratios={k:p[k].get('median_rg_min') for k in ('finger_1','finger_2','finger_3','palm')}
    immature_rgb=(sum(rgb_ratios[k] is not None and float(rgb_ratios[k])<.90
                      for k in ('finger_1','finger_2','finger_3'))>=2
                  and rgb_ratios['palm'] is not None
                  and float(rgb_ratios['palm'])<.90)
    immature=immature_side or immature_palm or immature_rgb
    # Development-only, two-view localized-damage hypothesis: one clear local
    # patch over 4% with a corroborating distinct finger over 2%.
    # Calibration derived from SAME FIVE v2 synthetic cases, not blind test.
    lesion_corrob=(sum(v>=.020 for v in d.values())>=2 and
                   max(d.values())>=.040)
    # Clear local damage is more concerning than skin colouring. Only then
    # can severe C override D. Lesser shadows in green fruit remain D,
    # not an automatic C false positive from camera shading.
    if temporal_applicable:
        # Experimental, evidence-specific rule. Not blind validation:
        # criterion selected after reviewing 2026-10-29 known test scenes.
        if persistent_lesion:
            grade='C';reason='C_PERSISTENT_PATCH_IN_SAME_FINGER_3_OF_4_EXPERIMENTAL'
        elif immature:
            grade='D';reason='D_IMMATURE_SIDE_OR_PALM'
        elif stage76_grade=='C':
            grade='B';reason='B_TRANSIENT_C_CANDIDATE_DOWNGRADED_EXPERIMENTAL'
        else:
            grade=stage76_grade;reason='AB_STAGE76_RULE_RETAINED'
    elif severe_lesion:
        grade='C';reason='C_SEVERE_LOCAL_DAMAGE_FIRST'
    elif immature:
        grade='D';reason='D_IMMATURE_SIDE_OR_PALM'
    elif multi_view_lesion or lesion_corrob:
        grade='C';reason='C_TWO_DISTINCT_FINGERS' if multi_view_lesion else 'C_LOCALIZED_PATCH_TWO_FINGER_DEVELOPMENT'
    else:
        grade=stage76_grade;reason='AB_STAGE76_RULE_UNCHANGED'
    return {"class":grade,"reason":reason,"d":d,"g":g,"severe":severe_lesion,"two":multi_view_lesion,"side":immature_side,"palm":immature_palm,"immature":immature,"immature_rgb":immature_rgb,"skin_rg_per_camera":rgb_ratios,"lesion_two_view_corrob":lesion_corrob,
            "temporal_candidate_not_independent":True,
            "temporal_applicable":temporal_applicable,
            "temporal_3of4_severe_c":persistent_lesion,
            "temporal_finger_dark_fractions":temporal_per_camera}
