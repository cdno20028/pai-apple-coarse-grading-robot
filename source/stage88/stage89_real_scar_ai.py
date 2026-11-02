# -*- coding: utf-8 -*-
"""Stage89: image-only, per-wound evidence from an ACTUAL-link camera.

All grades/cases/photo filenames are excluded from the image detector. This is
an experimental coarse visual *wound* heuristic. A single clearly visible
wound suffices to refer an apple to C; does not assert both source scars seen.
"""
from __future__ import annotations
import cv2
import numpy as np


def detect_dark_linear_wound(rgb):
    """Find a short, nearly horizontal, very dark skin incision.

    Evaluates full RGB, not the first-scan known-image LAB prototype.  The
    detector is uncalibrated for unseen orchard fruit.  The 12% upper exclusion
    avoids the calyx/stem zone; skin connected-component avoids blue gripper.
    """
    a = np.asarray(rgb, dtype=np.uint8)
    if a.ndim != 3 or a.shape[2] != 3 or min(a.shape[:2]) < 96:
        return {'detected': False, 'hits': [], 'reason': 'INVALID_RGB'}
    h, w = a.shape[:2]
    r, g, b = [a[:, :, k].astype(np.float32) for k in range(3)]
    skin = ((r > 60) & (r > g*1.15) & (r > b*1.10)).astype(np.uint8)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(skin, 8)
    if n <= 1:
        return {'detected': False, 'hits': [], 'reason': 'NO_FRUIT_SKIN'}
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    core = np.uint8(cc == idx)
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    core = cv2.erode(core, np.ones((5, 5), np.uint8))
    gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    dark = ((gray < 26) & (core > 0)).astype(np.uint8)
    n, component, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    hits = []
    for i in range(1, n):
        x, y, bw, bh, area = [int(t) for t in stats[i]]
        if y < int(0.12*h) or y > int(0.86*h):
            continue
        if bw < 7 or bw < 2.2*max(1, bh) or bh > 7 or area < 8:
            continue
        if x < 4 or x + bw > w - 4:
            continue
        region = (component == i)
        ring = cv2.dilate(region.astype(np.uint8), np.ones((15,15), np.uint8)) > 0
        ring &= ~ (cv2.dilate(region.astype(np.uint8), np.ones((3,3), np.uint8)) > 0)
        ring &= core > 0
        if int(ring.sum()) < 40:
            continue
        contrast = float(np.median(gray[ring]) - np.median(gray[region]))
        if contrast < 14:
            continue
        hits.append({'bbox_xyxy': [x, y, x+bw, y+bh],
                     'dark_pixel_area': area, 'aspect_ratio': round(bw/max(bh,1), 3),
                     'local_grayscale_contrast': round(contrast, 3)})
    return {'detected': bool(hits), 'hits': hits,
            'reason': 'DARK_LINEAR_WOUND_ON_RED_SKIN' if hits else 'NO_CONFIRMED_WOUND',
            'uses_case_or_photo_identifier': False, 'unknown_fruit_validation': False}


def grade_scar_evidence(frames):
    # Fixed optical pitch-20° for finger_2 throughout its continuous descent.
    # Only the TWO LATE live captures count; B stem/calyx, above the fruit,
    # must not become a false wound. No grades, filenames or photo IDs.
    observations = []
    for tick_index in (2, 3):
        img = frames.get(('finger',1,tick_index))
        result = detect_dark_linear_wound(img) if img is not None else {'detected':False,'hits':[], 'reason':'MISSING_LIVE_FRAME'}
        observations.append({'finger':'finger_2', 'capture_index': tick_index+1, **result})
    seen = [v for v in observations if v['detected']]
    return {'visible_wound':bool(seen), 'observations':observations,
            'sufficient_for_C_as_coarse_damage':bool(seen),
            'source_original_still_contains_two_scars':True,
            'both_source_scars_visible_in_runtime':False,
            'not_proof_of_generalization':True,
            'rule':'one clear physically photographed dark linear wound -> coarse C',
            'uses_case_or_photo_identifier':False}
