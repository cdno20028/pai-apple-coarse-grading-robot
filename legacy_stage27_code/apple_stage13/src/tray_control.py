"""v2.2 human-assisted grading stations: each A/B/C handoff tray has a virtual
load/occupancy switch represented ONLY by physical floor-contact impulses.
No simulator ground-truth fruit position is read by this controller.
"""
from __future__ import annotations
import os
from workbench_geometry import LAYOUT

TRAY_NAMES=tuple(f'{g}{i}' for g in 'ABCD' for i in (1,2,3))


def supported_load_newtons(contacts, tray_body_id, ignored_body_ids=()):
    """Measure dynamic load on the tray top using Bullet contact-force fields.

    `contacts` is p.getContactPoints(bodyA=tray_body_id).  In the physical
    apparatus this input corresponds to an independent load/presence sensor;
    it is *not* an ML decision or a safety-rated light curtain.
    """
    ignored=set(ignored_body_ids)
    total=0.
    for cp in contacts:
        if cp[1]!=tray_body_id:
            continue
        other=cp[2]
        if other in ignored or other == tray_body_id:
            continue
        total+=max(0.,float(cp[9]))
    return total


def tray_occupied(p, tray_body_id, robot_body_id, min_load_newtons=0.15):
    """Read the tray's modeled presence sensor, not hidden apple positions."""
    return supported_load_newtons(
        p.getContactPoints(bodyA=tray_body_id),tray_body_id,(robot_body_id,)
    ) >= min_load_newtons


def station_ready(occupied_by_tray):
    """Conservative interlock: don't begin the next apple until ALL are clear.

    This prevents holding a newly gripped apple during a human container swap.
    It deliberately trades throughput for simplicity and safety.
    """
    missing=set(TRAY_NAMES)-set(occupied_by_tray)
    if missing:
        raise ValueError(f'missing station(s): {sorted(missing)}')
    return not any(bool(occupied_by_tray[key]) for key in TRAY_NAMES)


def placement_allowed(destination, occupied_by_tray):
    return destination in TRAY_NAMES and not bool(occupied_by_tray[destination])


def next_free_nest(grade, occupied_by_tray):
    """Pick grade's first physically empty nest. Never reuse an occupied nest.

    Caller MUST pass a fresh per-nest physical presence-sensor reading.
    Only the inference result may supply `grade`.
    """
    if not isinstance(grade,str) or len(grade)!=1 or grade not in 'ABCD':
        return None
    missing=set(TRAY_NAMES)-set(occupied_by_tray)
    if missing:
        raise ValueError(f'missing station(s): {sorted(missing)}')
    slots=[f'{grade}{i}' for i in (1,2,3)]
    # Stage25 opt-in: select physically nearest empty station to the scale.
    # Existing station names / locations remain unchanged.
    if os.environ.get('APPLE_NEST_ORDER','legacy') == 'nearest':
        slots.sort(key=lambda name:(abs(LAYOUT.nest_centres_x_m[name]),name))
    return next((name for name in slots if not bool(occupied_by_tray[name])),None)
