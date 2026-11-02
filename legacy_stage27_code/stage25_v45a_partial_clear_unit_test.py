#!/usr/bin/env python3
"""Stage25 v4.5a offline occupancy/permutation tests. Does not claim physics success."""
import os,sys,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'apple_stage13'),str(ROOT/'apple_stage13/src')]
from tray_control import TRAY_NAMES,next_free_nest
from workbench_geometry import LAYOUT
import stage23_full_supervisor as sup

assert LAYOUT.validate()
assert len(TRAY_NAMES)==12

def occupied_with(grade,busy):
    occ={k:False for k in TRAY_NAMES}
    for slot in busy:occ[slot]=True
    return occ

os.environ['APPLE_NEST_ORDER']='nearest'
try:
    tested=0
    # Every possible occupancy combination, every grade.
    for grade in 'ABCD':
        slots=[f'{grade}{i}' for i in (1,2,3)]
        for flags in itertools.product([False,True],repeat=3):
            busy={name for name,f in zip(slots,flags) if f}
            occ=occupied_with(grade,busy)
            candidates=[name for name in slots if name not in busy]
            expected=(min(candidates,key=lambda n:(abs(LAYOUT.nest_centres_x_m[n]),n))
                      if candidates else None)
            got=next_free_nest(grade,occ)
            assert got==expected,(grade,busy,got,expected)
            if got is not None:assert not occ[got]
            tested+=1
        # Specifically: A1 and A3 stay occupied, C clears middle A2.
        occ=occupied_with(grade,[slots[0],slots[2]])
        assert next_free_nest(grade,occ)==slots[1]
        # A change between episodes is independent of which slot worker empties.
        for cleared in slots:
            occ=occupied_with(grade,slots)
            occ[cleared]=False
            assert next_free_nest(grade,occ)==cleared
            tested+=1

    # Test the actual supervisory logic with fake load-sensor readings:
    # one open middle slot must command placement, never full-stop.
    class FakeP:
        def __init__(self,states):self.states=states
        def removeBody(self,body):raise AssertionError('Unneeded human intervention: slot was open')
    old=sup.tray_occupied
    try:
        sup.tray_occupied=lambda p,body,robot:p.states[body]
        for grade in 'ABCD':
            states=occupied_with(grade,[f'{grade}1',f'{grade}3'])
            p=FakeP(states)
            scene={'robot':99,'tray_floors':{k:k for k in TRAY_NAMES},'placed_fruits':{}}
            ctrl=sup.FullGradeSupervisor()
            ctrl.process(p=p,scene=scene,grade=grade,
                inference={'status':'PREDICTED','class':grade,'confidence':1.,'scores':{grade:1.}},
                robot=99,apple=200,hold_step=lambda:3,pre_states=dict(states))
            assert [e['event'] for e in ctrl.events]==['AI_DECISION','GO_TO_NEST']
            assert ctrl.events[-1]['destination']==grade+'2'
            assert not any(e['event']=='THREE_FULL_STOP' for e in ctrl.events)
    finally:sup.tray_occupied=old
    print('PASS: 32 occupancy patterns + 12 arbitrary single-slot clear + 4 supervisor mid-slot tests;',
          tested,'placement checks; geometry unchanged.')
finally:
    os.environ.pop('APPLE_NEST_ORDER',None)
