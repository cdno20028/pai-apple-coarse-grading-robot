#!/usr/bin/env python3
"""Stage25 near-first allocation tests without PyBullet.
Original geometry, finger support, and all 12 station IDs are unchanged."""
import os,sys
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'apple_stage13/src'))
from workbench_geometry import LAYOUT
from tray_control import TRAY_NAMES,next_free_nest
import stage23_full_supervisor as sp

assert LAYOUT.validate()
assert len(LAYOUT.nest_centres_x_m)==12
for mode,expected in [
    ('legacy',{'A':['A1','A2','A3'],'B':['B1','B2','B3'],'C':['C1','C2','C3'],'D':['D1','D2','D3']}),
    ('nearest',{'A':['A3','A2','A1'],'B':['B3','B2','B1'],'C':['C1','C2','C3'],'D':['D1','D2','D3']})]:
    os.environ['APPLE_NEST_ORDER']=mode
    for grade,order in expected.items():
        occ={key:False for key in TRAY_NAMES}
        for slot in order:
            assert next_free_nest(grade,occ)==slot,(grade,mode,slot)
            occ[slot]=True
        assert next_free_nest(grade,occ) is None
        occ[order[0]]=False
        assert next_free_nest(grade,occ)==order[0]
    print('PASS',mode, expected)

# Simulated human clear and stop/resume for all 4 grades; no fruit teleportation.
os.environ['APPLE_NEST_ORDER']='nearest'
class P:
    def __init__(self,grade):
        self.grade=grade;self.occupied={name:False for name in TRAY_NAMES}
        self.removed=[]
    def removeBody(self,body):
        self.removed.append(body)
        self.occupied[self.slot_by_body[body]]=False
for grade in 'ABCD':
    p=P(grade)
    nests=[f'{grade}{j}' for j in (1,2,3)]
    p.occupied.update({k:True for k in nests})
    scene={'robot':99,'tray_floors':{k:k for k in TRAY_NAMES},
           'placed_fruits':{name:100+j for j,name in enumerate(nests)}}
    p.slot_by_body={v:k for k,v in scene['placed_fruits'].items()}
    old=sp.tray_occupied
    sp.tray_occupied=lambda p,n,robot:p.occupied[n]
    try:
        supervisor=sp.FullGradeSupervisor()
        supervisor.process(p=p,scene=scene,grade=grade,
          inference={'status':'PREDICTED','class':grade,'confidence':1.},
          robot=99,apple=200,hold_step=lambda:3,pre_states=dict(p.occupied))
        nearest=min(nests,key=lambda x:abs(LAYOUT.nest_centres_x_m[x]))
        assert p.removed==[100+nests.index(nearest)]
        assert p.occupied[nearest] is False
        assert supervisor.events[-1]['event']=='RESUME' and supervisor.events[-1]['destination']==nearest
        assert sum(e['event']=='THREE_FULL_STOP' for e in supervisor.events)==1
        print('PASS supervisor',grade,'nearest',nearest)
    finally: sp.tray_occupied=old
os.environ.pop('APPLE_NEST_ORDER',None)
