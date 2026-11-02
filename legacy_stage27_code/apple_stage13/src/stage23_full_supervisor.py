"""Stage23 v4.3: physically-sensed 3-slot pause/resume demonstration.

The classification MUST come from run_physics.infer(images, measured_mass).
Worker C is an EXTERNAL SCRIPTED EVENT, not a visually simulated person,
and robot hold does not imply safety certification.
"""
from __future__ import annotations
from dataclasses import dataclass,field
from typing import Callable
import os
from workbench_geometry import LAYOUT
from tray_control import next_free_nest,TRAY_NAMES,tray_occupied

@dataclass
class FullGradeSupervisor:
    events:list[dict]=field(default_factory=list)
    grade_counts:dict[str,int]=field(default_factory=lambda:{g:0 for g in 'ABCD'})
    simulated_worker_removals:int=0
    pause_steps:int=24
    def _event(self,kind,grade,**kw):
        e={'sequence':len(self.events)+1,'event':kind,'grade':grade,**kw}
        self.events.append(e)
        return e
    def reading(self,p,scene):
        robot=scene['robot']
        return {k:bool(tray_occupied(p,v,robot)) for k,v in scene['tray_floors'].items()}
    def process(self,*,p,scene,grade,inference,robot,apple,hold_step,pre_states):
        """Called AFTER real synthetic ML inference, BEFORE physical X transport.

        No fruit body/position used for AI destination: only predicted grade and
        separate load-sensor state. Previously placed IDs are used solely for
        scripted worker removal, not for AI inference.
        """
        if inference.get('status')!='PREDICTED' or grade not in 'ABCD':
            self._event('AI_REJECT',str(grade),status=inference.get('status'))
            return
        measured=self.reading(p,scene)
        if measured!=pre_states:
            self._event('SENSOR_REFRESH',grade,pre_states=pre_states,measured=measured)
        dest=next_free_nest(grade,measured)
        occupied=sum(bool(measured[f'{grade}{j}']) for j in (1,2,3))
        self._event('AI_DECISION',grade,confidence=inference.get('confidence'),
                    scores=inference.get('scores'),measured_mass_kg=inference.get('scale_mass_kg'),
                    occupied=occupied,capacity=3,destination=dest,
                    source='real camera+scale features; synthetic training data')
        if dest is not None:
            self._event('GO_TO_NEST',grade,destination=dest)
            return
        if occupied!=3:
            raise RuntimeError('FULL_INCONSISTENT_SENSOR_STATE: no destination but not 3 occupied')
        # Do not command transport to any occupied nest. Pause in lifted home pose.
        self._event('THREE_FULL_STOP',grade,occupied_slots=[f'{grade}{j}' for j in (1,2,3)],
                    robot_move_permitted=False,apple_held=True)
        self._event('HUMAN_ZONE_EVENT',grade,zone_occupied=True,
                    safety_note='virtual operator event; not a safety-rated detector')
        for _ in range(self.pause_steps):
            if hold_step()<2: raise RuntimeError('GRIP_SUPPORT_LOST_WHILE_FULL_STOP')
        # Abstract worker-C removes oldest fruit. This never teleports the NEW
        # held fruit and does not command robot to reach packing crates.
        # Oldest placed fruit for the chosen order is the first priority nest.
        # The stage25 opt-in uses the nearest nest; legacy keeps grade1.
        target=(min((f'{grade}{i}' for i in (1,2,3)),
                    key=lambda n:(abs(LAYOUT.nest_centres_x_m[n]),n))
                if os.environ.get('APPLE_NEST_ORDER','legacy')=='nearest'
                else f'{grade}1')
        if target not in scene['placed_fruits']:
            raise RuntimeError('WORKER_REMOVAL_BODY_UNAVAILABLE: '+target)
        old_id=scene['placed_fruits'][target]
        p.removeBody(old_id)
        scene['placed_fruits'].pop(target)
        self.simulated_worker_removals+=1
        self._event('WORKER_C_REMOVED',grade,nest=target,
                    operation='abstract removal of previously placed physical apple body')
        for _ in range(12):
            if hold_step()<2:raise RuntimeError('GRIP_SUPPORT_LOST_DURING_WORKER_EVENT')
        after=self.reading(p,scene)
        if after[target]:raise RuntimeError('WORKER_CLEAR_SENSOR_NOT_CONFIRMED')
        self._event('HUMAN_ZONE_EVENT',grade,zone_occupied=False)
        self._event('OPERATOR_RESET_ACK',grade,
                    operator_acknowledgement='scripted explicit reset event')
        dest=next_free_nest(grade,after)
        if dest!=target:raise RuntimeError('UNEXPECTED_DESTINATION_AFTER_WORKER_CLEAR')
        self._event('RESUME',grade,destination=dest,occupied_before=3,occupied_after=2,
                    motion_permitted=True)


def test_with_fake_physics():
    class FakeP:
        def __init__(self): self.active={j:False for j in TRAY_NAMES};self.deleted=[]
        def removeBody(self,body): self.deleted.append(body);self.active['A1']=False
    p=FakeP()
    p.active.update({'A1':True,'A2':True,'A3':True})
    scene={'robot':77,'tray_floors':{k:k for k in TRAY_NAMES},
           'placed_fruits':{'A1':11,'A2':12,'A3':13}}
    original=globals()['tray_occupied']
    globals()['tray_occupied']=lambda pp,b,robot:pp.active[b]
    try:
        s=FullGradeSupervisor()
        s.process(p=p,scene=scene,grade='A',inference={'status':'PREDICTED',
            'confidence':.96,'class':'A','scale_mass_kg':.18,'scores':{'A':.96}},
            robot=77,apple=8,hold_step=lambda:3,pre_states=dict(p.active))
        assert s.simulated_worker_removals==1
        assert p.deleted==[11]
        assert next_free_nest('A',p.active)=='A1'
        assert [e['event'] for e in s.events][-1]=='RESUME'
        assert [e['event'] for e in s.events].count('THREE_FULL_STOP')==1
        # Nonfull state must never remove old fruit.
        s.process(p=p,scene=scene,grade='B',inference={'status':'PREDICTED',
            'class':'B','scale_mass_kg':.22,'scores':{'B':1.}},
            robot=77,apple=9,hold_step=lambda:3,pre_states=dict(p.active))
        assert s.simulated_worker_removals==1
        assert [e['event'] for e in s.events][-1]=='GO_TO_NEST'
        for g in 'BC D'.replace(' ',''):
            for j in (1,2,3):p.active[f'{g}{j}']=True
            scene['placed_fruits'][f'{g}1']=101
            scene['placed_fruits'][f'{g}2']=102
            scene['placed_fruits'][f'{g}3']=103
            # Adapt fake body clearing for target grade.
            old_remove=p.removeBody
            p.removeBody=lambda body,grade=g: (p.deleted.append(body),p.active.__setitem__(f'{grade}1',False))
            s.process(p=p,scene=scene,grade=g,inference={'status':'PREDICTED',
                'class':g,'scale_mass_kg':.18},robot=77,apple=9,hold_step=lambda:3,
                pre_states=dict(p.active))
            p.removeBody=old_remove
            assert any(e['grade']==g and e['event']=='RESUME' for e in s.events)
        return True
    finally:
        globals()['tray_occupied']=original

if __name__=='__main__':
    assert test_with_fake_physics()
    print('PASS: four-grade occupancy, 3/3 stop, worker clear, ack, resume; no-full branch')
