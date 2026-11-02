"""Stage 05: shared, deterministic STATE-LOGIC demonstration (not safety-certified).

Accepts explicit simulated sensor/events, never reads hidden labels or simulates
real human detection. For final report, results are a logic/unit-test proof,
not physical worker protection or continuous PyBullet fruit-handling proof.

Japanese project timestamp: 2026/10/15 JST.
"""
from __future__ import annotations
from dataclasses import dataclass, field

GRADES = tuple('ABCD')
NESTS = tuple(f'{g}{i}' for g in GRADES for i in range(1, 4))

@dataclass
class InterlockController:
    state: str = 'WAIT_INPUT'
    human_in_work_zone: bool = False
    apple_on_scale: bool = False
    apple_grasped: bool = False
    grade_from_ai: str | None = None
    destination: str | None = None
    occupied: dict[str,bool] = field(default_factory=lambda:{s:False for s in NESTS})
    events: list[dict] = field(default_factory=list)
    _resume_state: str | None = None

    @property
    def motion_permitted(self) -> bool:
        return self.state not in ('WAIT_INPUT','WAIT_FULL','HOLD_FULL','STOP_HUMAN') and not self.human_in_work_zone

    def _log(self,event:str,detail:str='') -> str:
        self.events.append({'index':len(self.events)+1,'event':event,'state':self.state,
                            'destination':self.destination,'grasped':self.apple_grasped,
                            'motion_permitted':self.motion_permitted,'detail':detail})
        return self.state

    def _free_nest(self,grade:str) -> str | None:
        return next((f'{grade}{i}' for i in (1,2,3) if not self.occupied[f'{grade}{i}']),None)

    def place_on_scale(self):
        if self.apple_on_scale or self.apple_grasped:
            raise RuntimeError('fruit already in system')
        if self.state!='WAIT_INPUT': raise RuntimeError('not waiting for fruit')
        self.apple_on_scale=True;self.state='OBSERVE'
        return self._log('SCALE_OCCUPIED','Fruit presence sensor triggers observation')

    def classify(self,grade:str):
        if self.state!='OBSERVE' or grade not in GRADES:raise ValueError('classification unavailable')
        # Grade MUST come from the actual external inference result, never the case's ground truth.
        self.grade_from_ai=grade
        self.destination=self._free_nest(grade)
        self.state='GRASP' if self.destination else 'WAIT_FULL'
        return self._log('AI_GRADE',f'AI inference grade={grade}')

    def grasp_done(self):
        if self.state!='GRASP' or not self.apple_on_scale:raise RuntimeError('cannot grasp')
        self.apple_on_scale=False;self.apple_grasped=True
        self.state='MOVE' if self.destination and not self.occupied[self.destination] else 'HOLD_FULL'
        return self._log('GRASPED','Fruit now carried by gripper')

    def external_nest_occupied(self,nest:str):
        if nest not in NESTS:raise ValueError('unknown nest')
        self.occupied[nest]=True
        if self.apple_grasped and self.grade_from_ai==nest[0] and self.destination==nest:
            self.destination=self._free_nest(nest[0])
            if self.destination is None:self.state='HOLD_FULL'
        return self._log('NEST_BUSY',f'{nest} occupied by external event')

    def worker_removes(self,nest:str):
        if nest not in NESTS:raise ValueError('unknown nest')
        self.occupied[nest]=False
        if self.state in ('WAIT_FULL','HOLD_FULL') and self.grade_from_ai==nest[0]:
            self.destination=self._free_nest(self.grade_from_ai)
            self.state='MOVE' if self.apple_grasped else 'GRASP'
        return self._log('WORKER_REMOVES',f'{nest} now free')

    def enter_human_zone(self):
        self.human_in_work_zone=True
        if self.state!='STOP_HUMAN':
            self._resume_state=self.state
        self.state='STOP_HUMAN'
        return self._log('HUMAN_ZONE_OCCUPIED','virtual zone input; not safety-rated')

    def leave_human_zone(self):
        self.human_in_work_zone=False
        # No uncommanded auto-restart; explicit safety reset required.
        return self._log('HUMAN_ZONE_CLEAR','hold stop until reset acknowledgement')

    def acknowledge_reset(self):
        if self.state!='STOP_HUMAN' or self.human_in_work_zone:
            raise RuntimeError('safety reset not permitted')
        target=self._resume_state or 'WAIT_INPUT'
        if self.apple_grasped:
            self.destination=self._free_nest(self.grade_from_ai)
            target='MOVE' if self.destination else 'HOLD_FULL'
        elif self.grade_from_ai and self.apple_on_scale:
            self.destination=self._free_nest(self.grade_from_ai)
            target='GRASP' if self.destination else 'WAIT_FULL'
        self.state=target;self._resume_state=None
        return self._log('SAFETY_RESET_ACK','operator confirmation modeled')

    def release_into_nest(self):
        if self.state!='MOVE' or not self.apple_grasped or not self.destination:
            raise RuntimeError('no destination or fruit for release')
        if self.human_in_work_zone or self.occupied[self.destination]:
            raise RuntimeError('release blocked')
        slot=self.destination
        self.occupied[slot]=True
        self.apple_grasped=False;self.grade_from_ai=None;self.destination=None
        self.state='WAIT_INPUT'
        return self._log('RELEASED',f'placed into {slot}; robot returns to input')


def run_four_scenarios():
    """Synthetic sensor event traces: pure control demonstration (NO Bullet)."""
    out={}
    # Case 1: normal AI grade -> nest, then awaiting next fruit.
    c=InterlockController();c.place_on_scale();c.classify('C');c.grasp_done();c.release_into_nest()
    out['normal_sort']={'last':c.state,'trace':c.events,'filled':['C1']}
    # Case 2: unexpected concurrent nest fill after grasp: hold up high (logical hold).
    c=InterlockController();c.place_on_scale();c.classify('A');c.grasp_done()
    for n in ('A1','A2','A3'):c.external_nest_occupied(n)
    assert c.state=='HOLD_FULL' and not c.motion_permitted
    c.worker_removes('A2');c.release_into_nest()
    out['three_full_hold']={'last':c.state,'trace':c.events,'filled':['A1','A2','A3']}
    # Case 3: human zone -> stop -> exit -> explicit confirmation -> resume.
    c=InterlockController();c.place_on_scale();c.classify('B');c.grasp_done()
    c.enter_human_zone();c.leave_human_zone();assert c.state=='STOP_HUMAN'
    c.acknowledge_reset();c.release_into_nest()
    out['human_near_stop']={'last':c.state,'trace':c.events,'filled':['B1']}
    # Case 4: A away means no scale presence; remains in WAIT_INPUT until arrival.
    c=InterlockController();c._log('NO_INPUT','A is away; occupancy sensor still empty')
    c.place_on_scale();c.classify('D');c.grasp_done();c.release_into_nest()
    out['no_input_wait']={'last':c.state,'trace':c.events,'filled':['D1']}
    return out
