"""Stage27 deterministic scenario specifications; pure Python, no PyBullet.
2026/10/18 JST.  Robot geometry and original nest IDs are unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass

NEAREST = {'A': ('A3','A2','A1'), 'B': ('B3','B2','B1'),
           'C': ('C1','C2','C3'), 'D': ('D1','D2','D3')}

@dataclass(frozen=True)
class Scenario:
    name: str
    grades: tuple[str, ...]
    # episode number (1-based) -> existing tray nest(s) removed before that episode
    clear_before: tuple[tuple[int, tuple[str,...]], ...]
    description: str

SCENARIOS = {
    'rapid_pickup_8': Scenario(
        'rapid_pickup_8', tuple('ADBC CBDA'.replace(' ','')),
        tuple((i, ('PREVIOUS',)) for i in range(2,9)),
        'Routine: preceding apple cleared before every next trial; labels reordered'),
    'partial_gap_8': Scenario(
        'partial_gap_8', tuple('AAABABCD'), ((5, ('A2',)),),
        'A3/A2/A1 accumulate; clear A2 only, then route A to that exact gap'),
    'mixed_gap_12': Scenario(
        'mixed_gap_12', tuple('BDC AABDCBACD'.replace(' ','')),
        ((9, ('B3',)), (10, ('A2',)), (11, ('C1',)), (12, ('D2',))),
        'Cross-grade partial vacancies while all previously placed fruits persist'),
}


def plan_expected(scenario: Scenario):
    occupancy = {f'{g}{j}': False for g in 'ABCD' for j in (1,2,3)}
    clears = dict(scenario.clear_before)
    rows=[]
    previous=None
    for index,grade in enumerate(scenario.grades,1):
        cleared=[]
        for name in clears.get(index,()):
            name = previous if name == 'PREVIOUS' else name
            if not name or not occupancy[name]:
                raise AssertionError(f'planned clear not occupied: episode {index} {name}')
            occupancy[name] = False
            cleared.append(name)
        before=dict(occupancy)
        dest=next((s for s in NEAREST[grade] if not occupancy[s]),None)
        if dest is None: raise AssertionError(f'planned full grade: {index} {grade}')
        occupancy[dest]=True
        rows.append({'episode':index,'grade':grade,'clear_slots':cleared,
                     'expected_slot':dest,'occupied_before':before,
                     'occupied_after':dict(occupancy)})
        previous=dest
    return rows


def validate():
    expected = {
        'rapid_pickup_8': ['A3','D1','B3','C1','C1','B3','D1','A3'],
        'partial_gap_8': ['A3','A2','A1','B3','A2','B2','C1','D1'],
        'mixed_gap_12': ['B3','D1','C1','A3','A2','B2','D2','C2','B3','A2','C1','D2'],
    }
    for name,slots in expected.items():
        records=plan_expected(SCENARIOS[name])
        assert [r['expected_slot'] for r in records] == slots, (name,records)
        assert all(r['occupied_before'][r['expected_slot']] is False for r in records)
        assert all(r['occupied_after'][r['expected_slot']] is True for r in records)
    assert sum(len(s.grades) for s in SCENARIOS.values()) == 28
    return True

if __name__=='__main__':
    assert validate()
    for name,sc in SCENARIOS.items():
        print(name, ' '.join(f"{r['grade']}:{r['expected_slot']}" for r in plan_expected(sc)))
    print('STAGE27_PLAN_TEST_PASS: 28 episodes; 3 deterministic scenarios')
