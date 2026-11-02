"""Deterministic, illustrative 50-minute orchard *operations* model.

This is a queueing/capacity simulator, not PyBullet and not measured field
throughput.  It models workers A/C, arrivals D/E/F/G, one 3s robot B, one
single-fruit buffer per grade, and a pause on occupied downstream trays.

2026/10/14 22:06 JST
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from collections import Counter
from dataclasses import dataclass,asdict

@dataclass
class Settings:
    duration_s:float=3000
    dt:float=.5
    farmers:int=4
    apples_per_basket:int=30
    harvest_cycle_s:float=300
    a_inspect_s:float=2.5
    reject_every:int=6          # exactly 200/1200 rejected if all reach A
    robot_s:float=3.
    c_pack_s:float=2.
    container_capacity:int=30
    c_swap_s:float=15.
    grades:str='AAABBC'       # illustrative distribution only, NOT AI accuracy


def simulate(cfg:Settings):
    if min(cfg.dt,cfg.a_inspect_s,cfg.robot_s,cfg.harvest_cycle_s) <= 0:raise ValueError('invalid time')
    if cfg.reject_every<2 or cfg.container_capacity<1 or not cfg.grades or any(x not in 'ABC' for x in cfg.grades):
        raise ValueError('invalid workcell settings')
    source=0;total_delivered=0;screened=0;rejected=0;accepted=0
    a_done_at=None;pending_a=False;pending_grade=None
    b_done_at=None;b_grade=None
    trays={key:None for key in 'ABC'}
    c_done_at=None;c_action=None;c_grade=None
    c_packed=0;containers_packed=Counter();completed=Counter()
    pauses=0.;a_wait=0.;robot_idle=0.;worker_busy=0.;worker_swaps=0
    t=0.;jobs=0;last_batch=-1
    while t<cfg.duration_s-1e-9:
        batch=int(round(t/cfg.harvest_cycle_s))
        if abs(t-batch*cfg.harvest_cycle_s)<cfg.dt/2 and batch!=last_batch:
            inc=cfg.farmers*cfg.apples_per_basket
            source+=inc;total_delivered+=inc;last_batch=batch
        if c_done_at is not None and t>=c_done_at-1e-9:
            if c_action=='pack':
                c_packed+=1;containers_packed[c_grade]+=1
            elif c_action=='swap':
                containers_packed[c_grade]=0
            c_done_at=None;c_action=None;c_grade=None
        if b_done_at is not None and t>=b_done_at-1e-9:
            if trays[b_grade] is None:
                trays[b_grade]=t
                completed[b_grade]+=1
                b_done_at=None;b_grade=None
        # Worker C selects oldest fruit from A/B/C tray. If its matching
        # container is full, worker replaces container before handling fruit.
        if c_done_at is None:
            available=[(when,k) for k,when in trays.items() if when is not None]
            if available:
                _,chosen=min(available)
                c_grade=chosen
                if containers_packed[chosen]>=cfg.container_capacity:
                    c_action='swap';c_done_at=t+cfg.c_swap_s;worker_swaps+=1
                else:
                    trays[chosen]=None  # picked up by worker C; tray now free
                    c_action='pack';c_done_at=t+cfg.c_pack_s
        if a_done_at is not None and t>=a_done_at-1e-9:
            screened+=1;source-=1
            if screened%cfg.reject_every==0:
                rejected+=1
            else:
                pending_a=True
            a_done_at=None
        if pending_a and b_done_at is None:
            b_grade=cfg.grades[accepted%len(cfg.grades)]
            b_done_at=t+cfg.robot_s
            accepted+=1;pending_a=False
        if not pending_a and a_done_at is None and source>0:
            a_done_at=t+cfg.a_inspect_s
        if pending_a:a_wait+=cfg.dt
        if b_done_at is None:robot_idle+=cfg.dt
        elif t>=b_done_at-1e-9:pauses+=cfg.dt
        if c_done_at is not None:worker_busy+=cfg.dt
        t=round(t+cfg.dt,9)
        jobs+=1
    return {'simulation':'deterministic discrete-time queue, not field measurement',
        'settings':asdict(cfg),'duration_s':cfg.duration_s,
        'harvested_delivered':total_delivered,
        'worker_a_screened':screened,'worker_a_rejected':rejected,
        'worker_a_handoff_to_robot':accepted,
        'robot_delivered_to_trays':sum(completed.values()),
        'robot_delivered_by_grade':dict(completed),
        'worker_c_packed':c_packed,
        'worker_c_container_swaps_started':worker_swaps,
        'harvest_baskets_uninspected':source,
        'worker_a_holding_selected':int(pending_a),
        'robot_holding_for_occupied_tray':int(b_done_at is not None and b_done_at<=cfg.duration_s),
        'trays_occupied_end':{key:value is not None for key,value in trays.items()},
        'robot_paused_occupied_tray_s':round(pauses,2),
        'worker_a_waiting_robot_s':round(a_wait,2),
        'robot_idle_no_selected_fruit_s':round(robot_idle,2),
        'worker_c_busy_s':round(worker_busy,2),
        'ideal_robot_capacity_50min':int(cfg.duration_s/cfg.robot_s),
        'warning':'50min throughput depends on *assumed* A and C rates and tray clearances; 3s robot mechanics unverified.'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--c-pack-s',type=float,default=2.)
    parser.add_argument('--c-swap-s',type=float,default=15.)
    parser.add_argument('--output',default='output/workcell_50min.json')
    args=parser.parse_args()
    result=simulate(Settings(c_pack_s=args.c_pack_s,c_swap_s=args.c_swap_s))
    path=Path(__file__).resolve().parents[1]/args.output
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('settings','warning')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
