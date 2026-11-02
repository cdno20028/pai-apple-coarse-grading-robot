# -*- coding: utf-8 -*-
"""Stage68 camera evidence: per-tick real link Z, synthetic pole-provenance."""
from pathlib import Path
import json, numpy as np
from stage67_camera_ai import Stage67CameraAI
from stage68_surface import Stage68Surface

class Stage68CameraAI(Stage67CameraAI):
    def __init__(self,root,results,assets,object_ai):
        super().__init__(root,results,assets,object_ai)
        self.repair=Stage68Surface(assets,results)

    def texture_body(self,p,body,idx):
        photo=self.photo_for_case(idx)
        if photo is None:return
        if int(body) in self.material_ids:
            if self.material_ids[int(body)]!=idx:
                raise RuntimeError('STAGE68_NO_PHOTO_ID_REASSIGNMENT')
            return
        tex=p.loadTexture(str(self.repair.path(photo['asset_id'])))
        for vis in p.getVisualShapeData(body):
            p.changeVisualShape(body,int(vis[1]),textureUniqueId=tex,rgbaColor=[1,1,1,1])
        self.material_ids[int(body)]=idx

    def start_case(self,idx,typ,truth,rec):
        super().start_case(idx,typ,truth,rec)
        self.current['stage68_actual_z_trace']=[]
        self.current['stage68_motion_status']='NOT_RUN'

    def observe(self,p,robot,apple,plate,joints,stage,t,cfg,rp):
        super().observe(p,robot,apple,plate,joints,stage,t,cfg,rp)
        if stage=='scan_and_close' and self.current['kind']=='apple':
            z=[float(p.getLinkState(robot,joints[f'finger_camera_{k}_fixed'],
                       computeForwardKinematics=True)[4][2]) for k in range(3)]
            root=float(p.getJointState(robot,joints['stage_z_move'])[0])
            self.current['stage68_actual_z_trace'].append({'sim_s':round(float(t),6),
                 'root_m':round(root,7),'finger_z_m':[round(v,7) for v in z]})

    def conclude(self,case_name,rp_result,rec):
        row=self.current;history=row['stage68_actual_z_trace']
        measurement={'scan_ticks':len(history),'per_camera':{},'actual_kinematics_not_assumed':True}
        for k in range(3):
            values=np.array([x['finger_z_m'][k] for x in history],dtype=float)
            delta=np.diff(values)
            ups=delta[delta>.0005]
            measurement['per_camera'][f'finger_{k+1}']={
                'total_net_descent_mm':round(float((values[0]-values[-1])*1000),3) if len(values)>1 else None,
                'maximum_single_tick_up_mm':round(float(max(0,delta.max(initial=0))*1000),3),
                'upward_ticks_more_than_0p5mm':int(len(ups)),
                'total_upward_motion_mm':round(float(np.maximum(delta,0).sum()*1000),3)}
        measurement['monotone_with_0p5mm_single_tick_tolerance']=bool(len(history)>50 and
            all(v['upward_ticks_more_than_0p5mm']==0 for v in measurement['per_camera'].values()))
        row['stage68_motion_status']='MEASURED'
        row['stage68_motion_audit']=measurement
        # avoid bulky 85-tick traces in all JSON summaries but provide per-tick CSV.
        path=self.results/'stage68_motion_csv'/f'{case_name}.csv'
        path.parent.mkdir(exist_ok=True)
        with path.open('w',encoding='utf8') as f:
            f.write('time_s,root_z_m,finger_1_z_m,finger_2_z_m,finger_3_z_m\n')
            for x in history:
                f.write(','.join(map(str,[x['sim_s'],x['root_m'],*x['finger_z_m']]))+'\n')
        row['stage68_actual_z_trace_csv']=str(path.relative_to(self.results))
        row.pop('stage68_actual_z_trace')
        result=super().conclude(case_name,rp_result,rec)
        result['stage68_finger_monotone_observed']=measurement['monotone_with_0p5mm_single_tick_tolerance']
        return result
