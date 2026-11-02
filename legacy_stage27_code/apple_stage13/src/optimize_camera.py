"""v1.1 camera-placement search. Only geometrical visibility, NOT physics success.
Camera orientations are fixed to 3R wrist, never retargeted on the fly.
"""
from __future__ import annotations
from pathlib import Path
import sys, json, csv, math, time
from dataclasses import dataclass
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from camera_model import *

OUT=Path(__file__).resolve().parents[1]/'output'
OUT.mkdir(exist_ok=True)

def tilted_cam(f, pitch_deg=0., yaw_deg=0.):
    # fixed wrist optical mount. No dynamic look-at; wrist levels through q3.
    az=f['az']; tang=tangent3(az); inward=np.array([-math.cos(az),-math.sin(az),0.])
    yaw=math.radians(yaw_deg); pitch=math.radians(pitch_deg)
    forward=norm(math.cos(pitch)*(math.cos(yaw)*inward+math.sin(yaw)*tang)+math.sin(pitch)*np.array([0.,0.,1.]))
    right=norm(np.cross(np.array([0.,0.,1.]),forward))
    up=norm(np.cross(forward,right))
    return {'origin':f['camera'],'direction':forward,'right':right,'up':up}

def cam_simple(origin,pointing):
    forward=norm(np.asarray(pointing,float));right=norm(np.cross(np.array([0.,1.,0.]),forward))
    if np.linalg.norm(right)<.1:right=np.array([1.,0.,0.])
    up=norm(np.cross(forward,right))
    return {'origin':np.array(origin,float),'direction':forward,'right':right,'up':up}

def lifted_parts(parts,lift):
    arr=[]
    for f in parts:
        item={}
        for k,v in f.items():
            item[k]=v+np.array([0.,0.,lift]) if isinstance(v,np.ndarray) and v.shape==(3,) and k not in ('dir','right','up') else v
        arr.append(item)
    return arr

def coverage_masks(design:PadDesign, pitch=0., yaw=0., palm_height=.147,lift=.065,
                   npts=1800, times=None, with_links=True):
    pts=sphere_points(npts)
    union=np.zeros(npts,bool)
    per_cam=np.zeros((5,npts),bool)
    ts=np.linspace(0,G.scan_s,7) if times is None else times
    min_gap=1e9
    for t in ts:
        parts=finger_parts(float(t),design)
        min_gap=min(min_gap,pad_apple_clearance(parts,design))
        for k,f in enumerate(parts):
            seen=projected_surface_visibility(tilted_cam(f,pitch,yaw),pts,parts,design,with_links=with_links)
            union|=seen;per_cam[k]|=seen
    # Palm camera is mounted rigidly at a chosen (and explicitly listed) height.
    parts=finger_parts(.26,design)
    pcam=cam_simple([0,0,palm_height],[0,0,-1])
    seen=projected_surface_visibility(pcam,pts,parts,design,with_links=with_links)
    union|=seen;per_cam[3]|=seen
    pre=union.copy()
    post_parts=lifted_parts(finger_parts(.60,design),lift)
    plate=plate_spec()
    seen=projected_surface_visibility(plate,pts+np.array([0,0,lift]),post_parts,design,
                                     with_links=with_links,apple_center=APPLE_C+np.array([0,0,lift]))
    per_cam[4]|=seen;union|=seen
    return {'pre':pre,'post':union,'per_cam':per_cam,'min_gap':min_gap,'pts':pts}

def search():
    """Bounded camera layout sweep: preserve same-wrist pad, lens within 22mm,
    palm lens BELOW existing head, lift <=100 mm. Optical yaw fixed mechanically.
    Score gives 30% pre-lift surface +70% after-plate surface.
    """
    start=time.monotonic()
    results=[]
    for side in [7,9,11,13,15,17,19]:
      for cam_up in [0,3,5,8,11,14]:
        design=PadDesign(length_mm=18,cam_tangent_mm=side,cam_up_mm=cam_up)
        pieces=finger_parts(.26,design)
        lens_pad_dist=float(np.linalg.norm(pieces[0]['camera']-pieces[0]['padcenter'])*1000)
        if lens_pad_dist>22.0:continue
        for pitch in [-10.,0.,10.]:
          yaw=-math.degrees(math.atan(side/65.))
          result=coverage_masks(design,pitch,yaw,.160,.100,npts=2000,times=np.linspace(0,G.scan_s,7))
          pre=float(result['pre'].mean()*100);post=float(result['post'].mean()*100)
          results.append({'generated_at_jst':'2026/10/14 02:44 JST','side_mm':side,'up_mm':cam_up,'pitch_deg':pitch,
             'palm_z_mm':160.,'lift_mm':100.,'camera_pad_mm':lens_pad_dist,
             'before_pct':pre,'after_pct':post,'weighted_score':.3*pre+.7*post})
    results.sort(key=lambda r:r['weighted_score'],reverse=True)
    with (OUT/'candidate_search.csv').open('w',newline='',encoding='utf8') as f:
       w=csv.DictWriter(f,fieldnames=results[0].keys());w.writeheader();w.writerows(results)
    with (OUT/'search_summary.json').open('w',encoding='utf8') as f:
       json.dump({'candidate_count':len(results),'best':results[0],
                  'selected_first_order_design':{'side_mm':17,'up_mm':0,'pitch_deg':0},
                  'note':'Candidate search only, not validated physical grasp or recognition',
                  'seconds':time.monotonic()-start},f,ensure_ascii=False,indent=2)
    print(json.dumps({'n':len(results),'best':results[0],'top5':results[:5]},ensure_ascii=False,indent=2))
    return results
if __name__=='__main__':search()
