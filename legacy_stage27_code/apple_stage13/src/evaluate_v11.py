"""Evaluate mechanically constrained v1.1 against the same-calibration v1.0 reference.
NOTE: ray-cast visibility only, not camera recognition or physics simulation.
"""
from pathlib import Path
import json, csv, math, sys
import numpy as np
from optimize_camera import coverage_masks, tilted_cam, lifted_parts
from camera_model import PadDesign, finger_parts, render_cam, palm_spec, plate_spec, APPLE_C, G

OUT=Path(__file__).resolve().parents[1]/'output';OUT.mkdir(exist_ok=True)
SPECS=[
 ('v10_recomputed',PadDesign(length_mm=18,cam_tangent_mm=11,cam_up_mm=3),.147,.065),
 ('palm_only',PadDesign(length_mm=18,cam_tangent_mm=11,cam_up_mm=3),.160,.065),
 ('lift_only',PadDesign(length_mm=18,cam_tangent_mm=11,cam_up_mm=3),.147,.100),
 ('palm_and_lift',PadDesign(length_mm=18,cam_tangent_mm=11,cam_up_mm=3),.160,.100),
 ('v11_selected',PadDesign(length_mm=18,cam_tangent_mm=17,cam_up_mm=0),.160,.100)
]

def eval_cases():
 result=[]
 for name,d,h,l in SPECS:
  yaw=-math.degrees(math.atan(d.cam_tangent_mm/65.))
  m=coverage_masks(d,0,yaw,h,l,npts=12000,times=np.linspace(0,G.scan_s,13))
  dist=np.linalg.norm(finger_parts(.26,d)[0]['camera']-finger_parts(.26,d)[0]['padcenter'])*1000
  result.append({'generated_at_jst':'2026/10/14 02:44 JST','name':name,'cam_side_mm':d.cam_tangent_mm,'cam_up_mm':d.cam_up_mm,
     'palm_camera_z_mm':h*1000,'plate_capture_lift_mm':l*1000,
     'lens_to_pad_center_mm':float(dist),'pre_lift_visible_pct':float(m['pre'].mean()*100),
     'after_plate_visible_pct':float(m['post'].mean()*100),
     'unseen_pct':float((~m['post']).mean()*100),
     'scan_min_pad_gap_mm':float(m['min_gap']*1000)})
 (OUT/'v11_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 with (OUT/'v11_results.csv').open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,result[0]);w.writeheader();w.writerows(result)
 print(json.dumps(result,indent=2))
 return result
if __name__=='__main__':eval_cases()
