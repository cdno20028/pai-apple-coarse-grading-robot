"""Capture actual PyBullet camera frames with trace overlays.
Only called by run_physics on a real PyBullet client; cannot synthesize success.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

JP_FONTS=[
 '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
]

def choose_font(size):
 for f in JP_FONTS:
  if Path(f).exists(): return ImageFont.truetype(f,size)
 return ImageFont.load_default()

class VideoCapture:
 def __init__(self,p,output,label,width=640,height=440,apple_body_id=None,z_shift=0.0):
  self.p=p;self.output=Path(output);self.label=label;self.width=width;self.height=height
  self.apple_body_id=apple_body_id;self.z_shift=z_shift
  self.timestamp=datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y/%m/%d %H:%M JST')
  self.writer=imageio.get_writer(str(self.output),fps=20,codec='libx264',quality=7)
  self.font=choose_font(18);self.fsmall=choose_font(14);self.total=0
 def add_frame(self,t,stage,focus_xy=None):
  p=self.p;w=self.width;h=self.height
  if focus_xy is not None and stage in ('gentle_lower','supported_release','settle_in_bin'):
   cx,cy=focus_xy
   eye=[cx+.23,cy-.29,self.z_shift+.25];target=[cx,cy,self.z_shift+.075]
  elif self.apple_body_id is not None:
   # Camera following the apple is only for video framing, not a grading sensor.
   ax,ay,az=p.getBasePositionAndOrientation(self.apple_body_id)[0]
   eye=[ax+.46,ay-.54,az+.38];target=[ax,ay,az]
  else:
   eye=[.66,-.68,self.z_shift+.49];target=[0,.095,self.z_shift+.105]
  view=p.computeViewMatrix(cameraEyePosition=eye,cameraTargetPosition=target,cameraUpVector=[0,0,1])
  proj=p.computeProjectionMatrixFOV(fov=55,aspect=w/h,nearVal=.01,farVal=5)
  _,_,rgba,_,_=p.getCameraImage(w,h,viewMatrix=view,projectionMatrix=proj,renderer=p.ER_TINY_RENDERER)
  a=np.asarray(rgba,dtype=np.uint8).reshape(h,w,4)[:,:,:3]
  im=Image.fromarray(a).convert('RGB');dr=ImageDraw.Draw(im)
  dr.rectangle((0,0,w,53),fill=(13,31,58));dr.text((12,5),f'APPLE SORTING — PHYSICS TRIAL {self.label}',font=self.font,fill='white')
  dr.text((12,30),f't={t:.2f}s  |  {stage}',font=self.fsmall,fill=(194,219,248))
  dr.rectangle((0,h-28,w,h),fill=(13,31,58));dr.text((12,h-24),f'{self.timestamp} | PyBullet DIRECT, synthetic apple',font=self.fsmall,fill=(210,221,235))
  self.writer.append_data(np.asarray(im));self.total+=1
 def close(self,passed,elapsed):
  # Append short terminal condition card: based on measured contact and sorting state.
  im=Image.new('RGB',(self.width,self.height),(13,31,58));dr=ImageDraw.Draw(im)
  dr.text((25,60),'MEASURED PHYSICS RESULT',font=choose_font(25),fill='white')
  dr.text((25,116), 'PASS' if passed else 'FAIL — inspect JSON/CSV',font=choose_font(29),fill=(79,222,133) if passed else (247,125,119))
  dr.text((25,169),f'Measured simulation clock: {elapsed:.3f}s',font=self.font,fill='white')
  dr.text((25,210),'Synthetic colour/weight labels, NOT real fruit grading',font=self.fsmall,fill=(201,213,232))
  dr.text((25,360),self.timestamp,font=self.fsmall,fill='white')
  for _ in range(20):self.writer.append_data(np.asarray(im))
  self.writer.close()
