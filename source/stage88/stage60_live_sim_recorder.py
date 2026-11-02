#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage60: directly record REAL PyBullet time steps, NOT a diagnostic-photo replay.

At each simulated 1/20 s physics instant the actual PyBullet world is rendered.
The five sensor panes hold only the latest physically captured sensor images:
three finger cameras during the measured descent, the palm gate image, and
under-plate upward camera during lift. Holding latest sensor image is a UI
state, not a claim that the sensor captured at 20fps.
"""
from __future__ import annotations
import datetime, time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio

W,H=1600,900
FPS=20
JST=datetime.timezone(datetime.timedelta(hours=9))
# Japanese text and LARGE A/B/C/D labels REQUIRE a scalable CJK font.
# Do not silently fall back to PIL's tiny bitmap default (Stage60 v1 failure).
# Colab notebook installs fonts-noto-cjk before importing this module.
CJK_FONTS=('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
           '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc')

def locate_cjk_font():
    for src in CJK_FONTS:
        if Path(src).is_file():
            candidate=ImageFont.truetype(src,160)
            bbox=candidate.getbbox('判定 C')
            if bbox[2]-bbox[0]>170:
                return src
    raise RuntimeError('STAGE60_V2_MISSING_CJK_FONT: install fonts-noto-cjk BEFORE simulation; NEVER use tiny default bitmap font')

CJK_FONT_PATH=locate_cjk_font()
F={size:ImageFont.truetype(CJK_FONT_PATH,size)
   for size in (16,19,23,27,30,36,42,48,52,60,68,76,92,160)}
assert F[160].getbbox('C')[2]-F[160].getbbox('C')[0] > 80, 'Large GRADE font must be truly scalable'

class LiveSimulationRecorder:
    """Writes an MP4 from the LIVE physics callback; no replacement frames."""
    def __init__(self,output,source_id,reference_label,case_number):
        self.output=Path(output)
        if self.output.exists():raise FileExistsError(str(self.output))
        self.writer=imageio.get_writer(str(self.output),fps=FPS,codec='libx264',quality=7,macro_block_size=1)
        self.source_id=str(source_id)
        self.reference_label=str(reference_label)
        self.case_number=int(case_number)
        self.frames=0
        self.closed=False
        self.first_scan_sim=None
        self.last_sim=None
        self.creation_jst=datetime.datetime.now(JST).strftime('%Y/%m/%d %H:%M JST')
        self.sample_ids={}
    def record(self,pc,apple,stage,t_sim,state):
        if self.closed:raise RuntimeError('Recorder already closed')
        t_sim=float(t_sim)
        if self.last_sim is not None and t_sim<=self.last_sim:
            raise AssertionError('Video frame time is not strictly increasing')
        self.last_sim=t_sim
        # The observer already collected ORIGINAL sensor images and stage events
        # for this very physics step, before calling record().
        if self.first_scan_sim is None:
            ev=[x for x in state.get('events',[]) if x.get('phase')=='finger_approach']
            if ev:self.first_scan_sim=min(float(x['t_sim_s']) for x in ev)
        im=Image.new('RGB',(W,H),'#071521');d=ImageDraw.Draw(im)
        d.rectangle((0,0,W,70),fill='#0f263d')
        d.text((22,12),'Physical AI  |  PyBullet 実行中の撮影・判断',font=F[36],fill='#f4faff')
        d.text((1300,18),self.creation_jst,font=F[19],fill='#c4d7e5')
        # This is a REAL physical-world render at THIS simulation step.
        xyz=pc.getBasePositionAndOrientation(apple)[0]
        ax,ay,az=map(float,xyz)
        eye=[ax+.47,ay-.51,az+.36]
        target=[ax,ay,az]
        view=pc.computeViewMatrix(cameraEyePosition=eye,cameraTargetPosition=target,cameraUpVector=[0,0,1])
        ww,hh=1032,605
        proj=pc.computeProjectionMatrixFOV(fov=55,aspect=ww/hh,nearVal=.01,farVal=5)
        _,_,rgba,_,_=pc.getCameraImage(ww,hh,viewMatrix=view,projectionMatrix=proj,renderer=pc.ER_TINY_RENDERER)
        world=np.asarray(rgba,dtype=np.uint8).reshape(hh,ww,4)[:,:,:3]
        im.paste(Image.fromarray(world),(15,85))
        d=ImageDraw.Draw(im)
        d.rectangle((15,85,15+ww,85+hh),outline='#4b7d95',width=3)
        d.rectangle((15,85,15+ww,128),fill='#0a2338')
        d.text((29,91),'PyBullet 実際の物理状態  /  virtual overview',font=F[23],fill='#eff8ff')
        d.rectangle((1065,86,1584,883),fill='#11283f')
        d.text((1092,101),f'CASE {self.case_number:02d} / 08',font=F[36],fill='#e8f7ff')
        d.line((1090,150,1555,150),fill='#446078',width=2)
        gate=state.get('gate') or {}
        obj=gate.get('object') or {}
        accepted=gate.get('allow_descent')
        status=('APPLE  /  PASS' if accepted else 'HOLD  /  REJECT' if accepted is False else 'WAITING')
        d.text((1092,166),status,font=F[30],fill='#8df0c3' if accepted else '#d9e6f4')
        d.text((1092,222),'AI GRADE',font=F[27],fill='#c4d5e9')
        result=state.get('inference') or {}
        grade=result.get('class') if result else None
        d.rounded_rectangle((1090,263,1555,463),radius=18,fill='#1c3f59')
        if grade:
            d.text((1318,347),str(grade),font=F[160],fill='#fff6e8',anchor='mm')
        else:
            d.text((1318,360),'計測中',font=F[60],fill='#f5fbff',anchor='mm')
        # Time panel: simulation clock MUST advance with actual Bullet steps.
        d.text((1094,478),'シミュレーション時刻',font=F[27],fill='#c6d7e6')
        d.text((1094,514),f'{t_sim:6.2f} 秒',font=F[60],fill='#f1f9ff')
        d.text((1094,587),'撮影開始 → 現在 / 等級確定',font=F[23],fill='#bfd0df')
        if self.first_scan_sim is None:
            span='--.-- 秒'
        elif grade:
            span=f'{float(result["grade_inference_at_sim_t_s"])-self.first_scan_sim:.2f} 秒'
        else:
            span=f'{max(0.0,t_sim-self.first_scan_sim):.2f} 秒'
        d.text((1094,617),span,font=F[52],fill='#ffc174')
        d.text((1094,691),'物体識別 / 実測処理時間',font=F[23],fill='#bed0df')
        gms=obj.get('infer_wall_ms')
        d.text((1094,720),f'{float(gms)/1000:.4f} 秒' if gms is not None else '計測前',font=F[42],fill='#e9f7ff')
        d.text((1094,776),'等級分類 / 実測処理時間',font=F[23],fill='#bed0df')
        wms=result.get('grade_infer_wall_ms')
        d.text((1094,807),f'{float(wms)/1000:.4f} 秒' if wms is not None else '判定待機',font=F[42],fill='#e9f7ff')
        # Latest sampled sensor image, NOT continuous sensor video.
        labels=[('finger',0,'指先1'),('finger',1,'指先2'),('finger',2,'指先3'),('palm',0,'掌'),('bottom',0,'下部')]
        for i,(prefix,camera_number,caption) in enumerate(labels):
            x=19+i*210;y=708
            d.rectangle((x,y,x+200,y+174),fill='#172f46')
            files=[(k,v) for k,v in state.get('frames',{}).items() if k[0]==prefix and k[1]==camera_number]
            if files:
                _,arr=max(files,key=lambda kv:kv[0][2])
                tile=Image.fromarray(np.asarray(arr,dtype=np.uint8)).resize((144,144),Image.Resampling.BILINEAR)
                im.paste(tile,(x+6,y+23))
                index=max(k[2] for k,_ in files)+1
                text=f'{caption} #{index}' if prefix in ('finger','bottom') else caption
            else:
                text=f'{caption}  未撮影'
            d.text((x+8,y+2),text,font=F[19],fill='#f5f9fe')
        # Distinguish real-time sim capture from farm measurements.
        d.text((1095,857),'写真由来＋合成裏面 / 研究用 / 現場精度未検証',font=F[16],fill='#a9bcc9')
        self.writer.append_data(np.asarray(im))
        self.frames+=1
    def close(self):
        if not self.closed:
            self.writer.close()
            self.closed=True
        return {'video_file':self.output.name,'frames_rendered_from_pybullet':self.frames,
                'fps':FPS,'width':W,'height':H,'last_sim_t_s':self.last_sim,
                'first_finger_scan_t_s':self.first_scan_sim,
                'render_method':'LIVE_PYBULLET_WORLD_RENDER_EVERY_12_PHYSICS_STEPS',
                'sensor_method':'LAST_REAL_SENSOR_FRAME_OBSERVED_BY_THIS_TIME_NOT_FAKE_20FPS',
                'photo_uv_synthetic_back':True,'grade_processing_wall_ms_is_measured_during_observer':True}
