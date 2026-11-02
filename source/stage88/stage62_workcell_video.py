# -*- coding: utf-8 -*-
"""Real-time PyBullet scene + latest physically sampled five camera panes, one MP4.
All frames are rendered during physics execution; 20fps (240/12), 1.0x.
2026/10/22 JST
"""
import datetime
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont

# Stage71: explanatory overlay only. AI pixels and physics are unchanged.
# Stage75: presentation zoom keeps the Stage74 independent HUD; evidence HUD now occupies a dedicated LEFT GUTTER of the 1920x1080
# output frame. Workcell/cameras remain their original 1600x900 pixel resolution
# shifted intact by +320,+90. No overlay can cover a moving worker or robot.
# Never change camera sampling, physics world coordinates or inference.
STAGE73_EVIDENCE_PANEL=(15,251,273,804)
STAGE73_GRADE_PANEL=(15,106,273,242)

# Stage75: presentation-only zoom of the 3D workcell in the same 1035x710
# viewport. Do NOT touch either the five AI cameras or the physics camera
# sampling, the grade inference, collision model, camera HUD, or frame time.
# Crop corresponds to approximately 1.50x apparent linear magnification.
STAGE75_WORKCELL_ROI_XYWH = (145, 30, 690, 473)
STAGE75_WORKCELL_OUTPUT_WH = (1035, 710)
def stage75_zoom_workcell_rgb(rgb_image):
    assert rgb_image.size == STAGE75_WORKCELL_OUTPUT_WH
    x,y,w,h=STAGE75_WORKCELL_ROI_XYWH
    assert 0<=x and 0<=y and x+w<=rgb_image.width and y+h<=rgb_image.height
    return rgb_image.crop((x,y,x+w,y+h)).resize(STAGE75_WORKCELL_OUTPUT_WH, Image.Resampling.BICUBIC)

def draw_stage72_evidence(canvas, evidence, photo_uv, fonts, source_info=None):
    # Preserve interface for Stage71/72 regression tests. Presentation only.
    d=ImageDraw.Draw(canvas)
    x0,y0,x1,y1=STAGE73_EVIDENCE_PANEL
    d.rectangle((x0,y0,x1,y1),fill='#102c3d',outline='#7da4bd',width=2)
    x=24
    def say(y,t,font=16,color='#e8f2f8'):
        # Use compact individual lines; do not obscure the worker or camera panes.
        f=fonts[font]
        # Text is monitoring metadata, not AI input: a verbose label must
        # never abort the actual continuous physics simulation or its video.
        avail=x1-x-7
        if d.textlength(t,font=f)>avail:
            suffix='…'
            while t and d.textlength(t+suffix,font=f)>avail:
                t=t[:-1]
            t=t+suffix
        d.text((x,y),t,font=f,fill=color)
    say(260,'外観AI：判定根拠',22,'white')
    say(290,'暫定基準／実果未検証',16,'#ffcf97')
    say(321,'A：指先に傷候補なし')
    say(345,'B：指先1台／弱い候補2台')
    say(369,'C：指先2台／大傷7%以上')
    say(393,'D：掌・指2とも黄緑優勢')
    say(425,'傷閾値：指1・3 は3%',16,'#ffdb9d')
    say(449,'指2 は1%／弱い候補2%',16,'#ffdb9d')
    say(473,'掌の傷・下部は合成除外',16,'#ffdb9d')
    d.line((24,500,263,500),fill='#608aa5',width=2)
    if not evidence:
        say(517,'判定待機中：',19,'#b8cddf')
        say(546,'5カメラ撮影待ち',19,'#b8cddf')
        return canvas
    p=evidence['dark_pct']
    say(515,f"指1 {p['finger_1']:.1f}%  指2 {p['finger_2']:.1f}%",16,'white')
    say(540,f"指3 {p['finger_3']:.1f}%  掌 {p['palm']:.1f}%",16,'white')
    positives=evidence['positive']
    flag=bool(p['palm']>=7.0 and 'palm' not in positives)
    if flag:
        say(568,'掌の傷候補は合成面',16,'#ffae93')
        say(592,'暗部率は参考値に限定',16,'#ffae93')
    else:
        say(568,f"傷候補 {len(positives)}台／最大 {evidence['max_pct']:.1f}%",16,'#ffe0a2')
    if source_info:
        actual=source_info.get('actual_ai_grade')
        expected=source_info.get('reference_grade')
        if actual and expected:
            concordant=actual==expected
            say(617,f"AI {actual}／参考 {expected}："+('一致' if concordant else '不一致'),
                16,'#bde7bb' if concordant else '#ff957b')
    # Left-bottom reference UV removed: presentation-only, no AI or robot change.
    return canvas

def draw_stage71_evidence(canvas, evidence, source_uv_preview, fonts):
    return draw_stage72_evidence(canvas,evidence,source_uv_preview,fonts,None)

class Stage62Recorder:
    def __init__(self,p,out,fps=20,w=1920,h=1080):
        font=('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
        if not Path(font).is_file():raise RuntimeError('Noto CJK font missing (install fonts-noto-cjk)')
        self.font={i:ImageFont.truetype(font,i) for i in [16,19,22,25,30,36,48,64,100]}
        self.p=p;self.w=w;self.h=h;self.fps=fps
        self.writer=imageio.get_writer(str(out),fps=fps,codec='libx264',quality=7,macro_block_size=1)
        self.count=0;self.steps=0;self.state='開始';self.pred='認識待機';self.destination='—'
        self.occupied='—';self.input_count=0;self.proxy_note='作業員A/Cは模擬アバター・衝突モデルなし'
        self.date=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y/%m/%d %H:%M:%S JST')
        self.stage62_camera_frames={};self.stage62_grade=None
        self.stage71_evidence=None;self.stage71_source_preview=None;self.stage72_source_info=None
    def frame(self,phase,force=False):
        self.steps+=1
        if not force and self.steps%12!=0:return
        p=self.p
        W,H=1600,900;world_w,world_h=1035,710
        view=p.computeViewMatrix(cameraEyePosition=[.60,-3.10,3.00],cameraTargetPosition=[0.,.10,.72],cameraUpVector=[0,0,1])
        proj=p.computeProjectionMatrixFOV(fov=50.,aspect=world_w/world_h,nearVal=.02,farVal=6.)
        _,_,rgba,_,_=p.getCameraImage(world_w,world_h,viewMatrix=view,projectionMatrix=proj,renderer=p.ER_TINY_RENDERER)
        im=Image.new('RGB',(W,H),'#071521');raw_workcell=Image.fromarray(np.asarray(rgba,dtype=np.uint8).reshape(world_h,world_w,4)[:,:,:3]);im.paste(stage75_zoom_workcell_rgb(raw_workcell),(12,100))
        d=ImageDraw.Draw(im)
        d.rectangle((0,0,W,96),fill='#0f263d')
        d.text((16,9),'Physical AI｜りんご粗選果・連続PyBullet実行',font=self.font[36],fill='white')
        d.text((19,57),f'事例 {self.state}　動作 {phase}',font=self.font[22],fill='#caddec')
        d.rectangle((1060,100,1590,810),fill='#112a3b')
        cams=[('finger_1','指先1'),('finger_2','指先2'),('finger_3','指先3'),('palm','掌カメラ'),('inspection_under','検査台下部（上向き）')]
        positions=[(1070,112),(1327,112),(1070,340),(1327,340),(1197,564)]
        for (key,txt),(x,y) in zip(cams,positions):
            wid,hei=(246,190)
            d.rectangle((x,y,x+wid,y+hei+23),fill='#25394e')
            img=self.stage62_camera_frames.get(key)
            if img is not None:
                clip=Image.fromarray(np.asarray(img,dtype=np.uint8)).resize((186,186))
                im.paste(clip,(x+27,y+3))
            else:
                d.text((x+35,y+80),'未撮影',font=self.font[22],fill='#aec2d3')
            d.text((x+5,y+193),txt,font=self.font[19],fill='white')
            if self.stage71_evidence and key in self.stage71_evidence.get('positive',[]):
                d.rectangle((x-2,y-2,x+wid+2,y+hei+23),outline='#ffcd68',width=3)
                d.text((x+136,y+193),'傷候補',font=self.font[16],fill='#ffcd68')
        # STAGE74: Do NOT draw any HUD over the 3D workcell in the original frame.
        # The dedicated sidebar is composed after the workcell and camera panes.
        d.rectangle((0,816,W,900),fill='#10263c')
        d.text((20,822),f'AI: {self.pred}　搬送先: {self.destination}',font=self.font[22],fill='#ecf7e5')
        d.text((20,857),f'{self.date}  |  等速1.0倍・無編集・同一物理世界  |  仮想時刻 {self.steps/240:.2f}秒',font=self.font[19],fill='#b5c7d8')
        d.text((1075,833),'実果精度未検証／写真上端・裏面は合成',font=self.font[19],fill='#ffcda6')
        # 16:9 1920x1080 canvas: unscaled original 1600x900 scene shifted right
        # into a separate column. The left 320 px is *never* part of the
        # rendered physical workspace (worker C can walk anywhere in the scene).
        final=Image.new('RGB',(1920,1080),'#081c2d')
        final.paste(im,(320,90))
        hud=ImageDraw.Draw(final)
        hud.text((17,26),'AI判定・監査',font=self.font[30],fill='white')
        hud.text((18,65),'作業場とは独立した表示欄',font=self.font[16],fill='#bdcfe0')
        if self.stage62_grade is not None:
            gx0,gy0,gx1,gy1=STAGE73_GRADE_PANEL
            hud.rectangle((gx0,gy0,gx1,gy1),fill='#132e43',outline='#7da4bd',width=2)
            hud.text((25,112),'外観AI判定',font=self.font[22],fill='#c0e5f7')
            label=str(self.stage62_grade)
            bbox=hud.textbbox((0,0),label,font=self.font[100])
            tx=int((gx0+gx1-(bbox[2]-bbox[0]))/2)
            hud.text((tx,126),label,font=self.font[100],fill='#ffffff')
        draw_stage72_evidence(final,self.stage71_evidence,
                              self.stage71_source_preview,self.font,
                              self.stage72_source_info)
        hud.text((19,838),'3D作業場は無遮蔽',font=self.font[19],fill='#b9e4c5')
        hud.text((19,876),'物理・AI制御は従来版と同一',font=self.font[16],fill='#c4d7ea')
        hud.text((19,1016),'2026/10/22 工程75',font=self.font[19],fill='#c4d7ea')
        self.writer.append_data(np.asarray(final));self.count+=1
    def close(self):self.writer.close()
