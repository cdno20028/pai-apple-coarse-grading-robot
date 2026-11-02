#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P105 final video composition: one background photo, synchronous P104 physics
plus two genuine P105 observer-camera MP4s. NO old trial video is reused.

Preview: python p105_compose.py --physics P104...mp4 --photo p96_photo.jfif --preview P105_layout.png
Final:   python p105_compose.py --physics <P105-run original world MP4> \
         --grip <P105-run live grip MP4> --follow <P105-run live follow MP4> \
         --verdict <P105_CAPTURE_VERDICT.json> --photo p96_photo.jfif --out PAI_final_p105.mp4
"""
import argparse,datetime,hashlib,json,os,subprocess,tempfile
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
JST=datetime.timezone(datetime.timedelta(hours=9))
STAMP=datetime.datetime.now(JST).strftime('%Y/%m/%d %H:%M:%S JST')
W,H=1920,1080
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

def font(size):
    return ImageFont.truetype(FONT,size) if Path(FONT).is_file() else ImageFont.load_default()

def cover(image, size):
    return ImageOps.fit(image,size,method=Image.Resampling.LANCZOS,centering=(.5,.47))

def screen(photo,physics_frame,camera_frame=None,mode='本編',info='',out_size=(1920,1080)):
    # The background is ONE unmirrored photo. All panels are animated 3D or AI.
    # Reuse the approved P98R2 full-frame composite background without crop/cover.
    bg=photo.resize((W,H),Image.Resampling.LANCZOS).convert('RGB')
    shade=Image.new('RGBA',(W,H),(5,19,31,0))
    d=ImageDraw.Draw(shade)
    d.rectangle((0,427,W,H),fill=(5,17,29,239))
    d.rounded_rectangle((18,16,720,298),radius=14,fill=(8,24,37,203))
    bg=Image.alpha_composite(bg.convert('RGBA'),shade)
    d=ImageDraw.Draw(bg)
    d.text((42,39),'Physical AI｜りんご粗選果',font=font(42),fill='white')
    d.text((44,98),'外観AI → 3本指で把持 → 等級別に搬送',font=font(25),fill='#dbeef3')
    d.text((44,147),'4等級 A / B / C / D　・　5カメラ',font=font(23),fill='#e9dcb7')
    d.text((44,194),'工程98R2承認背景：果樹園＋装置イメージ（実機ではない）',font=font(18),fill='#f4dcb0')
    d.text((44,247),STAMP,font=font(16),fill='#c2c9c9')
    if physics_frame is None:raise ValueError('P104 physics frame required')
    p=Image.fromarray(cv2.cvtColor(physics_frame,cv2.COLOR_BGR2RGB))
    if p.size!=(1920,1080):raise ValueError(f'P104 expected 1920x1080; got {p.size}')
    world=p.crop((332,190,1367,900))
    cams=p.crop((1390,190,1910,900))
    audit=p.crop((0,8,320,900))
    if camera_frame is not None:
        world=Image.fromarray(cv2.cvtColor(camera_frame,cv2.COLOR_BGR2RGB))
    # Respect native aspect ratios; no distortion. Dedicated panels no overlap.
    boxes=[((16,440,1002,1004),world,'P104 物理動作' if camera_frame is None else mode),
           ((1021,440,1548,1004),cams,'同一時刻の5カメラ'),
           ((1568,440,1903,1004),audit,'AI判定・根拠')]
    d=ImageDraw.Draw(bg)
    for (x0,y0,x1,y1),frame,title in boxes:
        d.rounded_rectangle((x0-3,y0-3,x1+3,y1+3),radius=8,fill='#0d2b3f',outline='#91b9ca',width=2)
        ww,hh=x1-x0,y1-y0
        frame=ImageOps.contain(frame,(ww-8,hh-53),Image.Resampling.BICUBIC)
        bg.alpha_composite(frame.convert('RGBA'),(x0+(ww-frame.width)//2,y0+44+(hh-53-frame.height)//2))
        d=ImageDraw.Draw(bg)
        d.text((x0+11,y0+8),title,font=font(20),fill='white')
    d.rectangle((0,1017,W,H),fill=(7,22,35,248))
    d.text((24,1030),info or 'P104実録／写真は設置イメージ／同時刻AI入力',font=font(20),fill='white')
    return cv2.cvtColor(np.asarray(bg.convert('RGB')),cv2.COLOR_RGB2BGR)

def load_first(path,t=8):
    c=cv2.VideoCapture(str(path));fps=c.get(cv2.CAP_PROP_FPS)
    assert fps>0
    c.set(cv2.CAP_PROP_POS_MSEC,1000*t)
    ok,frame=c.read();c.release()
    if not ok:raise RuntimeError('VIDEO_FRAME_READ_FAILED '+str(path))
    return frame

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--physics',required=True)
    ap.add_argument('--photo',required=True)
    ap.add_argument('--preview')
    ap.add_argument('--grip');ap.add_argument('--follow');ap.add_argument('--verdict')
    ap.add_argument('--out')
    o=ap.parse_args()
    photo=Image.open(o.photo).convert('RGB')
    if o.preview:
        frame=screen(photo,load_first(o.physics,8),info='レイアウト確認｜P104成功実録（追従・把持カメラは未収録）')
        # The output file itself contains the Japan-time stamp above.
        cv2.imwrite(o.preview,frame)
        print('P105_LAYOUT_PREVIEW',o.preview,Path(o.preview).stat().st_size,sha(o.preview));return
    if not all([o.grip,o.follow,o.verdict,o.out]):
        raise SystemExit('STOP: Final requires REAL P105 grip/follow camera MP4s and P105 verdict; no fake follow crop.')
    v=json.loads(Path(o.verdict).read_text('utf8'))
    if not v.get('reproduction_claim') or not v.get('rerun_physics_10of10'):
        raise SystemExit('STOP: P105 10/10 camera+physics not independently verified')
    cams=v.get('extra_cameras',{})
    if cams.get('palm_x_motion_range_m',0)<=.08 or cams.get('errors'):
        raise SystemExit('STOP: no verified physically moving 3D camera')
    src={}
    for k,fn in [('main',o.physics),('grip',o.grip),('follow',o.follow)]:
        c=cv2.VideoCapture(str(fn))
        if not c.isOpened():raise RuntimeError('UNOPENABLE '+str(fn))
        fps=c.get(cv2.CAP_PROP_FPS);n=int(c.get(cv2.CAP_PROP_FRAME_COUNT))
        if abs(fps-20)>0.05:raise RuntimeError(f'FPS mismatch {k}: {fps}')
        src[k]={'video':c,'frames':n,'file':fn}
    n=src['main']['frames']
    if src['grip']['frames']!=n or src['follow']['frames']!=n:
        raise RuntimeError('CAMERA_FRAME_COUNT_OR_SYNC_MISMATCH')
    # Replays are clearly labelled; not new experiments. Max slowed motion 0.5x.
    plan=[
      ('本編',0,n,1.0,'P104連続10ケース／物理シミュレーション実録 1.0倍'),
      ('把持アップ',20,100,.5,'同じ物理試行の別視点／3本指把持 0.5倍再生'),
      ('スライダー追従',140,240,.5,'実際の3D追従カメラ／P104の同一動作 0.5倍再生'),
      ('C-1円形傷',max(0,n-92),n,.5,'C級円形傷をAI判定 → C2へ搬送／0.5倍リプレイ'),
    ]
    cmd=['ffmpeg','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo',
         '-s',f'{W}x{H}','-pix_fmt','bgr24','-r','20','-i','-',
         '-an','-c:v','libx264','-pix_fmt','yuv420p','-preset','veryfast','-crf','22',o.out]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    total=0
    try:
      for mode,a,b,speed,tag in plan:
        a=max(0,a);b=min(b,n)
        for original in range(a,b):
          frames={}
          for k,inp in src.items():
            cap=inp['video'];cap.set(cv2.CAP_PROP_POS_FRAMES,original)
            ok,frame=cap.read()
            if not ok:raise RuntimeError(f'FRAME_READ_FAILED {k} {original}')
            frames[k]=frame
          chosen=frames['grip'] if mode=='把持アップ' else (frames['follow'] if mode=='スライダー追従' else None)
          view=screen(photo,frames['main'],chosen,mode,tag)
          repetitions=2 if speed==.5 else 1
          for _ in range(repetitions):proc.stdin.write(view.tobytes());total+=1
    finally:
      try:proc.stdin.close()
      except:pass
      rc=proc.wait()
      for inp in src.values():inp['video'].release()
    if rc:raise RuntimeError('FFMPEG_ENCODER_FAILED')
    output=Path(o.out)
    assert output.is_file() and output.stat().st_size>10000
    audit={'date_jst':STAMP,'src':{k:{'file':z['file'],'sha256':sha(z['file'])} for k,z in src.items()},
           'one_photo_sha256':sha(o.photo),'output_file':str(output),'frames':total,
           'duration_s':total/20,'output_sha256':sha(output),'source_is_p105_10of10':True,
           'no_old_trial_video':True,'live_camera_3d':True}
    af=output.with_suffix('.json');af.write_text(json.dumps(audit,indent=2,ensure_ascii=False),'utf8')
    print('P105_FINAL_COMPOSED',output,output.stat().st_size,sha(output))
if __name__=='__main__':main()
