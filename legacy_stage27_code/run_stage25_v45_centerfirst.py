#!/usr/bin/env python3
"""工程25 v4.5: synthetic learned A/B/C/D dashboard + real Bullet 3/3 full gate.

Physical motion is inherited from Stage22 v4.2 65% X return, Stage20 ramp .08 s.
Worker C is a *scripted external action*, NOT a real simulated human body.
"""
from __future__ import annotations
import os,sys,shutil,hashlib,json,zipfile,traceback,statistics,subprocess,argparse,copy
from pathlib import Path
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo
BASE=Path(__file__).resolve().parent/'apple_stage13'
sys.path.insert(0,str(BASE));sys.path.insert(0,str(BASE/'src'))
from stage23_full_supervisor import FullGradeSupervisor,test_with_fake_physics
from workbench_geometry import LAYOUT
from run_stage03 import plan
from learned_classifier import LABELS
from PIL import Image,ImageDraw,ImageFont
import pybullet as p
import run_physics as rp

SOURCE22_SHA='d312fe0c77456056aca5a1509e3ce90ccbdd965c892c77701270042dab1a0115'
SOURCE23_SHA='2f9a4daba4ca77e8b5c78d0b6a9aad7a5710a290729f545b7bfde4befaae2dc9'
URDF_SHA='829f6e6823806ea4d2e3766c29c4330979406cb7ac14ecc79ed009a8f86cd08d'
NOW=lambda:datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y/%m/%d %H:%M JST')
# Three A slots fill; the fourth A triggers a stop and explicit worker-C event.
# B,C,D branch through the exact same trained classifier and 12-nest controller.
SEQUENCE=['A','B','C','D']*4

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify():
    assert sha(BASE/'src/run_physics.py')==SOURCE23_SHA
    assert sha(BASE/'robot_v14_phase30.urdf')==URDF_SHA
    assert len(LAYOUT.nest_centres_x_m)==12 and abs(LAYOUT.nest_pitch_m-.115)<1e-10
    assert 'STAGE20_V40D_LOWER_RAMP_S' in (BASE/'src/run_physics.py').read_text()
    assert 'STAGE22_V42_RETURN_SCALE' in (BASE/'src/run_physics.py').read_text()
    assert 'STAGE23_AFTER_INFERENCE_HOOK' in (BASE/'src/run_physics.py').read_text()
    assert len(SEQUENCE)==16 and all(SEQUENCE.count(g)==4 for g in 'ABCD')
    assert test_with_fake_physics()

FONTCAND=['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
def font(size):
    fp=next((p for p in FONTCAND if Path(p).is_file()),None)
    return ImageFont.truetype(fp,size) if fp else ImageFont.load_default()

def dashboard(trial,loc,episode,events,filename):
    """Faithful *post-run visual reconstruction*, not a realtime AI interface."""
    W,H=1280,720
    bg=(14,26,40);accent=(62,175,202);txt=(239,245,250);dim=(155,174,192)
    grade=str(trial.get('classification',{}).get('class') or '?')
    cls=trial.get('classification',{})
    fill_now=[r['nest'] for r in trial.get('retained_nests_after_return',[]) if r.get('ok')]
    predicted=[e for e in events if e['event']=='AI_DECISION']
    before=predicted[-1]['occupied'] if predicted else '?'
    stopped=any(e['event']=='THREE_FULL_STOP' for e in events)
    im=Image.new('RGB',(W,H),bg);d=ImageDraw.Draw(im)
    d.rectangle((0,0,W,85),fill=(22,46,66));d.text((30,15),'PHYSICAL AI | APPLE GRADING',font=font(37),fill=txt)
    d.text((960,26),f'EP {episode:02d} / 16',font=font(22),fill=dim)
    d.text((30,101),'5 simulated camera viewpoints | recorded PyBullet frames',font=font(23),fill=txt)
    pics=['finger_1','finger_2','finger_3','palm','plate']
    for j,name in enumerate(pics):
        x=30+j*247;y=139
        f=loc/f'{name}_lift.png'
        d.rounded_rectangle((x,y,x+221,y+192),radius=12,fill=(29,48,64))
        if f.is_file():
            pic=Image.open(f).convert('RGB');pic.thumbnail((176,150))
            im.paste(pic,(x+22,y+12))
        else:d.text((x+30,y+72),'CAMERA NOT SAVED',font=font(15),fill=dim)
        d.text((x+19,y+166),name.upper(),font=font(18),fill=txt)
    d.rectangle((30,352,598,642),fill=(23,44,60))
    d.text((48,369),'AI INFERENCE / SYNTHETIC TRAINING',font=font(23),fill=txt)
    scores=cls.get('scores') or {}
    for i,g in enumerate('ABCD'):
        y=414+i*50
        prob=float(scores.get(g,0))
        d.text((57,y),g,font=font(24),fill=txt)
        d.rectangle((104,y+7,500,y+30),fill=(51,71,86))
        d.rectangle((104,y+7,104+int(396*prob),y+30),fill=(68,189,198) if g==grade else (98,117,135))
        d.text((514,y),f'{prob:.1%}',font=font(19),fill=txt)
    d.rectangle((620,352,1247,642),fill=(23,44,60))
    mass=trial.get('scale_mass_kg_measured')
    d.text((644,367),f'PREDICTED: {grade}',font=font(37),fill=(108,228,163))
    d.text((644,420),f'Scale: {mass:.3f} kg' if isinstance(mass,(float,int)) else 'Scale: unavailable',font=font(23),fill=txt)
    d.text((644,455),f'Destination: {trial.get("assigned_nest") or "NONE"}',font=font(24),fill=txt)
    d.text((644,492),f'{grade} pre-placement: {before} / 3 occupied',font=font(23),fill=txt)
    d.text((644,528),'3/3 FULL -> STOP / C CLEAR / RESET' if stopped else 'FREE SLOT -> TRANSPORT',
           font=font(20),fill=(255,195,98) if stopped else (108,228,163))
    d.text((644,576),'12 nests / 3 per grade / 115 mm pitch',font=font(18),fill=dim)
    d.text((30,667),'Recorded measurements, dashboard reconstructed AFTER trial. Not real-fruit grading.  '+NOW(),font=font(17),fill=dim)
    im.save(filename)


def make_card(path,title,lines,color=(24,39,54)):
    im=Image.new('RGB',(1280,720),color);d=ImageDraw.Draw(im)
    d.rectangle((0,0,1280,13),fill=(79,173,195));d.text((62,71),title,font=font(52),fill='white')
    for j,line in enumerate(lines):d.text((65,190+j*66),line,font=font(27),fill=(209,228,238))
    d.text((65,655),NOW(),font=font(20),fill=(154,175,195))
    im.save(path)

def make_montage(root,records,events):
    ff=shutil.which('ffmpeg')
    if not ff:return {'status':'FFMPEG_NOT_INSTALLED'}
    cards=root/'cards';cards.mkdir(exist_ok=True)
    start=cards/'opening.png'
    make_card(start,'Physical AI / Stage25 v4.5',[
      '3-finger lower support  |  5 simulated camera viewpoints',
      'Synthetic learned colour+mass A/B/C/D inference',
      '12 concave nests  |  3-full stop and worker-C clear',
      'Test video: recorded Bullet episodes; not a continuous take'])
    end=cards/'ending.png'
    make_card(end,'STAGE24 RESULT / 4-GRADE FULL CAPACITY',[
      f'Physical trials passed: {sum(bool(r["success"]) for r in records)} / {len(records)}',
      'Worker C is a scripted external action, not a simulated person.',
      'Human-detection and injury prevention are not certified.',
      'Real-apple grading / bruising still unverified.'])
    sources=[(start,7.)]
    for row in records:
        if not row['success']:break
        movie=Path(row['physical_video']);dashboard_img=Path(row['dashboard'])
        assert movie.exists() and dashboard_img.exists()
        sources.append((movie,None))
        sources.append((dashboard_img,4.5))
        if row['full_stop']:
            full=cards/f'full_stop_{row["episode"]}.png'
            make_card(full,f'{row["predicted"]}1 {row["predicted"]}2 {row["predicted"]}3 : 3/3 FULL',[
               f'Actual synthetic AI predicted {row["predicted"]}; all three {row["predicted"]} nests occupied.',
               'No transfer into an occupied nest. Robot paused while holding fruit.',
               f'Worker C: external scripted removal of {row["predicted"]}1.',
               f'Physical sensor confirmed {row["predicted"]}1 clear; explicit reset and resume.',
               'No human body or safety-rated detection simulated.'],(51,37,32))
            sources.append((full,8.))
    sources.append((end,8.))
    clips=[]
    for i,(source,duration) in enumerate(sources):
        temp=cards/f'{i:02d}.mp4'
        cmd=[ff,'-y','-hide_banner','-loglevel','error']
        if duration is not None:
            cmd+=['-loop','1','-framerate','20','-i',str(source),'-t',str(duration)]
        else:cmd+=['-i',str(source)]
        cmd+=['-vf','scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=20,format=yuv420p',
              '-an','-c:v','libx264','-preset','veryfast','-crf','25','-movflags','+faststart',str(temp)]
        r=subprocess.run(cmd,capture_output=True,text=True)
        if r.returncode:raise RuntimeError('Video render failed: '+r.stderr[-600:])
        clips.append(temp)
    playlist=cards/'concat.txt'
    playlist.write_text(''.join(f"file '{f.absolute()}'\n" for f in clips),encoding='utf-8')
    final=root/'stage25_v4.5_centerfirst_integrated_demo_DRAFT.mp4'
    r=subprocess.run([ff,'-y','-hide_banner','-loglevel','error','-f','concat','-safe','0','-i',str(playlist),
        '-c','copy','-movflags','+faststart',str(final)],capture_output=True,text=True)
    if r.returncode:raise RuntimeError('Montage concat failed: '+r.stderr[-600:])
    return {'status':'CREATED','path':str(final),'bytes':final.stat().st_size}


def checkpoint(root,drive,report):
    (root/'progress.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    if drive:
        dest=Path(drive);dest.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/'progress.json',dest/'progress.json')
        # Store each completed episode's small logs and video immediately.
        for f in root.rglob('*'):
            if f.is_file() and f.suffix in ('.json','.png','.mp4') and 'cards' not in f.parts:
                if f.name=='progress.json':continue
                target=dest/f.relative_to(root)
                if target.is_file() and target.stat().st_size==f.stat().st_size:continue
                target.parent.mkdir(exist_ok=True,parents=True)
                shutil.copy2(f,target)

def run(drive_dir=None):
    verify()
    os.environ['APPLE_NEST_ORDER']='nearest'
    root=BASE/'output'/'stage25_v4_5_centerfirst_full'
    if root.exists():raise RuntimeError('Previous output exists; do not overwrite measured work: '+str(root))
    root.mkdir(parents=True)
    os.environ['STAGE20_V40D_LOWER_RAMP_S']='0.08'
    os.environ['STAGE22_V42_RETURN_SCALE']='0.65'
    prev=rp.MOTION_SETTINGS['settled_gate_09']
    rp.MOTION_SETTINGS['settled_gate_09']=replace(prev,x_effective_rate=1.45)
    supervisor=FullGradeSupervisor()
    rp.STAGE23_AFTER_INFERENCE_HOOK=supervisor.process
    base_cfg={g:cfg for _,g,cfg in plan('settled_gate_09') if g in LABELS}
    summary={'version':'4.5','stage':25,'timestamp_jst':NOW(),
      'study':'integrated physical 3/3 full-stop + synthetic camera/scale learned classification',
      'baseline_v4.2_return_scale':.65,'ramp_s':.08,
      'source22_sha256':SOURCE22_SHA,'source23_sha256':SOURCE23_SHA,
      'unchanged_10mm_urdf_sha256':URDF_SHA,
      'not_verified':['real human safety','actual worker C human animation','real fruit defect grading','fruit damage','true continuous single-take video'],
      'stop_trigger':'same predicted grade 3/3 sensed nests occupied AFTER camera+scale inference, BEFORE X transfer',
      'worker_event':'scripted removal; real previous apple body deleted, new apple never teleported',
      'recordings_montage':'separate physics episode recordings reconstructed as editing draft',
      'results':[],'worker_removals':0,'all_success':False}
    records=[];scene=None
    cid=p.connect(p.DIRECT);assert cid>=0
    try:
        for idx,label in enumerate(SEQUENCE,1):
            cfg=replace(base_cfg[label],profile='nominal',replicate=200+idx,
                scan_duration_s=.32,torque_Nm=12.)
            print('TRIAL',idx,'/',len(SEQUENCE),'expected synthetic label',label,flush=True)
            evstart=len(supervisor.events)
            try:
                row,scene=rp.run_trial(p,label,root/'episodes',record_video=True,
                    cfg=cfg,persistent_scene=scene,return_scene=True)
                good=bool(row.get('classification',{}).get('class')==label and (row.get('assigned_nest') or '').startswith(label) and row.get('success_physical_and_ai') and row.get('all_retained_fruits_ok')
                    and row.get('return_home_pass') and (row.get('post_return_check') or {}).get('ok'))
                event_slice=supervisor.events[evstart:]
                full=any(e['event']=='THREE_FULL_STOP' for e in event_slice)
                loc=root/'episodes'/f'{label}_{cfg.slug}'
                r={'episode':idx,'label_for_evaluation_only':label,
                   'predicted':row.get('classification',{}).get('class'),
                   'confidence':row.get('classification',{}).get('confidence'),
                   'assigned_nest':row.get('assigned_nest'),
                   'success':good,'full_stop':full,'cycle_s_including_pause':row.get('simulated_time_s'),
                   'retained_count':row.get('placed_fruit_count'),
                   'physical_video':str(loc/'physics_trial.mp4'),
                   'dashboard':str(root/f'dashboard_ep{idx:02d}.png'),
                   'events':event_slice}
                (root/f'events_ep{idx:02d}.json').write_text(json.dumps(event_slice,ensure_ascii=False,indent=2),encoding='utf8')
                dashboard(row,loc,idx,event_slice,root/f'dashboard_ep{idx:02d}.png')
                records.append(r)
                print('RESULT',idx,'predicted',r['predicted'],'nest',r['assigned_nest'],
                      'success',good,'FULL_STOP',full,'time',r['cycle_s_including_pause'],flush=True)
            except Exception as e:
                (root/f'exception_ep{idx:02d}.txt').write_text(traceback.format_exc(),encoding='utf8')
                records.append({'episode':idx,'label_for_evaluation_only':label,'success':False,'error':repr(e)})
                print('ERROR episode',idx,repr(e),flush=True)
                break
            finally:
                summary['results']=records
                summary['worker_removals']=supervisor.simulated_worker_removals
                (root/'events_all.json').write_text(json.dumps(supervisor.events,ensure_ascii=False,indent=2),encoding='utf8')
                checkpoint(root,drive_dir,summary)
            if not records[-1]['success']:break
    finally:
        p.disconnect()
        rp.MOTION_SETTINGS['settled_gate_09']=prev
        del rp.STAGE23_AFTER_INFERENCE_HOOK
        os.environ.pop('STAGE22_V42_RETURN_SCALE',None)
        os.environ.pop('STAGE20_V40D_LOWER_RAMP_S',None)
        os.environ.pop('APPLE_NEST_ORDER',None)
    summary['all_success']=len(records)==len(SEQUENCE) and all(r['success'] for r in records)
    summary['full_stop_confirmed']=bool(sum(r.get('full_stop',False) for r in records)==4)
    summary['worker_removal_confirmed']=bool(supervisor.simulated_worker_removals==4)
    summary['physical_recheck_confirmed']=bool(any(e['event']=='RESUME' for e in supervisor.events))
    summary['all_grades_full_stop_resume']={g:bool(sum(e['event']=='THREE_FULL_STOP' and e['grade']==g for e in supervisor.events)==1 and sum(e['event']=='RESUME' and e['grade']==g for e in supervisor.events)==1) for g in 'ABCD'}
    summary['all_grades_verified']=bool(all(summary['all_grades_full_stop_resume'].values()))
    if summary['all_success']:
        summary['mean_cycle_s_with_scripted_stop']=statistics.mean(r['cycle_s_including_pause'] for r in records)
        summary['mean_cycle_s_excluding_scripted_pause']=(sum(r['cycle_s_including_pause'] for r in records)-
            (supervisor.pause_steps+12)*rp.DT)/len(records)
        try:summary['video_draft']=make_montage(root,records,supervisor.events)
        except Exception as e:summary['video_draft']={'status':'FAILED','error':repr(e)}
    (root/'events_all.json').write_text(json.dumps(supervisor.events,ensure_ascii=False,indent=2),encoding='utf8')
    checkpoint(root,drive_dir,summary)
    archive=BASE/'apple_stage25_v4.5_centerfirst_full_results.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for f in sorted(root.rglob('*')):
            if f.is_file() and ('cards' not in f.parts) and (f.name in ('trial.json','history.csv','physics_trial.mp4')
                or f.name.startswith(('dashboard_','events_','exception_')) or f.name=='progress.json'
                or f.name=='stage25_v4.5_centerfirst_integrated_demo_DRAFT.mp4'):
                z.write(f,arcname=str(f.relative_to(BASE)))
        z.writestr('PROVENANCE.txt',NOW()+' | Stage25 v4.5 centre-first 16-trial physical test; Stage23 v4.3 model cached; inherited from Stage22 v4.2 return_065, 0.08s ramp, fixed 10mm supports. Synthetic learned classifier. Worker is abstract external event; not safety-certified.\n')
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print('RESULT ZIP',archive,'bytes',archive.stat().st_size,'success',summary['all_success'],flush=True)
    if drive_dir:
        dest=Path(drive_dir)/archive.name
        shutil.copy2(archive,dest)
        assert sha(dest)==sha(archive)
        print('DRIVE VERIFIED ZIP:',dest,'SHA256 MATCH',sha(dest),flush=True)
    if not (summary['all_success'] and summary['full_stop_confirmed']
            and summary['worker_removal_confirmed']
            and summary['physical_recheck_confirmed'] and summary['all_grades_verified']):
        raise RuntimeError('INTEGRATED_PHYSICS_NOT_PASSED: partial logs/ZIP retained; '
                           'never call incomplete/full-stop-untested run a success')
    return archive

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--drive-dir',default=None)
    args=parser.parse_args()
    x=run(args.drive_dir)
    print('DONE',x,flush=True)
