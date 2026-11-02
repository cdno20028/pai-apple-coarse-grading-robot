#!/usr/bin/env python3
"""工程22 v4.2: one-factor return-X time sensitivity + authentic episode video capture.
All other PyBullet control, safety gates and support model are unchanged.
"""
from __future__ import annotations
import os,sys,json,zipfile,hashlib,math,csv,subprocess,traceback,statistics,shutil
from pathlib import Path
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo
HERE=Path(__file__).resolve().parent/'apple_stage13'
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE/'src'))
import pybullet as bullet
import run_physics as rp
from run_stage03 import plan
from workbench_geometry import LAYOUT
SOURCE_BASE_SHA='953e6efd8847d42000a5263e1c761e2f27be9c6a42b32f13e7f5e381e171ec01'
SOURCE_MOD_SHA='d312fe0c77456056aca5a1509e3ce90ccbdd965c892c77701270042dab1a0115'
URDF_SHA='829f6e6823806ea4d2e3766c29c4330979406cb7ac14ecc79ed009a8f86cd08d'
PREVIOUS_AVERAGE_S=3.1079166666666667
TARGET=HERE/'output'/'stage22_v4_2_return_speed_and_video'
JST=lambda: datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y/%m/%d %H:%M:%S')

def verify():
    assert hashlib.sha256((HERE/'src/run_physics.py').read_bytes()).hexdigest()==SOURCE_MOD_SHA
    assert hashlib.sha256((HERE/'robot_v14_phase30.urdf').read_bytes()).hexdigest()==URDF_SHA
    s=(HERE/'src/run_physics.py').read_text()
    assert s.count('STAGE22_V42_RETURN_SCALE')==1
    assert 'STAGE20_V40D_LOWER_RAMP_S' in s
    assert len(LAYOUT.nest_centres_x_m)==12
    assert abs(LAYOUT.nest_pitch_m-.115)<1e-9
    assert len(plan('settled_gate_09'))==10

def record_name(cond,kind,ep):
    return f'{cond}_{kind}_ep{ep:02d}'

def run_all():
    verify()
    TARGET.mkdir(exist_ok=False,parents=True)
    os.environ['STAGE20_V40D_LOWER_RAMP_S']='0.08'
    prev_settings=rp.MOTION_SETTINGS['settled_gate_09']
    rp.MOTION_SETTINGS['settled_gate_09']=replace(prev_settings,x_effective_rate=1.45)
    # Return acceleration is proposed only after support-verified release and vertical retreat.
    variants=[('return_080',.80,False),('return_065',.65,True)]
    report={'study':'stage22_v4.2_provisional_one_factor_return_speed',
      'created_jst':JST(),'baseline_v4.1_mean_s':PREVIOUS_AVERAGE_S,
      'source_physics_v4.1_sha256':SOURCE_BASE_SHA,
      'source_physics_one_line_modified_sha256':SOURCE_MOD_SHA,
      'source_urdf_10mm_sha256':URDF_SHA,
      'unchanged':['3 fingers','5 cameras','10mm support shelves','12 concave static nests at 115mm pitch',
       'scan 0.32s','10cm lift','initial descent ramp 0.08s', 'stable-X arrival gate','release clearance gate','vertical retreat','actual home verification'],
      'variable':'X return commanded duration factor; actual motor feedback still used; min command time 0.20 s',
      'simulation_only':True,'no_apple_damage_claim':True,'video_montage_generated_only_if_selected_variant_20of20':True,
      'scenarios':{},'warnings':[]}
    p=bullet;cid=p.connect(p.DIRECT);assert cid>=0
    try:
      for key,scale,record in variants:
        os.environ['STAGE22_V42_RETURN_SCALE']=str(scale)
        group={'return_scale':scale,'record_all_20_episode_videos':record,'scenarios':{},'success_20of20':False}
        print('VARIANT',key,'scale',scale,'record_video',record,flush=True)
        for scenario,profile in [('nominal_control','nominal'),('original_stress_seed','stress_mix')]:
          scene=None; rows=[]
          for ep,grade,cfg0 in plan('settled_gate_09'):
            cfg=replace(cfg0,profile=profile,replicate=101+ep,scan_duration_s=.32,torque_Nm=12.)
            print(f'TRIAL {key}/{scenario} {ep}/10 {grade}',flush=True)
            try:
              trial,scene=rp.run_trial(p,grade,TARGET/key/scenario,record_video=record,
                                  cfg=cfg,persistent_scene=scene,return_scene=True)
              ok=bool(trial.get('success_physical_and_ai') and trial.get('all_retained_fruits_ok')
                    and trial.get('return_home_pass') and trial.get('post_return_check',{}).get('ok'))
              rec={'episode':ep,'grade':grade,'success':ok,'simulated_time_s':trial.get('simulated_time_s'),
                 'return_home_pass':trial.get('return_home_pass'),
                 'post_return_check':trial.get('post_return_check',{}).get('ok'),
                 'retained_fruits_ok':trial.get('all_retained_fruits_ok'),
                 'previous_apple_contact_steps':trial.get('diagnostics',{}).get('retained_fruit_robot_contact_steps'),
                 'placement_status':trial.get('placement',{}).get('status')}
              rows.append(rec)
              print('RESULT',key,scenario,ep,'success',ok,'cycle_s',rec['simulated_time_s'],flush=True)
              if not ok:
                report['warnings'].append(f'{key}/{scenario} stopped after episode {ep}: {rec["placement_status"]}')
                break
            except Exception as e:
              rows.append({'episode':ep,'grade':grade,'success':False,'error':repr(e)})
              (TARGET/key/scenario/f'exception_ep{ep}.txt').write_text(traceback.format_exc(),encoding='utf-8')
              report['warnings'].append(f'{key}/{scenario} exception episode {ep}: {e!r}')
              print('EXCEPTION',traceback.format_exc(),flush=True)
              break
          group['scenarios'][scenario]={'executed':len(rows),'success':sum(bool(r['success']) for r in rows),
            'mean_simulated_time_success_s':statistics.mean([r['simulated_time_s'] for r in rows if r['success']]) if any(r['success'] for r in rows) else None,
            'trials':rows}
        complete=all(v['success']==10 for v in group['scenarios'].values())
        group['success_20of20']=complete
        alltimes=[r['simulated_time_s'] for s in group['scenarios'].values() for r in s['trials'] if r['success']]
        group['mean_simulated_time_success_s']=statistics.mean(alltimes) if alltimes else None
        group['mean_target_under_3s']=complete and group['mean_simulated_time_success_s']<=3.
        report['scenarios'][key]=group
        (TARGET/f'{key}_summary.json').write_text(json.dumps(group,ensure_ascii=False,indent=2),encoding='utf-8')
        print('GROUP',key,'20of20',complete,'mean_s',group['mean_simulated_time_success_s'],flush=True)
    finally:
      rp.MOTION_SETTINGS['settled_gate_09']=prev_settings
      os.environ.pop('STAGE22_V42_RETURN_SCALE',None)
      p.disconnect()
    (TARGET/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (TARGET/'SHA256_provenance.txt').write_text(f'BASE v4.1 run_physics.py: {SOURCE_BASE_SHA}\n'
      f'Stage22 one-line return profile experiment run_physics.py: {SOURCE_MOD_SHA}\n'
      f'v4.1 10mm URDF: {URDF_SHA}\n',encoding='utf-8')
    selected=report['scenarios']['return_065']
    # Build a ~2-minute editing *draft* solely from actual per-episode captured frames.
    # If any episode failed, NEVER claim a successful 20-trial video.
    if selected['success_20of20']:
      print('Preparing draft video from measured episode recordings...',flush=True)
      try:
        movie=make_draft_movie()
        report['video_draft_path']=str(movie.relative_to(TARGET))
      except Exception as e:
        report['warnings'].append('Video compilation failed while experiment data remained valid: '+repr(e))
        (TARGET/'video_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
      (TARGET/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    archive=HERE/'apple_stage22_v4.2_return_speed_video_results.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
      for f in sorted(TARGET.rglob('*')):
        if f.is_file() and (f.name in ('trial.json','history.csv','physics_trial.mp4','summary.json',
          'SHA256_provenance.txt','video_draft_approx2min.mp4','video_error.txt') or
          f.name.endswith('_summary.json') or f.name.startswith('exception_ep')):
          z.write(f,arcname=str(f.relative_to(HERE)))
      z.writestr('PROVENANCE.txt','Stage22 v4.2: paired candidate return motion time; exact v4.1 10mm URDF; v4.0d 0.08 initial Z ramp. Candidate, not accepted until audited. Recorded clips are real PyBullet episodes; montage is stitched clips, not a simulated continuous camera take.\n')
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print('STAGE22 OUTPUT',archive,'bytes',archive.stat().st_size,flush=True)
    print('RESULTS',json.dumps({k:{'success':v['success_20of20'],'mean_s':v['mean_simulated_time_success_s']} for k,v in report['scenarios'].items()},ensure_ascii=False),flush=True)
    return archive

def make_draft_movie():
    import tempfile
    import PIL.Image as I
    import PIL.ImageDraw as D
    import PIL.ImageFont as F
    print('Video editor: attempting ffmpeg, without changing physics recordings',flush=True)
    exe=shutil.which('ffmpeg')
    if not exe:raise RuntimeError('ffmpeg not available; original MP4 files are still archived')
    frames=[]
    for scenario in ('nominal_control','original_stress_seed'):
      for ep,grade,cfg0 in plan('settled_gate_09'):
        cfg=replace(cfg0,profile='nominal' if scenario=='nominal_control' else 'stress_mix',
                    replicate=101+ep,scan_duration_s=.32,torque_Nm=12.)
        f=TARGET/'return_065'/scenario/f'{grade}_{cfg.slug}'/'physics_trial.mp4'
        assert f.is_file() and f.stat().st_size>1000,f
        frames.append((scenario,ep,f))
    output=TARGET/'video_draft_approx2min.mp4'
    work=TARGET/'video_cards';work.mkdir(exist_ok=True)
    font_candidates=['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf']
    font_path=next((f for f in font_candidates if Path(f).exists()),None)
    font=F.truetype(font_path,26) if font_path else F.load_default()
    big=F.truetype(font_path,37) if font_path else F.load_default()
    time_stamp=JST()+' JST'
    def slide(name,title,lines,duration):
      im=I.new('RGB',(640,448),(22,36,50));d=D.Draw(im)
      d.rectangle((0,0,640,9),fill=(66,157,198))
      d.text((32,40),title,font=big,fill=(245,250,255))
      for i,l in enumerate(lines):d.text((35,145+i*45),l,font=font,fill=(208,232,248))
      d.text((22,400),time_stamp,font=F.truetype(font_path,17) if font_path else font,fill=(169,190,208))
      png=work/f'{name}.png';vid=work/f'{name}.mp4';im.save(png)
      cmd=[exe,'-y','-hide_banner','-loglevel','error','-loop','1','-i',str(png),'-t',str(duration),
           '-vf','fps=20,format=yuv420p','-r','20','-c:v','libx264','-preset','veryfast','-crf','24',str(vid)]
      r=subprocess.run(cmd,capture_output=True,text=True)
      if r.returncode:raise RuntimeError('card failed: '+r.stderr[-400:])
      return vid
    opening=slide('intro','Matsuo Lab / Physical AI',[
       'Apple sorting robot — Stage22 v4.2',
       '3 fingers / 5 cameras / 12 concave nests',
       'Observed simulation footage; 20 episodes',
       '10-mm lower support ledges'],7)
    middle=slide('force','Stage21 v4.1 / support forces',[
       '20 / 20 placements succeeded',
       'Median upward normal force: 104.25% of weight',
       'Median signed friction: -4.69% of weight',
       'Force balance corrected; no bruise claim'],7)
    ending=slide('end','Submission video: editing draft',[
       'All videos captured from real Bullet runs.',
       'Episodes have been concatenated.',
       'Robot returns to center each cycle.',
       'Further review required before YouTube.'],8)
    ordered=[opening]+[f for s,e,f in frames if s=='nominal_control']+[middle]+[f for s,e,f in frames if s=='original_stress_seed']+[ending]
    playlist=work/'concat.txt';playlist.write_text(''.join("file '"+str(f).replace("'","'\\''")+"'\n" for f in ordered),encoding='utf-8')
    r=subprocess.run([exe,'-y','-hide_banner','-loglevel','error','-f','concat','-safe','0','-i',str(playlist),
        '-r','20','-vf','fps=20,scale=640:448,format=yuv420p','-c:v','libx264',
        '-preset','veryfast','-crf','23','-movflags','+faststart',str(output)],capture_output=True,text=True)
    if r.returncode or not output.is_file():raise RuntimeError('video concatenation failed: '+r.stderr[-1000:])
    print('VIDEO',output,'bytes',output.stat().st_size,flush=True)
    return output

if __name__=='__main__':run_all()
