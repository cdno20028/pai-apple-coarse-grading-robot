#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Join actually recorded PyBullet MP4 episodes, not graphs or animated paths."""
from pathlib import Path
import argparse,json,subprocess,shutil,hashlib,datetime,statistics,sys
from zoneinfo import ZoneInfo

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--source',required=True)
 ap.add_argument('--output',required=True)
 args=ap.parse_args()
 src=Path(args.source).resolve(); out=Path(args.output).resolve()
 rows=json.loads((src/'worker_results.json').read_text(encoding='utf8'))
 assert rows and len(rows)%4==0, 'No complete 4-grade rounds'
 assert all(r.get('physical_and_ai_success') for r in rows), 'Not all physical tests passed; inspect source before publication'
 videos=sorted((src/'episodes').rglob('physics_trial.mp4'))
 assert len(videos)==len(rows),f'videos={len(videos)} rows={len(rows)}'
 # Unique episode directories are indexed by Stage33 trial replicate (3201..).
 def idx(path):
  s=path.parent.name
  import re
  match=re.search(r'rep(?:licate)?[_-]?(\d+)',s)
  if match:return int(match.group(1))
  # read corresponding trial.json actual config
  d=json.loads((path.parent/'trial.json').read_text(encoding='utf8'))
  return int(d['experiment']['replicate'])
 pairs=sorted(((idx(v),v) for v in videos),key=lambda x:x[0])
 assert len({k for k,v in pairs})==len(pairs),'repeated case ids'
 assert [k for k,v in pairs]==list(range(3201,3201+len(rows))),'missing or unordered episodes'
 playlist=src/'stage35_concat.txt'
 with playlist.open('w',encoding='utf8') as f:
  for _,v in pairs: f.write("file '"+str(v).replace("'", "'\\''")+"'\n")
 if shutil.which('ffmpeg') is None: raise RuntimeError('ffmpeg unavailable')
 cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0','-i',str(playlist),'-c','copy','-movflags','+faststart',str(out)]
 subprocess.run(cmd,check=True)
 assert out.exists() and out.stat().st_size>10000
 fps=20; ts=datetime.datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y/%m/%d %H:%M JST')
 summary={'created_jst':ts,'source_physics':'PyBullet DIRECT TinyRenderer, actual frame capture during run_trial',
 'clips':len(videos),'all_pass':True,'separate_trials_not_uninterrupted_operation':True,
 'no_animation_reconstructed_from_logs':True,'no_voiceover_or_explanation_slides':True,
 'reused_same_controller_and_URDF':True,'grades':[r.get('predicted_grade') for r in rows],
 'apples_are_synthetic':True,'filename':out.name,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
 'simulation_time_sum_s':round(sum(float(r['simulation_cycle_s']) for r in rows),4)}
 (src/'stage35_video_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
 print('SUCCESS',out,'clips',len(videos),'size',out.stat().st_size)
 print('This film is edited from independent real PyBullet physical trial captures; not one continuous tray-filling run.')
 return 0

if __name__=='__main__':raise SystemExit(main())
