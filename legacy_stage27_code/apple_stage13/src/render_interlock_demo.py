"""Create an event-by-event state diagram video, NOT a PyBullet physics result."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from PIL import Image,ImageDraw,ImageFont
import numpy as np
import imageio.v2 as imageio
from workcell_interlocks import run_four_scenarios
from workbench_geometry import LAYOUT

TITLES={'normal_sort':'通常運転：画像判定→受け台','three_full_hold':'満杯：4個目保持で停止→解除',
        'human_near_stop':'人の接近：停止→確認後再開','no_input_wait':'投入なし：待機→再開'}
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
def F(n,b=False):
 for path in ((BOLD if b else FONT), '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
  if Path(path).is_file():return ImageFont.truetype(path,n)
 return ImageFont.load_default()


def render_frame(scenario,entry,timestamp,w=1080,h=610):
 g=LAYOUT
 im=Image.new('RGB',(w,h),'#f6f9fc');d=ImageDraw.Draw(im)
 def text(x,y,s,n=22,color='#203044',b=False):d.text((x,y),s,font=F(n,b),fill=color)
 d.rectangle([0,0,w,92],fill='#153335')
 text(23,13,'工程05｜停止・再開ロジック（仮想センサー入力）',30,'white',True)
 text(23,54,TITLES[scenario],21,'#c7f0e8')
 text(30,111,'作業台217.5cm｜コンテナ4個・クッション受け台12枠',21,b=True)
 x0=60;unit=432 # px/m
 xc=x0+(g.bench_length_m/2)*unit
 xx=lambda x:xc+x*unit
 yc=293
 d.rounded_rectangle([x0,yc-g.bench_depth_m*unit/2,x0+g.bench_length_m*unit,yc+g.bench_depth_m*unit/2],6,fill='#dbe1e9',outline='#697988',width=2)
 palette={'A':'#cae7d0','B':'#efdfad','C':'#ebc4b4','D':'#c8d9ed'}
 for grade,cx in g.crate_centres_x_m.items():
  x=xx(cx);w2=g.crate_x_m*unit/2
  d.rectangle([x-w2,yc-g.crate_y_m*unit/2,x+w2,yc+g.crate_y_m*unit/2],fill=palette[grade],outline='#596b79',width=2)
  text(x-34,yc+77,grade+'箱',19,b=True)
  for i in range(1,4):
   name=f'{grade}{i}';nx=xx(g.nest_centres_x_m[name]);yy=yc;r=20
   is_used=False
   detail=entry.get('detail','')
   if (scenario=='three_full_hold' and name.startswith('A')):
    # Visual occupancy for this scripted scenario (not a PyBullet sensor state).
    events=entry['event'];idx=entry['index']
    if (name=='A1' and idx>=4) or (name=='A2' and idx>=5 and idx<7) or (name=='A3' and idx>=6):is_used=True
    if idx>=8 and name=='A2':is_used=True
   if (scenario=='normal_sort' and idx_is_done(entry)):is_used=name=='C1'
   if (scenario=='human_near_stop' and idx_is_done(entry)):is_used=name=='B1'
   if (scenario=='no_input_wait' and idx_is_done(entry)):is_used=name=='D1'
   d.ellipse([nx-r,yy-r,nx+r,yy+r],fill='#d17338' if is_used else 'white',outline='#526878',width=2)
   d.text((nx,yy),name,font=F(15,True),fill='white' if is_used else '#243448',anchor='mm')
  # supports from rear exterior; cross-beams are above crate rim
  post_y=yc- g.support_post_y_m*unit
  d.rectangle([x-4,post_y-4,x+4,post_y+4],fill='#263544')
  d.line((x,post_y,x,yc),fill='#405161',width=2)
 # central scale
 d.rounded_rectangle([xc-40,yc-40,xc+40,yc+40],4,fill='#90b8d8',outline='#536c81',width=2)
 d.text((xc,yc),'計量',font=F(19,True),anchor='mm',fill='#213444')
 state=entry['state'];grasped=entry['grasped']
 colour='#a32233' if state in ('STOP_HUMAN','HOLD_FULL','WAIT_FULL') else '#1a6e77'
 d.rounded_rectangle((60,447,1010,530),9,fill='#e6edf3',outline='#bdcad6',width=2)
 text(80,461,f'入力イベント：{entry["event"]}',23,b=True)
 text(80,493,f'ロボット状態：{state}  ／  {"停止" if not entry["motion_permitted"] else "動作可"}',23,colour,True)
 if grasped:
  d.ellipse([xc-11,yc-63,xc+11,yc-41],fill='#e68c3b',outline='#8d4c1f',width=2)
  text(816,170,'● 把持中',18,color='#9a5724')
 if scenario=='human_near_stop' and (state=='STOP_HUMAN'):
  d.ellipse([xc+g.crate_centres_x_m['B']*unit-65,yc+70,xc+g.crate_centres_x_m['B']*unit-21,yc+114],fill='#eb9a8b',outline='#b24642',width=3)
  text(720,170,'危険領域入力=ON',19,color='#ac313d',b=True)
 text(28,558,'これは状態制御ロジックの可視化。物理シミュレーション・安全認証・連続選果の証拠ではない。',17,'#8a3a42')
 text(28,582,timestamp,16,'#526579')
 return np.asarray(im)

def idx_is_done(e):return e['event']=='RELEASED'

def main(outdir=None):
 out=Path(outdir or Path(__file__).resolve().parents[1]/'output'/'stage05')
 out.mkdir(parents=True,exist_ok=True)
 timestamp=datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y/%m/%d %H:%M JST')
 data=run_four_scenarios()
 (out/'stage05_state_logic_results.json').write_text(json.dumps({'timestamp_jst':timestamp,'kind':'deterministic virtual event/state controller, NOT PyBullet',
     'scenarios':data},ensure_ascii=False,indent=2),encoding='utf-8')
 mp4=out/'stage05_logic_demonstration_NOT_PHYSICS.mp4'
 with imageio.get_writer(mp4,fps=12,codec='libx264',quality=7,macro_block_size=2) as wr:
  for key,content in data.items():
   for item in content['trace']:
    frame=render_frame(key,item,timestamp)
    for _ in range(6):wr.append_data(frame)
   for _ in range(10):wr.append_data(frame)
 print('Video:',mp4,'Size:',mp4.stat().st_size)
 print('State traces:',out/'stage05_state_logic_results.json')
 return mp4
if __name__=='__main__':main()
