"""Dimensional top/side schematic (not a mechanical engineering approval)."""
from PIL import Image,ImageDraw,ImageFont
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from workbench_geometry import LAYOUT

def create_diagram(path=None):
 g=LAYOUT
 path=Path(path or Path(__file__).resolve().parents[1]/'stage04_dimensioned_layout.png')
 im=Image.new('RGB',(1680,1010),(247,249,252));d=ImageDraw.Draw(im)
 font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
 font2='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
 def f(s,b=False):return ImageFont.truetype(font2 if b else font,s)
 def txt(x,y,s,size=23,fill=(33,48,67),b=False,anchor=None):d.text((x,y),s,font=f(size,b),fill=fill,anchor=anchor)
 timestamp=datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y/%m/%d %H:%M JST')
 txt(75,30,'実寸に合わせた1列型りんご選果システム｜配置図 v2.4',34,b=True)
 txt(75,87,f'作成日時：{timestamp}',19,fill=(87,98,113))
 txt(75,129,'平面図（作業台の長辺を横方向へ表示・実寸比例）',24,b=True)
 x0=110; y0=206; scale=520; w=g.bench_length_m*scale; h=g.bench_depth_m*scale; xc=x0+w/2
 d.rounded_rectangle([x0,y0,x0+w,y0+h],radius=5,fill=(211,217,226),outline=(105,116,131),width=3)
 palette={'A':(205,230,210),'B':(237,227,186),'C':(242,206,190),'D':(203,220,242)}
 for grade,cx in g.crate_centres_x_m.items():
  center=xc+cx*scale; ca=palette[grade]
  x1=center-g.crate_x_m/2*scale; x2=center+g.crate_x_m/2*scale
  y1=y0+(g.bench_depth_m-g.crate_y_m)*scale/2; y2=y1+g.crate_y_m*scale
  d.rounded_rectangle([x1,y1,x2,y2],radius=5,fill=ca,outline=(102,117,126),width=2)
  txt(center,y2-43,f'{grade} コンテナ',20,b=True,anchor='mm')
  for k in range(1,4):
   name=f'{grade}{k}'; nx=xc+g.nest_centres_x_m[name]*scale
   r=g.nest_width_m*scale/2
   d.ellipse([nx-r,y0+107-r,nx+r,y0+107+r],fill=(251,253,254),outline=(77,91,105),width=3)
   txt(nx,y0+107,name,17,b=True,anchor='mm')
 # Central scale schematic: 20cm square assumption for display only
 sw=g.scale_width_m*scale
 d.rounded_rectangle([xc-sw/2,y0+(h-sw)/2,xc+sw/2,y0+(h+sw)/2],radius=8,
                     fill=(154,194,226),outline=(74,112,149),width=3)
 txt(xc,y0+h/2-16,'中央',21,b=True,anchor='mm')
 txt(xc,y0+h/2+16,'計量台',21,b=True,anchor='mm')
 # dimension line
 d.line([x0,182,x0+w,182],fill=(91,100,112),width=2)
 for x in (x0,x0+w):d.line([x,171,x,195],fill=(91,100,112),width=2)
 txt(xc,177,'作業台の長辺 217.5cm',23,b=True,anchor='ms')
 d.line([x0-22,y0,x0-22,y0+h],fill=(91,100,112),width=2)
 for y in (y0,y0+h):d.line([x0-32,y,x0-11,y],fill=(91,100,112),width=2)
 txt(x0-54,y0+h/2,'55cm',20,anchor='mm')
 txt(x0,y0+h+20,'0cm',18)
 txt(x0+w,y0+h+20,'217.5cm',18,anchor='ra')
 txt(75,531,'側面イメージ（高さは設計仮定を含む／上下方向は比較しやすいよう別縮尺）',24,b=True)
 # side projection
 side_base=879; side_scale=325
 def sy(z):return side_base-(z-.5)*side_scale
 # bench surface z=.70, 4 bins on table
 d.rectangle([x0,sy(g.bench_top_z_m),x0+w,sy(g.bench_top_z_m-.045)],fill=(131,141,153))
 for grade,cx in g.crate_centres_x_m.items():
  center=xc+cx*scale; x1=center-g.crate_x_m/2*scale; x2=center+g.crate_x_m/2*scale
  d.rectangle([x1,sy(g.rim_z_m),x2,sy(g.bench_top_z_m)],fill=palette[grade],outline=(87,102,118),width=2)
  txt(center,sy(g.rim_z_m)+12,f'{grade} 30.5cm',17,anchor='mt')
  # raised nest rail, with support posts shown diagrammatically
  d.line([center,sy(g.bench_top_z_m),center,sy(g.nest_top_z_m)],fill=(97,108,123),width=4)
  for k in (1,2,3):
   xx=xc+g.nest_centres_x_m[f'{grade}{k}']*scale
   d.rectangle([xx-g.nest_width_m*scale/2,sy(g.nest_top_z_m)-6,
                xx+g.nest_width_m*scale/2,sy(g.nest_top_z_m)+2],fill=(59,120,161))
 # center scale raised
 d.line([xc,sy(g.bench_top_z_m),xc,sy(g.nest_top_z_m+.01)],fill=(97,108,123),width=4)
 d.rectangle([xc-sw/2,sy(g.nest_top_z_m+.01)-5,xc+sw/2,sy(g.nest_top_z_m+.01)+2],fill=(70,114,175))
 # right-hand z labels
 for z,label in [(g.nest_top_z_m,'受け台上面：約120.5cm'),(g.rim_z_m,'コンテナ上縁：約100.5cm'),(g.bench_top_z_m,'作業台上面：70cm（仮定）')]:
  yy=sy(z);d.line([1246,yy,1280,yy],fill=(97,107,118),width=2);txt(1290,yy,label,16,anchor='lm')
 txt(80,935,'実測寸法：作業台 217.5×55cm／コンテナ 37×53.5×高さ30.5cm',20)
 txt(80,969,'仮定：計量台幅20cm、すき間2cm、受け台はコンテナ上縁より20cm高い。受け台はコンテナと別部品。',17,fill=(88,95,109))
 path.parent.mkdir(exist_ok=True,parents=True);im.save(path)
 return path

if __name__=='__main__':
 print(create_diagram())
