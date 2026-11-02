"""Render Stage 05 overview, indicating external rear posts and non-penetrating crossbeams."""
from PIL import Image,ImageDraw,ImageFont
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from workbench_geometry import LAYOUT

def create_diagram(path=None):
 g=LAYOUT
 path=Path(path or Path(__file__).resolve().parents[1]/'stage05_layout.png')
 im=Image.new('RGB',(1600,1080),'#f8fafc');d=ImageDraw.Draw(im)
 f='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
 fb='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
 def txt(x,y,s,size=23,fill='#203045',bold=False,anchor=None):
  d.text((x,y),s,font=ImageFont.truetype(fb if bold else f,size),fill=fill,anchor=anchor)
 x0=145; scale=575; X=lambda x: x0+(x+g.bench_length_m/2)*scale
 yy=345;Y=lambda y:yy-(y*scale)
 txt(70,30,'工程05｜作業台と受け台の配置（背面外付け支柱に修正）',34,bold=True)
 txt(70,84,'作成：'+datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y/%m/%d %H:%M JST'),21)
 txt(70,115,'平面図：コンテナの内部に支柱を入れない',25,bold=True)
 # table
 d.rounded_rectangle([X(-g.bench_length_m/2),Y(g.bench_depth_m/2),X(g.bench_length_m/2),Y(-g.bench_depth_m/2)],8,fill='#d4dae3',outline='#47576a',width=3)
 palette={'A':'#d1ead9','B':'#f0e4bd','C':'#f1d1c2','D':'#cddff4'}
 for grade,cx in g.crate_centres_x_m.items():
  d.rectangle([X(cx-g.crate_x_m/2),Y(g.crate_y_m/2),X(cx+g.crate_x_m/2),Y(-g.crate_y_m/2)],fill=palette[grade],outline='#617184',width=2)
  txt(X(cx),Y(-g.crate_y_m/2)-23,grade+'コンテナ',20,bold=True,anchor='mm')
  # exterior post: square beyond back long bench edge
  xx=X(cx);ry=Y(g.support_post_y_m);r=7
  d.rectangle([xx-r,ry-r,xx+r,ry+r],fill='#1f2937',outline='#1f2937')
  # elevated cantilever above the crate (in plan only)
  d.line((xx,ry,xx,Y(0)),fill='#273a4f',width=5)
  xa=X(cx-g.nest_pitch_m);xb=X(cx+g.nest_pitch_m)
  d.line((xa,Y(0),xb,Y(0)),fill='#273a4f',width=8)
  for slot in (1,2,3):
   nest=f'{grade}{slot}';x=X(g.nest_centres_x_m[nest]);r=g.nest_width_m*scale/2
   d.ellipse([x-r,Y(0)-r,x+r,Y(0)+r],fill='#ffffff',outline='#52667a',width=3)
   txt(x,Y(0),nest,17,bold=True,anchor='mm')
 sw=g.scale_width_m*scale
 d.rounded_rectangle([X(0)-sw/2,Y(.100),X(0)+sw/2,Y(-.100)],8,fill='#91bfde',outline='#406c8d',width=3)
 txt(X(0),Y(0),'計量台',21,bold=True,anchor='mm')
 # label on right
 txt(1450,215,'背面（+Y）',20,anchor='mm')
 d.line([X(-1.0875),170,X(1.0875),170],fill='#7b8793',width=2)
 txt(775,156,'作業台長辺 217.5cm',22,bold=True,anchor='mm')
 txt(145,534,'手前：作業員Cが受け台から取り出し、コンテナへ箱詰め',20)
 txt(70,587,'奥行き方向の断面：外付け支柱だけがコンテナの背後にある',25,bold=True)
 # right small diagram of Y-Z to show non penetration
 ox=400;oy=955;sy=440;sz=215
 yy2=lambda y:ox+y*sy
 zz=lambda z:oy-z*sz
 # bench centered z=.70
 d.rectangle([yy2(-.275),zz(.70),yy2(.275),zz(.66)],fill='#8793a3')
 # bin
 d.rectangle([yy2(-.2675),zz(1.005),yy2(.2675),zz(.70)],fill='#f1dcc4',outline='#8a6958',width=3)
 # column behind edge
 d.rectangle([yy2(.318),zz(1.178),yy2(.342),zz(.06)],fill='#263342')
 # beam above bin rim
 d.rectangle([yy2(0),zz(1.188),yy2(.33),zz(1.168)],fill='#263342')
 d.rectangle([yy2(-.05),zz(1.205),yy2(.05),zz(1.193)],fill='#387ca8')
 txt(685,709,'クッション受け台（上方）',19)
 txt(685,754,'上の横梁はコンテナの縁より約16cm高い（仮設計）',17)
 txt(685,806,'支柱は背面 +Y=33cm／台端=27.5cm（外側）',17)
 txt(685,856,'※ 支柱・梁は視覚モデル。強度と人間の安全は未検証。',17,fill='#8a3645')
 txt(70,1034,'実寸：作業台217.5×55cm、コンテナ37×53.5×高さ30.5cm。受け台高さとフレームは設計仮定。',17)
 path.parent.mkdir(exist_ok=True,parents=True)
 im.save(path)
 return path
if __name__=='__main__':print(create_diagram())
