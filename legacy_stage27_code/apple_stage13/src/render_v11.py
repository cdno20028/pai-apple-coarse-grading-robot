"""Render *computed* 5-camera RGB, occlusion coverage maps, and mechanism diagrams.
These are synthetic sphere and simplified solid-finger renderings; not physics-engine footage.
"""
import sys, math, json, subprocess
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"]="Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"]=False
import cv2
from camera_model import *
from optimize_camera import coverage_masks, tilted_cam, lifted_parts
from evaluate_v11 import SPECS

OUT=Path(__file__).resolve().parents[1]/'output';OUT.mkdir(exist_ok=True)
STAMP='2026/10/14 02:40 JST'

def note(fig,txt='幾何学・光線シミュレーション（物理把持・傷識別は未検証）'):
    fig.text(.02,.008,txt+' | '+STAMP,ha='left',va='bottom',fontsize=9,color='#555555')

def camset(d,palm_z,lift,t=.32):
    parts=finger_parts(t,d)
    yaw=-math.degrees(math.atan(d.cam_tangent_mm/65.))
    cs=[tilted_cam(f,0,yaw) for f in parts]
    pcs={'origin':np.array([0.,0.,palm_z]),'direction':np.array([0.,0.,-1.]),
        'right':np.array([1.,0.,0.]),'up':np.array([0.,1.,0.])}
    lp=lifted_parts(finger_parts(.60,d),lift)
    return [(f'指カメラ {i+1}',cs[i],parts,APPLE_C) for i in range(3)]+[
       ('手のひらカメラ',pcs,parts,APPLE_C),
       ('台カメラ（持上げ後）',plate_spec(),lp,APPLE_C+np.array([0.,0.,lift]))]

def make_cameras():
    name,d,h,l=SPECS[-1]
    cases=camset(d,h,l)
    fig,axes=plt.subplots(1,5,figsize=(15,3.8),constrained_layout=False)
    images=[]
    for ax,(label,cam,parts,c) in zip(axes,cases):
       rgb,rr=render_cam(cam,parts,d,n=256,apple_center=c)
       images.append(rgb)
       ax.imshow(rgb);ax.axis('off');ax.set_title(label,fontsize=12)
    fig.suptitle('v1.1   5台の仮想カメラから計算した撮影画像',fontsize=17,y=.96)
    note(fig,'球体りんご・簡略化した指。橙は把持面／青灰はリンク（カメラRGBの擬似描画）')
    fig.subplots_adjust(left=.01,right=.99,bottom=.10,top=.85,wspace=.06)
    fig.savefig(OUT/'v11_five_cameras.png',dpi=155);plt.close(fig)
    return images

def make_coverage():
    res=[]
    for name,d,h,l in [SPECS[0],SPECS[-1]]:
      yaw=-math.degrees(math.atan(d.cam_tangent_mm/65.))
      m=coverage_masks(d,0,yaw,h,l,npts=14000,times=np.linspace(0,G.scan_s,15))
      res.append((name,m))
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.1))
    for ax,(label,m) in zip(axes,res):
      normal=(m['pts']-APPLE_C)/APPLE_R
      az=(np.degrees(np.arctan2(normal[:,1],normal[:,0]))+360)%360
      ze=normal[:,2]
      num=np.histogram2d(ze,az,bins=(30,60),range=((-1,1),(0,360)),weights=m['post'].astype(int))[0]
      den=np.histogram2d(ze,az,bins=(30,60),range=((-1,1),(0,360)))[0]
      im=ax.imshow(num/np.maximum(den,1),origin='lower',extent=[0,360,-1,1],aspect='auto',vmin=0,vmax=1,cmap='viridis')
      ax.set_xlabel('果実表面の方位角（度）');ax.set_ylabel('表面の高さ（球の中心を0）')
      ax.set_title(('従来配置（同一光学モデル）' if label=='v10_recomputed' else 'v1.1 改良配置') + f"  {m['post'].mean()*100:.1f}%")
    fig.colorbar(im,ax=axes.tolist(),fraction=.025,pad=.05,label='その領域を撮影できた割合')
    fig.suptitle('持ち上げ後を含む果皮の観察可能領域（未観察部も表示）',fontsize=15,y=.97)
    note(fig)
    fig.subplots_adjust(left=.075,right=.89,bottom=.16,top=.83,wspace=.23)
    fig.savefig(OUT/'v11_surface_coverage.png',dpi=150);plt.close(fig)

def make_layout_and_ablation():
    res=json.loads((OUT/'v11_results.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(13,5.1),gridspec_kw={'width_ratios':[1,1.2]})
    # Actual 3D location projected to plane: same hand frame and camera-pad rigidity.
    ax=axes[0];d=SPECS[-1][1]
    pts=finger_parts(.32,d);f=pts[0]
    r=np.linalg.norm(f['wrist'][:2]); z=f['wrist'][2]
    ax.add_patch(plt.Circle((0,G.apple_z),G.apple_r,facecolor='#d85d53',edgecolor='#8d302f',alpha=.8))
    for a,b,color in [(f['root'],f['elbow'],'#53769e'),(f['elbow'],f['wrist'],'#288ac0'),(f['pad_a'],f['pad_b'],'#dd932f')]:
      # plot radial coordinate, physical 3D pad/camera projections
      ax.plot([np.linalg.norm(a[:2]),np.linalg.norm(b[:2])],[a[2],b[2]],lw=6,color=color,solid_capstyle='round')
    for position,color,mark in [(f['camera'],'#096ac2','o'),(f['padcenter'],'#db8b14','D')]:
       ax.scatter(np.linalg.norm(position[:2]),position[2],s=85,color=color,marker=mark,zorder=5)
    camr=np.linalg.norm(f['camera'][:2]);camz=f['camera'][2]
    ax.annotate('レンズ（同じ指先）',xy=(camr,camz),xytext=(.077,.139),arrowprops={'arrowstyle':'->'},fontsize=11)
    ax.annotate('把持面 18mm',xy=(np.linalg.norm(f['padcenter'][:2]),f['padcenter'][2]),xytext=(.074,.050),arrowprops={'arrowstyle':'->'},fontsize=11)
    ax.hlines(G.plate_z,-.048,.14,color='#999999',lw=2)
    ax.set_xlim(-.048,.14);ax.set_ylim(.035,.19);ax.set_aspect('equal')
    ax.set_title('1本の指：リンク・レンズ・把持面',fontsize=12)
    ax.set_xlabel('半径方向（m）');ax.set_ylabel('高さ（m）')
    ax=axes[1];labs=['従来配置','手のひら高だけ変更','持上げ量だけ変更','両方変更','v1.1（横位置も調整）']
    ypos=np.arange(len(res));pre=np.array([r['pre_lift_visible_pct'] for r in res]);post=np.array([r['after_plate_visible_pct'] for r in res]);
    ax.barh(ypos+.18,pre,height=.32,label='持ち上げ前');ax.barh(ypos-.18,post,height=.32,label='台カメラ後')
    ax.set_yticks(ypos,labs);ax.invert_yaxis();ax.set_xlim(0,105);ax.set_xlabel('可視果皮率（%）');ax.grid(axis='x',alpha=.2);ax.legend(loc='lower right')
    for y,v in zip(ypos,post):ax.text(v+1,y-.18,f'{v:.1f}%',va='center',fontsize=9)
    fig.suptitle('v1.1 近接した1本2役指を維持した配置改善',fontsize=16,y=.97)
    note(fig)
    fig.subplots_adjust(left=.07,right=.97,bottom=.15,top=.86,wspace=.44)
    fig.savefig(OUT/'v11_design_and_ablation.png',dpi=150);plt.close(fig)

def make_video():
    d=SPECS[-1][1];h=SPECS[-1][2];l=SPECS[-1][3]
    outfile=OUT/'v11_camera_scan.mp4'
    fourcc=cv2.VideoWriter_fourcc(*'mp4v');video=cv2.VideoWriter(str(outfile),fourcc,24,(960,420))
    if not video.isOpened(): raise RuntimeError('VideoWriter unavailable')
    for t in np.linspace(0,G.scan_s,36):
       parts=finger_parts(float(t),d)
       cams=[tilted_cam(p,0,-math.degrees(math.atan(d.cam_tangent_mm/65.))) for p in parts]
       rgb=[render_cam(c,parts,d,n=190)[0] for c in cams]
       canvas=np.full((420,960,3),250,dtype=np.uint8)
       for k,img in enumerate(rgb):
         x=65+k*302
         dst=cv2.resize(img,(238,238),interpolation=cv2.INTER_NEAREST)
         canvas[100:338,x:x+238]=dst
         cv2.putText(canvas,f'FINGER CAM {k+1}',(x,82),cv2.FONT_HERSHEY_SIMPLEX,.55,(42,60,83),2,cv2.LINE_AA)
       cv2.putText(canvas,f'v1.1 RAY-TRACED VIEWS   t={t:.3f}s',(40,40),cv2.FONT_HERSHEY_SIMPLEX,.7,(25,38,55),2,cv2.LINE_AA)
       cv2.putText(canvas,'SIMULATED SPHERE / NOT PHYSICAL GRASP',(45,374),cv2.FONT_HERSHEY_SIMPLEX,.56,(70,70,70),1,cv2.LINE_AA)
       cv2.putText(canvas,'2026/10/14 02:40 JST',(645,402),cv2.FONT_HERSHEY_SIMPLEX,.56,(70,70,70),1,cv2.LINE_AA)
       video.write(cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR))
    video.release()

if __name__=='__main__':
    make_cameras();make_coverage();make_layout_and_ablation();make_video()
    print('Created images and MP4')
