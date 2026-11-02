# -*- coding: utf-8 -*-
"""Stage91: alternate *mount calibration only* from three existing link poses.

NOT inference inputs, NOT new/extra operational cameras, NOT physical proof that
several hypothetical lens locations are installed simultaneously.  Diagnostic
only; label/case IDs are used to decide what audit PNGs to save, never in AI.
"""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from pathlib import Path
import hashlib, json, math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

JST = timezone(timedelta(hours=9))

def _font(size):
    for f in ('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(f).is_file():return ImageFont.truetype(f,size)
    return ImageFont.load_default()

def _view(p,robot,joints,k,shift_z,yaw_deg,pitch_deg,side=180):
    """Only optical extrinsics wrt a REAL fixed link. Does not move the link."""
    from mechanism_v09_reference import G
    pos,rot=p.getLinkState(robot,joints[f'finger_camera_{k}_fixed'],
                           computeForwardKinematics=True)[4:6]
    R=np.asarray(p.getMatrixFromQuaternion(rot),dtype=float).reshape(3,3)
    eye=np.asarray(pos,dtype=float)+R@np.array([0.,0.,shift_z])
    yaw,pitch=map(math.radians,(yaw_deg,pitch_deg))
    aim_local=np.array([math.cos(pitch)*math.cos(yaw),
                        math.cos(pitch)*math.sin(yaw),-math.sin(pitch)])
    up_local=np.array([math.sin(pitch)*math.cos(yaw),
                       math.sin(pitch)*math.sin(yaw),math.cos(pitch)])
    view=p.computeViewMatrix(eye.tolist(),(eye+R@aim_local*.20).tolist(),
                             (R@up_local).tolist())
    projection=p.computeProjectionMatrixFOV(fov=G.camera_fov_deg,aspect=1,
                                             nearVal=.005,farVal=.8)
    _,_,rgba,_,_=p.getCameraImage(side,side,viewMatrix=view,
            projectionMatrix=projection,renderer=p.ER_TINY_RENDERER)
    rgb=np.asarray(rgba,dtype=np.uint8).reshape(side,side,4)[:,:,:3].copy()
    return rgb,eye.tolist()

def capture_mount_grid(p,robot,apple,joints,case,root,current_frames,side=180):
    """B case02 / C-2 case03 ONLY for audit. No grades or labels output to AI."""
    root=Path(root)/'stage91_mount_grid'; root.mkdir(parents=True,exist_ok=True)
    out=root/f'case{int(case):02d}';out.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(JST).strftime('%Y/%m/%d %H:%M:%S JST')
    before=(p.getBasePositionAndOrientation(apple),p.getBaseVelocity(apple))
    rows=[]
    for k in range(3):
        panels=[]
        # Exactly 30 optical mount hypotheses per existing physical finger link.
        # Vertical offset is the critical missing variable from Stage88/90.
        for dz in (0.,-.015,-.030):
            for pitch in ((0.,20.) if k!=1 else (20.,35.)):
                for yaw in (-40.,-20.,0.,20.,40.):
                    rgb,eye=_view(p,robot,joints,k,dz,yaw,pitch,side)
                    fid=f'f{k+1}_z{int(round(-dz*1000)):02d}_p{int(pitch):02d}_y{int(yaw):+03d}'
                    file=fid+'.png'
                    Image.fromarray(rgb).save(out/file)
                    # Fixed mount baseline MUST be pixel-identical to live image.
                    baseline=bool(dz==0 and yaw==0 and pitch==(20. if k==1 else 0.))
                    matches=None
                    if baseline:
                        live=current_frames.get(('finger',k,3))
                        assert live is not None,'STAGE91_LIVE_SCAN70_MISSING'
                        matches=bool(np.array_equal(rgb,live))
                        assert matches,'STAGE91_ACTUAL_LINK_BASELINE_PIXEL_MISMATCH'
                    r,g,b=[rgb[:,:,i].astype(np.float32) for i in range(3)]
                    mask=(r>60)&(r>g*1.15)&(r>b*1.1)
                    pixels=np.argwhere(mask)
                    bbox=[int(pixels[:,1].min()),int(pixels[:,0].min()),
                          int(pixels[:,1].max()),int(pixels[:,0].max())] if len(pixels) else None
                    dat={'file':file,'physical_link':f'finger_{k+1}',
                         'offset_local_z_mm':int(round(dz*1000)),
                         'yaw_deg':int(yaw),'down_pitch_deg':int(pitch),
                         'optical_eye_world_m':[round(float(x),5) for x in eye],
                         'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),
                         'red_skin_bbox':bbox,'red_skin_pixel_ratio':round(float(mask.mean()),4),
                         'matches_actual_live_scan70_baseline':matches}
                    rows.append(dat)
                    tile=Image.fromarray(rgb).resize((150,150))
                    panel=Image.new('RGB',(178,186),'#10273a')
                    panel.paste(tile,(14,7))
                    draw=ImageDraw.Draw(panel)
                    draw.text((5,158),f'z{int(-dz*1000):02d} p{int(pitch):02d} y{int(yaw):+03d}',
                              font=_font(12),fill='white')
                    if baseline:draw.rectangle((0,0,177,185),outline='#00ff66',width=3)
                    panels.append(panel)
        sheet=Image.new('RGB',(5*178,6*186+55),'#082032')
        d=ImageDraw.Draw(sheet)
        d.text((10,7),f'Physical finger {k+1} / B & C diagnostic only / {stamp}',
               font=_font(17),fill='white')
        for i,panel in enumerate(panels):sheet.paste(panel,((i%5)*178,(i//5)*186+53))
        sheet.save(root/f'case{int(case):02d}_finger{k+1}_contact.png')
    after=(p.getBasePositionAndOrientation(apple),p.getBaseVelocity(apple))
    def close(a,b):
        return bool(np.allclose(np.asarray(a[0][0]),np.asarray(b[0][0]),atol=1e-9) and
                    np.allclose(np.asarray(a[0][1]),np.asarray(b[0][1]),atol=1e-9) and
                    np.allclose(np.asarray(a[1][0]),np.asarray(b[1][0]),atol=1e-9) and
                    np.allclose(np.asarray(a[1][1]),np.asarray(b[1][1]),atol=1e-9))
    assert close(before,after),'STAGE91_MOUNT_PROBE_CHANGED_PHYSICS'
    audit={'case_index_for_evidence_only':int(case),'date_jst':stamp,
           'n_probed_optical_configs':len(rows),'expected':90,
           'num_actual_physical_finger_links':3,'total_operational_cameras':5,
           'camera_mount_grid_is_diagnostic_only':True,
           'not_used_by_ai_for_grading':True,'not_used_for_motor_control':True,
           'no_case_truth_or_user_photo_loaded_by_camera_probe':True,
           'baseline_camera_rgb_pixel_exact':True,
           'apple_position_velocity_unchanged':True,'observations':rows}
    assert len(rows)==90,'STAGE91_WRONG_GRID_COUNT'
    (root/f'case{int(case):02d}_mount_grid.json').write_text(
        json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
    return root
