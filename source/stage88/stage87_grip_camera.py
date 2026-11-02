# -*- coding: utf-8 -*-
"""Stage88: Three synchronized live views of the unmodified Stage87 gripper action.
Two observer cameras + existing fifth inspection-table upward camera geometry.
No animation, no re-used footage, no changes to simulation stepping or AI.
All video frames come from the same live Bullet world / simulation tick.
"""
from pathlib import Path
import datetime
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import imageio.v2 as imageio

JST = datetime.timezone(datetime.timedelta(hours=9))
PHASES = (
    'scale_settle', 'scale_settle_scan_overlap', 'scan_and_close',
    'contact_settle', 'contact_feedback', 'gravity_lift',
    'lift_clearance_verify',
)
COLORS = ((0.27,-0.32,0.22),(-0.30,-0.29,0.22))
CAPTURE_CASES = (1,5)

class Stage87ActualGripCamera:
    def __init__(self, bullet, output_root, *, fps=20, sample_every_ticks=12):
        self.p=bullet
        self.base=Path(output_root)/'stage87_actual_grip'
        self.base.mkdir(parents=True,exist_ok=True)
        self.fps=int(fps)
        self.skip=int(sample_every_ticks)
        self.active={}
        self.records={}
        self.timestamp=datetime.datetime.now(JST).strftime('%Y/%m/%d %H:%M JST')
        # This is the existing fifth AI inspection camera, not a sixth AI camera.
        # Exact eye/target/up/FOV match run_physics.images_from_actual_links('plate').
        import run_physics as _physics
        self.underplate_eye=[0.,0.,float(_physics.G.plate_z)+.0005+
                             float(_physics.LAYOUT.relative_world_origin_z_m)]
        self.underplate_fov=float(_physics.G.camera_fov_deg)
        self.font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
        self.font = ImageFont.truetype(self.font_path,22) if Path(self.font_path).exists() else ImageFont.load_default()

    @staticmethod
    def _writer(path,fps):
        return imageio.get_writer(str(path),fps=fps,codec='libx264',quality=8,
                                  macro_block_size=1)

    def _open_case(self,case_id,base_position):
        tag=f'{case_id:02d}'
        target=[float(base_position[0]),float(base_position[1]),float(base_position[2])+.070]
        writers={
            'view_a':self._writer(self.base/f'case{tag}_grip_oblique_A_PYBULLET.mp4',self.fps),
            'view_b':self._writer(self.base/f'case{tag}_grip_oblique_B_PYBULLET.mp4',self.fps),
            'dual':self._writer(self.base/f'case{tag}_grip_dual_ACTUAL_PYBULLET.mp4',self.fps),
            'under':self._writer(self.base/f'case{tag}_inspection_table_under_ACTUAL_PYBULLET.mp4',self.fps),
            'three':self._writer(self.base/f'case{tag}_grip_plus_table_3views_ACTUAL_PYBULLET.mp4',self.fps)
        }
        self.active[case_id]=dict(ticks=0,frames=0,base=tuple(base_position),
                                  target=target,writers=writers,phases=[],first_z=None,
                                  max_lift=0.,contacts_seen=0,examples=[],seen_phases=set(),
                                  under_visible_pixel_samples=[],under_snapshot_count=0)
        self.records[case_id]=[]

    def _render(self,target,eyeoffset):
        p=self.p
        x,y,z=target
        eye=[x+eyeoffset[0],y+eyeoffset[1],z+eyeoffset[2]]
        view=p.computeViewMatrix(cameraEyePosition=eye,
                                 cameraTargetPosition=target,cameraUpVector=[0.,0.,1.])
        projection=p.computeProjectionMatrixFOV(
            fov=38.,aspect=8/7,nearVal=.012,farVal=3.)
        w,h,rgba,depth,seg=p.getCameraImage(
            880,770,viewMatrix=view,projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER)
        return np.asarray(rgba,dtype=np.uint8).reshape(h,w,4)[:,:,:3].copy()

    def _render_underplate(self,apple):
        # Live physical camera, looking upward from the fixed inspection table.
        # These *exact* geometrical parameters are also used by the existing
        # AI underside camera (run_physics.images_from_actual_links -> 'plate').
        p=self.p
        eye=self.underplate_eye
        vm=p.computeViewMatrix(eye,[eye[0],eye[1],eye[2]+.30],[0.,1.,0.])
        proj=p.computeProjectionMatrixFOV(fov=self.underplate_fov,
              aspect=1,nearVal=.005,farVal=.8)
        w,h,rgba,_,seg=p.getCameraImage(640,640,viewMatrix=vm,
              projectionMatrix=proj,renderer=p.ER_TINY_RENDERER)
        rgb=np.asarray(rgba,dtype=np.uint8).reshape(h,w,4)[:,:,:3].copy()
        ids=np.asarray(seg,dtype=np.int64).reshape(h,w)
        # Count only a physically rendered fruit ID; never synthesize visibility.
        apple_pixels=int(np.count_nonzero((ids & ((1 << 24)-1))==int(apple)))
        return rgb,apple_pixels

    def _three_panel_frame(self,case_id,stage,pa,pb,under,under_pixels):
        # Presentation only. Source images from three getCameraImage calls
        # during the SAME observer callback, before the next physics step.
        im=Image.new('RGB',(1920,900),(9,28,43))
        d=ImageDraw.Draw(im)
        labels=('A: GRIP OBSERVER','B: GRIP OBSERVER',
                '5th AI CAMERA: TABLE UNDERSIDE (UPWARD)')
        images=(pa,pb,under)
        for k,(label,pixels) in enumerate(zip(labels,images)):
            x=10+k*638
            d.rounded_rectangle((x,60,x+625,807),radius=7,
                                fill=(23,43,62),outline=(105,133,159),width=2)
            d.text((x+12,16),label,font=self.font,fill='white')
            fit=ImageOps.contain(Image.fromarray(pixels),(603,700))
            im.paste(fit,(x+(626-fit.width)//2,74+(700-fit.height)//2))
        d.text((22,830),f'ACTUAL PyBullet | CASE {case_id:02d} | {stage}',
               font=self.font,fill='white')
        d.text((22,861),self.timestamp+' | same simulated time | table view is real camera #5',
               font=self.font,fill='#d4e5ed')
        # NOTE: this label is a sample of physical segmentation, not a diagnosis.
        return np.asarray(im)

    def on_tick(self,case_id,stage,robot,apple,joints):
        if case_id not in CAPTURE_CASES or stage not in PHASES:return
        try:
            bp=self.p.getBasePositionAndOrientation(apple)[0]
        except Exception:
            return
        if case_id not in self.active:self._open_case(case_id,bp)
        info=self.active[case_id]
        info['ticks']+=1
        info['seen_phases'].add(stage)
        if info['ticks'] % self.skip: return

        pa=self._render(info['target'],COLORS[0])
        pb=self._render(info['target'],COLORS[1])
        under,under_pixels=self._render_underplate(apple)
        info['under_visible_pixel_samples'].append(under_pixels)
        if under_pixels>0: info['under_snapshot_count']+=1
        info['writers']['under'].append_data(under)
        info['writers']['three'].append_data(
            self._three_panel_frame(case_id,stage,pa,pb,under,under_pixels))
        info['writers']['view_a'].append_data(pa)
        info['writers']['view_b'].append_data(pb)

        # Two actual Bullet camera views side by side. Minimal phase label only.
        im=Image.new('RGB',(1920,950),(9,28,43))
        im.paste(Image.fromarray(pa).resize((935,818)),(12,52))
        im.paste(Image.fromarray(pb).resize((935,818)),(973,52))
        d=ImageDraw.Draw(im)
        d.text((22,12),f'ACTUAL PyBullet | CASE {case_id:02d} | {stage}',
               font=self.font,fill='white')
        d.text((984,12),'SECOND REAL CAMERA / DIFFERENT ANGLE',
               font=self.font,fill='white')
        d.text((20,895),self.timestamp+' | 20fps capture from same simulated physics clock',
               font=self.font,fill='#d4e5ed')
        d.text((20,922),'Two p.getCameraImage views / no synthetic finger movement',
               font=self.font,fill='#d4e5ed')
        info['writers']['dual'].append_data(np.asarray(im))
        info['frames']+=1
        if info['first_z'] is None:info['first_z']=float(bp[2])
        info['max_lift']=max(info['max_lift'],float(bp[2])-info['first_z'])
        contacts=len(self.p.getContactPoints(robot,apple))
        if contacts:info['contacts_seen']+=1
        zjoint=float(self.p.getJointState(robot,joints['stage_z_move'])[0])
        record={'case':case_id,'frame':info['frames'],'phase':stage,
                'sim_sample_step_within_selected_phases':info['ticks'],
                'fruit_xyz_m':[float(v) for v in bp],
                'stage_z_joint_m':zjoint,'robot_apple_contact_count':int(contacts),
                'underplate_camera_fruit_pixels':under_pixels,
                'underplate_camera_eye_xyz_m':self.underplate_eye,
                'underplate_camera_fov_deg':self.underplate_fov,
                'rendering':'three live PyBullet views: 2 observers + existing underside camera geometry'}
        self.records[case_id].append(record)
        if info['frames'] in (1,5,10,15,20,25,30,40):
            filename=f'case{case_id:02d}_frame{info["frames"]:03d}_{stage}.png'
            Image.fromarray(pa).save(self.base/filename)
            info['examples'].append(filename)
            underfile=f'case{case_id:02d}_under_frame{info["frames"]:03d}_{stage}.png'
            Image.fromarray(under).save(self.base/underfile)
            info['examples'].append(underfile)

    def close(self):
        stats=[]
        for case_id,info in self.active.items():
            for writer in info['writers'].values():writer.close()
            data={'case':case_id,'captured_frames':info['frames'],
                  'fps':self.fps,'elapsed_capture_s':info['frames']/self.fps,
                  'phases_sampled':sorted(info['seen_phases']),
                  'fruit_max_height_change_m':round(info['max_lift'],5),
                  'frames_with_robot_apple_contacts':info['contacts_seen'],
                  'first_fruit_position_m':list(info['base']),
                  'fixed_closeup_camera_target_m':info['target'],
                  'underplate_camera_eye_xyz_m':self.underplate_eye,
                  'underplate_camera_fov_deg':self.underplate_fov,
                  'underplate_frames_with_visible_apple_id':info['under_snapshot_count'],
                  'underplate_per_frame_apple_visible_pixel_counts':info['under_visible_pixel_samples'],
                  'underplate_camera_is_existing_fifth_ai_camera':True,
                  'all_three_views_generated_in_same_simulation_observer_callback':True,
                  'sample_pngs':info['examples'],
                  'not_a_new_AI_or_physics_controller':True,
                  'rendered_from_live_pybullet_camera':True,
                  'no_interpolation_or_repeated_video_frames':True}
            stats.append(data)
            (self.base/f'case{case_id:02d}_physical_camera_frames.json').write_text(
                json.dumps(self.records[case_id],ensure_ascii=False,indent=2),encoding='utf8')
        (self.base/'camera_capture_audit.json').write_text(
            json.dumps({'time_jst':self.timestamp,'cases':stats,
                        'note':'3 live views from the same simulated physics run; 2 observer cameras do NOT enter AI inference, the table-under view duplicates the geometry of existing 5th AI input',
                        'CAUTION':'3-finger descent, grip and lift must be inspected in rendered footage before claims; no real robot was operated'},
                       ensure_ascii=False,indent=2),encoding='utf8')
