"""Independent acceptance checks for mechanical geometry and synthetic vision.
Does not assert rigid-body grasp success without PyBullet execution.
"""
from pathlib import Path
import sys,math,unittest
import numpy as np
from xml.etree import ElementTree as ET
from scipy.spatial.transform import Rotation as Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from camera_model import G, APPLE_R, PadDesign, finger_parts, pad_apple_clearance, projected_surface_visibility,sphere_points
from optimize_camera import tilted_cam, coverage_masks
ROOT=ET.parse(Path(__file__).resolve().parents[1]/'robot_v13.urdf').getroot()
JOINTS={j.find('child').get('link'):j for j in ROOT.findall('joint')}
DESIGN=PadDesign(length_mm=18,cam_tangent_mm=17,cam_up_mm=0)

def urdf_fk(leaf,values):
 H=np.eye(4); chain=[]
 while leaf!='base':
  j=JOINTS[leaf];chain.append(j);leaf=j.find('parent').get('link')
 for j in reversed(chain):
  org=j.find('origin');xyz=np.array([float(x) for x in org.get('xyz','0 0 0').split()]);rpy=np.array([float(x) for x in org.get('rpy','0 0 0').split()]);M=np.eye(4);M[:3,:3]=Rotation.from_euler('xyz',rpy).as_matrix();M[:3,3]=xyz;H=H@M
  if j.get('type') in ('revolute','prismatic'):
   A=np.eye(4);axis=np.array([float(x) for x in j.find('axis').get('xyz').split()]);q=values[j.get('name')]
   if j.get('type')=='revolute':A[:3,:3]=Rotation.from_rotvec(axis*q).as_matrix()
   else:A[:3,3]=axis*q
   H=H@A
 return H

class MechanismTests(unittest.TestCase):
 def values(self,t):
  from mechanism_v09_reference import q_at
  q=q_at(t);v={'stage_x_move':0.,'stage_y_move':0.,'stage_z_move':G.root_z}
  for k in range(3):v.update({f'shoulder_{k}':q[0],f'elbow_{k}':q[1],f'wrist_{k}_joint':q[2]})
  return v
 def test_three_fingers_camera_same_wrist(self):
  children=[j.get('name') for j in ROOT.findall('joint')]
  self.assertEqual(sum(x.startswith('shoulder_') for x in children),3)
  self.assertEqual(sum(x.startswith('elbow_') for x in children),3)
  self.assertEqual(sum(x.startswith('finger_camera_') and x.endswith('_fixed') for x in children),3)
  for k in range(3):
   self.assertEqual(JOINTS[f'finger_camera_{k}'].find('parent').get('link'),f'wrist_{k}')
 def test_fixed_links_and_urdf_pose_agree(self):
  for t in np.linspace(0,.60,13):
   v=self.values(t);parts=finger_parts(t,DESIGN)
   for k,p in enumerate(parts):
    cam=urdf_fk(f'finger_camera_{k}',v);wrist=urdf_fk(f'wrist_{k}',v)
    np.testing.assert_allclose(cam[:3,3],p['camera'],atol=1e-8)
    np.testing.assert_allclose(cam[:3,:3]@np.array([1,0,0]),tilted_cam(p,0,-math.degrees(math.atan(17/65)))['direction'],atol=2e-8)
    np.testing.assert_allclose((wrist@np.array([.008,0,-.010,1]))[:3],p['padcenter'],atol=2e-8)
 def test_camera_close_to_pad(self):
  for t in (0,.2,.4,.52):
   for p in finger_parts(t,DESIGN):
    gap=np.linalg.norm(p['camera']-p['padcenter'])*1000
    self.assertLessEqual(gap,22.)
    self.assertGreater(gap,14.)
 def test_before_contact_clearance(self):
  for t in np.linspace(0,.52,101):
   self.assertGreaterEqual(pad_apple_clearance(finger_parts(t,DESIGN),DESIGN),.0029)
  self.assertLess(pad_apple_clearance(finger_parts(.60,DESIGN),DESIGN),0)
 def test_camera_lens_above_plate(self):
  for t in np.linspace(0,.52,25):
   for p in finger_parts(t,DESIGN):
    self.assertGreater(p['camera'][2]-G.plate_z,.020)
 def test_palm_camera_below_head(self):
  v=self.values(.26);pos=urdf_fk('palm_camera',v)[:3,3]
  self.assertAlmostEqual(pos[2],.160,places=6)
  self.assertLess(pos[2],G.root_z-.005)
 def test_recomputed_coverage(self):
  masks=coverage_masks(DESIGN,0,-math.degrees(math.atan(17/65)),.160,.100,npts=4000,times=np.linspace(0,.52,9))
  self.assertGreater(masks['pre'].mean(),.77)
  self.assertGreater(masks['post'].mean(),.90)
  self.assertLess(masks['post'].mean(),1.)
 def test_finger_camera_sees_synthetic_fruit(self):
  pts=sphere_points(900)
  parts=finger_parts(.26,DESIGN)
  for p in parts:
   visible=projected_surface_visibility(tilted_cam(p,0,-math.degrees(math.atan(17/65))),pts,parts,DESIGN)
   self.assertGreater(visible.sum(),50)

if __name__=='__main__':unittest.main()
