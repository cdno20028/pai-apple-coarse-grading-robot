"""Offline structural checks. Real Bullet contact physics requires Colab."""
import sys,unittest
from pathlib import Path
import trimesh
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cushion_texture_assets import (ASSETS,ensure_assets,CUP_DEPTH_M,CUP_DIAMETER_M,
                                    create_cushion,spawn_textured_apple,capture_in_cup)
from workbench_geometry import LAYOUT

class FakeBullet:
    GEOM_MESH=5;GEOM_SPHERE=2; GEOM_FORCE_CONCAVE_TRIMESH=1
    def __init__(self): self.collision=[];self.visual=[];self.body=[];self.texture=[];self.dynamics=[]
    def createCollisionShape(self,shape,**kw):
        self.collision.append((shape,kw));return len(self.collision)
    def createVisualShape(self,shape,**kw):
        self.visual.append((shape,kw));return len(self.visual)
    def createMultiBody(self,**kw):
        self.body.append(kw);return len(self.body)
    def changeDynamics(self,*a,**kw):self.dynamics.append((a,kw))
    def loadTexture(self,path):self.texture.append(path);return 123
    def changeVisualShape(self,*a,**kw):self.texture.append(kw)

class Stage13Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):ensure_assets()
    def test_watertight_cup_and_real_dip(self):
        m=trimesh.load(str(ASSETS/'cushion_cup.obj'),process=False,force='mesh')
        self.assertTrue(m.is_watertight)
        self.assertAlmostEqual(m.bounds[1][2],CUP_DEPTH_M,places=6)
        self.assertAlmostEqual(m.bounds[0][0],-CUP_DIAMETER_M/2-.002,places=6)
    def test_12_single_apple_nests(self):
        self.assertEqual(len(LAYOUT.nest_centres_x_m),12)
        self.assertAlmostEqual(LAYOUT.nest_pitch_m,.115)
        self.assertEqual(list(LAYOUT.nest_centres_x_m),
             [g+str(i) for g in 'ABCD' for i in (1,2,3)])
    def test_four_distinct_texture_files(self):
        images=[]
        for label in 'ABCD':
            im=Image.open(ASSETS/f'apple_{label}.png');self.assertEqual(im.size,(512,256))
            images.append(im.getpixel((int(512*.75),130)))
        self.assertEqual(len(set(images)),4)
    def test_uv_sphere_file_contains_uv(self):
        obj=(ASSETS/'apple_uv.obj').read_text()
        self.assertIn('vt ',obj);self.assertIn('f ',obj)
    def test_cup_is_collision_mesh_not_box(self):
        fake=FakeBullet();body=create_cushion(fake,'A',.1,0,1.2)
        self.assertEqual(fake.collision[0][0],fake.GEOM_MESH)
        self.assertEqual(fake.collision[0][1]['flags'],fake.GEOM_FORCE_CONCAVE_TRIMESH)
        self.assertEqual(body,1)
        self.assertEqual(fake.body[0]['basePosition'],[.1,0,1.2])
    def test_apple_physical_sphere_visual_uv_mesh(self):
        fake=FakeBullet();body=spawn_textured_apple(fake,'C',.18,.04,[0,0,1.2],1.0)
        self.assertEqual(fake.collision[0][0],fake.GEOM_SPHERE)
        self.assertEqual(fake.visual[0][0],fake.GEOM_MESH)
        self.assertEqual(fake.visual[0][1]['meshScale'],[.04]*3)
        self.assertEqual(body,1)
        self.assertTrue(any('apple_C.png' in str(v) for v in fake.texture))
    def test_two_independent_capture_metrics(self):
        self.assertTrue(capture_in_cup((0,0),(.025,0,0),True,.03)[0])
        self.assertFalse(capture_in_cup((0,0),(.035,0,0),True,.03)[0])
        self.assertFalse(capture_in_cup((0,0),(0,0,0),False,0)[0])
        self.assertFalse(capture_in_cup((0,0),(0,0,0),True,.2)[0])

if __name__=='__main__':unittest.main()
