"""Stage 04 measured dimensional geometry, without requiring PyBullet."""
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from workbench_geometry import LAYOUT,make_static_geometry

class MockBullet:
 GEOM_BOX=3
 def __init__(self):self.collisions=[];self.visuals=[];self.bodies=[]
 def createCollisionShape(self,shape,halfExtents):
  self.collisions.append((shape,tuple(halfExtents)));return len(self.collisions)
 def createVisualShape(self,shape,halfExtents,rgbaColor):
  self.visuals.append((shape,tuple(halfExtents),tuple(rgbaColor)));return len(self.visuals)
 def createMultiBody(self,baseMass,baseCollisionShapeIndex,baseVisualShapeIndex,basePosition):
  self.bodies.append({'m':baseMass,'collision':baseCollisionShapeIndex,'pos':tuple(basePosition)})
  return len(self.bodies)

class GeometryTests(unittest.TestCase):
 def test_exact_measurements(self):
  self.assertAlmostEqual(LAYOUT.bench_length_m,2.175)
  self.assertAlmostEqual(LAYOUT.bench_depth_m,.550)
  self.assertAlmostEqual(LAYOUT.crate_x_m,.370)
  self.assertAlmostEqual(LAYOUT.crate_y_m,.535)
  self.assertAlmostEqual(LAYOUT.crate_height_m,.305)
  self.assertAlmostEqual(LAYOUT.scale_width_m,.20)
  self.assertAlmostEqual(LAYOUT.margin_each_end_m,.2075)
 def test_four_crates_straddle_central_scale(self):
  expected=[-.695,-.305,.305,.695]
  for got,want in zip(LAYOUT.crate_centres_x_m.values(),expected):
   self.assertAlmostEqual(got,want)
  self.assertEqual(LAYOUT.nest_count_per_grade,3)
  self.assertEqual(len(LAYOUT.nest_centres_x_m),12)
 def test_nests_on_top_of_corresponding_crate(self):
  self.assertTrue(LAYOUT.validate())
  for name,x in LAYOUT.nest_centres_x_m.items():
   self.assertLessEqual(abs(x-LAYOUT.crate_centres_x_m[name[0]])+.05,.185+1e-10)
  self.assertAlmostEqual(LAYOUT.farthest_nest_from_scale_m,.695+LAYOUT.nest_pitch_m)
 def test_elevated_nests_are_above_rim(self):
  self.assertAlmostEqual(LAYOUT.nest_top_z_m-LAYOUT.rim_z_m,.200)
  self.assertAlmostEqual(LAYOUT.bench_top_z_m,.700)
  self.assertAlmostEqual(LAYOUT.relative_world_origin_z_m,1.165)
 def test_realistic_bench_and_crate_geometry_bodies(self):
  fake=MockBullet();bodies=make_static_geometry(fake)
  self.assertEqual(len(bodies),33) # bench, 4*(bottom+4walls), 4*(external post+cantilever+rail)
  self.assertEqual(len(fake.bodies),33)
  self.assertEqual(len(fake.collisions),21) # posts are deliberately visual only
  self.assertAlmostEqual(fake.bodies[0]['pos'][2],.68)
  self.assertAlmostEqual(fake.bodies[1]['pos'][0],-.695)
  self.assertEqual(fake.bodies[-1]['collision'],-1)
 def test_supports_are_outside_bench_and_crate(self):
  fake=MockBullet();bodies=make_static_geometry(fake)
  for grade in 'ABCD':
   post=fake.bodies[bodies[f'{grade}_nest_support_post']-1]
   self.assertGreater(post['pos'][1]-LAYOUT.support_post_halfwidth_m,LAYOUT.bench_depth_m/2)
   self.assertGreater(post['pos'][1]-LAYOUT.support_post_halfwidth_m,LAYOUT.crate_y_m/2)
   self.assertEqual(post['collision'],-1)
   for kind in ('cantilever','crossrail'):
    beam=fake.bodies[bodies[f'{grade}_nest_support_{kind}']-1]
    self.assertGreater(beam['pos'][2]-.010,LAYOUT.rim_z_m)
    self.assertEqual(beam['collision'],-1)


if __name__=='__main__':unittest.main()
