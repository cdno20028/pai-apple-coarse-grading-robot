import unittest, sys, math
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from bin_vision import choose_clear_floor

class BinVisionTests(unittest.TestCase):
    def setUp(self):
        t=np.linspace(-.1,.1,101)
        self.floor=np.array([(x,y,.04) for x in t for y in t])
    def test_empty_bin(self):
        q=choose_clear_floor(self.floor,(0.,0.))
        self.assertTrue(q.free)
        self.assertEqual(q.target_xy,(0.,0.))
    def test_reject_observed_center_occupied_and_choose_other(self):
        # visual depth pixels representing an existing apple near the center
        points=self.floor.copy()
        for i,(x,y,z) in enumerate(points):
            if math.hypot(x,y)<.046: points[i,2]=.085
        q=choose_clear_floor(points,(0.,0.))
        self.assertFalse(q.free)  # no guaranteed 80-mm diameter free footprint
    def test_existing_apple_on_left_selects_clear_right(self):
        points=self.floor.copy()
        left=np.linalg.norm(points[:,:2]-[-.052,0.],axis=1)<.041
        points[left,2]=.092  # a measured occupied region, not a fruit GT coordinate
        q=choose_clear_floor(points,(0.,0.))
        self.assertTrue(q.free)
        self.assertGreater(q.target_xy[0],0.)

    def test_unknown_not_free(self):
        self.assertFalse(choose_clear_floor(np.array([[0.,0.,.04]]),(0.,0.)).free)
    def test_nan_pixels_not_free(self):
        self.assertFalse(choose_clear_floor(np.full((100,3),float('nan')),(0.,0.)).free)
    def test_small_spot_alternative(self):
        points=self.floor.copy()
        points[np.linalg.norm(points[:,:2]-[.052,.052],axis=1)<.013,2]=.10
        q=choose_clear_floor(points,(0.,0.))
        self.assertTrue(q.free)
        self.assertEqual(q.target_xy,(0.,0.))
    def test_too_small_crate(self):
        q=choose_clear_floor(self.floor,(0.,0.),inside_half=.043)
        self.assertFalse(q.free)

if __name__=='__main__': unittest.main()
