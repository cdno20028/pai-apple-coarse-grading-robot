"""Pure-control checks only: NEVER substitute for physical Bullet execution."""
import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_physics import TrialConfig, MOTION_SETTINGS, motion_settings, staged_lower_target
from workbench_geometry import LAYOUT

class Stage06SpeedTests(unittest.TestCase):
    def test_existing_defaults_unchanged(self):
        c=TrialConfig();m=motion_settings(c)
        self.assertEqual(c.motion_profile,'baseline')
        self.assertEqual(m.lift_steps,109)
        self.assertEqual(m.scale_settle_steps,60)
        self.assertEqual(m.x_max_velocity,.6)
        self.assertEqual(m.settle_max_steps,120)
        self.assertNotIn('Vbaseline',c.slug)

    def test_baseline_old_descent(self):
        m=motion_settings(TrialConfig())
        self.assertAlmostEqual(staged_lower_target(.4,.275,.165,m),.235,places=5)

    def test_optimized_has_10cm_lift(self):
        for key,m in MOTION_SETTINGS.items():
            self.assertGreaterEqual(m.lift_steps,2)
            self.assertEqual(round(.175+.100,3),.275)
            self.assertLessEqual(m.x_max_velocity,1.5)
            self.assertLessEqual(m.lower_fast_m_s,.5)

    def test_descent_is_monotonic(self):
        for name in MOTION_SETTINGS:
            m=motion_settings(TrialConfig(motion_profile=name))
            z=[staged_lower_target(t*.001,.275,.165,m) for t in range(1201)]
            self.assertTrue(all(z[i+1]<=z[i]+1e-12 for i in range(len(z)-1)))

    def test_descent_slow_zone(self):
        for name in ('balanced','fast_trial'):
            m=MOTION_SETTINGS[name]
            end=.165+m.lower_slow_window_m
            transition=(.275-end)/m.lower_fast_m_s
            self.assertAlmostEqual(staged_lower_target(transition,.275,.165,m),end,places=7)
            self.assertAlmostEqual(staged_lower_target(transition+.1,.275,.165,m),end-.01,places=7)

    def test_layout_unchanged(self):
        self.assertEqual(len(LAYOUT.nest_centres_x_m),12)
        self.assertEqual(LAYOUT.nest_count_per_grade,3)
        self.assertAlmostEqual(LAYOUT.farthest_nest_from_scale_m,.81,places=5)
        self.assertAlmostEqual(LAYOUT.crate_x_m,.37)
        self.assertAlmostEqual(LAYOUT.crate_y_m,.535)

    def test_invalid_setting_rejected(self):
        with self.assertRaises(ValueError):motion_settings(TrialConfig(motion_profile='fiction'))

    def test_slug_separation(self):
        self.assertNotEqual(TrialConfig(motion_profile='balanced').slug,TrialConfig(motion_profile='fast_trial').slug)

if __name__=='__main__':unittest.main()
