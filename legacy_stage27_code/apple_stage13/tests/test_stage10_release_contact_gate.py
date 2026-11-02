"""Offline checks for the Stage10 release gate (not PyBullet physics tests)."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_physics import extra_open_joint_goals,contact_gate_accept,motion_settings,TrialConfig

class Stage10ReleaseTests(unittest.TestCase):
    def test_radial_open_goals_exist_and_differ(self):
        a=extra_open_joint_goals(0);b=extra_open_joint_goals(9)
        self.assertEqual(len(a),9)
        self.assertNotEqual(a['shoulder_0'],b['shoulder_0'])
        self.assertEqual(a['shoulder_0'],a['shoulder_2'])
    def test_gate_requires_contact_zero(self):
        self.assertFalse(contact_gate_accept(1,True,.001,0,8))
    def test_gate_requires_full_stable_window(self):
        self.assertFalse(contact_gate_accept(0,True,.001,0,7))
        self.assertTrue(contact_gate_accept(0,True,.001,0,8))
    def test_gate_rejects_drift_and_lost_support(self):
        self.assertFalse(contact_gate_accept(0,True,.022,0,10))
        self.assertFalse(contact_gate_accept(0,False,.001,0,10))
        self.assertFalse(contact_gate_accept(0,True,.001,.2,10))
    def test_lift_settings_and_release_lengths(self):
        a=motion_settings(TrialConfig(motion_profile='contact_gate_39'))
        b=motion_settings(TrialConfig(motion_profile='contact_gate_50'))
        self.assertEqual(a.lift_steps,b.lift_steps)
        self.assertEqual(a.lift_steps,83)
        self.assertEqual(a.release_steps,39)
        self.assertEqual(b.release_steps,50)
    def test_excessive_spread_rejected(self):
        with self.assertRaises(ValueError): extra_open_joint_goals(13)
