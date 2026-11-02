"""Offline validation of Stage11 measured-motor release gate."""
import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from run_physics import finger_goal_settled,clearance_accept,TrialConfig,motion_settings,extra_open_joint_goals
class StubBullet:
    def __init__(self, positions, speeds):
        self.positions=positions;self.speeds=speeds
    def getJointStates(self,robot,indices):
        return [(self.positions[i],self.speeds[i],None,None) for i in indices]
class Stage11Tests(unittest.TestCase):
    def test_position_not_reached_cannot_pass(self):
        goals=extra_open_joint_goals(9)
        joints={k:i for i,k in enumerate(goals)}
        p=StubBullet({i:v+.05 for i,(k,v) in enumerate(goals.items())},{i:0 for i in range(9)})
        self.assertFalse(finger_goal_settled(p,1,joints,goals))
    def test_motor_still_moving_cannot_pass(self):
        goals=extra_open_joint_goals(9);joints={k:i for i,k in enumerate(goals)}
        p=StubBullet({i:v for i,(k,v) in enumerate(goals.items())},{i:.25 for i in range(9)})
        self.assertFalse(finger_goal_settled(p,1,joints,goals))
    def test_converged_joints_pass(self):
        goals=extra_open_joint_goals(9);joints={k:i for i,k in enumerate(goals)}
        p=StubBullet({i:v for i,(k,v) in enumerate(goals.items())},{i:0 for i in range(9)})
        self.assertTrue(finger_goal_settled(p,1,joints,goals))
    def test_extra_open_below_minimum_cannot_pass(self):
        self.assertFalse(clearance_accept(0,True,.001,0,8,3,9,True))
        self.assertTrue(clearance_accept(0,True,.001,0,8,9,9,True))
    def test_extra_open_requires_joints_settled(self):
        self.assertFalse(clearance_accept(0,True,.001,0,8,12,9,False))
    def test_no_contact_8_consecutive(self):
        self.assertFalse(clearance_accept(0,True,.001,0,7,12,9,True))
    def test_safety_preserves_lift_and_velocity(self):
        a=motion_settings(TrialConfig(motion_profile='settled_gate_09'))
        b=motion_settings(TrialConfig(motion_profile='settled_gate_12'))
        self.assertEqual(a.lift_steps,83)
        self.assertEqual(b.lift_steps,83)
        self.assertEqual(a.release_steps,39)
        self.assertEqual(b.release_steps,39)
