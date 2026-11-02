import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from workcell_sim import Settings, simulate

class OrchardWorkflowTests(unittest.TestCase):
    def test_four_farmers_1200_delivered(self):
        r=simulate(Settings())
        self.assertEqual(r['harvested_delivered'],1200)
    def test_mass_balance(self):
        r=simulate(Settings())
        self.assertEqual(r['worker_a_screened']+r['harvest_baskets_uninspected'],1200)
        self.assertEqual(r['worker_a_screened'],r['worker_a_rejected']+r['worker_a_handoff_to_robot']+r['worker_a_holding_selected'])
        self.assertLessEqual(r['robot_delivered_to_trays'],r['worker_a_handoff_to_robot'])
        self.assertLessEqual(r['worker_c_packed'],r['robot_delivered_to_trays'])
    def test_slower_worker_c_creates_robot_pauses(self):
        fast=simulate(Settings(c_pack_s=1.,c_swap_s=0.))
        slow=simulate(Settings(c_pack_s=6.,c_swap_s=30.))
        self.assertGreater(slow['robot_paused_occupied_tray_s'],fast['robot_paused_occupied_tray_s'])
        self.assertLess(slow['robot_delivered_to_trays'],fast['robot_delivered_to_trays'])
    def test_no_full_container_swap_without_capacity(self):
        r=simulate(Settings(container_capacity=100000))
        self.assertEqual(r['worker_c_container_swaps_started'],0)

if __name__=='__main__': unittest.main()
