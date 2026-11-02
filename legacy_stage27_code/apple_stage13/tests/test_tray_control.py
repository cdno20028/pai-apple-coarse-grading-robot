import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tray_control import station_ready,placement_allowed,supported_load_newtons,tray_occupied,TRAY_NAMES


def mock_contact(body_a,body_b,normal):
    v=[0]*10;v[1]=body_a;v[2]=body_b;v[9]=normal
    return tuple(v)

class BulletMock:
    def __init__(self,contacts):self.contacts=contacts
    def getContactPoints(self,bodyA):
        return [c for c in self.contacts if c[1]==bodyA]

class TrayInterlockTests(unittest.TestCase):
    def test_all_empty(self):
        self.assertTrue(station_ready({x:False for x in TRAY_NAMES}))
    def test_any_tray_occupied_stops_next_fruit(self):
        for occupied in TRAY_NAMES:
            statuses={k:k==occupied for k in TRAY_NAMES}
            self.assertFalse(station_ready(statuses))
    def test_worker_removes_apple_then_restarts(self):
        statuses={x:x=='B2' for x in TRAY_NAMES}
        self.assertFalse(station_ready(statuses))
        statuses['B2']=False
        self.assertTrue(station_ready(statuses))
    def test_no_unrecognized_destination(self):
        self.assertFalse(placement_allowed('E1',{'A1':False}))
        self.assertFalse(placement_allowed('A1',{'A1':True,'B1':False}))
        self.assertTrue(placement_allowed('A1',{'A1':False,'B1':True}))
    def test_contact_load_sensor_filters_robot(self):
        sensed=[mock_contact(8,23,1.7), mock_contact(8,99,100.)]
        self.assertAlmostEqual(supported_load_newtons(sensed,8,(99,)),1.7)
        self.assertTrue(tray_occupied(BulletMock(sensed),8,99))
    def test_empty_tray_and_robot_touch_not_occupied(self):
        self.assertFalse(tray_occupied(BulletMock([mock_contact(8,99,55)]),8,99))
    def test_small_contact_noise_is_not_fruit(self):
        self.assertFalse(tray_occupied(BulletMock([mock_contact(8,23,.03)]),8,99))
    def test_missing_tray_is_error(self):
        with self.assertRaises(ValueError):station_ready({'A':False,'B':False})

if __name__=='__main__': unittest.main()
