"""GX16 hand driver against a fake Dynamixel SDK (no serial port)."""
from pathlib import Path
import time
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from gx4m_real.robot import hand as H

ROOT = Path(__file__).resolve().parents[3]
URDF = ROOT/'asset/gx7_right/urdf/gx7_right_gx4m_cam_xl330.urdf'


class FakeSDK:
    """Records register writes in order; positions live in self.present (motor units)."""
    COMM_SUCCESS = 0

    def __init__(self, present):
        self.present, self.log, self.goals, self.fail_read = dict(present), [], [], False
        sdk = self

        class PortHandler:
            def __init__(self, name):
                self.name, self.closed = name, False

            def openPort(self):
                return True

            def setBaudRate(self, baud):
                sdk.baud = baud
                return True

            def closePort(self):
                sdk.log.append(('close',))

        class PacketHandler:
            def __init__(self, protocol):
                sdk.protocol = protocol

            def write1ByteTxRx(self, port, motor, address, value):
                sdk.log.append((motor, address, value))
                return 0, 0

            write2ByteTxRx = write1ByteTxRx

            def getTxRxResult(self, result):
                return f'comm error {result}'

            def getRxPacketError(self, error):
                return f'packet error {error}'

        class GroupSyncRead:
            def __init__(self, port, packet, address, size):
                assert (address, size) == (H.PRESENT_POSITION, 4)
                self.ids = []

            def addParam(self, motor):
                self.ids.append(motor)
                return True

            def txRxPacket(self):
                return -3001 if sdk.fail_read else 0

            def isAvailable(self, motor, address, size):
                return motor in self.ids

            def getData(self, motor, address, size):
                return sdk.present[motor] % 2**32        # the SDK returns the unsigned register value

        class GroupSyncWrite:
            def __init__(self, port, packet, address, size):
                assert (address, size) == (H.GOAL_POSITION, 4)
                self.params = {}

            def clearParam(self):
                self.params = {}

            def addParam(self, motor, data):
                self.params[motor] = int.from_bytes(bytes(data), 'little', signed=True)
                return True

            def txPacket(self):
                sdk.goals.append(dict(self.params))
                sdk.log.append(('goal',))
                return 0

        self.PortHandler, self.PacketHandler = PortHandler, PacketHandler
        self.GroupSyncRead, self.GroupSyncWrite = GroupSyncRead, GroupSyncWrite


def make_driver(present, **kw):
    config = H.load_hand_config()
    sdk = FakeSDK(present)
    bus = H.DynamixelBus('test-only', config['ids'], config['baudrate'], config['protocol'], sdk=sdk)
    return H.HandDriver(bus=bus, **kw), sdk


class HandDriverTests(unittest.TestCase):
    def test_config_matches_gex16(self):
        c = H.load_hand_config()
        self.assertEqual((c['ids'], c['baudrate'], c['protocol']), (list(range(1, 17)), 1_000_000, 2.))
        np.testing.assert_array_equal(c['directions'], [1, 1, -1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1])

    def test_limits_match_urdf(self):
        joints = {j.get('name'): j.find('limit') for j in ET.parse(URDF).getroot().iter('joint')}
        for k, name in enumerate(H.HAND_JOINTS):
            self.assertAlmostEqual(H.LOWER[k], float(joints[name].get('lower')))
            self.assertAlmostEqual(H.UPPER[k], float(joints[name].get('upper')))

    def test_angle_convention(self):
        d = H.load_hand_config()['directions']
        units = np.full(16, round(90/H.DEG_PER_UNIT))       # 90 deg motor reading = URDF zero
        np.testing.assert_allclose(H.units_to_urdf(units, d), 0., atol=1e-3)
        q = np.linspace(0., .2, 16)
        np.testing.assert_allclose(H.units_to_urdf(H.urdf_to_units(q, d), d), q, atol=np.deg2rad(H.DEG_PER_UNIT))
        # A -1 joint turns the motor the other way: +10 deg URDF reads 80 deg on the motor.
        self.assertEqual(H.urdf_to_units(np.full(16, np.deg2rad(10.)), d)[2], round(80/H.DEG_PER_UNIT))

    def test_connect_configures_like_gex16_and_holds_before_torque(self):
        present = {i: 1100 + i for i in range(1, 17)}
        driver, sdk = make_driver(present, curr_limit=800, goal_current=500, goal_pwm=300)
        q = driver.connect()
        setup = [e for e in sdk.log if e[0] == 1][:5]
        self.assertEqual(setup, [(1, H.TORQUE_ENABLE, 0), (1, H.CURRENT_LIMIT, 800), (1, H.OPERATING_MODE, 5),
                                 (1, H.GOAL_CURRENT, 500), (1, H.GOAL_PWM, 300)])
        first_goal = sdk.log.index(('goal',))
        torque_on = [k for k, e in enumerate(sdk.log) if e[1:] == (H.TORQUE_ENABLE, 1)]
        self.assertEqual(len(torque_on), 16)
        self.assertLess(first_goal, min(torque_on))           # goal = present before any torque-on
        self.assertEqual(sdk.goals[0], present)
        np.testing.assert_allclose(q, H.units_to_urdf([present[i] for i in range(1, 17)], driver.directions))
        driver.close()
        self.assertEqual(sum(e[1:] == (H.TORQUE_ENABLE, 0) for e in sdk.log), 32)   # setup + close
        self.assertEqual(sdk.log[-1], ('close',))

    def test_negative_present_position(self):
        driver, _ = make_driver({i: -5 for i in range(1, 17)})
        q = driver.connect(torque=False)
        np.testing.assert_allclose(q, H.units_to_urdf(np.full(16, -5), driver.directions))

    def test_command_clips_to_limits(self):
        driver, sdk = make_driver({i: 1024 for i in range(1, 17)})
        driver.connect()
        sent = driver.command(np.full(16, 3.))
        np.testing.assert_allclose(sent, H.UPPER)
        np.testing.assert_array_equal([sdk.goals[-1][i] for i in range(1, 17)], H.urdf_to_units(H.UPPER, driver.directions))
        with self.assertRaises(ValueError):
            driver.command(np.zeros(15))
        with self.assertRaises(ValueError):
            driver.command(np.full(16, np.nan))

    def test_read_failure_raises_and_close_turns_torque_off(self):
        driver, sdk = make_driver({i: 1024 for i in range(1, 17)})
        driver.connect()
        sdk.fail_read = True
        with self.assertRaises(RuntimeError):
            driver.read()
        driver.close()
        self.assertEqual(sdk.log[-1], ('close',))
        with self.assertRaises(RuntimeError):
            driver.read()

    def test_mock_moves_toward_the_command(self):
        with H.HandDriver(mock=True) as driver:
            target = np.full(16, .1)
            driver.command(target)
            time.sleep(.05)
            self.assertTrue(np.all(np.abs(driver.read() - target) < np.abs(np.zeros(16) - target) + 1e-9))


if __name__ == '__main__':
    unittest.main()
