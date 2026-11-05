"""robot_server.py actions and IPC, and robot_panel.py's button logic, on the simulated arm and hand."""
import dataclasses
from multiprocessing.connection import Client
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tasks/dexlong_gear3/perception'))

from gx4m_real.camera.frame_ipc import receive, send                    # noqa: E402
from gx4m_real.robot.arm import ArmDriver                               # noqa: E402
from gx4m_real.robot.config import load_config                          # noqa: E402
from gx4m_real.robot.hand import HandDriver                             # noqa: E402
from robot_panel import enabled_actions, tactile_markdown                                # noqa: E402
from tasks.dexlong_gear3.robot_server import RobotController, serve     # noqa: E402
from tasks.dexlong_gear3.run_step1 import (CONTROL_CONFIG, MODEL, CameraToEnv, FixedGearPose,  # noqa: E402
                                           Step1Runner, arm_config, load_actor)
from tasks.dexlong_gear3.step1 import Step1Interface                    # noqa: E402


class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iface = Step1Interface()
        cls.actor = load_actor(MODEL, cls.iface)

    def controller(self, gear_pos=None):
        config = dataclasses.replace(load_config(CONTROL_CONFIG), velocity_limit=1.5)
        gear = FixedGearPose(self.iface.ref['object_pos'] if gear_pos is None else gear_pos,
                             self.iface.ref['object_quat'])
        runner = Step1Runner(self.iface, self.actor, ArmDriver(config, mock=True), None, gear, CameraToEnv(),
                             approach_speed=2., log=lambda message: None)
        out = Path(tempfile.mkdtemp(prefix='robot_server_test_'))
        controller = RobotController(runner, lambda: HandDriver(mock=True), out, log=lambda message: None)
        self.addCleanup(controller.close)
        self.addCleanup(runner.arm.disconnect)
        return controller

    def run_action(self, controller, action, timeout=10.):
        controller.submit(action)
        deadline = time.monotonic() + timeout
        while controller.status()['busy'] is not None:
            self.assertLess(time.monotonic(), deadline, f'{action} did not finish')
            time.sleep(.01)
        return controller.status()

    def test_buttons_in_order(self):
        c = self.controller()
        self.assertEqual(enabled_actions(c.status()), {'connect_arm', 'connect_hand'})
        self.assertIn('没有连接', self.run_action(c, 'release')['message'])
        s = self.run_action(c, 'connect_arm')
        self.assertEqual((s['arm']['connected'], s['arm']['mode']), (True, 'hold'))
        self.assertEqual(sorted(s['q']), sorted(self.iface.joint_order[:7]))
        self.assertIn('先连接灵巧手', self.run_action(c, 'hand_home')['message'])
        s = self.run_action(c, 'connect_hand')
        self.assertTrue(s['hand']['connected'])
        self.assertEqual(len(s['q']), 23)
        self.assertEqual(self.run_action(c, 'release')['arm']['mode'], 'released')
        self.assertEqual(enabled_actions(c.status()),
                         {'disconnect_arm', 'disconnect_hand', 'hold', 'arm_home', 'hand_home'})
        self.assertEqual(self.run_action(c, 'hold')['arm']['mode'], 'hold')
        self.assertIn('机械臂回初始位置', self.run_action(c, 'run_policy')['message'])
        # The arm moves alone: the hand keeps its pose until hand_home.
        hand_before = np.array([c.status()['q'][name] for name in self.iface.joint_order[7:]])
        s = self.run_action(c, 'arm_home')
        self.assertEqual(s['message'], '机械臂已回到初始位置')
        q = np.array([s['q'][name] for name in self.iface.joint_order])
        np.testing.assert_allclose(q[:7], self.iface.ref['initial_joint_pos'][:7], atol=.02)
        np.testing.assert_allclose(q[7:], hand_before, atol=1e-3)
        self.assertIn('灵巧手回初始位置', self.run_action(c, 'run_policy')['message'])
        s = self.run_action(c, 'hand_home')
        self.assertEqual(s['message'], '灵巧手已回到初始位置')
        q = np.array([s['q'][name] for name in self.iface.joint_order])
        np.testing.assert_allclose(q[7:], self.iface.ref['initial_joint_pos'][7:], atol=.1)
        # The policy runs until stopped, then holds and saves its log.
        c.submit('run_policy')
        time.sleep(.5)
        self.assertEqual(c.status()['busy'], 'run_policy')
        self.assertEqual(enabled_actions(c.status()), {'stop'})
        s = self.run_action(c, 'stop')
        self.assertIn('已停止', s['message'])
        self.assertEqual(s['arm']['mode'], 'hold')
        self.assertTrue((c.out/'episode_000.npz').is_file())
        s = self.run_action(c, 'disconnect_hand')
        self.assertFalse(s['hand']['connected'])
        s = self.run_action(c, 'disconnect_arm')
        self.assertEqual((s['arm']['connected'], s['q']), (False, {}))

    def test_tactile_tare_and_status(self):
        c = self.controller()
        self.assertIn('触觉没有打开', self.run_action(c, 'tare_tactile')['message'])

        class Tactile:
            threshold_n = .3

            def tare(self, duration):
                return np.array([1., 2., 3., 4.])

            def forces(self):
                return np.array([0., .4, 0., 0.])

        c.runner.tactile = Tactile()
        status = self.run_action(c, 'tare_tactile')
        self.assertIn('零点 1.0, 2.0, 3.0, 4.0 N', status['message'])
        self.assertEqual((status['tactile'], status['tactile_threshold']), ([0., .4, 0., 0.], .3))
        self.assertIn('tare_tactile', enabled_actions(status))
        self.run_action(c, 'connect_hand')
        message = self.run_action(c, 'hand_home', timeout=20.)['message']
        self.assertIn('灵巧手已回到初始位置', message)
        self.assertIn('触觉已去皮', message)        # the hand open at the start pose: a fresh zero

    def test_policy_needs_the_gear_at_the_start(self):
        c = self.controller(np.asarray(self.iface.ref['object_pos']) + [.03, 0, 0])
        self.run_action(c, 'connect_arm')
        self.run_action(c, 'connect_hand')
        self.run_action(c, 'arm_home')
        self.run_action(c, 'hand_home')
        message = self.run_action(c, 'run_policy')['message']
        self.assertIn('齿轮离训练起点 x +30', message)
        self.assertFalse(any(c.out.iterdir()))

    def test_start_height_tolerance(self):
        c = self.controller(np.asarray(self.iface.ref['object_pos']) + [0, 0, .015])
        for action in ('connect_arm', 'connect_hand', 'arm_home', 'hand_home'):
            self.run_action(c, action)
        self.assertIn('z +15 mm（x、y 各 ±20 mm，z ±5 mm）', self.run_action(c, 'run_policy')['message'])
        c.start_z_tolerance = .02           # --start-z-tolerance 0.02: the episode starts
        c.submit('run_policy')
        time.sleep(1.)                      # past the start checks, into the episode
        self.assertIn('策略结束：已停止', self.run_action(c, 'stop', timeout=20.)['message'])

    def test_stop_interrupts_the_move_to_start(self):
        c = self.controller()
        c.runner.approach_speed = .1
        self.run_action(c, 'connect_arm')
        c.submit('arm_home')
        time.sleep(.3)
        s = self.run_action(c, 'stop')
        self.assertEqual((s['message'], s['arm']['mode']), ('已停止，机械臂保持当前位置', 'hold'))
        self.run_action(c, 'connect_hand')
        c.submit('hand_home')
        time.sleep(.3)
        self.assertEqual(self.run_action(c, 'stop')['message'], '已停止，灵巧手保持当前位置')

    def test_viewer_connection(self):
        c = self.controller()
        key, stop = os.urandom(16), threading.Event()
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            address = s.getsockname()
        thread = threading.Thread(target=serve, args=(c, address, key), kwargs=dict(stop=stop), daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2.)
        self.addCleanup(stop.set)
        for _ in range(50):
            try:
                connection = Client(address, authkey=key)
                break
            except ConnectionRefusedError:
                time.sleep(.05)
        with connection:
            self.assertEqual(receive(connection)[0]['type'], 'status')
            send(connection, {'type': 'command', 'action': 'connect_arm'})
            deadline = time.monotonic() + 5
            while not receive(connection)[0]['arm']['connected']:
                self.assertLess(time.monotonic(), deadline)


class ArmConfigTests(unittest.TestCase):
    def test_hand_eye_compensation_replaces_the_control_one(self):
        handeye = ROOT/'apps/charuco_handeye/runtime/he_20261102_185319_j1p87/calibration/handeye_result.json'
        config = arm_config(CONTROL_CONFIG, Step1Interface(), handeye)
        self.assertEqual(config.compensation.offsets_deg[0], 8.7)
        self.assertEqual(config.velocity_limit, 1.5)
        self.assertEqual(arm_config(CONTROL_CONFIG, Step1Interface()).compensation,
                         load_config(CONTROL_CONFIG).compensation)


class PanelLogicTests(unittest.TestCase):
    def status(self, arm=False, mode='disconnected', hand=False, busy=None):
        return {'arm': {'connected': arm, 'mode': mode, 'error': None}, 'hand': {'connected': hand, 'error': None},
                'busy': busy, 'message': '', 'q': {}}

    def test_enabled_actions(self):
        self.assertEqual(enabled_actions(None), set())
        self.assertEqual(enabled_actions(self.status(True, 'hold', True)),
                         {'disconnect_arm', 'disconnect_hand', 'release', 'arm_home', 'hand_home', 'run_policy'})
        self.assertEqual(enabled_actions(self.status(hand=True)), {'connect_arm', 'disconnect_hand', 'hand_home'})
        self.assertEqual(enabled_actions(self.status(True, 'fault')), {'disconnect_arm', 'connect_hand'})
        self.assertEqual(enabled_actions(self.status(True, 'hold', True, busy='hand_home')), {'stop'})
        self.assertEqual(enabled_actions(self.status(True, 'hold', True, busy='connect_hand')), set())
        self.assertIn('tare_tactile', enabled_actions(dict(self.status(), tactile=[0., .1, .2, .3])))
        self.assertNotIn('tare_tactile', enabled_actions(dict(self.status(), tactile='PaXini reading too old')))
        self.assertNotIn('tare_tactile', enabled_actions(dict(self.status(busy='run_policy'), tactile=[0.]*4)))

    def test_tactile_markdown(self):
        text = tactile_markdown(dict(self.status(), tactile=[.1, .5, 0., -.2], tactile_threshold=.3))
        self.assertIn('| 食指 | +0.50 N | 🔴 是 |', text)
        self.assertIn('| 拇指 | +0.10 N | 否 |', text)
        self.assertIn('读取失败', tactile_markdown(dict(self.status(), tactile='timeout')))
        self.assertIn('没有打开触觉', tactile_markdown(self.status()))


if __name__ == '__main__':
    unittest.main()
