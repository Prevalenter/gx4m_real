"""Control contracts tested without CAN hardware."""
from pathlib import Path
import sys
import time
import types
import tempfile
import json
import threading
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arm import ArmDriver
from config import ARM_JOINTS, load_config, validate_position
from trajectory import load_trajectory, save_trajectory


class FakeCAN:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeRobot:
    def __init__(self, q):
        self.q = q.copy()
        self.running = False
        self.last_feedback_at = time.monotonic()
        self.last_error = None
        self.events = []

    def getJP(self):
        return self.q.copy()

    def setJPVTs(self, q, velocities, torques):
        self.events.append(('send', np.array(q), velocities, torques))

    def run(self):
        self.events.append(('run',))
        self.running = True

    def stop(self):
        self.events.append(('stop',))
        self.running = False

    def disable(self):
        self.events.append(('disable',))

    def set_released(self, released):
        self.events.append(('released', released))


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / 'config.yaml')

    def make_real(self):
        measured = self.config.initial_position + .02
        robot, can = FakeRobot(measured), FakeCAN()
        driver = ArmDriver(self.config, mock=False, factory=lambda _: (robot, can))
        return driver, robot, can, measured

    def test_constructor_and_mock_never_touch_hardware(self):
        def forbidden(*_):
            self.fail('Hardware factory accessed')
        driver = ArmDriver(self.config, mock=False, factory=forbidden)
        self.assertEqual(driver.mode, 'disconnected')
        mock = ArmDriver(self.config, mock=True, factory=forbidden)
        mock.connect()
        mock.command(self.config.initial_position + .03)
        mock.update()
        mock.hold()
        mock.disconnect()
        self.assertEqual(mock.mode, 'disconnected')

    def test_first_command_is_measured_hold_before_run(self):
        driver, robot, can, measured = self.make_real()
        current = driver.connect()
        self.assertEqual([e[0] for e in robot.events], ['send', 'run'])
        np.testing.assert_allclose(robot.events[0][1], measured)
        np.testing.assert_allclose(current, measured)
        self.assertEqual(robot.events[0][2], [.25] * 7)
        driver.disconnect()
        self.assertTrue(can.closed)

    def test_hardware_factory_pins_adapter_channel_and_keeps_limits(self):
        from arm import hardware_factory
        options = {}
        can = FakeCAN()
        can.init_can = lambda: None
        robot = FakeRobot(self.config.initial_position)
        robot.setup = lambda: robot.events.append(('setup',))
        def make_can(**kwargs):
            options['can'] = kwargs
            return can
        def make_robot(connection, **kwargs):
            self.assertIs(connection, can)
            options['robot'] = kwargs
            return robot
        stub = types.SimpleNamespace(VCICAN=make_can, GX7=make_robot)
        with patch.dict(sys.modules, {'libgx7': stub}):
            result = hardware_factory(self.config)
        self.assertEqual(result, (robot, can))
        self.assertEqual(options['can']['device_serial_number'], '31F1000632C')
        self.assertTrue(options['can']['enforce_single_io_thread'])
        self.assertEqual(options['robot']['can_channel'], 0)
        self.assertTrue(options['robot']['soft_limit'])
        self.assertEqual(options['robot']['control_mode'], 'pvt')
        self.assertEqual(robot.events, [('setup',)])

    def test_control_module_import_is_independent_of_tracker_and_hardware(self):
        import subprocess
        code = ('import sys; import arm; '
                'assert "libgx7" not in sys.modules; '
                'assert not any(n.startswith("tracker_app") for n in sys.modules)')
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True)

    def test_invalid_targets_rejected_without_send_or_state_change(self):
        driver, robot, _, measured = self.make_real()
        driver.connect()
        for invalid in ([0] * 6, [0] * 8, [float('nan')] * 7,
                        [float('inf')] * 7, self.config.upper + .01, self.config.lower - .01):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    driver.command(invalid)
                self.assertEqual(len(robot.events), 2)
                np.testing.assert_allclose(driver.requested, measured)
        driver.disconnect()

    def test_command_is_staged_then_limited_and_hold_uses_feedback(self):
        driver, robot, _, measured = self.make_real()
        driver.connect()
        driver.command(measured + .3)
        self.assertEqual(len(robot.events), 2)
        driver.update()
        np.testing.assert_allclose(robot.events[-1][1], measured + self.config.max_joint_step_rad)
        robot.q = measured + .01
        driver.hold()
        np.testing.assert_allclose(robot.events[-1][1], robot.q)
        np.testing.assert_allclose(driver.requested, robot.q)
        driver.disconnect()

    def test_stale_feedback_latches_fault_and_blocks_more_commands(self):
        driver, robot, can, measured = self.make_real()
        driver.connect()
        driver.command(measured + .02)
        robot.last_feedback_at -= 1
        sends = sum(e[0] == 'send' for e in robot.events)
        driver.update()
        self.assertEqual(driver.mode, 'fault')
        self.assertFalse(robot.running)
        driver.update()
        with self.assertRaises(RuntimeError):
            driver.command(measured)
        self.assertEqual(sum(e[0] == 'send' for e in robot.events), sends)
        self.assertIn('反馈', driver.error)
        driver.disconnect()
        self.assertTrue(can.closed)

    def test_failed_connect_releases_resources(self):
        driver, robot, can, _ = self.make_real()
        robot.q[:] = float('nan')
        with self.assertRaises(ValueError):
            driver.connect()
        self.assertTrue(can.closed)
        self.assertFalse(robot.running)
        self.assertIsNone(driver.robot)
        self.assertIn(('disable',), robot.events)

    def test_mock_respects_velocity_and_reaches_target(self):
        driver = ArmDriver(self.config)
        driver.connect()
        initial = driver.q.copy()
        driver.command(initial + .1)
        for _ in range(20):
            driver.robot.last_feedback_at = time.monotonic() - .05
            old = driver.q.copy()
            driver.update()
            self.assertLessEqual(float(np.max(np.abs(driver.q - old))), .013)
        np.testing.assert_allclose(driver.q, initial + .1)
        driver.hold()
        self.assertEqual(driver.mode, 'hold')
        driver.disconnect()

    def test_model_paths_joint_mapping_and_fixed_hand(self):
        from app import load_model
        model, names, positions = load_model(self.config)
        self.assertEqual([name for name in names if name in ARM_JOINTS], list(ARM_JOINTS))
        self.assertGreater(len(names), 7)
        for name, q in zip(names, positions):
            self.assertEqual(q, self.config.initial_joints[name])
        # Loading every mesh must have succeeded without fallback primitives.
        self.assertGreater(len(model.scene.geometry), 7)
        hand_indices = [i for i, name in enumerate(names) if name.startswith('gx4m_')]
        original_hand = positions[hand_indices].copy()
        positions[[names.index(name) for name in ARM_JOINTS]] += .1
        np.testing.assert_array_equal(positions[hand_indices], original_hand)

    def test_config_rejects_implicit_adapter_selection(self):
        import tempfile
        import yaml
        raw = yaml.safe_load((ROOT / 'config.yaml').read_text())
        raw['model']['urdf'] = str(self.config.urdf)
        raw['model']['initial_pose'] = str(ROOT.parent / 'step0/initial_pose.json')
        raw['arm']['motor_config'] = str(self.config.motor_config)
        raw['arm']['can_device_serial'] = ''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text(yaml.safe_dump(raw))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_released_recording_tracks_feedback_without_position_commands(self):
        driver, robot, _, measured = self.make_real()
        driver.connect()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'record.json'
            driver.start_recording(path)
            self.assertEqual(robot.events[-1], ('released', True))
            sends = len(robot.events)
            robot.q = measured + .02
            driver.update()
            self.assertEqual(len(robot.events), sends)
            with self.assertRaises(RuntimeError):
                driver.command(measured)
            with self.assertRaises(RuntimeError):
                driver.hold()
            driver.stop_recording()
            times, positions = load_trajectory(self.config, path)
            self.assertGreater(times[-1], 0)
            np.testing.assert_allclose(positions[-1], robot.q)
            self.assertEqual(driver.mode, 'released')
            driver.hold()
            self.assertEqual(robot.events[-1], ('released', False))
            np.testing.assert_allclose(robot.events[-2][1], robot.q)
        driver.disconnect()

    def test_disconnect_and_fault_save_partial_recording(self):
        for fault in (False, True):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                driver, robot, can, _ = self.make_real()
                driver.connect()
                path = Path(directory) / 'partial.json'
                driver.start_recording(path)
                driver.update()
                if fault:
                    robot.last_feedback_at -= 1
                    driver.update()
                    self.assertEqual(driver.mode, 'fault')
                driver.disconnect()
                self.assertTrue(can.closed)
                self.assertIsNone(driver.recording)
                load_trajectory(self.config, path)

    def test_recording_never_overwrites_and_bad_replay_never_enables(self):
        driver, robot, _, _ = self.make_real()
        driver.connect()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'record.json'
            path.write_text('{}')
            count = len(robot.events)
            with self.assertRaises(ValueError):
                driver.start_recording(path)
            with self.assertRaises(ValueError):
                driver.play(path)
            self.assertEqual(len(robot.events), count)
            self.assertEqual(path.read_text(), '{}')
        driver.disconnect()

    def test_playback_returns_to_start_limits_speed_and_holds_at_end(self):
        driver, robot, _, measured = self.make_real()
        driver.connect()
        start = measured + .1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'record.json'
            save_trajectory(path, [0., .1], [start, start + .1])
            driver.play(path, speed=2.)
            self.assertAlmostEqual(driver.playback['speed'], .25)
            driver.playback['last'] -= .1
            driver.update()
            self.assertEqual(driver.playback['time'], 0.)
            np.testing.assert_allclose(driver.requested, start)
            robot.q = start.copy()
            driver.update()
            self.assertTrue(driver.playback['at_start'])
            for _ in range(10):
                robot.q = driver.requested.copy()
                driver.playback['last'] -= .1
                driver.update()
                if driver.playback is None:
                    break
            self.assertIsNone(driver.playback)
            self.assertEqual(driver.mode, 'hold')
            np.testing.assert_allclose(driver.requested, start + .1)
            driver.play(path)
            driver.hold()
            self.assertIsNone(driver.playback)
        driver.disconnect()

    def test_trajectory_rejects_bad_times_units_joints_and_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'record.json'
            q = self.config.initial_position
            save_trajectory(path, [0., 1.], [q, q])
            data = json.loads(path.read_text())
            for key, value in [('times_sec', [0., 0.]), ('times_sec', [0., float('nan')]),
                               ('units', 'deg'), ('joint_names', list(reversed(ARM_JOINTS))),
                               ('positions_rad', [q.tolist(), (self.config.upper + 1).tolist()])]:
                with self.subTest(key=key, value=value):
                    path.write_text(json.dumps({**data, key: value}))
                    with self.assertRaises(ValueError):
                        load_trajectory(self.config, path)

    def test_sdk_released_polling_and_resume_seed_before_enable(self):
        from libgx7.robot.robot import Robot
        robot = Robot.__new__(Robot)
        robot.command_lock = threading.RLock()
        robot._release_applied = threading.Event()
        robot._release_requested = True
        robot._motors_released = False
        robot.pvt_control_velocities = [.25] * 7
        robot.pvt_control_torques = [.85] * 7
        events = []
        q = self.config.initial_position
        robot.robot_motors = types.SimpleNamespace(
            disable_all=lambda: events.append('disable'),
            enable_all=lambda: events.append('enable'))
        robot.update_status = lambda _: None
        robot.getJP = lambda: q.copy()
        robot.setJPVTs = lambda positions, *_: events.append(('seed', positions))
        self.assertTrue(robot._poll_released())
        self.assertTrue(robot._poll_released())
        self.assertEqual(events, ['disable', 'disable'])
        robot._release_requested = False
        self.assertFalse(robot._poll_released())
        self.assertEqual(events[-3], 'disable')
        np.testing.assert_allclose(events[-2][1], q)
        self.assertEqual(events[-1], 'enable')


if __name__ == '__main__':
    unittest.main()
