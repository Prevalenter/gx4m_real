import argparse
import json
from pathlib import Path
import tempfile
import socket
import time
import unittest
from unittest import mock

import numpy as np

from gx4m_real.robot.config import load_config
from tasks.beam_lift_replay.app import CommandRate, ROOT, ReplayApp, load_targets, parse_args, replay_config
from tasks.beam_lift_replay.beam_view import BeamView, world_camera
from tasks.dexlong_gear3.arm_setup import DEFAULT_HANDEYE


class CompensationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/'handeye_result.json'
        self.recorded = {'directions': [-1, 1, -1, 1, -1, -1, -1],
                         'offsets_deg': [4.5, 0., -.5254, .3368, -.3779, -1.4705, 7.5802]}
        self.path.write_text(json.dumps({'joint_compensation': self.recorded}))
        self.config_path = ROOT/'apps/control/config.yaml'

    def test_identified_compensation_replaces_control_config(self):
        config = replay_config(self.config_path, self.path)
        self.assertEqual(config.compensation.offsets_deg, tuple(self.recorded['offsets_deg']))
        self.assertEqual(config.compensation.directions, tuple(self.recorded['directions']))
        self.assertEqual(config.compensation_path, self.path)
        baseline = load_config(self.config_path)
        self.assertEqual(config.velocity_limit, baseline.velocity_limit)
        np.testing.assert_array_equal(config.lower, baseline.lower)
        np.testing.assert_array_equal(config.upper, baseline.upper)

    def test_identified_config_validates_trajectory_without_changing_model_angles(self):
        config = replay_config(self.config_path, self.path)
        path = ROOT/'tasks/beam_lift_replay/trajectory/joint_targets.json'
        _, times, positions = load_targets(path, config)
        source = json.loads(path.read_text())
        self.assertEqual(positions.shape, (int(np.ceil(source['times_sec'][-1]*30))+1, 23))
        np.testing.assert_allclose(positions[0, :7], source['positions_rad'][0][:7])
        np.testing.assert_allclose(positions[-1, :7], source['positions_rad'][-1][:7])
        self.assertAlmostEqual(times[-1], (len(positions)-1)/30)

    def test_send_and_measured_apply_compensation_once(self):
        app = ReplayApp.__new__(ReplayApp)
        app.mode, app.play_hz, app.hand_target_speed_limit = 'play', 30, 1.5
        app.last_target = np.zeros(23)
        app.config = replay_config(self.config_path, self.path)
        app.arm, app.hand = mock.Mock(), mock.Mock()
        app.arm_rate, app.hand_rate = CommandRate(), CommandRate()
        model = np.zeros(23)
        model[:7] = np.linspace(-.1, .1, 7)
        motor = np.asarray(self.recorded['directions']) * (
            model[:7] - np.deg2rad(self.recorded['offsets_deg']))
        app.arm.snapshot.return_value = {'mode': 'hold', 'positions': motor}
        app.hand.read.return_value = model[7:]
        app.send(model)
        np.testing.assert_allclose(app.arm.command.call_args.args[0], motor)
        np.testing.assert_allclose(app.hand.command.call_args.args[0], model[7:])
        np.testing.assert_allclose(app.measured(), model, atol=1e-12)
        self.assertEqual(len(app.arm_rate.samples), 1)
        self.assertEqual(len(app.hand_rate.samples), 1)

    def test_missing_or_invalid_calibration_compensation_is_rejected(self):
        for document in ({}, {'joint_compensation': {}},
                         {'joint_compensation': dict(self.recorded, offsets_deg=[0.])}):
            with self.subTest(document=document):
                self.path.write_text(json.dumps(document))
                with self.assertRaises(ValueError):
                    replay_config(self.config_path, self.path)

    def test_default_handeye_and_explicit_override(self):
        self.assertEqual(parse_args(['--real']).handeye, DEFAULT_HANDEYE)
        self.assertEqual(parse_args(['--real', '--handeye', str(self.path)]).handeye, self.path)
        self.assertEqual(parse_args([]).handeye, DEFAULT_HANDEYE)

    def test_default_calibration_matches_deployment(self):
        from tasks.dexlong_gear3.arm_setup import arm_config_for_reference

        handeye = parse_args([]).handeye
        reference = json.loads((ROOT/'tasks/dexlong_gear3/config/beam1_reference.json').read_text())
        replay = replay_config(self.config_path, handeye, reference)
        deployment = arm_config_for_reference(self.config_path, reference, handeye)
        self.assertEqual(replay.compensation, deployment.compensation)
        self.assertEqual(replay.compensation.offsets_deg,
                         (11.1842, .4693, -2.1378, -.192, .54, -1.2316, 7.8836))
        load_targets(ROOT/'tasks/beam_lift_replay/trajectory/joint_targets.json', replay)

    def test_deployment_entrypoints_default_handeye_and_override(self):
        from tasks.dexlong_gear3 import robot_server, run_step1

        original_parse_args = argparse.ArgumentParser.parse_args
        for entrypoint in (run_step1, robot_server):
            for argv, expected in (([], DEFAULT_HANDEYE), (['--handeye', str(self.path)], self.path)):
                with self.subTest(entrypoint=entrypoint.__name__, argv=argv):
                    def check_args(parser, arguments):
                        parsed = original_parse_args(parser, arguments)
                        self.assertEqual(parsed.handeye, expected)
                        raise InterruptedError

                    with mock.patch.object(argparse.ArgumentParser, 'parse_args', check_args):
                        with self.assertRaises(InterruptedError):
                            entrypoint.main(argv)

    def test_offline_without_handeye_keeps_control_compensation(self):
        config = replay_config(self.config_path)
        self.assertEqual(config.compensation, load_config(self.config_path).compensation)


class CommandRateTests(unittest.TestCase):
    def test_actual_frequency_and_idle_expiry(self):
        counter = CommandRate()
        self.assertEqual(counter.hz(0.), 0.)
        for stamp in np.arange(31) / 30:
            counter.mark(float(stamp))
        self.assertAlmostEqual(counter.hz(1.), 30.)
        self.assertEqual(counter.hz(2.1), 0.)
        for stamp in 3. + np.arange(11) / 10:
            counter.mark(float(stamp))
        self.assertAlmostEqual(counter.hz(4.), 10.)

    def app(self, real=False):
        app = ReplayApp.__new__(ReplayApp)
        app.args = mock.Mock(real=real)
        app.arm = app.hand = None
        app.arm_rate, app.hand_rate = CommandRate(), CommandRate()
        app.arm_hz, app.hand_hz = mock.Mock(value=''), mock.Mock(value='')
        app.rates_shown = -float('inf')
        return app

    def test_frequency_display_distinguishes_preview_mock_and_real(self):
        app = self.app()
        app.refresh_rates(0.)
        self.assertIn('离线预览', app.arm_hz.value)
        app.arm, app.hand = mock.Mock(), mock.Mock()
        for stamp in (0., .1, .2, .3):
            app.arm_rate.mark(stamp)
        for stamp in (0., .2):
            app.hand_rate.mark(stamp)
        app.refresh_rates(.3)
        self.assertEqual(app.arm_hz.value, '模拟 · 10.0 Hz')
        self.assertEqual(app.hand_hz.value, '模拟 · 3.3 Hz')
        app.args.real = True
        app.refresh_rates(2.)
        self.assertEqual(app.arm_hz.value, '实机 · 0.0 Hz')
        self.assertEqual(app.hand_hz.value, '实机 · 0.0 Hz')

    def test_hand_success_is_counted_when_arm_update_faults(self):
        app = self.app(real=True)
        app.mode, app.play_hz, app.hand_target_speed_limit = 'play', 30, 1.5
        app.last_target = np.zeros(23)
        app.config = load_config(ROOT/'apps/control/config.yaml')
        app.arm, app.hand = mock.Mock(), mock.Mock()
        app.arm.snapshot.return_value = {'mode': 'fault', 'error': 'test fault'}
        with self.assertRaisesRegex(RuntimeError, 'test fault'):
            app.send(np.zeros(23))
        self.assertEqual(len(app.arm_rate.samples), 0)
        self.assertEqual(len(app.hand_rate.samples), 1)

    def test_failed_hand_command_does_not_count_success(self):
        app = self.app(real=True)
        app.mode, app.play_hz, app.hand_target_speed_limit = 'play', 30, 1.5
        app.last_target = np.zeros(23)
        app.config = load_config(ROOT/'apps/control/config.yaml')
        app.arm, app.hand = mock.Mock(), mock.Mock()
        app.hand.command.side_effect = RuntimeError('serial failure')
        with self.assertRaisesRegex(RuntimeError, 'serial failure'):
            app.send(np.zeros(23))
        self.assertEqual(len(app.arm_rate.samples), 0)
        self.assertEqual(len(app.hand_rate.samples), 0)


class BeamViewTests(unittest.TestCase):
    def setUp(self):
        import viser
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/'handeye.json'
        transform = np.eye(4)
        transform[:3, 3] = [.1, .2, .3]
        self.path.write_text(json.dumps({'mode': 'eye-to-hand', 'T_base_cam': transform.tolist()}))
        self.config = load_config(ROOT/'apps/control/config.yaml')
        self.trajectory = ROOT/'tasks/beam_lift_replay/trajectory/joint_targets.json'
        self.data = json.loads(self.trajectory.read_text())
        self.server = viser.ViserServer(host='127.0.0.1', port=0)
        self.addCleanup(self.server.stop)

    def test_initial_and_recorded_positions_are_distinct_from_live_pose(self):
        view = BeamView(self.server, self.data, self.trajectory, self.config)
        self.addCleanup(view.close)
        view.update(0.)
        np.testing.assert_allclose(view.start.position, self.data['initial_object_pose_xyz_wxyz'][:3])
        self.assertFalse(view.current.visible)
        self.assertTrue(view.recorded.visible)
        with np.load(self.trajectory.with_name('contact_force.npz')) as archive:
            first, last = archive['object_pos_m'][0], archive['object_pos_m'][-1]
            end_time = float(archive['time_s'][-1])
        np.testing.assert_allclose(view.recorded.position, first)
        view.update(end_time + 1.)
        np.testing.assert_allclose(view.recorded.position, last)
        view.show_start.value = view.show_recorded.value = False
        view.update(0.)
        self.assertFalse(view.start.visible)
        self.assertFalse(view.recorded.visible)

    def test_live_pose_world_transform_expiry_and_invalid_packets(self):
        view = BeamView(self.server, self.data, self.trajectory, self.config, handeye=self.path, pose_port=0)
        self.addCleanup(view.close)
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addCleanup(sender.close)
        stamp = time.monotonic()
        transform = np.eye(4)
        transform[:3, 3] = [.01, -.02, .5]
        address = view.sock.getsockname()
        sender.sendto(json.dumps({'timestamp_monotonic': stamp, 'object_in_camera': transform.tolist()}).encode(), address)
        time.sleep(.01)
        view.update(0., now=stamp + .02)
        expected = world_camera(self.config, self.path) @ transform
        self.assertTrue(view.current.visible)
        np.testing.assert_allclose(view.current.position, expected[:3, 3])
        self.assertIn('Δxyz', view.delta_text.value)
        original = np.asarray(view.current.position).copy()
        for value in (None, np.zeros((4, 4)).tolist(), [[1, 2]]):
            sender.sendto(json.dumps({'timestamp_monotonic': stamp + .03, 'object_in_camera': value}).encode(), address)
        sender.sendto(b'invalid json', address)
        time.sleep(.01)
        view.update(0., now=stamp + .04)
        np.testing.assert_allclose(view.current.position, original)
        view.update(0., now=stamp + 2.)
        self.assertFalse(view.current.visible)
        self.assertIn('过期', view.current_text.value)

    def test_unsupported_calibration_and_invalid_initial_pose_rejected(self):
        self.path.write_text(json.dumps({'mode': 'eye-in-hand'}))
        with self.assertRaises(ValueError):
            world_camera(self.config, self.path)
        with self.assertRaises(ValueError):
            BeamView(self.server, {}, self.trajectory, self.config)


if __name__ == '__main__':
    unittest.main()
