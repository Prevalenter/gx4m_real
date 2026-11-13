import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import queue
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

from tasks.beam_lift_replay.analyze import analyze_trial, main as analyze_main
from tasks.beam_lift_replay.app import ROOT, ReplayApp, load_targets, parse_args, replay_config
from tasks.beam_lift_replay.recording import ReplayRecorder


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.args = parse_args(['--record-dir', self.temporary.name, '--record-trials', '2', '--record-tail', '0'])
        self.config = replay_config(self.args.config)
        self.data, self.times, self.targets = load_targets(self.args.trajectory, self.config)
        self.recorder = ReplayRecorder(self.args.record_dir, self.args, self.config, self.times, self.targets)

    def app(self):
        app = ReplayApp.__new__(ReplayApp)
        app.args, app.config = self.args, self.config
        app.recorder = self.recorder
        app.arm, app.hand = mock.Mock(), mock.Mock()
        app.beam = mock.Mock(latest=None)
        app.status, app.record_status, app.frame = mock.Mock(), mock.Mock(), mock.Mock()
        app.mode, app.index = 'idle', 0
        app.play_hz = 30
        app.target_hz = mock.Mock(value='30 Hz')
        app.q, app.times = self.targets[:2], self.times[:2]
        app.last_target = app.q[0].copy()
        app.events = queue.Queue()
        app.draw = mock.Mock()
        app.measured = mock.Mock(return_value=app.q[0].copy())
        app.send = lambda target: setattr(app, 'last_target', target.copy())
        app.arm.snapshot.return_value = {'mode': 'hold'}
        app.arm.feedback_sample.side_effect = lambda: {
            'timestamp_monotonic': time.monotonic(),
            'positions_rad': self.config.compensation.to_motor(app.last_target[:7]).tolist()}
        app.hand.read.side_effect = lambda: app.last_target[7:].copy()
        return app

    def test_manual_play_records_settling_and_stops_after_limit(self):
        app = self.app()
        for trial in range(2):
            app.action('play')
            app.tick()
            app.tick()
            self.assertEqual(app.mode, 'settle')
            app.tick()
            self.assertEqual(app.mode, 'idle')
            self.assertEqual(self.recorder.completed, trial+1)
        with self.assertRaisesRegex(RuntimeError, 'trial count reached'):
            app.action('play')
        with np.load(self.recorder.last_path, allow_pickle=False) as data:
            self.assertEqual(str(data['outcome']), 'completed')
            self.assertEqual(data['hand_rad'].shape, (4, 16))
            self.assertEqual(data['arm_rad'].shape, (4, 7))
            self.assertTrue(np.isnan(data['beam_world']).all())
        with redirect_stdout(io.StringIO()):
            analyze_main([str(self.recorder.directory), '--plot'])
        report = json.loads((self.recorder.directory/'summary.json').read_text())
        self.assertEqual(report['completed_trials'], 2)
        self.assertTrue((self.recorder.directory/'comparison.svg').is_file())
        self.assertTrue((self.recorder.directory/'reference/joint_targets.npz').is_file())
        session = json.loads((self.recorder.directory/'session.json').read_text())
        self.assertFalse(session['real_hardware'])
        self.assertIn('joint_targets.json', session['trajectory_sha256'])

    def test_stop_saves_partial_without_counting_completion(self):
        app = self.app()
        app.action('play')
        app.tick()
        app.action('stop')
        self.assertEqual(self.recorder.completed, 0)
        with np.load(self.recorder.last_path, allow_pickle=False) as data:
            self.assertEqual(str(data['outcome']), 'stopped')
        app.action('play')
        self.assertEqual(self.recorder.attempts, 2)

    def test_10hz_preserves_frames_and_records_selected_rate(self):
        app = self.app()
        app.action('play10')
        self.assertEqual(app.play_hz, 10)
        self.assertEqual(app.target_hz.value, '10 Hz')
        self.assertAlmostEqual(app.tick_period(), .1)
        self.assertIn('10 Hz', app.status.value)
        with self.assertRaisesRegex(RuntimeError, 'Stop the current operation'):
            app.action('play')
        self.assertEqual(app.play_hz, 10)
        for target in app.q:
            app.tick()
            np.testing.assert_array_equal(app.last_target, target)
        self.assertEqual(app.mode, 'settle')
        self.assertAlmostEqual(app.tick_period(), 1/30)
        app.tick()
        with np.load(self.recorder.last_path, allow_pickle=False) as data:
            self.assertEqual(float(data['requested_replay_hz']), 10.)
            commands = np.isfinite(data['command_monotonic_s'])
            np.testing.assert_array_equal(data['target_rad'][commands], app.q)
            np.testing.assert_array_equal(data['phase_s'][commands], app.times)
        app.measured.return_value = app.q[0].copy()
        app.action('play')
        self.assertEqual(app.play_hz, 30)
        self.assertAlmostEqual(app.tick_period(), 1/30)

    def test_10hz_uses_existing_safety_checks(self):
        app = self.app()
        app.measured.return_value = app.q[0] + 1.
        with self.assertRaisesRegex(RuntimeError, 'Move to start first'):
            app.action('play10')
        self.assertIsNone(self.recorder.active)
        app.measured.return_value = app.q[0].copy()
        app.args = argparse.Namespace(**vars(self.args))
        app.args.real = True
        with self.assertRaisesRegex(RuntimeError, 'fresh beam pose'):
            app.action('play10')
        self.assertIsNone(self.recorder.active)

    def test_run_loop_uses_selected_period(self):
        for rate in (10, 30):
            with self.subTest(rate=rate):
                app = self.app()
                app.mode, app.play_hz = 'play', rate
                app.tick = mock.Mock()
                app.refresh_rates = mock.Mock()
                app.hold = mock.Mock()
                app.finish_recording = mock.Mock()
                app.server = mock.Mock()
                clock = [100.]
                sleeps = []

                def sleep(duration):
                    sleeps.append(duration)
                    clock[0] += duration
                    if len(sleeps) == 3:
                        raise KeyboardInterrupt

                with mock.patch('tasks.beam_lift_replay.app.time.monotonic', side_effect=lambda: clock[0]), \
                        mock.patch('tasks.beam_lift_replay.app.time.sleep', side_effect=sleep):
                    app.run()
                np.testing.assert_allclose(sleeps, [1/rate]*3)
                self.assertEqual(app.tick.call_count, 3)

    def test_empty_interrupted_attempt_and_analysis(self):
        self.recorder.start()
        self.recorder.finish('interrupted')
        with redirect_stdout(io.StringIO()):
            analyze_main([str(self.recorder.directory)])
        report = json.loads((self.recorder.directory/'summary.json').read_text())
        self.assertEqual(report['completed_trials'], 0)
        self.assertEqual(report['trials'][0]['error'], 'No samples saved')

    def test_real_recording_requires_fresh_perception(self):
        app = self.app()
        app.args = argparse.Namespace(**vars(self.args))
        app.args.real = True
        with self.assertRaisesRegex(RuntimeError, 'fresh beam pose'):
            app.action('play')
        self.assertIsNone(self.recorder.active)

    def test_invalid_recording_arguments(self):
        for args in (['--record-trials', '0'], ['--record-tail', '-1'], ['--record-tail', 'nan']):
            with redirect_stdout(io.StringIO()), mock.patch('sys.stderr', io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(args)


class AnalysisTests(unittest.TestCase):
    def test_phase_and_wall_time_distinguish_slowdown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'trial_001.npz'
            phase = np.arange(31)/30
            clock = 100.+phase*2
            targets = np.repeat(phase[:, None], 23, axis=1)
            beam = np.repeat(np.eye(4)[None], len(phase), axis=0)
            beam[:, 2, 3] = phase
            np.savez_compressed(path, outcome='completed', phase_s=phase, command_monotonic_s=clock,
                                sample_monotonic_s=clock, arm_monotonic_s=clock,
                                hand_read_start_monotonic_s=clock, hand_read_end_monotonic_s=clock,
                                target_rad=targets, arm_rad=targets[:, :7], hand_rad=targets[:, 7:],
                                beam_monotonic_s=clock, beam_world=beam)
            reference = {'time_s': phase, 'actual_rad': targets}
            beam_reference = {'time_s': phase, 'object_pos_m': beam[:, :3, 3]}
            result = analyze_trial(path, reference, beam_reference)
            self.assertAlmostEqual(result['command_hz'], 15.)
            self.assertAlmostEqual(result['wall_to_nominal_ratio'], 2.)
            self.assertAlmostEqual(result['arm']['vs_command']['rmse_deg'], 0.)
            self.assertAlmostEqual(result['arm']['vs_sim_phase_aligned']['rmse_deg'], 0.)
            self.assertGreater(result['arm']['vs_sim_wall_time']['rmse_deg'], 1.)
            self.assertAlmostEqual(result['beam']['phase_position_rmse_mm'], 0.)
            self.assertAlmostEqual(result['beam_fresh_fraction'], 1.)
            np.savez_compressed(path, outcome='completed', phase_s=phase, command_monotonic_s=clock,
                                sample_monotonic_s=clock, arm_monotonic_s=clock,
                                hand_read_start_monotonic_s=clock, hand_read_end_monotonic_s=clock,
                                target_rad=targets, arm_rad=targets[:, :7], hand_rad=targets[:, 7:],
                                beam_monotonic_s=clock-1, beam_world=beam)
            stale = analyze_trial(path, reference, beam_reference)
            self.assertEqual(stale['beam_fresh_fraction'], 0.)
            self.assertNotIn('beam', stale)


if __name__ == '__main__':
    unittest.main()
