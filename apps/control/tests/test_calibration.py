"""Software timing and lifecycle contracts; no USB camera or CAN bus used."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gx4m_real.camera.frames import ColorFrame
from gx4m_real.robot.recorder import JointHistory, PlaybackRecorder
from gx4m_real.robot.config import load_config
from gx4m_real.robot.trajectory import save_trajectory


def sample(timestamp, value):
    return {'timestamp_monotonic': timestamp, 'positions_rad': [value] * 7,
            'mode': 'playback', 'target_rad': [9.] * 7, 'playback_phase': 'trajectory',
            'trajectory_time_sec': value}


class FakeCamera:
    def __init__(self, fail_after=None):
        self.metadata = {'synthetic': True, 'K': [[1, 0, 1], [0, 1, 1], [0, 0, 1]],
                         'color_order': 'RGB'}
        self.sequence = 0
        self.fail_after = fail_after
        self.closed = False

    def open(self):
        pass

    def read(self, timeout=2.):
        if self.fail_after is not None and self.sequence >= self.fail_after:
            raise RuntimeError('camera disconnected')
        time.sleep(.01)
        self.sequence += 1
        return ColorFrame(self.sequence, np.full((3, 4, 3), [210, 20, 5], dtype=np.uint8),
                          time.monotonic(), self.sequence * 10000)

    def close(self):
        self.closed = True


def wait_until(predicate, timeout=3.):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError('timed out waiting for background capture')


class AlignmentTests(unittest.TestCase):
    def test_interpolates_actual_feedback_instead_of_targets(self):
        history = JointHistory(max_gap=.1)
        history.append(sample(1., 0.))
        history.append(sample(1.1, .2))
        result = history.align(1.025, timeout=0)
        self.assertTrue(result['alignment_valid'])
        np.testing.assert_allclose(result['positions_rad'], [.05] * 7)
        np.testing.assert_allclose(result['joint_time_offsets_sec'], [.025, .075])
        self.assertEqual(result['playback_phase'], 'trajectory')

    def test_missing_and_wide_brackets_are_not_extrapolated(self):
        history = JointHistory(max_gap=.05)
        history.append(sample(1., 0.))
        self.assertFalse(history.append(sample(1., 1.)))
        history.append(sample(2., 1.))
        for timestamp in (.9, 1.5, 2.1):
            with self.subTest(timestamp=timestamp):
                result = history.align(timestamp, timeout=0)
                self.assertFalse(result['alignment_valid'])
                self.assertIsNone(result['positions_rad'])
        self.assertTrue(history.align(1., timeout=0)['alignment_valid'])
        exact = history.align(2., timeout=0)
        self.assertTrue(exact['alignment_valid'])
        self.assertEqual(exact['positions_rad'], [1.] * 7)
        self.assertEqual(exact['joint_time_offsets_sec'], [0., 0.])

    def test_sdk_feedback_snapshot_is_coherent(self):
        from libgx7.robot.robot import Robot
        robot = Robot.__new__(Robot)
        robot.status_lock = threading.RLock()
        from types import SimpleNamespace
        robot.global_motors_status = SimpleNamespace(positions=list(range(7)))
        robot.last_feedback_at = 12.5
        timestamp, q = robot.get_joint_sample()
        self.assertEqual(timestamp, 12.5)
        np.testing.assert_equal(q, np.arange(7))
        q[:] = 0
        self.assertEqual(robot.global_motors_status.positions, list(range(7)))


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.trajectory = self.root / 'source.json'
        self.trajectory.write_text('{"example": true}\n')
        self.epoch = time.monotonic()

    def make_recorder(self, camera):
        def reader():
            timestamp = time.monotonic()
            return sample(timestamp, (timestamp - self.epoch) * .1)
        recorder = PlaybackRecorder(reader, camera_factory=lambda: camera, sample_hz=200)
        self.addCleanup(lambda: recorder.stop('test_cleanup') if recorder.active else None)
        return recorder

    def test_rgb_manifest_alignment_and_clean_shutdown(self):
        camera = FakeCamera()
        recorder = self.make_recorder(camera)
        output = self.root / 'capture'
        recorder.start(output, self.trajectory, {'mock': True})
        wait_until(lambda: recorder.snapshot()['aligned_frames'] >= 3)
        recorder.stop('completed')
        self.assertTrue(camera.closed)
        self.assertFalse(recorder.frame_thread.is_alive())
        self.assertFalse(recorder.sample_thread.is_alive())
        frames = [json.loads(line) for line in (output / 'frames.jsonl').read_text().splitlines()]
        for frame in frames:
            pixels = np.array(Image.open(output / frame['rgb_path']))
            np.testing.assert_equal(pixels[0, 0], [210, 20, 5])
            if frame['alignment_valid']:
                expected = (frame['timestamp_monotonic'] - self.epoch) * .1
                np.testing.assert_allclose(frame['positions_rad'], [expected] * 7, atol=1e-8)
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual(summary['stop_reason'], 'completed')
        self.assertIsNone(summary['error'])
        self.assertEqual(summary['frames'], len(frames))
        saved_image, saved_record = recorder.latest_recorded()
        self.assertEqual(saved_record, frames[-1])
        self.assertEqual(saved_image.received_at, saved_record['timestamp_monotonic'])
        self.assertEqual(saved_image.sequence, saved_record['camera_sequence'])
        np.testing.assert_array_equal(saved_image.rgb, np.array(Image.open(output / saved_record['rgb_path'])))
        self.assertTrue((output / 'joints.jsonl').stat().st_size)
        self.assertEqual((output / 'source_trajectory.json').read_text(), self.trajectory.read_text())
        metadata = json.loads((output / 'session.json').read_text())
        self.assertTrue(metadata['mock'])
        self.assertTrue(metadata['camera']['synthetic'])

    def test_camera_startup_failure_closes_and_records_error(self):
        camera = FakeCamera(fail_after=0)
        recorder = self.make_recorder(camera)
        with self.assertRaisesRegex(RuntimeError, 'camera disconnected'):
            recorder.start(self.root / 'failed', self.trajectory, {'mock': True})
        self.assertFalse(recorder.active)
        self.assertTrue(camera.closed)
        summary = json.loads((self.root / 'failed' / 'summary.json').read_text())
        self.assertEqual(summary['stop_reason'], 'startup_failed')

    def test_mid_capture_camera_failure_preserves_partial_data(self):
        camera = FakeCamera(fail_after=4)
        recorder = self.make_recorder(camera)
        recorder.start(self.root / 'partial', self.trajectory, {'mock': True})
        wait_until(lambda: recorder.snapshot()['error'])
        recorder.stop('capture_failed')
        self.assertGreater(recorder.frames, 0)
        self.assertEqual(recorder.error, 'camera disconnected')
        self.assertTrue(camera.closed)

    def test_image_write_failure_is_reported(self):
        recorder = self.make_recorder(FakeCamera())
        with patch('PIL.Image.Image.save', side_effect=OSError('disk full')):
            recorder.start(self.root / 'disk', self.trajectory, {'mock': True})
            wait_until(lambda: recorder.snapshot()['error'])
            recorder.stop('capture_failed')
        self.assertEqual(recorder.error, 'disk full')
        self.assertEqual(recorder.frames, 0)
        self.assertIsNone(recorder.latest_recorded())

    def test_existing_output_is_never_overwritten(self):
        recorder = self.make_recorder(FakeCamera())
        with self.assertRaises(FileExistsError):
            recorder.start(self.root, self.trajectory, {})
        self.assertEqual(self.trajectory.read_text(), '{"example": true}\n')


class PlaybackCaptureTests(unittest.TestCase):
    def setUp(self):
        from app import ControlApp
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.app = ControlApp(replace(load_config(ROOT / 'config.yaml'), port=0))
        self.addCleanup(self.app.close)
        self.app.driver.connect()
        source = self.root / 'trajectory.json'
        q = self.app.driver.q.copy()
        save_trajectory(source, [0., .15], [q, q])
        self.app.play_path.value = str(source)
        self.app.capture_path.value = str(self.root / 'capture')

    def test_completed_playback_automatically_saves_capture(self):
        self.app.start_playback()
        def finished():
            self.app.update()
            return not self.app.capture.active
        wait_until(finished)
        self.assertEqual(self.app.driver.mode, 'hold')
        self.assertGreater(self.app.capture.aligned_frames, 0)
        self.assertFalse(self.app.error.value)
        summary = json.loads((self.root / 'capture' / 'summary.json').read_text())
        self.assertEqual(summary['stop_reason'], 'completed')

    def test_capture_failure_stops_motion_and_closes_logs(self):
        with patch('app.SimulatedRGBCamera', return_value=FakeCamera(fail_after=2)):
            self.app.start_playback()
        wait_until(lambda: self.app.capture.snapshot()['error'])
        self.app.update()
        self.assertEqual(self.app.driver.mode, 'hold')
        self.assertIsNone(self.app.driver.playback)
        self.assertFalse(self.app.capture.active)
        self.assertIn('camera disconnected', self.app.error.value)

    def test_stop_button_finalizes_capture_and_holds(self):
        self.app.start_playback()
        self.app.stop_playback()
        self.assertEqual(self.app.driver.mode, 'hold')
        self.assertFalse(self.app.capture.active)
        self.assertTrue((self.root / 'capture' / 'summary.json').is_file())

    def test_startup_failure_prevents_playback(self):
        with patch('app.SimulatedRGBCamera', return_value=FakeCamera(fail_after=0)):
            with self.assertRaisesRegex(RuntimeError, 'camera disconnected'):
                self.app.start_playback()
        self.assertEqual(self.app.driver.mode, 'hold')
        self.assertIsNone(self.app.driver.playback)
        self.assertNotEqual(self.app.capture_path.value, str(self.root / 'capture'))

    def start_manual(self):
        self.app.record_path.value = str(self.root / 'manual.json')
        self.app.start_recording()

    def test_manual_recording_saves_rgb_and_previews_matching_joints(self):
        self.start_manual()
        def ready():
            self.app.update()
            return self.app.capture.aligned_frames >= 3
        wait_until(ready)
        self.assertEqual(self.app.driver.mode, 'recording')
        self.assertTrue(self.app.camera_image.visible)
        self.assertTrue(self.app.recorded_image.visible)
        self.assertIn('对齐有效', self.app.recorded_status.content)
        self.assertIn('无力录制同步采集中', self.app.camera_status.content)
        self.app.save_recording()
        self.app.refresh()
        self.assertEqual(self.app.driver.mode, 'released')
        self.assertFalse(self.app.capture.active)
        self.assertTrue(self.app.recorded_image.visible)
        self.assertTrue((self.root / 'manual.json').is_file())
        self.assertTrue((self.root / 'capture' / 'frames.jsonl').is_file())
        session = json.loads((self.root / 'capture' / 'session.json').read_text())
        self.assertEqual(session['capture_mode'], 'recording')
        self.assertEqual(session['joint_recording_path'], str(self.root / 'manual.json'))

    def test_manual_capture_failure_saves_trajectory_without_enabling_arm(self):
        with patch('app.SimulatedRGBCamera', return_value=FakeCamera(fail_after=2)):
            self.start_manual()
        wait_until(lambda: self.app.capture.snapshot()['error'])
        self.app.update()
        self.assertEqual(self.app.driver.mode, 'released')
        self.assertFalse(self.app.capture.active)
        self.assertTrue((self.root / 'manual.json').is_file())
        summary = json.loads((self.root / 'capture' / 'summary.json').read_text())
        self.assertEqual(summary['stop_reason'], 'capture_failed')

    def test_manual_startup_failure_prevents_releasing_arm(self):
        with patch('app.SimulatedRGBCamera', return_value=FakeCamera(fail_after=0)):
            with self.assertRaisesRegex(RuntimeError, 'camera disconnected'):
                self.start_manual()
        self.assertEqual(self.app.driver.mode, 'hold')
        self.assertFalse(self.app.driver.snapshot()['recording'])

    def test_manual_disconnect_finalizes_images_and_trajectory(self):
        self.start_manual()
        self.app.disconnect()
        self.assertFalse(self.app.capture.active)
        self.assertFalse(self.app.driver.snapshot()['connected'])
        self.assertTrue((self.root / 'manual.json').is_file())
        self.assertTrue((self.root / 'capture' / 'summary.json').is_file())

    def test_joint_only_recording_does_not_open_camera(self):
        self.app.record_capture_enabled.value = False
        with patch('app.SimulatedRGBCamera') as camera:
            self.start_manual()
            self.app.save_recording()
        camera.assert_not_called()
        self.assertIsNone(self.app.capture)


if __name__ == '__main__':
    unittest.main()
