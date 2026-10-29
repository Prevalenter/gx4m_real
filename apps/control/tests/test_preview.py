"""Camera preview and recording ownership checks using synthetic frames."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import ControlApp
from gx4m_real.camera.frames import ColorFrame
from gx4m_real.camera.preview import CameraPreview
from gx4m_real.robot.config import load_config
from gx4m_real.robot.trajectory import save_trajectory


def wait_until(predicate):
    deadline = time.monotonic() + 3.
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    raise AssertionError('Camera preview did not become ready')


class FakePreviewCamera:
    def __init__(self, fail=False):
        self.metadata = {'synthetic': True, 'device': 'test camera'}
        self.sequence = 0
        self.fail = fail
        self.closed = False

    def open(self):
        pass

    def read(self, timeout=1.):
        if self.fail:
            raise RuntimeError('camera disconnected')
        time.sleep(.01)
        self.sequence += 1
        return ColorFrame(self.sequence, np.full((3, 4, 3), [200, 30, 10], dtype=np.uint8),
                          time.monotonic())

    def close(self):
        self.closed = True


class PreviewWorkerTests(unittest.TestCase):
    def test_latest_rgb_and_close(self):
        camera = FakePreviewCamera()
        preview = CameraPreview(lambda: camera)
        self.addCleanup(preview.stop)
        preview.start()
        preview.start()
        wait_until(lambda: preview.snapshot()['frame'] is not None)
        snapshot = preview.snapshot()
        self.assertTrue(snapshot['running'])
        np.testing.assert_array_equal(snapshot['frame'].rgb[0, 0], [200, 30, 10])
        self.assertTrue(snapshot['metadata']['synthetic'])
        preview.stop()
        self.assertTrue(camera.closed)
        self.assertFalse(preview.snapshot()['running'])

    def test_camera_failure_is_reported_and_closed(self):
        camera = FakePreviewCamera(fail=True)
        preview = CameraPreview(lambda: camera)
        self.addCleanup(preview.stop)
        preview.start()
        wait_until(lambda: not preview.snapshot()['running'])
        self.assertEqual(preview.snapshot()['error'], 'camera disconnected')
        self.assertTrue(camera.closed)

    def test_restart_clears_old_errors(self):
        cameras = iter([FakePreviewCamera(fail=True), FakePreviewCamera()])
        preview = CameraPreview(lambda: next(cameras))
        self.addCleanup(preview.stop)
        preview.start()
        wait_until(lambda: not preview.snapshot()['running'])
        preview.start()
        wait_until(lambda: preview.snapshot()['frame'] is not None)
        self.assertIsNone(preview.snapshot()['error'])


class PreviewUITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = replace(load_config(ROOT / 'config.yaml'), port=0)
        self.app = ControlApp(self.config)
        self.addCleanup(self.app.close)

    def test_real_preview_is_default_even_with_simulated_robot(self):
        self.assertTrue(self.app.driver.mock)
        self.assertEqual(self.app.camera_source.value, '真实相机')
        with patch('app.ProcessColorCamera', return_value='real camera') as factory:
            self.assertEqual(self.app.make_preview_camera(), 'real camera')
        factory.assert_called_once_with(0)

    def test_preview_does_not_connect_robot(self):
        self.app.camera_source.value = '模拟画面'
        with patch('app.SimulatedRGBCamera', side_effect=FakePreviewCamera):
            self.app.start_camera_preview()
            wait_until(lambda: self.app.camera_preview.snapshot()['frame'] is not None)
        self.app.refresh()
        self.assertTrue(self.app.camera_image.visible)
        np.testing.assert_array_equal(self.app.camera_image.image[0, 0], [200, 30, 10])
        self.assertIn('模拟画面', self.app.camera_status.content)
        self.assertFalse(self.app.driver.snapshot()['connected'])
        self.app.camera_preview.stop()
        self.app.refresh()
        self.assertFalse(self.app.camera_image.visible)

    def test_preview_error_is_separate_from_robot_state(self):
        self.app.camera_source.value = '模拟画面'
        with patch('app.SimulatedRGBCamera', return_value=FakePreviewCamera(fail=True)):
            self.app.start_camera_preview()
            wait_until(lambda: not self.app.camera_preview.snapshot()['running'])
        self.app.refresh()
        self.assertIn('camera disconnected', self.app.camera_status.content)
        self.assertFalse(self.app.camera_image.visible)
        self.assertEqual(self.app.driver.snapshot()['mode'], 'disconnected')

    def test_playback_reuses_recording_camera_after_releasing_preview(self):
        self.app.camera_source.value = '模拟画面'
        active = []
        cameras = []

        class ExclusiveCamera(FakePreviewCamera):
            def open(self):
                if active:
                    raise RuntimeError('Camera already in use')
                active.append(self)

            def close(self):
                if self in active:
                    active.remove(self)
                super().close()

        def factory():
            camera = ExclusiveCamera()
            cameras.append(camera)
            return camera

        self.app.connect_driver()
        trajectory = Path(self.temp.name) / 'trajectory.json'
        positions = self.app.driver.q.copy()
        save_trajectory(trajectory, [0., 1.], [positions, positions])
        self.app.play_path.value = str(trajectory)
        self.app.capture_path.value = str(Path(self.temp.name) / 'capture')
        with patch('app.SimulatedRGBCamera', side_effect=factory):
            self.app.start_camera_preview()
            wait_until(lambda: self.app.camera_preview.snapshot()['frame'] is not None)
            self.app.start_playback()
        self.assertTrue(cameras[0].closed)
        self.assertFalse(self.app.camera_preview.snapshot()['running'])
        self.assertEqual(active, [cameras[1]])
        self.app.refresh()
        self.assertTrue(self.app.camera_image.visible)
        self.assertIn('回放同步采集中', self.app.camera_status.content)
        self.assertTrue(self.app.camera_start_button.disabled)
        with self.assertRaisesRegex(RuntimeError, '复用'):
            self.app.start_camera_preview()
        self.app.stop_playback()
        self.assertEqual(active, [])


if __name__ == '__main__':
    unittest.main()
