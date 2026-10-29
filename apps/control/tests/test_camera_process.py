"""Camera subprocess lifecycle and frame transfer without hardware."""
import os
import time
import unittest

import numpy as np

from gx4m_real.camera.frames import ColorFrame
from gx4m_real.camera.process import ProcessColorCamera


class TestCamera:
    def __init__(self, device_index=0, fail=None):
        self.fail = fail
        self.metadata = {'pid': os.getpid(), 'device_index': device_index, 'synthetic': True}

    def open(self):
        if self.fail == 'open':
            raise RuntimeError('camera unavailable')
        if self.fail == 'hang':
            time.sleep(30)

    def read(self, timeout=2.):
        if self.fail == 'read':
            raise TimeoutError('no camera frame')
        return ColorFrame(1, np.full((3, 4, 3), 42, dtype=np.uint8), time.monotonic(), 123, 456)

    def close(self):
        pass


class CameraProcessTests(unittest.TestCase):
    def make_camera(self, **options):
        camera = ProcessColorCamera(2, camera_factory=TestCamera, **options)
        self.addCleanup(camera.close)
        return camera

    def test_separate_process_transfers_frame_and_can_reopen(self):
        camera = self.make_camera()
        camera.open()
        self.assertNotEqual(camera.metadata['pid'], os.getpid())
        self.assertEqual(camera.metadata['device_index'], 2)
        started = time.monotonic()
        frame = camera.read()
        self.assertGreaterEqual(frame.received_at, started)
        self.assertLessEqual(frame.received_at, time.monotonic())
        self.assertEqual(frame.device_timestamp_us, 123)
        self.assertEqual(frame.system_timestamp_us, 456)
        np.testing.assert_array_equal(frame.rgb, np.full((3, 4, 3), 42, dtype=np.uint8))
        camera.close()
        self.assertIsNone(camera.process)
        self.assertIsNone(camera.connection)
        camera.open()
        self.assertEqual(camera.read().sequence, 1)

    def test_startup_error_cleans_up(self):
        camera = self.make_camera(fail='open')
        with self.assertRaisesRegex(RuntimeError, 'camera unavailable'):
            camera.open()
        self.assertIsNone(camera.process)
        self.assertIsNone(camera.connection)

    def test_read_error_cleans_up(self):
        camera = self.make_camera(fail='read')
        camera.open()
        with self.assertRaisesRegex(TimeoutError, 'no camera frame'):
            camera.read()
        self.assertIsNone(camera.process)
        self.assertIsNone(camera.connection)

    def test_startup_timeout_terminates_worker(self):
        camera = self.make_camera(fail='hang', startup_timeout=.2)
        with self.assertRaisesRegex(TimeoutError, 'response timed out'):
            camera.open()
        self.assertIsNone(camera.process)
        self.assertIsNone(camera.connection)


if __name__ == '__main__':
    unittest.main()
