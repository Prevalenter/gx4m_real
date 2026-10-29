"""Fixed-camera scene checks without camera access or GPU inference."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from viser_scene import (DEFAULT_INITIAL, SceneViewer, initial_base, load_calibration,
                         read_detection, rigid, world_object)


def make_transform(position, angle):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('z', angle, degrees=True).as_matrix()
    T[:3, 3] = position
    return T


class SceneTests(unittest.TestCase):
    def setUp(self):
        self.initial = json.loads(DEFAULT_INITIAL.read_text())
        self.calibration = {'T_world_base': make_transform([.1, -.2, .3], 90),
                            'T_base_cam': make_transform([.4, .5, .6], -30),
                            'base_link': 'base_link', 'synthetic': True}

    def test_transform_order_and_mesh_origin(self):
        pose = make_transform([.02, .03, .5], 15)
        expected = self.calibration['T_world_base'] @ self.calibration['T_base_cam'] @ pose
        np.testing.assert_allclose(world_object(self.calibration, pose), expected)
        point = np.array([.01, .02, .03, 1])
        np.testing.assert_allclose(world_object(self.calibration, pose) @ point, expected @ point)

    def test_invalid_transform_is_rejected(self):
        for bad in (np.zeros((4, 4)), np.full((4, 4), np.nan), np.diag([2, 1, 1, 1])):
            with self.assertRaises(ValueError):
                rigid(bad)

    def test_calibration_loading_and_wrist_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'handeye.json'
            path.write_text(json.dumps({'mode': 'eye-to-hand',
                                        'T_base_cam': self.calibration['T_base_cam'].tolist()}))
            loaded = load_calibration(path, self.initial)
            np.testing.assert_allclose(loaded['T_world_base'], initial_base(self.initial))
            path.write_text(json.dumps({'mode': 'eye-in-hand'}))
            with self.assertRaisesRegex(ValueError, 'fixed-camera'):
                load_calibration(path, self.initial)

    def test_existing_detection_reads_rgb_and_intrinsics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'result.json').write_text(json.dumps({'object_in_camera': np.eye(4).tolist()}))
            rgb = np.full((20, 30, 3), [12, 40, 150], dtype=np.uint8)
            Image.fromarray(rgb).save(path / 'rgb.png')
            K = np.array([[100., 0, 15], [0, 100, 10], [0, 0, 1]])
            np.savez(path / 'rgbd.npz', K=K)
            record, image, intrinsics = read_detection(path)
            np.testing.assert_equal(image, rgb)
            np.testing.assert_equal(intrinsics, K)
            self.assertIn('object_in_camera', record)

    def test_lost_invalid_and_expired_poses_are_hidden(self):
        viewer = SceneViewer(self.calibration, initial=self.initial, port=0, live=True, stale_after=.5)
        try:
            pose = make_transform([.01, .02, .5], 10)
            record = {'status': 'TRACK', 'object_in_camera': pose.tolist()}
            viewer.update(record)
            self.assertTrue(viewer.part.visible)
            np.testing.assert_allclose(viewer.part.position, world_object(self.calibration, pose)[:3, 3])
            viewer.update({'status': 'SEARCHING', 'object_in_camera': None})
            self.assertFalse(viewer.part.visible)
            viewer.update(record)
            viewer.expire(time.monotonic() + 2)
            self.assertFalse(viewer.part.visible)
            self.assertIn('过期', viewer.status.value)
            viewer.update({'object_in_camera': np.zeros((4, 4))})
            self.assertFalse(viewer.part.visible)
            self.assertIn('无效', viewer.status.value)
        finally:
            viewer.close()

    def test_camera_serial_mismatch_rejected(self):
        self.calibration['camera_metadata'] = {'serial': 'expected'}
        viewer = SceneViewer(self.calibration, initial=self.initial, port=0)
        try:
            with self.assertRaisesRegex(ValueError, 'serial'):
                viewer.set_camera_metadata({'serial': 'other'})
        finally:
            viewer.close()

    def test_import_does_not_load_inference_or_camera(self):
        subprocess.run([sys.executable, '-c',
            'import sys; import viser_scene; '
            'assert "torch" not in sys.modules; '
            'assert "pyorbbecsdk" not in sys.modules; '
            'assert "estimater" not in sys.modules'], cwd=HERE, check=True)


if __name__ == '__main__':
    unittest.main()
