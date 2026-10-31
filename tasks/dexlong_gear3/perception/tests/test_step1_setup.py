"""step1 placement check without camera or GPU."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from gears import GEARS, check_gear_mesh, gear_mesh_path, yaw_symmetry          # noqa: E402
from step1_setup import (REFERENCE, Step1Setup, median_pose, placement_report, pose)  # noqa: E402
from viser_scene import DEFAULT_INITIAL                                          # noqa: E402


def tilted(T, degrees):
    T = T.copy()
    T[:3, :3] = T[:3, :3] @ Rotation.from_euler('x', degrees, degrees=True).as_matrix()
    return T


class PlacementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = json.loads(REFERENCE.read_text(encoding='utf-8'))
        cls.final = pose(cls.ref['final_pos'], cls.ref['final_quat'])
        cls.start = pose(cls.ref['object_pos'], cls.ref['object_quat'])

    def report(self, T):
        return placement_report(T[:3, 3], T[:3, 2], self.ref)

    def test_gear_on_pin(self):
        T = self.final.copy()
        T[:3, 3] += [.002, -.001, .001]
        r = self.report(T)
        self.assertTrue(r['on_pin'] and r['pin']['ok'])
        self.assertAlmostEqual(r['pin']['dx'], .002)
        T[:3, 3] += [.003, 0, 0]
        self.assertFalse(self.report(T)['pin']['ok'])
        self.assertFalse(self.report(tilted(self.final, 5))['pin']['ok'])

    def test_gear_at_start(self):
        T = self.start.copy()
        T[:3, 3] += [.015, -.019, 0]
        T[:3, :3] = Rotation.from_euler('z', 77, degrees=True).as_matrix()     # any yaw
        r = self.report(T)
        self.assertFalse(r['on_pin'])
        self.assertTrue(r['start']['ok'])
        T[:3, 3] += [.01, 0, 0]
        self.assertFalse(self.report(T)['start']['ok'])
        lifted = self.start.copy()
        lifted[:3, 3] += [0, 0, .01]
        self.assertFalse(self.report(lifted)['start']['ok'])

    def test_median_pose_rejects_one_outlier(self):
        poses = [self.final.copy() for _ in range(5)]
        poses[2][:3, 3] += [.05, 0, 0]
        position, axis = median_pose(poses)
        np.testing.assert_allclose(position, self.final[:3, 3])
        np.testing.assert_allclose(axis, self.final[:3, 2])

    def test_viewer_checks_live_poses(self):
        initial = json.loads(DEFAULT_INITIAL.read_text())
        C = np.eye(4)
        C[:3, :3] = Rotation.from_euler('x', 180, degrees=True).as_matrix()   # looking down
        C[:3, 3] = [.0, -.25, .6]
        calibration = {'T_world_base': np.eye(4), 'T_base_cam': C, 'synthetic': True, 'base_link': 'base_link'}
        viewer = Step1Setup(calibration, reference=self.ref, initial=initial, port=0, live=True)
        try:
            T_cam = np.linalg.inv(C) @ self.final
            for _ in range(3):
                viewer.update({'status': 'TRACK', 'object_in_camera': T_cam.tolist()})
            self.assertTrue(viewer.report['pin']['ok'])
            self.assertIn('✅', viewer.check.content)
            viewer.update({'status': 'LOST - reacquiring', 'object_in_camera': None})
            self.assertIsNone(viewer.report)
            self.assertEqual(len(viewer.history), 0)
        finally:
            viewer.close()


class GearTests(unittest.TestCase):
    def test_meshes_and_symmetry(self):
        import trimesh
        for gear, (teeth, _) in GEARS.items():
            check_gear_mesh(trimesh.load(gear_mesh_path(gear), force='mesh'), gear)
            symmetry = yaw_symmetry(gear)
            self.assertEqual(symmetry.shape, (teeth, 4, 4))
            np.testing.assert_allclose(symmetry[1, :2, :2],
                                       Rotation.from_euler('z', 360/teeth, degrees=True).as_matrix()[:2, :2])
        with self.assertRaises(ValueError):
            check_gear_mesh(trimesh.load(gear_mesh_path('gear36'), force='mesh'), 'gear24')

    def test_color_mask_picks_each_gear(self):
        import cv2
        from detect_gear36 import color_gear_mask
        from gears import hsv_ranges
        # Grey table with a red (both hue ends), a blue and a yellow gear; each mask finds its own disc.
        rgb = np.full((120, 360, 3), 128, dtype=np.uint8)
        hsv = np.zeros((1, 4, 3), dtype=np.uint8)
        hsv[0] = [(2, 200, 180), (176, 200, 180), (115, 200, 180), (27, 200, 200)]
        red_low, red_high, blue, yellow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0].tolist()
        cv2.circle(rgb, (60, 60), 30, red_low, -1)
        cv2.rectangle(rgb, (55, 30), (65, 90), red_high, -1)      # the other end of the red hue range
        cv2.circle(rgb, (180, 60), 30, blue, -1)
        cv2.circle(rgb, (300, 60), 30, yellow, -1)
        depth = np.ones(rgb.shape[:2], dtype=np.float32)
        for gear, x in (('gear24', 60), ('gear36', 180), ('gear48', 300)):
            mask = color_gear_mask(rgb, depth, hsv_ranges(gear))
            ys, xs = np.nonzero(mask)
            self.assertAlmostEqual(xs.mean(), x, delta=2, msg=gear)
            self.assertGreater(mask.sum(), 2500)
        with self.assertRaises(ValueError):
            color_gear_mask(np.full_like(rgb, 128), depth, hsv_ranges('gear24'))
        self.assertEqual(hsv_ranges('gear24', [1, 2, 3], [4, 5, 6]), (((1, 2, 3), (4, 5, 6)),))

    def test_import_does_not_load_inference_or_camera(self):
        subprocess.run([sys.executable, '-c',
            'import sys; import step1_setup; '
            'assert "torch" not in sys.modules; '
            'assert "pyorbbecsdk" not in sys.modules; '
            'assert "estimater" not in sys.modules'], cwd=HERE, check=True)


if __name__ == '__main__':
    unittest.main()
