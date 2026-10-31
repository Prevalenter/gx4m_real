"""step1 placement check without camera or GPU."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from gears import GEARS, check_gear_mesh, gear_mesh_path, yaw_symmetry          # noqa: E402
from detect_plate import PlateDetector, fuse_poses, mask_metrics, plate_mask, table_height  # noqa: E402
from step1_setup import (REFERENCE, Step1Setup, median_pose, placement_report, plate_report, pose)  # noqa: E402
from viser_scene import DEFAULT_INITIAL                                          # noqa: E402


def tilted(T, degrees):
    T = T.copy()
    T[:3, :3] = T[:3, :3] @ Rotation.from_euler('x', degrees, degrees=True).as_matrix()
    return T


class FakeBackend:
    """FoundationPose stand-in: a fixed pose and a fixed rendered depth."""

    def __init__(self, rendered):
        self.rendered, self.pose = rendered, np.eye(4)
        self.pose[2, 3] = .6
        self.registers = 0

    def register(self, frame, mask):
        self.registers += 1
        return self.pose

    def render(self, frame, pose):
        return self.rendered


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


class PlateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = json.loads(REFERENCE.read_text(encoding='utf-8'))
        cls.trained = pose(cls.ref['fixture_pos'], cls.ref['fixture_quat'])

    def scene(self):
        """Camera 0.6 m above the table looking straight down at a plate with a red gear on it."""
        K = np.array([[500., 0, 320], [0, 500, 240], [0, 0, 1]])
        T_world_cam = np.eye(4)
        T_world_cam[:3, :3] = Rotation.from_euler('x', 180, degrees=True).as_matrix()
        table = table_height(self.ref)
        T_world_cam[:3, 3] = [0, 0, table + .6]
        depth = np.full((480, 640), .6, dtype=np.float32)
        depth[140:340, 220:420] = .6 - .012       # plate top, 12 mm above the table
        rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
        rgb[140:340, 220:420] = 25                # black plate on a grey table
        return rgb, depth, K, T_world_cam, table

    def test_mask_finds_plate_above_table(self):
        rgb, depth, K, T, table = self.scene()
        mask, ignore = plate_mask(rgb, depth, K, T, table)
        self.assertEqual(mask.sum(), 200*200)
        self.assertTrue(mask[240, 320] and not mask[50, 50] and not ignore.any())

    def test_mask_ignores_gear_and_hand_height(self):
        rgb, depth, K, T, table = self.scene()
        depth[100:130, 100:130] = .4              # something far above the table, such as the hand
        depth[0:5, 0:5] = 0                       # depth hole
        rgb[200:280, 280:360] = (220, 30, 30)     # red gear on the plate
        depth[200:280, 280:360] = .6 - .05
        mask, ignore = plate_mask(rgb, depth, K, T, table)
        self.assertFalse(mask[240, 320] or mask[115, 115])
        self.assertTrue(mask[150, 320] and ignore[240, 320] and ignore[2, 2])

    def test_mask_rejects_non_black_object_at_plate_height(self):
        rgb, depth, K, T, table = self.scene()
        depth[10:400, 10:200] = .6 - .012         # larger than the plate, same height, but grey
        mask, _ = plate_mask(rgb, depth, K, T, table)
        self.assertTrue(mask[240, 320] and not mask[100, 100])
        self.assertEqual(mask.sum(), 200*200)

    def test_mask_keeps_depth_holes_and_highlights_on_plate(self):
        rgb, depth, K, T, table = self.scene()
        depth[140:340, 300:340] = 0               # a hole across the whole plate must not split it
        rgb[180:182, 240:260] = 200               # thin highlight
        mask, ignore = plate_mask(rgb, depth, K, T, table)
        self.assertEqual(mask.sum(), 200*200)
        self.assertTrue(ignore[240, 320] and not ignore[240, 250])

    def test_mask_without_plate_or_depth_raises(self):
        rgb, depth, K, T, table = self.scene()
        flat = np.full_like(depth, .6)
        with self.assertRaises(ValueError):
            plate_mask(rgb, flat, K, T, table)        # black on the table, nothing above it
        sparse = depth.copy()
        sparse[140:340, 220:420] = 0
        sparse[140:150, 220:230] = .6 - .012
        with self.assertRaises(ValueError):
            plate_mask(rgb, sparse, K, T, table)      # black but almost no depth

    def test_metrics_do_not_penalize_ignored_pixels(self):
        silhouette = np.zeros((100, 100), dtype=bool)
        silhouette[20:80, 20:80] = True
        ignore = np.zeros_like(silhouette)
        ignore[40:60, 40:60] = True               # gear on the plate
        mask = silhouette & ~ignore
        depth = np.full(silhouette.shape, .5)
        iou, error = mask_metrics(silhouette, mask, ignore, np.where(silhouette, .502, 0), depth)
        self.assertEqual(iou, 1.)
        self.assertAlmostEqual(error, .002)
        self.assertLess(mask_metrics(silhouette, mask, np.zeros_like(ignore), np.where(silhouette, .502, 0), depth)[0], 1.)

    def test_fuse_rejects_flipped_frame(self):
        T = self.trained.copy()
        poses = [T.copy() for _ in range(4)]
        poses[1][:3, 3] += [.002, 0, 0]
        fused, spread = fuse_poses(poses)
        np.testing.assert_allclose(fused, T, atol=1e-9)
        self.assertAlmostEqual(spread['position_m'], .002)
        flipped = T.copy()
        flipped[:3, :3] = Rotation.from_euler('z', 180, degrees=True).as_matrix()
        with self.assertRaises(ValueError):
            fuse_poses([T, T, T, flipped])

    def test_report_is_measured_at_the_left_pin(self):
        self.assertTrue(plate_report(self.trained, self.ref)['ok'])
        moved = self.trained.copy()
        moved[:3, 3] += [.002, -.001, .001]
        r = plate_report(moved, self.ref)
        self.assertTrue(r['ok'])
        self.assertAlmostEqual(r['dx'], .002)
        moved[:3, 3] += [.003, 0, 0]
        self.assertFalse(plate_report(moved, self.ref)['ok'])
        # A yaw about the pin keeps the pin in place and only reports the yaw.
        pin = np.asarray(self.ref['final_pos'])
        spin = np.eye(4)
        spin[:3, :3] = Rotation.from_euler('z', 10, degrees=True).as_matrix()
        spin[:3, 3] = pin - spin[:3, :3] @ pin
        r = plate_report(spin @ self.trained, self.ref)
        self.assertTrue(r['ok'])
        self.assertAlmostEqual(r['yaw'], 10)
        self.assertAlmostEqual(r['xy'], 0)
        self.assertFalse(plate_report(tilted(self.trained, 5), self.ref)['ok'])

    def detector_scene(self, **kwargs):
        rgb, depth, K, T, _ = self.scene()
        backend = FakeBackend(np.where(depth < .6, .588, 0).astype(np.float32))
        detector = PlateDetector(backend, self.ref, T, fuse_frames=3, **kwargs)
        return detector, backend, SimpleNamespace(rgb=rgb, depth=depth, K=K, sequence=0)

    def test_detector_idle_until_started_then_measures_once(self):
        detector, backend, frame = self.detector_scene(every=2)
        for _ in range(5):
            self.assertIsNone(detector.update(frame))
        self.assertEqual(backend.registers, 0)
        detector.start()
        self.assertEqual(detector.update(frame)['status'], 'MEASURING')
        self.assertIsNone(detector.update(frame))          # the gear loop runs between registrations
        detector.update(frame)
        detector.update(frame)
        self.assertEqual(detector.update(frame)['status'], 'DONE')
        self.assertEqual(backend.registers, 3)
        for _ in range(5):
            self.assertIsNone(detector.update(frame))      # stays put until the next press
        self.assertEqual(backend.registers, 3)
        np.testing.assert_allclose(detector.T_camera_plate, backend.pose)

    def test_detector_ignores_occluders(self):
        detector, backend, frame = self.detector_scene(every=1)
        frame.depth = frame.depth.copy()
        frame.depth[140:340, 220:300] = .3                 # the hand over part of the plate
        detector.start()
        for _ in range(3):
            state = detector.update(frame)
        self.assertEqual(state['status'], 'DONE')

    def test_detector_failure_keeps_previous_pose(self):
        detector, backend, frame = self.detector_scene(every=1, max_attempts=4)
        detector.start()
        for _ in range(3):
            detector.update(frame)
        previous = detector.T_camera_plate.copy()
        backend.rendered = backend.rendered - .05*(backend.rendered > 0)     # depth does not match
        detector.start()
        for _ in range(4):
            state = detector.update(frame)
        self.assertEqual(state['status'], 'FAILED')
        self.assertFalse(detector.active)
        self.assertIn('depth error', state['reason'])
        np.testing.assert_allclose(state['T_camera_plate'], previous)

    def test_detector_restarts_on_disagreeing_frames(self):
        detector, backend, frame = self.detector_scene(every=1)
        detector.start()
        detector.update(frame)
        detector.update(frame)
        backend.pose = backend.pose.copy()
        backend.pose[0, 3] += .02
        self.assertEqual(detector.update(frame)['status'], 'MEASURING')
        self.assertEqual(len(detector.window), 1)
        detector.update(frame)
        self.assertEqual(detector.update(frame)['status'], 'DONE')
        np.testing.assert_allclose(detector.T_camera_plate, backend.pose)

    def test_viewer_shows_saved_and_measured_plate(self):
        initial = json.loads(DEFAULT_INITIAL.read_text())
        C = np.eye(4)
        C[:3, :3] = Rotation.from_euler('x', 180, degrees=True).as_matrix()
        C[:3, 3] = [.0, -.25, .6]
        calibration = {'T_world_base': np.eye(4), 'T_base_cam': C, 'synthetic': True, 'base_link': 'base_link'}
        T_cam = np.linalg.inv(C) @ self.trained
        viewer = Step1Setup(calibration, reference=self.ref, plate=dict(T_camera_plate=T_cam.tolist(), measured_at='now'),
                            initial=initial, port=0, live=True)
        try:
            self.assertTrue(viewer.plate_report['ok'] and viewer.detected_plate.visible)
            state = dict(status='MEASURING', reason='', fused=3, fuse_frames=5, T_camera_plate=T_cam, spread=None)
            viewer.update_plate(state)
            self.assertIn('检测中 3/5', viewer.plate_status.content)
            viewer.update_plate({**state, 'status': 'FAILED', 'reason': 'x'})
            self.assertIn('上次结果', viewer.plate_status.content)
            viewer.update_plate({**state, 'status': 'DONE'})
            self.assertEqual(viewer.plate_status.content, '')
            self.assertIn('检测）', viewer.plate_check.content)
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
            'import sys; import step1_setup; import detect_plate; '
            'assert "torch" not in sys.modules; '
            'assert "pyorbbecsdk" not in sys.modules; '
            'assert "estimater" not in sys.modules'], cwd=HERE, check=True)


if __name__ == '__main__':
    unittest.main()
