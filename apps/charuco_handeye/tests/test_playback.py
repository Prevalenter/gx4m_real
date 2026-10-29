"""Synthetic offline tests. No cameras, robot connections or real calibration data."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from board import make_board
from playback_data import DEFAULT_URDF, JOINTS, camera_parameters, load_session
from robot import UrdfChainFK
from solve_handeye import inv, rot_deg
from solve_session import calibrate_session, check_motion
from view_result import ResultViewer, poses_for_sample


def synthetic_session(root, mode, count=14):
    root.mkdir()
    (root / 'rgb').mkdir()
    rng = np.random.default_rng(41)
    fk = UrdfChainFK(str(DEFAULT_URDF), 'base_link', 'Link7')
    pose = json.loads((HERE.parents[1] / 'tasks/dexlong_gear3/config/initial_pose.json').read_text())
    q0 = np.array([pose['joint_positions'][name] for name in JOINTS])
    E0 = fk.fk(q0.tolist())
    K = np.array([[1000., 0, 640], [0, 1000., 480], [0, 0, 1]])
    if mode == 'eye-to-hand':
        C0 = np.eye(4)
        C0[:3, 3] = E0[:3, 3] + [.65, -.4, .4]
        z = E0[:3, 3] - C0[:3, 3]
        z /= np.linalg.norm(z)
        x = np.cross(z, [0, 0, 1])
        x /= np.linalg.norm(x)
        C0[:3, :3] = np.column_stack([x, np.cross(z, x), z])
        B0 = C0.copy()
        B0[:3, 3] = E0[:3, 3] - C0[:3, :3] @ [.09, .1125, 0]
        U, V = C0, inv(E0) @ B0
    else:
        U = np.eye(4)
        U[:3, 3] = [.01, .02, .06]
        C0 = E0 @ U
        V = C0.copy()
        V[:3, 3] = C0[:3, 3] + C0[:3, :3] @ [-.09, -.1125, .7]
    board = make_board()
    image = board.generateImage((720, 900))
    outer = np.array([[0, 0, 0], [.18, 0, 0], [.18, .225, 0], [0, .225, 0]], dtype=float)
    rows = []
    for attempt in range(1000):
        q = q0 + rng.uniform(-1, 1, 7) * np.deg2rad([25, 15, 15, 20, 30, 20, 30])
        E = fk.fk(q.tolist())
        C, B = (U, E @ V) if mode == 'eye-to-hand' else (E @ U, V)
        T = inv(C) @ B
        points = outer @ T[:3, :3].T + T[:3, 3]
        if points[:, 2].min() < .25 or T[2, 2] < .45:
            continue
        uv = (points @ K.T)
        uv = uv[:, :2] / uv[:, 2:]
        if (uv < 15).any() or (uv[:, 0] > 1265).any() or (uv[:, 1] > 945).any():
            continue
        H = K @ np.column_stack([T[:3, 0], T[:3, 1], T[:3, 3]]) @ np.diag([.18/720, .225/900, 1])
        frame = cv2.warpPerspective(image, H, (1280, 960), borderValue=160)
        filename = f'rgb/{len(rows):06d}.png'
        cv2.imwrite(str(root / filename), frame)
        rows.append({'frame_index': len(rows), 'rgb_path': filename, 'alignment_valid': True,
                     'positions_rad': q.tolist(), 'timestamp_monotonic': 10 + len(rows),
                     'playback_phase': 'trajectory'})
        if len(rows) >= count:
            break
    if len(rows) < count:
        raise AssertionError('Could not generate enough visible synthetic board poses')
    # Nonzero world translation and rotation catch implicit iiwa/world offsets.
    metadata = {'version': 1, 'mock': True, 'joint_names': JOINTS, 'joint_units': 'rad',
                'robot_urdf': str(DEFAULT_URDF), 'base_position': [.12, -.58, .72],
                'base_wxyz': [.70710678, 0, 0, -.70710678],
                'fixed_model_joints': pose['joint_positions'],
                'camera': {'K': K.tolist(), 'distortion': [0.] * 8, 'undistorted': False,
                           'resolution': [1280, 960], 'synthetic': True}}
    (root / 'session.json').write_text(json.dumps(metadata))
    (root / 'frames.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return U, V


class PlaybackTests(unittest.TestCase):
    def test_fk_matches_control_urdf(self):
        import yourdfpy
        model = yourdfpy.URDF.load(str(DEFAULT_URDF), load_meshes=False)
        fk = UrdfChainFK(str(DEFAULT_URDF), 'base_link', 'Link7')
        q = [.2, -.3, .1, -1.5, .3, .4, -.2]
        model.update_cfg(dict(zip(JOINTS, q)))
        np.testing.assert_allclose(fk.fk(q), model.get_transform('Link7', 'base_link'), atol=1e-10)

    def test_raw_distortion_order_and_rectified_camera(self):
        camera = {'K': [[500, 0, 320], [0, 500, 240], [0, 0, 1]], 'resolution': [640, 480],
                  'distortion': dict(zip(['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6'], range(8)))}
        _, dist, size = camera_parameters(camera)
        np.testing.assert_equal(dist, np.arange(8))
        self.assertEqual(size, (640, 480))
        _, dist, _ = camera_parameters({**camera, 'undistorted': True})
        np.testing.assert_equal(dist, np.zeros(8))

    def test_degenerate_motion_rejected(self):
        poses = []
        for i in range(8):
            T = np.eye(4)
            T[:3, :3] = Rotation.from_euler('z', i * 10, degrees=True).as_matrix()
            poses.append(T)
        with self.assertRaisesRegex(ValueError, 'diversity'):
            check_motion(poses)

    def test_both_mounting_modes_from_recorded_images(self):
        for mode in ('eye-to-hand', 'eye-in-hand'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                U, V = synthetic_session(root / 'session', mode)
                with self.assertRaisesRegex(ValueError, 'Synthetic'):
                    load_session(root / 'session')
                result = calibrate_session(root / 'session', root / 'result', mode=mode,
                                           min_interval=0, min_rotation_deg=0, min_translation=0,
                                           reject=0, allow_synthetic=True)
                self.assertGreaterEqual(len(result['samples']), 10)
                key1, key2 = (('T_base_cam', 'T_flange_board') if mode == 'eye-to-hand'
                              else ('T_flange_cam', 'T_base_board'))
                recovered = np.array(result[key1])
                self.assertLess(np.linalg.norm(recovered[:3, 3] - U[:3, 3]), .01)
                self.assertLess(rot_deg(recovered[:3, :3].T @ U[:3, :3]), 1.)
                self.assertLess(np.linalg.norm(np.array(result[key2])[:3, 3] - V[:3, 3]), .01)
                self.assertLess(result['residual_summary']['reproj_px']['rms'], 1.)
                sample = result['samples'][0]
                poses = poses_for_sample(result, sample)
                np.testing.assert_allclose(poses['camera'] @ np.array(sample['T_cam_board']),
                                           poses['observed_board'])
                viewer = ResultViewer(root / 'result' / 'handeye_result.json', port=0)
                try:
                    viewer.index.value = len(result['samples']) - 1
                    viewer.refresh()
                    self.assertIn('相机世界位置', viewer.details.content)
                    self.assertEqual(viewer.image.image.shape[:2], (960, 1280))
                finally:
                    viewer.server.stop()
                if mode == 'eye-in-hand':
                    self.assertNotIn('T_base_cam', result)
                    self.assertFalse((root / 'result' / 'T_RC.npy').exists())

    def test_invalid_alignment_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synthetic_session(root / 'session', 'eye-to-hand', count=2)
            rows = [json.loads(line) for line in (root / 'session/frames.jsonl').read_text().splitlines()]
            rows[0].update(alignment_valid=False, positions_rad=None)
            (root / 'session/frames.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            loaded = load_session(root / 'session', allow_synthetic=True)
            self.assertNotIn('T_base_flange', loaded['frames'][0])
            with self.assertRaisesRegex(ValueError, 'valid views'):
                calibrate_session(root / 'session', root / 'result', mode='eye-to-hand', allow_synthetic=True)
            report = json.loads((root / 'result/detection_report.json').read_text())
            self.assertEqual(report[0]['reason'], 'invalid_alignment')


if __name__ == '__main__':
    unittest.main()
