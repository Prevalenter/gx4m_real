"""Synthetic tests: mock arm + rendered ChArUco camera → collector → session → solver. No hardware."""
import dataclasses
import json
from pathlib import Path
import socket
import sys
import tempfile
import time
import unittest

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from board import make_board
from gx4m_real.camera.frames import ColorFrame
from gx4m_real.robot.compensation import JointCompensation
from gx4m_real.robot.config import load_config
from handeye_session import DEFAULT_URDF, JOINTS, SessionWriter, camera_parameters, load_session
from robot import UrdfChainFK
from solve_handeye import inv, rot_deg
from solve_session import calibrate_session, check_motion
from view_result import ResultViewer, poses_for_sample

CONTROL_CONFIG = ROOT / 'apps/control/config.yaml'
# Same directions as the real arm, plus offsets, so FK on raw motor angles would fail the tests.
COMPENSATION = JointCompensation((-1, 1, -1, 1, -1, -1, -1), (0., 1.5, -2., 3., -2., -6.5, 0.))
K = np.array([[1000., 0, 640], [0, 1000., 480], [0, 0, 1]])
SIZE = (1280, 960)
BOARD_IMAGE = make_board().generateImage((720, 900))
BOARD_OUTLINE = np.array([[0, 0, 0], [.18, 0, 0], [.18, .225, 0], [0, .225, 0]], dtype=float)


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def render(T_cam_board):
    if T_cam_board is None:
        return np.full((SIZE[1], SIZE[0], 3), 160, dtype=np.uint8)
    T = T_cam_board
    H = K @ np.column_stack([T[:3, 0], T[:3, 1], T[:3, 3]]) @ np.diag([.18 / 720, .225 / 900, 1])
    gray = cv2.warpPerspective(BOARD_IMAGE, H, SIZE, borderValue=160)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


class BoardCamera:
    """Renders the board as seen from the current pose_fn() at read time."""

    def __init__(self, pose_fn):
        self.pose_fn, self.sequence = pose_fn, 0
        self.metadata = {'synthetic': True, 'device': 'rendered ChArUco', 'resolution': list(SIZE),
                         'K': K.tolist(), 'distortion': [0.] * 8, 'undistorted': False}

    def open(self):
        pass

    def read(self, timeout=1.):
        time.sleep(1 / 30)
        rgb = render(self.pose_fn())
        self.sequence += 1
        return ColorFrame(self.sequence, rgb, time.monotonic())

    def close(self):
        pass


class Rig:
    """Ground-truth mounting and visible poses for one mode."""

    def __init__(self, mode, config, seed=41):
        self.mode = mode
        self.fk = UrdfChainFK(str(DEFAULT_URDF), 'base_link', 'Link7')
        pose = json.loads((ROOT / 'tasks/dexlong_gear3/config/initial_pose.json').read_text())
        self.q0 = np.array([pose['joint_positions'][name] for name in JOINTS])
        E0 = self.fk.fk(self.q0.tolist())
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
            self.U, self.V = C0, inv(E0) @ B0
        else:
            self.U = np.eye(4)
            self.U[:3, 3] = [.01, .02, .06]
            C0 = E0 @ self.U
            self.V = C0.copy()
            self.V[:3, 3] = C0[:3, 3] + C0[:3, :3] @ [-.09, -.1125, .7]
        self.lower, self.upper = config.lower, config.upper
        self.rng = np.random.default_rng(seed)

    def cam_board(self, q):
        E = self.fk.fk(list(q))
        C, B = (self.U, E @ self.V) if self.mode == 'eye-to-hand' else (E @ self.U, self.V)
        return inv(C) @ B

    def poses(self, count):
        out = []
        for _ in range(2000):
            q = self.q0 + self.rng.uniform(-1, 1, 7) * np.deg2rad([25, 15, 15, 20, 30, 20, 30])
            motor = COMPENSATION.to_motor(q)
            if np.any(motor < self.lower + .05) or np.any(motor > self.upper - .05):
                continue
            T = self.cam_board(q)
            points = BOARD_OUTLINE @ T[:3, :3].T + T[:3, 3]
            if points[:, 2].min() < .25 or T[2, 2] < .45:
                continue
            uv = points @ K.T
            uv = uv[:, :2] / uv[:, 2:]
            if (uv < 15).any() or (uv[:, 0] > SIZE[0] - 15).any() or (uv[:, 1] > SIZE[1] - 15).any():
                continue
            out.append(q)
            if len(out) == count:
                return out
        raise AssertionError('Could not generate enough visible synthetic board poses')


class CollectorCase(unittest.TestCase):
    def make_app(self, root, mode, *, blank=False):
        import collect_app

        config = dataclasses.replace(load_config(CONTROL_CONFIG), compensation=COMPENSATION, port=free_port())
        rig = Rig(mode, config)
        holder = {}

        def pose():
            app = holder.get('app')
            if blank or app is None or app.driver.robot is None:
                return None
            return rig.cam_board(COMPENSATION.to_model(app.driver.robot.q))

        app = collect_app.CollectApp(config, real=False, camera_factory=lambda: BoardCamera(pose), mode=mode,
                                     sample_sec=.3, session_root=root / 'sessions')
        holder['app'] = app
        self.addCleanup(app.close)
        app.action(app.connect)
        self.assertEqual(app.error.value, '')
        return app, rig

    def teleport(self, app, q):
        motor = COMPENSATION.to_motor(q)
        app.driver.robot.q = motor.copy()
        app.driver.requested = motor.copy()
        time.sleep(.08)                              # let frames rendered at the old pose age out

    def sample(self, app, during=None):
        app.action(app.start_sample)
        self.assertEqual(app.error.value, '')
        while app.pending is not None:
            if during is not None and time.monotonic() > app.pending['start'] + .15:
                during()
                during = None
            app.update()
            time.sleep(.01)


class CollectToSolveTests(CollectorCase):
    def test_both_mounting_modes(self):
        for mode in ('eye-to-hand', 'eye-in-hand'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                app, rig = self.make_app(root, mode)
                poses = rig.poses(14)
                for k, q in enumerate(poses):
                    self.teleport(app, q)
                    app.validation.value = k >= 12
                    self.sample(app)
                    self.assertEqual(app.error.value, '', app.message)
                self.assertEqual(len(app.session.samples), 14)
                self.assertIsNotNone(app.solution, app.solution_note)
                self.assertLess(np.linalg.norm(app.solution['U'][:3, 3] - rig.U[:3, 3]), .01)
                session = app.session.path
                app.close()

                metadata = json.loads((session / 'session.json').read_text(encoding='utf-8'))
                self.assertEqual(metadata['mounting_mode'], mode)
                self.assertEqual(metadata['joint_compensation']['directions'], list(COMPENSATION.directions))
                result = calibrate_session(session, root / 'result', reject=0, allow_synthetic=True)
                key1, key2 = (('T_base_cam', 'T_flange_board') if mode == 'eye-to-hand'
                              else ('T_flange_cam', 'T_base_board'))
                recovered = np.array(result[key1])
                self.assertLess(np.linalg.norm(recovered[:3, 3] - rig.U[:3, 3]), .01)
                self.assertLess(rot_deg(recovered[:3, :3].T @ rig.U[:3, :3]), 1.)
                self.assertLess(np.linalg.norm(np.array(result[key2])[:3, 3] - rig.V[:3, 3]), .01)
                self.assertLess(result['residual_summary']['reproj_px']['rms'], 1.)
                self.assertEqual(len(result['samples']), 12)
                self.assertEqual(len(result['validation_samples']), 2)
                self.assertLess(result['validation_summary']['reproj_px']['rms'], 1.)
                self.assertEqual(result['joint_compensation']['source'], 'session')
                sample = result['samples'][0]
                np.testing.assert_allclose(sample['positions_rad'],
                                           COMPENSATION.to_model(sample['motor_positions_rad']))
                poses_ = poses_for_sample(result, sample)
                np.testing.assert_allclose(poses_['camera'] @ np.array(sample['T_cam_board']),
                                           poses_['observed_board'])
                viewer = ResultViewer(root / 'result' / 'handeye_result.json', port=0)
                try:
                    viewer.index.value = len(result['samples']) - 1
                    viewer.refresh()
                    self.assertIn('相机世界位置', viewer.details.content)
                    self.assertEqual(viewer.image.image.shape[:2], (SIZE[1], SIZE[0]))
                finally:
                    viewer.server.stop()
                if mode == 'eye-in-hand':
                    self.assertNotIn('T_base_cam', result)
                    self.assertFalse((root / 'result' / 'T_RC.npy').exists())


class RejectionTests(CollectorCase):
    def test_requires_motors_holding(self):
        with tempfile.TemporaryDirectory() as directory:
            app, rig = self.make_app(Path(directory), 'eye-in-hand')
            app.action(app.driver.release)
            app.action(app.start_sample)
            self.assertIn('保持当前位置', app.error.value)
            self.assertIsNone(app.pending)

    def test_moving_arm_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            app, rig = self.make_app(Path(directory), 'eye-in-hand')
            q = rig.poses(1)[0]
            self.teleport(app, q)

            def nudge():                             # the mock arm follows its target, so move both
                moved = app.driver.robot.q + np.deg2rad([0, 0, 0, 0, 0, .2, 0])
                app.driver.robot.q, app.driver.requested = moved.copy(), moved.copy()
            self.sample(app, during=nudge)
            self.assertIn('未静止', app.error.value)
            self.assertEqual(app.session.samples, [])

    def test_missing_board_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            app, rig = self.make_app(Path(directory), 'eye-in-hand', blank=True)
            self.teleport(app, rig.poses(1)[0])
            self.sample(app)
            self.assertIn('标定板', app.error.value)
            self.assertEqual(app.session.samples, [])


def write_session(root, directions=COMPENSATION.directions):
    writer = SessionWriter(root, {
        'mock': True, 'mounting_mode': 'eye-in-hand', 'joint_names': JOINTS, 'urdf': str(DEFAULT_URDF),
        'joint_compensation': {'directions': list(directions), 'offsets_deg': list(COMPENSATION.offsets_deg)},
        'camera': {'K': K.tolist(), 'distortion': [0.] * 8, 'resolution': list(SIZE), 'synthetic': True},
        'board': {'square_length_m': .045}})
    writer.append({'validation': False, 'motor_positions_rad': [.1] * 7}, render(None))
    return writer


class SessionTests(unittest.TestCase):
    def test_compensation_override_keeps_directions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_session(root / 's')
            loaded = load_session(root / 's', allow_synthetic=True)
            np.testing.assert_allclose(loaded['samples'][0]['positions_rad'], COMPENSATION.to_model([.1] * 7))
            newer = JointCompensation(COMPENSATION.directions, (1.,) * 7)
            newer.save(root / 'newer.yaml')
            loaded = load_session(root / 's', allow_synthetic=True, compensation=root / 'newer.yaml')
            self.assertEqual(loaded['compensation_source'], str(root / 'newer.yaml'))
            np.testing.assert_allclose(loaded['samples'][0]['positions_rad'], newer.to_model([.1] * 7))
            JointCompensation().save(root / 'wrong.yaml')
            with self.assertRaisesRegex(ValueError, 'directions differ'):
                load_session(root / 's', allow_synthetic=True, compensation=root / 'wrong.yaml')
            with self.assertRaisesRegex(ValueError, 'Synthetic'):
                load_session(root / 's')

    def test_undo_removes_image_and_old_format_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = write_session(root / 's')
            image = writer.path / writer.samples[0]['image']
            self.assertTrue(image.is_file())
            writer.remove_last()
            self.assertFalse(image.is_file())
            self.assertEqual((writer.path / 'samples.jsonl').read_text(), '')
            (root / 'old').mkdir()
            (root / 'old/session.json').write_text(json.dumps({'version': 1}))
            with self.assertRaisesRegex(ValueError, 'v2'):
                load_session(root / 'old')


class GeometryTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
