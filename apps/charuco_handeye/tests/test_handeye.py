"""Synthetic tests: mock arm + rendered ChArUco camera → collector → session → solver. No hardware."""
import dataclasses
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from board import make_board
from gx4m_real.camera.frames import ColorFrame, RGBDFrame
from gx4m_real.robot.compensation import JointCompensation
from gx4m_real.robot.config import load_config
from handeye_session import DEFAULT_URDF, JOINTS, SessionWriter, camera_parameters, load_session
from robot import UrdfChainFK
from solve_handeye import inv, rot_deg
from solve_session import calibrate_session, check_motion
from view_result import GearWorker, ResultViewer, load_pointcloud_capture, poses_for_sample

CONTROL_CONFIG = ROOT / 'apps/control/config.yaml'
# Same directions as the real arm, plus offsets, so FK on raw motor angles would fail the tests.
COMPENSATION = JointCompensation((-1, 1, -1, 1, -1, -1, -1), (0., 1.5, -2., 3., -2., -6.5, 0.))
K = np.array([[1000., 0, 640], [0, 1000., 480], [0, 0, 1]])
SIZE = (1280, 960)
BOARD_IMAGE = make_board().generateImage((720, 900))
BOARD_OUTLINE = np.array([[0, 0, 0], [.18, 0, 0], [.18, .225, 0], [0, .225, 0]], dtype=float)
# Stands in for tasks/dexlong_gear3/perception/gear_worker.py: same protocol, gear at the frame's max depth.
FAKE_GEAR_WORKER = '''
import os, sys
sys.path.insert(0, {root!r})
from multiprocessing.connection import Listener
import numpy as np
from gx4m_real.camera.frame_ipc import receive, send
key = bytes.fromhex(os.environ['GEAR_WORKER_AUTHKEY'])
with Listener(('127.0.0.1', int(os.environ['GEAR_WORKER_PORT'])), authkey=key) as listener, listener.accept() as c:
    send(c, {{'type': 'ready'}})
    while True:
        try:
            header, arrays = receive(c)
        except EOFError:
            break
        if header['type'] == 'frame':
            pose = np.eye(4)
            pose[2, 3] = float(arrays[1].max())
            send(c, {{'type': 'pose', 'sequence': header['sequence'], 'status': 'TRACK', 'mode': 'TRACK',
                      'mask_iou': .9, 'median_depth_error_m': .001, 'processing_ms': 1.,
                      'object_in_camera': pose.tolist()}})
'''


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
                    self.assertIn('Link7', viewer.model.link_map)
                    self.assertFalse([name for name in viewer.model.link_map if name.startswith('gx4m_')])
                finally:
                    viewer.server.stop()
                if mode == 'eye-to-hand':
                    self.check_live_viewer(root / 'result' / 'handeye_result.json', sample)
                if mode == 'eye-in-hand':
                    self.assertNotIn('T_base_cam', result)
                    self.assertFalse((root / 'result' / 'T_RC.npy').exists())

    def check_live_viewer(self, result_path, sample):
        """Live mode maps raw motor feedback with the result's compensation and streams the cloud."""
        stop = threading.Event()
        depth = np.zeros((4, 6))
        depth[1:3, 2:5] = 1.
        frame = RGBDFrame(3, np.full((4, 6, 3), 200, np.uint8), depth, K, 0., 0.)

        class Source:
            """Context manager whose first read() returns value and stops the loop."""
            metadata = {}

            def __init__(self, value=None):
                self.value = value

            def __enter__(self):
                if self.value is None:
                    stop.set()
                    raise TimeoutError('no CAN reply')
                return self

            def __exit__(self, *_):
                pass

            def read(self, after=-1, timeout=5):
                stop.set()
                return self.value

        motor = sample['motor_positions_rad']
        worker = Path(tempfile.mkdtemp()) / 'fake_gear_worker.py'
        worker.write_text(FAKE_GEAR_WORKER.format(root=str(ROOT)))
        viewer = ResultViewer(result_path, port=0, live=True, cloud_stride=1,
                              camera_factory=lambda: Source(frame), reader_factory=lambda: Source(np.array(motor)),
                              gear_factory=lambda cancel: GearWorker([sys.executable, str(worker)], cancel=cancel,
                                                                     startup_timeout=30))
        try:
            self.assertFalse(viewer.show_boards.value)
            viewer._joint_loop(stop)
            np.testing.assert_allclose(viewer.live_q, COMPENSATION.to_model(motor))
            np.testing.assert_allclose(viewer.live_q, sample['positions_rad'])
            self.assertIn('J1', viewer.joint_status.content)
            stop.clear()
            viewer._cloud_loop(stop)
            self.assertEqual(viewer.cloud.points.shape, (6, 3))
            np.testing.assert_allclose(viewer.cloud.points[:, 2], 1.)
            self.assertIn('第 3 帧', viewer.cloud_status.content)
            stop.clear()
            viewer.reader_factory = Source
            viewer._joint_loop(stop)
            self.assertIn('关节读取失败', viewer.joint_status.content)
            np.testing.assert_allclose(viewer.live_q, sample['positions_rad'])
            # The cloud loop handed its frame to the gear tracker; the fake worker puts the gear at the max depth.
            self.assertIs(viewer.latest_frame, frame)
            stop.clear()
            thread = threading.Thread(target=viewer._gear_loop, args=(stop,))
            thread.start()
            deadline = time.monotonic() + 30
            while not viewer.gear.visible and time.monotonic() < deadline:
                time.sleep(.05)
            stop.set()
            thread.join(timeout=15)
            self.assertFalse(thread.is_alive())
            self.assertTrue(viewer.gear.visible, viewer.gear_status.content)
            np.testing.assert_allclose(viewer.gear.position, [0, 0, 1.])
            self.assertIn('世界坐标', viewer.gear_status.content)
        finally:
            viewer.server.stop()
            worker.unlink()
            worker.parent.rmdir()
        # Without a camera: no cloud widgets; with the hand reader, the hand is drawn at its live angles.
        hand = np.linspace(0., .3, 16)
        stop.clear()
        class Tactile:
            def __init__(self, normal):
                self.normal, self.closed = normal, False

            def read_frame(self):
                stop.set()
                if self.normal is None:
                    raise RuntimeError('PaXini read failed with status 0x01')
                frame = np.zeros((135, 3))
                frame[:3, 2] = self.normal
                return frame

            def read_total(self):
                return np.array([-1., 2., 9.])

            def close(self):
                self.closed = True

        tactile = {'A': Tactile(10.), 'B': Tactile(None)}
        viewer = ResultViewer(result_path, port=0, live=True, reader_factory=lambda: Source(np.array(motor)),
                              hand_factory=lambda: Source(hand), tactile_factory=lambda: tactile)
        try:
            self.assertFalse(hasattr(viewer, 'cloud_status'))
            self.assertIn('gx4m_link1', viewer.model.link_map)
            viewer._hand_loop(stop)
            self.assertEqual(viewer.hand_q['gx4m_joint16'], hand[15])
            self.assertIn('16 +17.2°', viewer.hand_status.content)
            stop.clear()
            viewer._joint_loop(stop)                    # an arm update keeps the hand angles
            self.assertEqual(viewer.hand_q['gx4m_joint1'], hand[0])
            np.testing.assert_allclose(viewer.arm_shown, sample['positions_rad'])
            stop.clear()
            viewer._tactile_loop(stop)                  # one failing sensor does not hide the other
            self.assertIn('| A | 30 | 10 | 3 | -1 / +2 / 9 |', viewer.tactile_status.content)
            self.assertIn('| B | 读取失败：PaXini read failed with status 0x01', viewer.tactile_status.content)
            self.assertTrue(all(s.closed for s in tactile.values()))
            # A sensor plugged in later shows up: a changed label set reopens the sensors.
            opened = []

            def reopen():
                opened.append(1)
                if len(opened) == 2:
                    stop.set()
                return {'A': Tactile(1.)}

            stop.clear()
            viewer.tactile_factory, viewer.tactile_labels = reopen, lambda: ['A', 'C']
            with patch('view_result.RETRY_SEC', 0.):
                viewer._tactile_loop(stop)
            self.assertEqual(len(opened), 2)
        finally:
            viewer.server.stop()


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
            JointCompensation(COMPENSATION.directions, (0.,) * 7).save(root / 'zero.yaml')
            loaded = load_session(root / 's', allow_synthetic=True, compensation=root / 'zero.yaml')
            np.testing.assert_allclose(loaded['samples'][0]['positions_rad'], np.array(COMPENSATION.directions) * .1)
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
    def test_capture_cloud_matches_pixels_in_camera_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {'undistorted': True, 'depth_unit': 'metre'}
            (root / 'capture.json').write_text(json.dumps(metadata))
            rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
            depth = np.zeros((3, 4))
            depth[1, 2], depth[2, 0] = 1., 2.
            depth[0] = [np.nan, np.inf, -.5, 3.]
            K = np.array([[2., 0., 1.], [0., 4., 1.], [0., 0., 1.]])
            np.savez(root / 'rgbd.npz', rgb=rgb, depth=depth, K=K)
            points, colors, _, _ = load_pointcloud_capture(root, stride=1)
            np.testing.assert_allclose(points, [[.5, 0., 1.], [-1., .5, 2.]])
            np.testing.assert_array_equal(colors, rgb[[1, 2], [2, 0]])
            with self.assertRaisesRegex(ValueError, 'stride'):
                load_pointcloud_capture(root, stride=0)
            metadata['undistorted'] = False
            (root / 'capture.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'undistorted'):
                load_pointcloud_capture(root)

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


class J7OffsetTests(unittest.TestCase):
    def test_save_writes_the_result_and_its_yaml(self):
        from view_result import save_j7_offset
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            yaml_path = folder/'joint_compensation.yaml'
            yaml_path.write_text('version: 1\n# keep me\njoints:\n'
                                 + ''.join(f'  joint{i}:\n    direction: -1\n    offset_deg: {i}.5\n' for i in range(1, 8)))
            result = folder/'handeye_result.json'
            result.write_text(json.dumps({'joint_compensation': {'directions': [-1]*7, 'offsets_deg': [1.5]*7,
                                                                 'source': str(yaml_path)}, 'other': 1}, indent=2))
            result.chmod(0o644)
            save_j7_offset(result, -12.3)
            self.assertEqual(result.stat().st_mode & 0o777, 0o644)
            data = json.loads(result.read_text())
            self.assertEqual(data['joint_compensation']['offsets_deg'], [1.5]*6 + [-12.3])
            self.assertEqual(data['other'], 1)
            loaded = JointCompensation.load(yaml_path)
            self.assertEqual(loaded.offsets_deg, (1.5, 2.5, 3.5, 4.5, 5.5, 6.5, -12.3))
            self.assertIn('# keep me', yaml_path.read_text())


if __name__ == '__main__':
    unittest.main()
