"""FK calibration contracts, tested with synthetic data and the simulated arm (no CAN hardware)."""
import dataclasses
import json
from pathlib import Path
import socket
import sys
import tempfile
import time
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import yaml

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from calibrate import evaluate, solve
from chain import ArmChain, delta_matrix
from pivot import pivot_fit
from session import SessionWriter, load_plate, load_session
import selftest_synthetic
from gx4m_real.robot.config import ARM_JOINTS, load_config
from gx4m_real.robot.kinematics import UrdfFK, quat_matrix

CONTROL_CONFIG = HERE.parent / 'control' / 'config.yaml'


def synthetic(poses_per_seat=10, truth='offsets', elevated=False, seed=0):
    config = load_config(CONTROL_CONFIG)
    chain = ArmChain(config.urdf)
    R = quat_matrix(config.base_wxyz)
    raw = yaml.safe_load(CONTROL_CONFIG.read_text())
    pose = json.loads((CONTROL_CONFIG.parent / raw['model']['initial_pose']).read_text())
    centre = R.T @ (np.asarray(pose['object_position']) - config.base_position)
    limits = config.compensation.model_limits(config.lower, config.upper)
    data, validation, truth_errors = selftest_synthetic.generate(
        chain, limits, R.T @ [0., 0., 1.], centre, truth=truth, poses_per_seat=poses_per_seat,
        elevated=elevated, seed=seed)
    return config, chain, data, validation, truth_errors


class ChainTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONTROL_CONFIG)
        self.chain = ArmChain(self.config.urdf)

    def test_matches_urdf_fk(self):
        fk = UrdfFK(self.config.urdf)
        Q = np.random.default_rng(1).uniform(-2, 2, (5, 7))
        for q, T in zip(Q, self.chain.fk(Q)):
            poses = fk.link_poses(dict(zip(ARM_JOINTS, q)))
            np.testing.assert_allclose(np.linalg.inv(poses[self.chain.base_link]) @ poses[self.chain.tip_link], T,
                                       atol=1e-12)

    def test_written_urdf_reproduces_geometric_errors(self):
        rng = np.random.default_rng(2)
        errors = np.zeros((7, 7))
        errors[1:, 1:4] = rng.normal(0, 1e-3, (6, 3))
        errors[1:, 4:] = rng.normal(0, 2e-3, (6, 3))
        errors[1:6, 0] = .01                         # dθ must not end up in the URDF
        with tempfile.TemporaryDirectory() as tmp:
            path = self.chain.write_urdf(errors, Path(tmp) / 'out' / 'cal.urdf')
            written = ArmChain(path)
            mesh = next(ET.parse(path).getroot().iter('mesh')).get('filename')
            self.assertTrue((path.parent / mesh).is_file())
        geometric = errors.copy()
        geometric[:, 0] = 0
        Q = rng.uniform(-2, 2, (4, 7))
        np.testing.assert_allclose(written.fk(Q), self.chain.fk(Q, geometric), atol=1e-9)

    def test_delta_matrix_is_rigid(self):
        D = delta_matrix(np.array([.001, -.002, .003, .01, -.02, .03]))
        np.testing.assert_allclose(D[:3, :3] @ D[:3, :3].T, np.eye(3), atol=1e-12)


class PivotTest(unittest.TestCase):
    def test_recovers_tip_and_centres(self):
        rng = np.random.default_rng(3)
        tip, centres = np.array([.01, -.02, .15]), {'A': np.array([.3, 0, 0]), 'B': np.array([.3, .2, .1])}
        poses, seats = [], []
        for seat, c in centres.items():
            for _ in range(6):
                T = np.eye(4)
                T[:3, :3] = np.linalg.qr(rng.normal(size=(3, 3)))[0]
                if np.linalg.det(T[:3, :3]) < 0:
                    T[:3, 0] *= -1
                T[:3, 3] = c - T[:3, :3] @ tip
                poses.append(T)
                seats.append(seat)
        fit_tip, fit_centres, residuals = pivot_fit(np.array(poses), seats)
        np.testing.assert_allclose(fit_tip, tip, atol=1e-12)
        for seat in centres:
            np.testing.assert_allclose(fit_centres[seat], centres[seat], atol=1e-12)
        self.assertLess(np.abs(residuals).max(), 1e-12)

    def test_single_orientation_is_rejected(self):
        T = np.eye(4)
        self.assertIsNone(pivot_fit(np.array([T, T, T]), ['A', 'A', 'A']))


class SolverTest(unittest.TestCase):
    def test_offsets_model_recovers_joint_zero_errors(self):
        config, chain, data, validation, truth = synthetic()
        train = data.subset(~validation)
        nominal = evaluate(chain, solve(chain, train, 'nominal'), data.subset(validation))
        result = solve(chain, train, 'offsets')
        calibrated = evaluate(chain, result, data.subset(validation))
        np.testing.assert_allclose(result.errors[1:6, 0], truth[1:6, 0], atol=np.deg2rad(.02))
        self.assertLess(calibrated['pivot_rms_mm'], .3)
        self.assertLess(calibrated['pivot_rms_mm'], nominal['pivot_rms_mm'] / 3)
        self.assertEqual(result.errors[0, 0], 0.)    # J1 and J7 zero offsets are never estimated
        self.assertEqual(result.errors[6, 0], 0.)


class SessionTest(unittest.TestCase):
    def test_writer_round_trip_and_undo(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = SessionWriter(Path(tmp) / 's', {'joint_compensation': {'directions': [1] * 7,
                                                                            'offsets_deg': [0.] * 7}})
            writer.append({'seat': 'H1', 'motor_positions_rad': [0.] * 7})
            writer.append({'seat': 'H2', 'motor_positions_rad': [.1] * 7})
            writer.remove_last()
            _, samples = load_session(writer.path)
            self.assertEqual([s['seat'] for s in samples], ['H1'])
            with self.assertRaises(FileExistsError):
                SessionWriter(writer.path, {})

    def test_plate_ignores_unmeasured_distances(self):
        plate = load_plate(HERE / 'plate.yaml')
        self.assertEqual(plate['distances_m'], {})
        self.assertEqual(list(plate['plate_seats']), ['H1', 'H2', 'H3', 'H4'])


class ReleaseTest(unittest.TestCase):
    def test_release_then_hold_without_recording(self):
        from gx4m_real.robot.arm import ArmDriver

        driver = ArmDriver(load_config(CONTROL_CONFIG), mock=True)
        driver.connect()
        driver.release()
        self.assertEqual(driver.snapshot()['mode'], 'released')
        self.assertTrue(driver.robot.released)
        self.assertIsNone(driver.recording)
        driver.hold()
        self.assertEqual(driver.snapshot()['mode'], 'hold')
        self.assertFalse(driver.robot.released)
        driver.disconnect()


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class CollectToSolveTest(unittest.TestCase):
    """Simulated arm → collector → session files → solve.py."""

    def test_end_to_end(self):
        import collect_app
        import solve

        config, chain, data, validation, truth = synthetic(poses_per_seat=8)
        config = dataclasses.replace(config, port=free_port())
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plate = yaml.safe_load((HERE / 'plate.yaml').read_text(encoding='utf-8'))
            plate['distances_mm'] = {f'{a}-{b}': float(d * 1000) for a, b, d in data.distances}
            (tmp / 'plate.yaml').write_text(yaml.safe_dump(plate), encoding='utf-8')
            app = collect_app.CollectApp(config, tmp / 'plate.yaml', real=False, sample_sec=.1,
                                         session_root=tmp / 'sessions')
            try:
                app.action(app.connect)
                for q, seat, val in zip(data.q, data.seats, validation):
                    motor = config.compensation.to_motor(q)
                    app.driver.robot.q = motor.copy()       # teleport the simulated arm onto the pose
                    app.driver.requested = motor.copy()
                    app.seat.value, app.validation.value = seat, bool(val)
                    app.action(app.start_sample)
                    self.assertEqual(app.error.value, '')
                    while app.pending is not None:
                        app.update()
                        time.sleep(.01)
                    self.assertEqual(app.error.value, '', app.message)
                session = app.session.path
                self.assertEqual(len(app.session.samples), len(data.q))
                self.assertLess(np.linalg.norm(app.fit['tip'] - selftest_synthetic.TRUE_TIP), 2e-3)
            finally:
                app.close()
            solve.main([str(session), '--plate', str(tmp / 'plate.yaml'), '--models', 'nominal', 'offsets',
                        '--out', str(tmp / 'calib')])
            report = json.loads((tmp / 'calib' / 'result.json').read_text(encoding='utf-8'))
            self.assertLess(report['models']['offsets']['validation']['pivot_rms_mm'],
                            report['models']['nominal']['validation']['pivot_rms_mm'] / 3)
            saved = yaml.safe_load((tmp / 'calib' / 'offsets' / 'joint_compensation.yaml').read_text())
            offsets = [saved['joints'][n]['offset_deg'] for n in ARM_JOINTS]
            expected = np.asarray(config.compensation.offsets_deg) + np.rad2deg(truth[:, 0])
            np.testing.assert_allclose(offsets, expected, atol=.03)


if __name__ == '__main__':
    unittest.main()
