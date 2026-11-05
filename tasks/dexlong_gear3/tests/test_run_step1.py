"""run_step1.py loop on the simulated arm and hand (no hardware, no perception process)."""
import dataclasses
import json
from pathlib import Path
import socket
import sys
import time
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from gx4m_real.policy import RslRlActor                                 # noqa: E402
from gx4m_real.robot.arm import ArmDriver                               # noqa: E402
from gx4m_real.robot.config import load_config                          # noqa: E402
from gx4m_real.robot.hand import HandDriver                             # noqa: E402
from gx4m_real.robot.tactile import FingertipContacts                   # noqa: E402
from gx4m_real.robot.tests.test_tactile import FakeTotal                # noqa: E402
from tasks.dexlong_gear3.run_step1 import (CONTROL_CONFIG, MODEL, CameraToEnv, FixedGearPose,  # noqa: E402
                                           Step1Runner, UdpGearPose, pose_matrix)
from tasks.dexlong_gear3.step1 import (ESTIMATORS, MODELS, REFERENCES, Step1Interface,  # noqa: E402
                                       load_reference)

HANDEYE = ROOT/'apps/charuco_handeye/runtime/calibration_20261031_fkzero/handeye_result.json'


class StaleGearPose(FixedGearPose):
    def get(self):
        return self.T.copy(), 2.


class Step1RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iface = Step1Interface()
        cls.actor = RslRlActor(MODEL)

    def runner(self, gear=None, velocity=1.5):
        config = dataclasses.replace(load_config(CONTROL_CONFIG), velocity_limit=velocity)
        arm, hand = ArmDriver(config, mock=True), HandDriver(mock=True)
        arm.connect()
        hand.connect()
        self.addCleanup(arm.disconnect)
        self.addCleanup(hand.close)
        gear = gear or FixedGearPose(self.iface.ref['object_pos'], self.iface.ref['object_quat'])
        return Step1Runner(self.iface, self.actor, arm, hand, gear, CameraToEnv(), approach_speed=1.,
                           log=lambda message: None)

    def test_episode_runs_at_30_hz_within_limits(self):
        runner = self.runner()
        q = runner.move_to_start()
        np.testing.assert_allclose(q[:7], self.iface.ref['initial_joint_pos'][:7], atol=.02)
        start = time.monotonic()
        outcome, rows = runner.episode(max_steps=30)
        self.assertEqual((outcome, len(rows)), ('max_steps', 30))
        self.assertAlmostEqual((time.monotonic() - start)/30, 1/30, delta=.01)
        targets = np.array([r['targets'] for r in rows])
        self.assertTrue((targets >= self.iface.lower - 1e-9).all() and (targets <= self.iface.upper + 1e-9).all())
        # Arm targets move at most the trained speed per step.
        steps = np.abs(np.diff(targets[:, :7], axis=0))
        self.assertLessEqual(steps.max(), self.iface.arm_speed_scale*self.iface.dt + 1e-9)
        self.assertEqual(rows[0]['obs'].shape, (self.iface.obs_dim,))
        self.assertEqual(runner.arm.mode, 'hold')

    def test_pose_age_limit(self):
        runner = self.runner(StaleGearPose(self.iface.ref['object_pos'], self.iface.ref['object_quat']))
        q = runner.joints()
        self.assertEqual(runner.max_pose_age, 1.)
        self.assertIsNone(runner.gear_pose(q)[0])           # 2 s old
        runner.max_pose_age = 2.5
        self.assertIsNotNone(runner.gear_pose(q)[0])

    def test_stale_pose_stops_and_holds(self):
        runner = self.runner(StaleGearPose(self.iface.ref['object_pos'], self.iface.ref['object_quat']))
        runner.move_to_start()
        outcome, rows = runner.episode(max_steps=30)
        self.assertEqual((outcome, rows), ('pose_lost', []))
        self.assertEqual(runner.arm.mode, 'hold')

    def test_arm_lag_stops_the_episode(self):
        runner = self.runner(velocity=.05)
        runner.max_tracking_error = .02
        runner.approach_speed = .05
        outcome, rows = runner.episode(max_steps=90)
        self.assertEqual(outcome, 'tracking_error')
        self.assertLess(len(rows), 90)

    def tactile_runner(self, normals):
        sensors = [FakeTotal(n) for n in normals]
        tactile = FingertipContacts(sensors, period=.001)
        self.addCleanup(tactile.close)
        runner = self.runner()
        runner.tactile = tactile
        return runner, sensors

    def test_contact_reaches_the_observation(self):
        """The resting force is tared away at the episode start; a press on thumb and ring finger becomes 1."""
        runner, sensors = self.tactile_runner([3, 3, 3, 3])
        outcome, rows = runner.episode(max_steps=1)
        self.assertEqual(outcome, 'max_steps')
        np.testing.assert_array_equal(rows[0]['contact'], [0, 0, 0, 0])
        np.testing.assert_allclose(runner.tactile.offset, [.3]*4)
        sensors[0].normal = sensors[3].normal = 9               # +0.6 N on thumb and ring finger
        time.sleep(.02)
        contact, force = runner.contacts()
        np.testing.assert_array_equal(contact, [1, 0, 0, 1])
        np.testing.assert_allclose(force, [.6, 0, 0, .6], atol=1e-9)
        start = sum(w for n, w in self.iface.fields[:[n for n, _ in self.iface.fields].index('fingertip_object_contact')])
        obs = self.iface.observation(self.iface.ref['initial_joint_pos'], self.iface.ref['object_pos'], [0, 0, 1.],
                                     contact)
        np.testing.assert_array_equal(obs[start:start + 4], [1, 0, 0, 1])

    def test_lost_tactile_stops_and_holds(self):
        runner, sensors = self.tactile_runner([0, 0, 0, 0])
        runner.max_tactile_age = .05
        runner.move_to_start()
        sensors[1].fail = True
        time.sleep(.1)
        outcome, rows = runner.episode(max_steps=30)
        self.assertEqual((outcome, rows), ('tactile_lost', []))
        self.assertEqual(runner.arm.mode, 'hold')

    def test_pose_estimator_keeps_running_while_the_gear_is_lost(self):
        """step2: a gear pose older than valid_pose_age is held and flagged instead of stopping the episode."""
        iface = Step1Interface(load_reference(REFERENCES[2]), estimator=ESTIMATORS[2])
        config = dataclasses.replace(load_config(CONTROL_CONFIG), velocity_limit=1.5)
        arm, hand = ArmDriver(config, mock=True), HandDriver(mock=True)
        arm.connect()
        hand.connect()
        self.addCleanup(arm.disconnect)
        self.addCleanup(hand.close)
        tactile = FingertipContacts([FakeTotal(0) for _ in range(4)], period=.001)
        self.addCleanup(tactile.close)
        runner = Step1Runner(iface, RslRlActor(MODELS[2]), arm, hand,
                             StaleGearPose(iface.ref['object_pos'], iface.ref['object_quat']), CameraToEnv(),
                             tactile=tactile, approach_speed=1., log=lambda message: None)
        runner.move_to_start()
        outcome, rows = runner.episode(max_steps=10)
        self.assertEqual((outcome, len(rows)), ('max_steps', 10))
        self.assertFalse(rows[0]['pose_valid'])
        self.assertEqual(rows[0]['obs'].shape, (80,))
        self.assertEqual(rows[0]['pad_force'].shape, (4, 3))
        self.assertIn('estimated_pos', rows[-1])
        runner.max_lost_age = 1.                                     # 2 s old: now too old even for the estimator
        self.assertEqual(runner.episode(max_steps=10)[0], 'pose_lost')


class GearPoseTests(unittest.TestCase):
    def test_udp_pose_and_lost_tracking(self):
        receiver = UdpGearPose(0)
        self.addCleanup(receiver.close)
        address = receiver.sock.getsockname()
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addCleanup(sender.close)
        self.assertEqual(receiver.get(), (None, float('inf')))
        T = pose_matrix([.1, .2, .3], [np.cos(.2), 0., 0., np.sin(.2)])
        sender.sendto(json.dumps({'timestamp_monotonic': time.monotonic(), 'object_in_camera': T.tolist()}).encode(),
                      address)
        sender.sendto(b'not json', address)
        for _ in range(100):
            pose, age = receiver.get()
            if pose is not None:
                break
            time.sleep(.01)
        np.testing.assert_allclose(pose, T)
        self.assertLess(age, .5)
        # Lost tracking keeps the last valid pose; its age grows until max_pose_age rejects it.
        sender.sendto(json.dumps({'timestamp_monotonic': time.monotonic(), 'object_in_camera': None}).encode(),
                      address)
        time.sleep(.1)
        pose, later = receiver.get()
        np.testing.assert_allclose(pose, T)
        self.assertGreater(later, age + .05)

    @unittest.skipUnless(HANDEYE.exists(), 'no hand-eye result')
    def test_eye_to_hand_camera_to_env(self):
        iface = Step1Interface()
        offset = pose_matrix([0., 0., .01], [1., 0., 0., 0.])
        camera = CameraToEnv(HANDEYE, iface, offset)
        d = json.loads(HANDEYE.read_text(encoding='utf-8'))
        T_env_cam = pose_matrix(iface.ref['base_pos'], iface.ref['base_quat']) @ np.asarray(d['T_base_cam'])
        T_env_object = pose_matrix(iface.ref['object_pos'], iface.ref['object_quat'])
        T_cam_mesh = np.linalg.inv(T_env_cam) @ T_env_object @ np.linalg.inv(offset)
        np.testing.assert_allclose(camera(T_cam_mesh, np.zeros(23)), T_env_object, atol=2e-3)


if __name__ == '__main__':
    unittest.main()
