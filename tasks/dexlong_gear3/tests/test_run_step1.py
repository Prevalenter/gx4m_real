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
from tasks.dexlong_gear3.run_step1 import (CONTROL_CONFIG, MODEL, CameraToEnv, FixedGearPose,  # noqa: E402
                                           Step1Runner, UdpGearPose, pose_matrix)
from tasks.dexlong_gear3.step1 import Step1Interface                    # noqa: E402

HANDEYE = ROOT/'apps/charuco_handeye/runtime/calibration_20261031_fkzero/handeye_result.json'


class StaleGearPose(FixedGearPose):
    def get(self):
        return self.T.copy(), 1.


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
        sender.sendto(json.dumps({'timestamp_monotonic': time.monotonic(), 'object_in_camera': None}).encode(),
                      address)
        for _ in range(100):
            if receiver.get()[0] is None:
                break
            time.sleep(.01)
        self.assertIsNone(receiver.get()[0])

    @unittest.skipUnless(HANDEYE.exists(), 'no hand-eye result')
    def test_eye_to_hand_camera_to_env(self):
        iface = Step1Interface()
        offset = pose_matrix([0., 0., .01], [1., 0., 0., 0.])
        camera = CameraToEnv(HANDEYE, iface, offset)
        d = json.loads(HANDEYE.read_text(encoding='utf-8'))
        T_env_cam = np.asarray(d['T_world_base']) @ np.asarray(d['T_base_cam'])
        T_env_object = pose_matrix(iface.ref['object_pos'], iface.ref['object_quat'])
        T_cam_mesh = np.linalg.inv(T_env_cam) @ T_env_object @ np.linalg.inv(offset)
        np.testing.assert_allclose(camera(T_cam_mesh, np.zeros(23)), T_env_object, atol=2e-3)


if __name__ == '__main__':
    unittest.main()
