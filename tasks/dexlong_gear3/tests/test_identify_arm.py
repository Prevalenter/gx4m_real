"""tools/identify_arm.py: plan construction, clearance pruning and the operator-driven runner (simulated arm)."""
import dataclasses
from pathlib import Path
import sys
import tempfile
import time
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tasks/dexlong_gear3/tools'))

import identify_arm as A                                                # noqa: E402
from tasks.dexlong_gear3.run_step1 import CONTROL_CONFIG, arm_config    # noqa: E402
from tasks.dexlong_gear3.step1 import URDF, Step1Interface              # noqa: E402


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iface = Step1Interface()
        config = arm_config(CONTROL_CONFIG, cls.iface, A.HANDEYE)
        cls.config = config
        cls.lower, cls.upper = config.compensation.model_limits(config.lower, config.upper)
        cls.lower, cls.upper = cls.lower + A.LIMIT_MARGIN, cls.upper - A.LIMIT_MARGIN
        cls.start = np.asarray(cls.iface.ref['initial_joint_pos'][:7], float)
        cls.geometry = A.RobotGeometry(cls.iface, URDF, A.HANDEYE)
        cls.hand = np.asarray(cls.iface.ref['initial_joint_pos'][7:], float)

    def test_plan_moves_one_joint_at_a_time_within_speed(self):
        targets, index, segments, _, skipped = A.build_plan([('start', self.start)], self.lower, self.upper)
        self.assertEqual(skipped, [])
        np.testing.assert_allclose(targets[0], self.start)
        np.testing.assert_allclose(targets[-1], self.start)
        self.assertTrue((targets >= self.lower - 1e-9).all() and (targets <= self.upper + 1e-9).all())
        for s, d in enumerate(segments):
            if 'joint' in d:
                moved = np.abs(targets[index == s] - self.start).max(axis=0) > 1e-9
                self.assertEqual(np.nonzero(moved)[0].tolist(), [d['joint'] - 1])
        moves = np.abs(np.diff(targets, axis=0)).max(axis=1)
        smooth = np.array([segments[s]['kind'] != 'step' for s in index[1:]]) & (index[1:] == index[:-1])
        self.assertLess(moves[smooth].max()*A.COMMAND_HZ, .3 + .02)   # ramps, chirp, transfers
        self.assertLessEqual(moves.max(), np.radians(5) + 1e-9)       # steps jump 2 or 5 deg once

    def test_clearance_prunes_and_flips(self):
        limits = dict(table=.05, plate=.03, camera=0., self=.01)
        check = A.segment_checker(self.geometry, self.hand, limits)
        _, _, segments, clear, skipped = A.build_plan([('start', self.start)], self.lower, self.upper, check=check)
        for c in clear:
            for key, (d, _) in c.items():
                self.assertGreaterEqual(d, limits[key])
        self.assertTrue(all(s['why'] for s in skipped))
        ramps = [s for s in segments if s['kind'] == 'ramp' and s['joint'] == 4]
        self.assertEqual(len(ramps), len(A.RAMP_SPEEDS))     # J4 ramps kept, flipped away from the table

    def test_geometry_sees_the_table(self):
        q = np.r_[self.start, self.hand]
        start = self.geometry.clearance(q)['table'][0]
        self.assertGreater(start, .05)                      # the step1 start pose: the hand about 11 cm up
        self.geometry.table_z += .2
        try:
            self.assertAlmostEqual(self.geometry.clearance(q)['table'][0], start - .2)
        finally:
            self.geometry.table_z -= .2


class RunnerTests(unittest.TestCase):
    def test_runner_only_moves_after_start_and_stops(self):
        iface = Step1Interface()
        config = arm_config(CONTROL_CONFIG, iface, A.HANDEYE)
        start = np.asarray(iface.ref['initial_joint_pos'][:7], float)
        lower, upper = config.compensation.model_limits(config.lower, config.upper)
        targets, index, segments, clear, skipped = A.build_plan([('start', start)], lower + A.LIMIT_MARGIN,
                                                                upper - A.LIMIT_MARGIN)
        geometry = A.RobotGeometry(iface, URDF, A.HANDEYE)
        plan = dict(iface=iface, config=config, targets=targets, index=index, segments=segments, geometry=geometry,
                    hand_q=np.asarray(iface.ref['initial_joint_pos'][7:]), per=dict(enumerate(clear)),
                    skipped=skipped, problems=[], limits=dict(table=-1., plate=-1., camera=-1., self=-1.),
                    poses=[('start', start)], urdf=URDF)
        from gx4m_real.robot.hand import HandDriver
        with tempfile.TemporaryDirectory() as out:
            runner = A.PlanRunner(plan, mock=True, out_root=Path(out), log=lambda m: None,
                                  hand_factory=lambda: HandDriver(mock=True))
            with self.assertRaises(RuntimeError):
                runner.go_home()                            # not connected: nothing moves
            runner.connect()
            self.assertIsNotNone(runner.hand)
            q0 = runner.measured()
            time.sleep(.2)
            np.testing.assert_allclose(runner.measured(), q0)   # connected and holding, not moving
            self.assertFalse(runner.at_start())
            with self.assertRaisesRegex(RuntimeError, '初始位置'):
                runner.start()                              # collecting only from the start pose
            runner.go_home()
            runner.thread.join(30.)
            self.assertTrue(runner.at_start(), runner.status)
            np.testing.assert_allclose(runner.measured_hand(), plan['hand_q'], atol=.02)
            runner.start()
            time.sleep(1.)
            self.assertTrue(runner.running)
            runner.stop()
            runner.thread.join(5.)
            self.assertIn('操作员停止', runner.status)
            data = np.load(runner.out/'run.npz')
            self.assertGreater(len(data['t']), 30)
            runner.disconnect()
            self.assertIsNone(runner.hand)


if __name__ == '__main__':
    unittest.main()
