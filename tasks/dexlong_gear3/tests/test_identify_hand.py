"""tools/identify_hand.py: target plans, clearance checks and response metrics (no hardware)."""
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tasks/dexlong_gear3/tools'))

from gx4m_real.robot.hand import LOWER, UPPER                           # noqa: E402
from identify_hand import (COMMAND_HZ, MARGIN, Clearance, joint_plan, step_metrics,  # noqa: E402
                           xcorr_delay)
from tasks.dexlong_gear3.step1 import Step1Interface                    # noqa: E402


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.iface = Step1Interface()
        cls.base = np.array(cls.iface.ref['initial_joint_pos'][7:])

    def test_plans_move_one_joint_inside_the_margin(self):
        for j in range(16):
            for scale in (.25, 1.):
                targets, index, segments = joint_plan(j, self.base, scale)
                others = np.delete(targets, j, axis=1)
                np.testing.assert_allclose(others, np.delete(self.base, j)[None].repeat(len(targets), 0))
                self.assertTrue((targets[:, j] >= LOWER[j] + MARGIN - 1e-9).all())
                self.assertTrue((targets[:, j] <= UPPER[j] - MARGIN + 1e-9).all())
                self.assertAlmostEqual(targets[-1, j], np.clip(self.base[j], LOWER[j] + MARGIN, UPPER[j] - MARGIN))
                self.assertEqual(index.max() + 1, len(segments))
                self.assertIn('chirp', [k for k, _ in segments])
                # The chirp never asks for more than its speed limit (one 30 Hz sample of slack).
                chirp = targets[index == len(segments) - 1, j]
                self.assertLess(np.abs(np.diff(chirp)).max()*COMMAND_HZ, 2.2)

    def test_clearance(self):
        start_arm = np.array(self.iface.ref['initial_joint_pos'][:7])
        self.assertIsNone(Clearance(self.iface, start_arm).check(self.base))
        self.assertIn('above the table', Clearance(self.iface, start_arm, table_clearance=.3).check(self.base))
        self.assertIn('fingers', Clearance(self.iface, start_arm, finger_clearance=.2).check(self.base))


class MetricTests(unittest.TestCase):
    def test_step_and_delay_of_a_known_response(self):
        t = np.arange(0., 2.5, .014)
        target = np.where(t < 1.5, .3, 0.)
        delay, tau = .04, .05
        q = np.where(t > delay, .3*(1 - np.exp(-np.clip(t - delay, 0, None)/tau)), 0.)
        m = step_metrics(t, 0., .3, q)
        self.assertAlmostEqual(m['t10_ms'], (delay + tau*np.log(1/.9))*1000, delta=15)
        self.assertAlmostEqual(m['rise_ms'], tau*np.log(9)*1000, delta=30)
        self.assertLess(abs(m['steady_error_deg']), .1)
        sine = np.sin(2*np.pi*1.5*t)
        self.assertAlmostEqual(xcorr_delay(t, sine, np.sin(2*np.pi*1.5*(t - .06))), 60, delta=3)


if __name__ == '__main__':
    unittest.main()
