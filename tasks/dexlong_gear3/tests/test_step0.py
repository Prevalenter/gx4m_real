"""Step0 interface and policy against a deterministic Isaac Lab rollout (no Isaac, no hardware).

tests/data/step0_isaac_rollout.npz is gx4m_learning's export_gear3_split_sim2sim_reference.py --step 0 output
for models/step0_mlp_20261027_iter750.pt: per step k the Isaac policy observation obs[k], its action[k], and
the joint positions q[k], joint targets targets[k] and gear pose after that action, which the next
observation obs[k+1] is built from.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from gx4m_real.policy import RslRlActor  # noqa: E402
from gx4m_real.robot.kinematics import quat_matrix  # noqa: E402
from step0 import Step0Interface  # noqa: E402

ROLLOUT = HERE/'data'/'step0_isaac_rollout.npz'
MODEL = HERE.parent/'models'/'step0_mlp_20261027_iter750.pt'


def slices(fields):
    out, start = {}, 0
    for name, width in fields:
        out[name] = slice(start, start + width)
        start += width
    return out


class Step0RolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rollout = dict(np.load(ROLLOUT))
        cls.iface = Step0Interface()
        cls.cols = slices(cls.iface.fields)

    def rebuilt(self, k):
        """Observation k from the state Isaac recorded after action k-1 (or the reset state for k = 0)."""
        r, iface = self.rollout, self.iface
        if k == 0:
            q, pos, quat = iface.ref['initial_joint_pos'], iface.ref['object_pos'], iface.ref['object_quat']
            iface.targets = np.array(iface.ref['initial_joint_pos'])
        else:
            q, pos, quat = r['q'][k-1], r['object_pos'][k-1], r['object_quat'][k-1]
            iface.targets = r['targets'][k-1]
        contact = r['obs'][k][self.cols['fingertip_object_contact']]   # needs PhysX contacts: taken as recorded
        return iface.observation(q, pos, quat_matrix(quat)[:, 2], contact)

    def test_observation_matches_isaac(self):
        self.iface.reset(self.iface.ref['object_pos'])
        worst = {name: 0. for name, _ in self.iface.fields}
        for k in range(len(self.rollout['obs'])):
            diff = np.abs(self.rebuilt(k) - self.rollout['obs'][k])
            for name, cols in self.cols.items():
                worst[name] = max(worst[name], float(diff[cols].max()))
        # float32 Isaac state vs float64 here; positions are metres, 1e-4 = 0.1 mm.
        for name, err in worst.items():
            with self.subTest(field=name):
                self.assertLess(err, 1e-4, f'{name}: max |rebuilt - isaac| = {err:.2e}')

    def test_action_targets_match_isaac(self):
        r, iface = self.rollout, self.iface
        iface.reset(iface.ref['object_pos'])
        initial = np.array(iface.ref['initial_joint_pos'])
        err = 0.
        for k in range(len(r['action'])):
            iface.targets = initial if k == 0 else r['targets'][k-1]
            targets = iface.apply_action(r['action'][k])
            if r['done'][k]:            # the episode ended (success after 85 steps): Isaac reset the targets
                np.testing.assert_allclose(r['targets'][k], initial, atol=1e-6)
                continue
            err = max(err, float(np.abs(targets - r['targets'][k]).max()))
        self.assertGreater(int(r['done'].sum()), 0)
        self.assertLess(err, 1e-5, f'max |targets - isaac targets| = {err:.2e} rad')

    def test_policy_matches_isaac(self):
        actor = RslRlActor(MODEL)
        self.assertEqual((actor.kind, actor.obs_dim, actor.action_dim), ('mlp', self.iface.obs_dim, 23))
        err = max(float(np.abs(actor(o) - a).max()) for o, a in zip(self.rollout['obs'], self.rollout['action']))
        self.assertLess(err, 1e-4, f'max |action - isaac action| = {err:.2e}')

    def test_closed_loop_policy_on_rebuilt_observations(self):
        """Policy fed with rebuilt observations reproduces Isaac's actions (interface + policy end to end)."""
        actor = RslRlActor(MODEL)
        self.iface.reset(self.iface.ref['object_pos'])
        err = max(float(np.abs(actor(self.rebuilt(k)) - self.rollout['action'][k]).max())
                  for k in range(len(self.rollout['obs'])))
        self.assertLess(err, 1e-3, f'max |action(rebuilt obs) - isaac action| = {err:.2e}')

    def test_goal_follows_measured_start(self):
        start = np.array(self.iface.ref['object_pos']) + (.013, -.008, 0.)
        self.iface.reset(start)
        np.testing.assert_allclose(self.iface.goal, start + (0., 0., .1), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
