"""Step1 interface and policy against a deterministic Isaac Lab rollout (no Isaac, no hardware).

tests/data/step1_isaac_rollout.npz is gx4m_learning's export_gear3_split_sim2sim_reference.py --step 1 output
for models/step1_lstm_20261031_iter500.pt (domain randomization off, nominal start, 400 steps, four
episodes): per step k the Isaac policy observation obs[k], its action[k], and the joint positions q[k],
joint targets targets[k], gear pose and stage after that action, which the next observation obs[k+1] is
built from. On a step with done[k] the recorded state is already the reset one.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from gx4m_real.policy import RslRlActor  # noqa: E402
from gx4m_real.robot.kinematics import quat_matrix  # noqa: E402
from step1 import Step1Interface  # noqa: E402

ROLLOUT = HERE/'data'/'step1_isaac_rollout.npz'
MODEL = HERE.parent/'models'/'step1_lstm_20261031_iter500.pt'


def slices(fields):
    out, start = {}, 0
    for name, width in fields:
        out[name] = slice(start, start + width)
        start += width
    return out


class Step1RolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rollout = dict(np.load(ROLLOUT))
        cls.iface = Step1Interface()
        cls.cols = slices(cls.iface.fields)

    def replay(self):
        """Yield (k, rebuilt observation k) with the interface's own stage tracker fed the recorded states."""
        r, iface = self.rollout, self.iface
        iface.reset()
        for k in range(len(r['obs'])):
            if k == 0 or r['done'][k-1]:
                iface.reset()
                q, pos, quat = iface.ref['initial_joint_pos'], iface.ref['object_pos'], iface.ref['object_quat']
            else:
                q, pos, quat = r['q'][k-1], r['object_pos'][k-1], r['object_quat'][k-1]
                iface.targets = r['targets'][k-1]
                info = iface.update(q, pos, quat)
                self.assertIsNone(info['outcome'], f'step {k-1}: tracker ended the episode, Isaac did not')
                self.assertEqual(info['stage'], r['stage'][k-1], f'stage after step {k-1}')
            contact = r['obs'][k][self.cols['fingertip_object_contact']]   # needs PhysX contacts: taken as recorded
            yield k, iface.observation(q, pos, quat_matrix(quat)[:, 2], contact)

    def test_rollout_covers_every_stage(self):
        r = self.rollout
        self.assertEqual(self.iface.ref['task'], 'gear3_split_step1')
        self.assertTrue(self.iface.ref['yaw_free_goal'])
        self.assertEqual(set(r['stage'].tolist()), {0, 1, 2})
        self.assertGreaterEqual(int(r['done'].sum()), 2)

    def test_observation_and_stage_match_isaac(self):
        worst = {name: 0. for name, _ in self.iface.fields}
        for k, obs in self.replay():
            diff = np.abs(obs - self.rollout['obs'][k])
            for name, cols in self.cols.items():
                worst[name] = max(worst[name], float(diff[cols].max()))
        # float32 Isaac state vs float64 here; positions are metres, 1e-4 = 0.1 mm.
        for name, err in worst.items():
            with self.subTest(field=name):
                self.assertLess(err, 1e-4, f'{name}: max |rebuilt - isaac| = {err:.2e}')

    def test_action_targets_match_isaac(self):
        r, iface = self.rollout, self.iface
        initial = np.array(iface.ref['initial_joint_pos'])
        self.assertAlmostEqual(iface.smoothing, .36)          # 30 Hz filter, not control['smoothing'] (60 Hz)
        err = 0.
        for k in range(len(r['action'])):
            iface.targets = initial if k == 0 or r['done'][k-1] else r['targets'][k-1]
            targets = iface.apply_action(r['action'][k])
            if r['done'][k]:            # the episode ended: Isaac reset the targets
                np.testing.assert_allclose(r['targets'][k], initial, atol=1e-6)
                continue
            err = max(err, float(np.abs(targets - r['targets'][k]).max()))
        self.assertLess(err, 1e-5, f'max |targets - isaac targets| = {err:.2e} rad')

    def test_policy_matches_isaac(self):
        """The LSTM run step by step over the recorded observations, reset after every episode end."""
        actor = RslRlActor(MODEL)
        self.assertEqual((actor.kind, actor.obs_dim, actor.action_dim), ('lstm', self.iface.obs_dim, 23))
        err = 0.
        for o, a, done in zip(self.rollout['obs'], self.rollout['action'], self.rollout['done']):
            err = max(err, float(np.abs(actor(o) - a).max()))
            if done:
                actor.reset()
        self.assertLess(err, 1e-4, f'max |action - isaac action| = {err:.2e}')

    def test_closed_loop_policy_on_rebuilt_observations(self):
        """Policy fed with rebuilt observations (own stage tracker) reproduces Isaac's actions end to end."""
        actor = RslRlActor(MODEL)
        err = 0.
        for k, obs in self.replay():
            if k > 0 and self.rollout['done'][k-1]:
                actor.reset()
            # Compared after the [-1, 1] clamp the robot applies. The rebuilt observations differ from Isaac's by
            # float32 rounding (<1e-6), which the recurrent policy (raw outputs up to ~55) amplifies to a few 1e-3:
            # ~1e-4 rad of arm target per step, far below the trained observation noise.
            err = max(err, float(np.abs(np.clip(actor(obs), -1, 1) - np.clip(self.rollout['action'][k], -1, 1)).max()))
        self.assertLess(err, 5e-3, f'max |action(rebuilt obs) - isaac action| = {err:.2e}')

    def test_yaw_does_not_count_but_tilt_does(self):
        iface = self.iface
        iface.reset()
        pos, _ = iface.goal
        yaw = np.array([np.cos(np.radians(3.5)), 0., 0., np.sin(np.radians(3.5))])     # 7 deg about z
        tilt = np.array([np.cos(np.radians(3.5)), np.sin(np.radians(3.5)), 0., 0.])    # 7 deg about x
        self.assertLess(iface.keypoint_distance(pos, yaw), 1e-9)
        self.assertGreater(iface.keypoint_distance(pos, tilt), .003)

    def test_stage_needs_five_near_steps_and_switches_goal(self):
        iface = self.iface
        iface.reset()
        q = iface.ref['initial_joint_pos']
        pos, quat = iface.goal
        for step in range(4):
            self.assertEqual(iface.update(q, pos, quat)['stage'], 0)
        info = iface.update(q, pos, quat)
        self.assertEqual(info['stage'], 1)
        np.testing.assert_allclose(iface.goal[0], iface.ref['final_pos'])


if __name__ == '__main__':
    unittest.main()
