"""Step1 / step2 interface and policy against deterministic Isaac Lab rollouts (no Isaac, no hardware).

tests/data/step1_isaac_rollout.npz is gx4m_learning's export_gear3_split_sim2sim_reference.py --step 1 output
for models/step1_lstm_20261103_iter4750.pt (domain randomization off, nominal start, 400 steps, three
episodes): per step k the Isaac policy observation obs[k], its action[k], and the joint positions q[k],
joint targets targets[k], gear pose and stage after that action, which the next observation obs[k+1] is
built from. On a step with done[k] the recorded state is already the reset one.

step2_isaac_rollout.npz is the --step 2 output for the occlusion-aware models/step2_lstm_20261105_iter4500.pt with
its pose estimator (600 steps, the camera losing the gear in every episode; this early checkpoint does not finish
the task). It also records what produced each observation: the tracked gear pose (held while lost), valid flag and
age, pad forces, 0/1 contact, and the estimator's input and output.

tests/data/step{1,2}_isaac_reference.json is the reference exported with each rollout. The deployed
config/step{1,2}_reference.json may differ from it in the robot base pose (moved after the export), which the
observations depend on, so the rollouts are checked against their own export.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from gx4m_real.policy import RslRlActor  # noqa: E402
from gx4m_real.robot.kinematics import quat_matrix  # noqa: E402
from step1 import ESTIMATORS, MODELS, REFERENCES, Step1Interface, load_reference  # noqa: E402


def slices(fields):
    out, start = {}, 0
    for name, width in fields:
        out[name] = slice(start, start + width)
        start += width
    return out


class Step1RolloutTests(unittest.TestCase):
    STEP, SMOOTHING = 1, .36        # finger filter at 30 Hz, not control['smoothing'] (60 Hz)

    @classmethod
    def setUpClass(cls):
        cls.rollout = dict(np.load(HERE/'data'/f'step{cls.STEP}_isaac_rollout.npz'))
        cls.model = MODELS[cls.STEP]
        cls.iface = Step1Interface(load_reference(HERE/'data'/f'step{cls.STEP}_isaac_reference.json'))
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

    def test_deployed_reference_matches_export(self):
        deployed = load_reference(REFERENCES[self.STEP])
        for key in ('base_pos', 'base_quat'):
            deployed.pop(key)
        self.assertEqual(deployed, {k: v for k, v in self.iface.ref.items() if k not in ('base_pos', 'base_quat')})

    def test_rollout_covers_every_stage(self):
        r = self.rollout
        self.assertEqual(self.iface.ref['task'], f'gear3_split_step{self.STEP}')
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
        self.assertAlmostEqual(iface.smoothing, self.SMOOTHING)
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
        actor = RslRlActor(self.model)
        self.assertEqual((actor.kind, actor.obs_dim, actor.action_dim), ('lstm', self.iface.obs_dim, 23))
        err = 0.
        for o, a, done in zip(self.rollout['obs'], self.rollout['action'], self.rollout['done']):
            err = max(err, float(np.abs(actor(o) - a).max()))
            if done:
                actor.reset()
        self.assertLess(err, 1e-4, f'max |action - isaac action| = {err:.2e}')

    def test_closed_loop_policy_on_rebuilt_observations(self):
        """Policy fed with rebuilt observations (own stage tracker) reproduces Isaac's actions end to end."""
        actor = RslRlActor(self.model)
        err = 0.
        for k, obs in self.replay():
            if k > 0 and self.rollout['done'][k-1]:
                actor.reset()
            # Compared after the [-1, 1] clamp the robot applies. The rebuilt observations differ from Isaac's by
            # float32 rounding (<1e-6), which the recurrent policy (raw outputs up to ~55) amplifies to a few 1e-3:
            # ~1e-4 rad of arm target per step, far below the trained observation noise.
            err = max(err, float(np.abs(np.clip(actor(obs), -1, 1) - np.clip(self.rollout['action'][k], -1, 1)).max()))
        self.assertLess(err, 5e-3, f'max |action(rebuilt obs) - isaac action| = {err:.2e}')

    def test_palm_rot_keeps_its_sign_across_w_zero(self):
        """Isaac's palm quaternion changes continuously with the joints; joint3 sweeps its w through zero."""
        q, previous = np.array(self.iface.ref['initial_joint_pos']), None
        for angle in np.linspace(self.iface.lower[2], self.iface.upper[2], 81):
            q[2] = angle
            poses = self.iface.fk.link_poses(dict(zip(self.iface.joint_order, q)))
            quat = self.iface.palm_and_tips(q)[1]
            np.testing.assert_allclose(quat_matrix(quat), poses[self.iface.palm_body][:3, :3], atol=1e-9)
            if previous is not None:
                self.assertGreater(float(quat @ previous), .9)
            previous = quat
        self.assertLess(previous[0], 0.)

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


class Step2EstimatorRolloutTests(unittest.TestCase):
    """step2 (gear48 onto the right pin) with the gear pose estimator, against the recorded Isaac rollout."""
    STEP = 2

    @classmethod
    def setUpClass(cls):
        cls.rollout = dict(np.load(HERE/'data'/f'step{cls.STEP}_isaac_rollout.npz'))
        cls.reference = load_reference(HERE/'data'/f'step{cls.STEP}_isaac_reference.json')
        cls.cols = slices(list(cls.reference['policy_fields'].items()))

    def interface(self):
        iface = Step1Interface(self.reference, estimator=ESTIMATORS[self.STEP])
        inputs = []
        step = iface.estimator.step
        iface.estimator.step = lambda x: (inputs.append(np.array(x)), step(x))[1]
        return iface, inputs

    def replay(self, iface):
        """Yield (k, rebuilt observation k) from the recorded tracked pose, valid flag, age and pad forces, the
        stage tracker fed the recorded true states as in Isaac."""
        r, pe = self.rollout, self.reference['pose_estimator']
        for k in range(len(r['obs'])):
            if k == 0 or r['done'][k-1]:
                iface.reset()
                q = iface.ref['initial_joint_pos']
            else:
                q = r['q'][k-1]
                iface.targets = r['targets'][k-1]
                iface.update(q, r['object_pos'][k-1], r['object_quat'][k-1])
            axis = quat_matrix(r['tracked_quat'][k])[:, 2]
            yield k, iface.observation(q, r['tracked_pos'][k], axis, r['contact'][k], valid=bool(r['valid'][k]),
                                       age=float(r['age'][k])*pe['max_age_s'],
                                       pad_force=r['pad_force'][k].reshape(4, 3)*pe['tactile_force_scale'])

    def test_deployed_reference_matches_export(self):
        self.assertEqual(load_reference(REFERENCES[self.STEP]), self.reference)
        self.assertEqual(sum(self.reference['policy_fields'].values()), 80)

    def test_rollout_loses_the_gear(self):
        r = self.rollout
        self.assertTrue(r['est_first'][0])
        self.assertGreater(r['valid'].mean(), .05)
        self.assertLess(r['valid'].mean(), .95)

    def test_estimator_input_output_and_observation_match_isaac(self):
        iface, inputs = self.interface()
        r = self.rollout
        worst = dict(input=0., pos=0., axis=0., log_std=0.)
        fields = {name: 0. for name in self.cols}
        for k, obs in self.replay(iface):
            worst['input'] = max(worst['input'], float(np.abs(inputs[-1] - r['est_x'][k]).max()))
            pos, axis = iface.estimate[0], iface.estimate[1]
            worst['log_std'] = max(worst['log_std'], float(np.abs(iface.estimate[2] - r['est_log_std'][k]).max()))
            diff = np.abs(obs - r['obs'][k])
            for name, cols in self.cols.items():
                fields[name] = max(fields[name], float(diff[cols].max()))
        # float32 Isaac state vs float64 here: 1e-4 is 0.01 mm of an input in dm, 0.1 mm of a position in m.
        self.assertLess(worst['input'], 1e-4, f'estimator input: {worst["input"]:.2e}')
        self.assertLess(worst['log_std'], 1e-3, f'estimator log std: {worst["log_std"]:.2e}')
        for name, err in fields.items():
            with self.subTest(field=name):
                self.assertLess(err, 1e-4 if name != 'object_pose_std' else 1e-3, f'{name}: {err:.2e}')

    def test_action_targets_match_isaac(self):
        r = self.rollout
        iface = Step1Interface(self.reference, estimator=ESTIMATORS[self.STEP])
        initial = np.array(iface.ref['initial_joint_pos'])
        self.assertAlmostEqual(iface.smoothing, .1)
        err = 0.
        for k in range(len(r['action'])):
            iface.targets = initial if k == 0 or r['done'][k-1] else r['targets'][k-1]
            targets = iface.apply_action(r['action'][k])
            if r['done'][k]:
                continue
            err = max(err, float(np.abs(targets - r['targets'][k]).max()))
        self.assertLess(err, 1e-5, f'max |targets - isaac targets| = {err:.2e} rad')

    def test_policy_matches_isaac(self):
        actor = RslRlActor(MODELS[self.STEP])
        self.assertEqual((actor.kind, actor.obs_dim, actor.action_dim), ('lstm', 80, 23))
        err = 0.
        for o, a, done in zip(self.rollout['obs'], self.rollout['action'], self.rollout['done']):
            err = max(err, float(np.abs(actor(o) - a).max()))
            if done:
                actor.reset()
        self.assertLess(err, 1e-4, f'max |action - isaac action| = {err:.2e}')

    def test_estimated_pose_has_the_estimated_axis(self):
        iface, _ = self.interface()
        for k, _ in self.replay(iface):
            if k == 30:
                break
        pos, quat = iface.estimated_pose()
        np.testing.assert_allclose(quat_matrix(quat)[:, 2], iface.estimate[1], atol=1e-9)
        np.testing.assert_allclose(pos, iface.estimate[0])


class TrainingUrdfTests(unittest.TestCase):
    def test_each_step_uses_its_training_urdf(self):
        self.assertEqual(Step1Interface(load_reference(REFERENCES[1])).urdf.name, 'gx7_right_gx4m_cam_xl330.urdf')
        self.assertEqual(Step1Interface(load_reference(REFERENCES[2]), estimator=ESTIMATORS[2]).urdf.name,
                         'gx7_right_gx4m_cam_xl330_pad.urdf')

    def test_pad_urdf_has_the_same_palm_and_fingertips(self):
        ref = load_reference(REFERENCES[2])
        plain = Step1Interface(ref, urdf=Step1Interface(load_reference(REFERENCES[1])).urdf, estimator=ESTIMATORS[2])
        pad = Step1Interface(ref, estimator=ESTIMATORS[2])
        q = np.asarray(ref['initial_joint_pos']) + np.random.default_rng(0).uniform(-.2, .2, len(ref['joint_order']))
        for a, b in zip(plain.palm_and_tips(q), pad.palm_and_tips(q)):
            np.testing.assert_allclose(a, b, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
