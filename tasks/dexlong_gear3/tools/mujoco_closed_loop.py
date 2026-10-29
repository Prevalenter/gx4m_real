"""Run the step0 deployment code closed-loop against gx4m_learning's MuJoCo replica as stand-in hardware.

MuJoCo only provides physics: measured joints, gear pose and fingertip contacts. Observation, policy and
joint targets come from tasks/dexlong_gear3/step0.py and gx4m_real.policy, as on the real robot. Each step
also compares the rebuilt observation with the replica's own.

Needs gx4m_learning (scripts/sim2sim_gear3_split_mujoco.py and its MJCF assets) and MuJoCo:
    C:/mjfit/python.exe tasks/dexlong_gear3/tools/mujoco_closed_loop.py --learning ../gx4m_learning --episodes 20
"""
import argparse
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]

from gx4m_real.policy import RslRlActor  # noqa: E402
from gx4m_real.robot.kinematics import quat_matrix  # noqa: E402
from step0 import Step0Interface, load_reference  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--learning', type=Path, required=True, help='gx4m_learning checkout')
    parser.add_argument('--model', type=Path, default=HERE.parent/'models'/'step0_mlp_20261027_iter750.pt')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--substeps', type=int, default=20, help='MuJoCo substeps per 1/60 s step')
    parser.add_argument('--position-noise', type=float, default=.02, help='Gear start offset, uniform +-m in x, y')
    parser.add_argument('--random-yaw', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--zero-contact', action='store_true', help='Feed 0 for every fingertip contact')
    parser.add_argument('--arm-speed-scale', type=float, default=None, help='Override the trained 3 rad/s')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.learning.resolve()/'scripts'))
    import mujoco
    import sim2sim_gear3_split_mujoco as replica

    ref = load_reference()
    nominal_pos, nominal_quat = np.array(ref['object_pos']), np.array(ref['object_quat'])
    lift = ref['preinsert_pos'][2] - ref['object_pos'][2]
    env = replica.MujocoGearEnv(ref, 0, args.substeps)
    iface, actor = Step0Interface(ref), RslRlActor(args.model)
    rng = np.random.default_rng(args.seed)
    threshold = ref['binary_tactile_threshold_n']
    print(f'{actor.kind} actor {args.model.name}; {args.episodes} episodes, start noise +-{args.position_noise} m, '
          f'random yaw {args.random_yaw}, zero contact {args.zero_contact}, substeps {args.substeps}', flush=True)
    outcomes, obs_err = [], 0.
    for episode in range(args.episodes):
        pos = nominal_pos + (*rng.uniform(-args.position_noise, args.position_noise, 2), 0.)
        half = rng.uniform(0, math.pi) if args.random_yaw else 0.
        quat = replica.quat_mul(np.array([math.cos(half), 0., 0., math.sin(half)]), nominal_quat)
        ref['object_pos'], ref['object_quat'] = pos.tolist(), quat.tolist()
        ref['preinsert_pos'] = (pos + (0., 0., lift)).tolist()      # the replica's goal follows the gear too
        env.reset()
        mujoco.mj_forward(env.model, env.data)
        measured_pos, _ = env.object_pose()
        iface.reset(measured_pos)
        actor.reset()
        for step in range(ref['p2p']['episode_steps']):
            q = env.data.qpos[env.qadr].copy()
            obj_pos, obj_quat = env.object_pose()
            contact = (env.fingertip_object_forces() > threshold).astype(float)
            if args.zero_contact:
                contact[:] = 0.
            obs = iface.observation(q, obj_pos, quat_matrix(obj_quat)[:, 2], contact)
            if not args.zero_contact:
                obs_err = max(obs_err, float(np.abs(obs - env.observation()).max()))
            targets = iface.apply_action(actor(obs), args.arm_speed_scale)
            env.targets = targets                       # drive the replica with our targets
            env.data.ctrl[env.act] = targets
            for _ in range(env.substeps):
                mujoco.mj_step(env.model, env.data)
                v = env.data.qvel[env.vadr]
                env.data.qvel[env.vadr] = np.clip(v, -env.velocity_limit, env.velocity_limit)
            info = stage_update(env, replica)
            if info['done']:
                break
        outcomes.append((info['outcome'], step + 1, info['distance'], obj_pos[2]))
        print(f'episode {episode}: {info["outcome"]:8s} after {step + 1:3d} steps, keypoint distance '
              f'{info["distance"]*1000:5.1f} mm, gear z {env.object_pose()[0][2]:.3f} m', flush=True)
    success = sum(o[0] == 'success' for o in outcomes)
    print(f'\nsuccess {success}/{len(outcomes)}', flush=True)
    if not args.zero_contact:
        print(f'max |rebuilt obs - replica obs| = {obs_err:.2e}', flush=True)


def stage_update(env, replica):
    """The replica's step0 stage and termination logic after the physics step (MujocoGearEnv.step without control)."""
    p = env.p2p
    distance = env.keypoint_distance()
    obj_pos, obj_quat = env.object_pose()
    env.stage_steps += 1
    if distance <= p['insertion_success_tolerance']*p['keypoint_scale']:
        env.near_steps += 1
    success = env.near_steps >= p['success_steps']
    grasp = obj_pos + replica.rotate(obj_quat, np.array(env.ref['grasp_position']))
    tips = np.linalg.norm(env.fingertips() - grasp, axis=-1)
    fall = obj_pos[2] < p['fall_height']
    hand_far = tips.max() > p['hand_far_distance']
    timeout = env.stage_steps >= p['episode_steps']
    outcome = 'success' if success else 'fall' if fall else 'hand_far' if hand_far else 'timeout' if timeout else None
    return dict(done=outcome is not None, outcome=outcome, distance=distance)


if __name__ == '__main__':
    main()
