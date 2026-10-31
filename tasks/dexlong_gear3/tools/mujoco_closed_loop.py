"""Run the step1 deployment code closed-loop against gx4m_learning's MuJoCo replica as stand-in hardware.

MuJoCo only provides physics: measured joints, gear pose and fingertip contacts. Observation, policy, joint
targets and the stage logic come from tasks/dexlong_gear3/step1.py and gx4m_real.policy, as on the real robot.
Each step also compares the rebuilt observation and keypoint distance with the replica's own.

Needs gx4m_learning (scripts/sim2sim_gear3_split_mujoco.py and its MJCF assets) and MuJoCo:
    C:/mjfit/python.exe tasks/dexlong_gear3/tools/mujoco_closed_loop.py --learning ../gx4m_learning --episodes 20
"""
import argparse
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE.parent)]

from gx4m_real.policy import RslRlActor  # noqa: E402
from gx4m_real.robot.kinematics import quat_matrix  # noqa: E402
from step1 import Step1Interface, load_reference  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--learning', type=Path, required=True, help='gx4m_learning checkout')
    parser.add_argument('--model', type=Path, default=HERE.parent/'models'/'step1_lstm_20261031_iter500.pt')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--substeps', type=int, default=40, help='MuJoCo substeps per 1/30 s step (1200 Hz)')
    parser.add_argument('--position-noise', type=float, default=.02, help='Gear start offset, uniform +-m in x, y')
    parser.add_argument('--fixed-start', action='store_true', help='Nominal gear start pose (no offset, no yaw)')
    parser.add_argument('--zero-contact', action='store_true', help='Feed 0 for every fingertip contact')
    parser.add_argument('--arm-speed-scale', type=float, default=None, help='Override the trained arm speed (reference control.arm_speed_scale)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.learning.resolve()/'scripts'))
    import mujoco
    import sim2sim_gear3_split_mujoco as replica

    ref = load_reference()
    env = replica.MujocoGearEnv(ref, 1, args.substeps)
    iface, actor = Step1Interface(ref), RslRlActor(args.model)
    rng = None if args.fixed_start else np.random.default_rng(args.seed)
    threshold = ref['binary_tactile_threshold_n']
    print(f'{actor.kind} actor {args.model.name}; {args.episodes} episodes, '
          f'{"fixed start" if rng is None else f"start +-{args.position_noise} m and any yaw"}, '
          f'zero contact {args.zero_contact}, substeps {args.substeps}', flush=True)
    outcomes, obs_err, dist_err = [], 0., 0.
    for episode in range(args.episodes):
        env.reset(rng, args.position_noise)
        iface.reset()
        actor.reset()
        max_stage, steps = 0, 0
        while True:
            q = env.data.qpos[env.qadr].copy()
            obj_pos, obj_quat = env.object_pose()
            contact = (env.fingertip_object_forces() > threshold).astype(float)
            if args.zero_contact:
                contact[:] = 0.
            obs = iface.observation(q, obj_pos, quat_matrix(obj_quat)[:, 2], contact)
            env.stage = iface.stage                     # the replica's goal follows our stage
            if not args.zero_contact:
                obs_err = max(obs_err, float(np.abs(obs - env.observation()).max()))
            targets = iface.apply_action(actor(obs), args.arm_speed_scale)
            env.targets = targets                       # drive the replica with our targets
            env.data.ctrl[env.act] = targets
            for _ in range(env.substeps):
                mujoco.mj_step(env.model, env.data)
                v = env.data.qvel[env.vadr]
                env.data.qvel[env.vadr] = np.clip(v, -env.velocity_limit, env.velocity_limit)
            obj_pos, obj_quat = env.object_pose()
            env.stage = iface.stage
            dist_err = max(dist_err, abs(env.keypoint_distance() - iface.keypoint_distance(obj_pos, obj_quat)))
            info = iface.update(env.data.qpos[env.qadr].copy(), obj_pos, obj_quat)
            max_stage, steps = max(max_stage, info['stage']), steps + 1
            if info['outcome']:
                break
        outcomes.append(info['outcome'])
        print(f'episode {episode}: {info["outcome"]:8s} after {steps:3d} steps, max stage {max_stage}, keypoint distance '
              f'{info["distance"]*1000:5.1f} mm, closest fingertip {info["away"]*100:4.1f} cm from the hub', flush=True)
    print(f'\nsuccess {outcomes.count("success")}/{len(outcomes)}', flush=True)
    if not args.zero_contact:
        # The replica reads body poses (d.xpos) one substep behind qpos (mj_step computes kinematics before it
        # integrates): ~1 mm on the palm and fingertip fields while the arm moves; 1e-16 after mj_forward.
        print(f'max |rebuilt obs - replica obs| = {obs_err:.2e}', flush=True)
    print(f'max |keypoint distance - replica keypoint distance| = {dist_err:.2e} m', flush=True)


if __name__ == '__main__':
    main()
