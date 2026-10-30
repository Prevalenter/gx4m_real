"""Synthetic end-to-end check: perturb the URDF chain, generate pivot poses by IK, calibrate, compare.

  python apps/fk_calibration/selftest_synthetic.py                # joint zero errors only
  python apps/fk_calibration/selftest_synthetic.py --truth full   # plus link geometry errors
  python apps/fk_calibration/selftest_synthetic.py --no-elevated  # only the four coplanar seats
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibrate import MODELS, Data, evaluate, solve
from chain import ArmChain
from gx4m_real.robot.config import load_config
from gx4m_real.robot.kinematics import quat_matrix

TRUE_TIP = np.array([.004, -.003, .17])


def ik(chain, errors, tip, centre, direction, seeds, limits, rng, tries=30):
    """Joint angles that put the probe ball at `centre` with the stem pointing along `direction`.

    Each try starts from one of `seeds` (nearby reachable postures) and pulls the redundant DOF
    towards it, so elbow and wrist configurations vary between poses.
    """
    def residual(q):
        T = chain.fk(q, errors)[0]
        stem = -T[:3, :3] @ tip
        return np.concatenate([T[:3, :3] @ tip + T[:3, 3] - centre, .1 * (stem / np.linalg.norm(stem) - direction)])

    for _ in range(tries):
        reference = seeds[rng.integers(len(seeds))]
        q, damping = reference.copy(), 1e-3
        r = residual(q)
        for _ in range(300):
            if np.linalg.norm(r) < 1e-11:
                break
            J = np.stack([(residual(q + h) - r) / 1e-7 for h in np.eye(7) * 1e-7], axis=1)
            J_pinv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(6))
            candidate = q - J_pinv @ r + .2 * (np.eye(7) - J_pinv @ J) @ (reference - q)
            r_new = residual(candidate)
            if np.linalg.norm(r_new) < np.linalg.norm(r):
                q, r, damping = candidate, r_new, max(damping / 3, 1e-12)
            else:
                damping *= 4
        if np.linalg.norm(r) < 1e-9 and np.all((q > limits[0]) & (q < limits[1])):
            return q
    raise RuntimeError('IK failed')


def generate(chain, limits, up, centre, *, truth='offsets', poses_per_seat=25, elevated=True, seed=0,
             q_noise=1e-4, seat_noise=3e-5, distance_noise=3e-5):
    rng = np.random.default_rng(seed)
    errors = np.zeros((7, 7))
    errors[1:6, 0] = np.deg2rad(rng.normal(0, .3, 5))
    if truth == 'full':
        errors[1:7, 1:4] = rng.normal(0, 5e-4, (6, 3))
        errors[1:7, 4:7] = np.deg2rad(rng.normal(0, .1, (6, 3)))
    x = np.cross(up, [1., 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(up, x)
    seats = {name: centre + (sx - .12) * x + (sy - .095) * y for name, (sx, sy) in
             {'H1': (.025, .025), 'H2': (.215, .025), 'H3': (.025, .165), 'H4': (.215, .165)}.items()}
    distances = [(a, b, np.linalg.norm(seats[a] - seats[b]) + rng.normal(0, distance_noise))
                 for a, b in [('H1', 'H2'), ('H3', 'H4'), ('H1', 'H3'), ('H2', 'H4'), ('H1', 'H4'), ('H2', 'H3')]]
    if elevated:
        seats['S1'] = centre + .1 * up + .03 * x
    pool = rng.uniform(limits[0], limits[1], (100000, 7))     # reachable postures to start IK from
    T = chain.fk(pool)
    pool_tips, pool_stems = T[:, :3, :3] @ TRUE_TIP + T[:, :3, 3], -(T[:, :3, :3] @ TRUE_TIP) / np.linalg.norm(TRUE_TIP)
    q, names = [], []
    for name, c in seats.items():
        count = 0
        for _ in range(poses_per_seat * 5):
            if count == poses_per_seat:
                break
            tilt, azimuth = np.deg2rad(rng.uniform(0, 50)), rng.uniform(0, 2 * np.pi)
            direction = np.cos(tilt) * up + np.sin(tilt) * (np.cos(azimuth) * x + np.sin(azimuth) * y)
            score = np.linalg.norm(pool_tips - c, axis=1) + .1 * np.linalg.norm(pool_stems - direction, axis=1)
            seeds = pool[np.argsort(score)[:20]]
            try:
                q.append(ik(chain, errors, TRUE_TIP, c + rng.normal(0, seat_noise, 3), direction, seeds, limits, rng,
                            tries=5))
            except RuntimeError:
                continue                            # this tilt is out of reach for this seat
            names.append(name)
            count += 1
        if count < poses_per_seat:
            raise RuntimeError(f'{name}: only {count} reachable poses')
    q = np.array(q) + rng.normal(0, q_noise, (len(q), 7))
    validation = rng.random(len(q)) < .2
    return Data(q, names, distances, 1e-4, 5e-5), validation, errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--robot-config', type=Path, default=HERE.parent / 'control' / 'config.yaml')
    parser.add_argument('--truth', choices=('offsets', 'full'), default='offsets')
    parser.add_argument('--poses-per-seat', type=int, default=25)
    parser.add_argument('--no-elevated', dest='elevated', action='store_false')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    config = load_config(args.robot_config)
    chain = ArmChain(config.urdf)
    R_world_base = quat_matrix(config.base_wxyz)
    up = R_world_base.T @ [0., 0., 1.]
    # Put the plate where the task happens: the gear object position of initial_pose.json.
    raw = yaml.safe_load(args.robot_config.read_text())
    pose = json.loads((args.robot_config.parent / raw['model']['initial_pose']).read_text())
    centre = R_world_base.T @ (np.asarray(pose['object_position']) - config.base_position)
    limits = config.compensation.model_limits(config.lower, config.upper)
    data, validation, truth = generate(chain, limits, up, centre, truth=args.truth,
                                       poses_per_seat=args.poses_per_seat, elevated=args.elevated, seed=args.seed)
    train, val = data.subset(~validation), data.subset(validation)
    print(f'{len(data.q)} poses ({int(validation.sum())} held out), truth = {args.truth}')
    print(f'{"model":8s} {"rank":>9s} {"train RMS":>10s} {"val RMS":>8s} {"val max":>8s} {"dist RMS":>9s} {"dθ err max":>11s}')
    worst = {}
    for model in MODELS:
        result = solve(chain, train, model)
        tr, va = evaluate(chain, result, train), evaluate(chain, result, val)
        dtheta = np.rad2deg(np.abs(result.errors[1:6, 0] - truth[1:6, 0])).max() if model != 'nominal' else np.nan
        worst[model] = va['pivot_rms_mm']
        print(f'{model:8s} {result.rank:4d}/{result.n_unknowns:<4d} {tr["pivot_rms_mm"]:8.3f}mm {va["pivot_rms_mm"]:6.3f}mm '
              f'{va["pivot_max_mm"]:6.3f}mm {tr["distance_rms_mm"]:7.3f}mm {dtheta:9.4f}°')
        if model == 'full':
            weak = [result.param_label(j) for j in range(len(result.free)) if result.identified[j] < .5]
            print(f'         {len(weak)} 个参数基本不可辨识（保持名义值）：{", ".join(weak)}')
    print('\ntrue dθ J2–J6 (deg):', np.rad2deg(truth[1:6, 0]).round(4).tolist())
    return worst


if __name__ == '__main__':
    main()
