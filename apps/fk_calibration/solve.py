"""Calibrate GX7 FK from one or more pivot sessions recorded by collect_app.py.

  python apps/fk_calibration/solve.py runtime/sessions/fk_20261030_150000 [more sessions ...]
      [--holdout-seat H4] [--models nominal offsets full] [--reject-outliers]

Seat centres are per session, so move the plate only between sessions. Distances come from
plate.yaml at solve time, so they may be measured after the data was collected.
Writes <first session>/calib_<timestamp>/ with result.json and, per model, a joint_compensation.yaml
(and for `full` a calibrated URDF). Nothing in the repository is overwritten.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibrate import MODELS, Data, evaluate, solve
from chain import ArmChain, PARAM_NAMES
from gx4m_real.robot.compensation import JointCompensation
from gx4m_real.robot.config import ARM_JOINTS, load_config
from session import file_sha256, load_plate, load_session, session_compensation


def load_data(paths, plate, urdf):
    q, seats, validation, compensation, sessions = [], [], [], None, []
    urdf_hash = file_sha256(urdf)
    for path in paths:
        metadata, samples = load_session(path)
        name = Path(path).resolve().name
        if metadata.get('mock'):
            print(f'警告：{name} 是模拟模式数据，不能用于真机标定')
        if metadata.get('urdf_sha256') != urdf_hash:
            print(f'警告：{name} 采集时的 URDF 与当前 {urdf} 不同')
        comp = session_compensation(metadata)
        if compensation is None:
            compensation = comp
        elif comp != compensation:
            raise ValueError('各会话采集时的关节补偿不同，不能合并；请分开标定')
        for s in samples:
            q.append(comp.to_model(s['motor_positions_rad']))
            seats.append(f"{name}/{s['seat']}")
            validation.append(bool(s.get('validation')))
        sessions.append({'path': str(Path(path).resolve()), 'samples': len(samples)})
    distances = [(f'{s}/{a}', f'{s}/{b}', d) for s in (Path(p).resolve().name for p in paths)
                 for (a, b), d in plate['distances_m'].items()]
    data = Data(np.array(q), seats, distances, plate['sigma_pivot_m'], plate['sigma_distance_m'])
    return data, np.array(validation), compensation, sessions


def held_out(seats, specs):
    """A spec is 'H4' (that seat in every session) or '<session>/H4'."""
    return np.array([any(s == spec or s.split('/', 1)[1] == spec for spec in specs) for s in seats], dtype=bool)


def joint_table(result):
    rows = []
    for i, name in enumerate(ARM_JOINTS):
        e = result.errors[i]
        if not np.any(e):
            continue
        rows.append(f'  J{i + 1}: dθ {np.rad2deg(e[0]):+.4f}°  dxyz [{", ".join(f"{v * 1000:+.3f}" for v in e[1:4])}] mm  '
                    f'drot [{", ".join(f"{np.rad2deg(v):+.4f}" for v in e[4:7])}]°')
    return rows


def summarise(result, train_metrics, val_metrics):
    params = [{'joint': ARM_JOINTS[i], 'param': PARAM_NAMES[k],
               'value': float(np.rad2deg(result.errors[i, k]) if k in (0, 4, 5, 6) else result.errors[i, k] * 1000),
               'unit': 'deg' if k in (0, 4, 5, 6) else 'mm',
               'std': float(np.rad2deg(s) if k in (0, 4, 5, 6) else s * 1000), 'identified': float(f)}
              for (i, k), s, f in zip(result.free, result.std, result.identified)]
    strip = lambda m: {k: v for k, v in m.items() if k != 'sample_errors_mm'}
    return {'model': result.model, 'rank': result.rank, 'unknowns': result.n_unknowns,
            'iterations': result.iterations, 'parameters': params,
            'tip_offset_mm': (result.tip * 1000).tolist(),
            'seat_centres_mm': {k: (v * 1000).tolist() for k, v in result.centres.items()},
            'singular_values': result.singular_values.tolist(),
            'train': strip(train_metrics), 'validation': strip(val_metrics) if val_metrics else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sessions', nargs='+', type=Path)
    parser.add_argument('--robot-config', type=Path, default=HERE.parent / 'control' / 'config.yaml')
    parser.add_argument('--plate', type=Path, default=HERE / 'plate.yaml')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--holdout-seat', nargs='*', default=[], help="e.g. H4 or fk_20261030_150000/H4")
    parser.add_argument('--reject-outliers', action='store_true', help='drop training samples > 3×RMS once and refit')
    parser.add_argument('--rcond', type=float, default=1e-3, help='relative SVD cut-off for unidentifiable directions')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)

    config = load_config(args.robot_config)
    chain = ArmChain(config.urdf)
    plate = load_plate(args.plate)
    data, validation, compensation, sessions = load_data(args.sessions, plate, config.urdf)
    holdout = held_out(data.seats, args.holdout_seat)
    train_mask = ~validation & ~holdout
    if not plate['distances_m']:
        print('警告：plate.yaml 没有实测孔距，尺度不可观，只能修正角度类误差')
    print(f'{len(data.q)} 个样本：训练 {int(train_mask.sum())}，验证 {int((~train_mask).sum())}；'
          f'座：{", ".join(dict.fromkeys(data.seats))}')

    out = args.out or (args.sessions[0].resolve() / f'calib_{datetime.now():%Y%m%d_%H%M%S}')
    out.mkdir(parents=True, exist_ok=False)
    report = {'created': datetime.now().isoformat(timespec='seconds'), 'sessions': sessions,
              'urdf': str(config.urdf), 'plate': plate['raw'], 'holdout_seats': args.holdout_seat,
              'joint_compensation_at_collection': {'directions': list(compensation.directions),
                                                   'offsets_deg': list(compensation.offsets_deg)},
              'models': {}}
    print(f'\n{"模型":6s} {"秩/未知数":>9s} {"训练RMS":>9s} {"验证RMS":>9s} {"验证max":>9s} {"孔距RMS":>9s}')
    for model in args.models:
        train = data.subset(train_mask)
        result = solve(chain, train, model, rcond=args.rcond)
        train_metrics = evaluate(chain, result, train)
        if args.reject_outliers:
            errors = train_metrics['sample_errors_mm']
            keep = errors <= 3 * train_metrics['pivot_rms_mm']
            if not keep.all():
                print(f'  {model}: 剔除 {int((~keep).sum())} 个离群样本后重算')
                train = train.subset(keep)
                result = solve(chain, train, model, rcond=args.rcond)
                train_metrics = evaluate(chain, result, train)
        val_metrics = evaluate(chain, result, data.subset(~train_mask)) if (~train_mask).any() else None
        v = val_metrics or {'pivot_rms_mm': float('nan'), 'pivot_max_mm': float('nan')}
        print(f'{model:8s} {result.rank:4d}/{result.n_unknowns:<4d} {train_metrics["pivot_rms_mm"]:7.3f}mm '
              f'{v["pivot_rms_mm"]:7.3f}mm {v["pivot_max_mm"]:7.3f}mm {train_metrics["distance_rms_mm"]:7.3f}mm')
        summary = summarise(result, train_metrics, val_metrics)
        report['models'][model] = summary
        if model == 'nominal':
            continue
        for row in joint_table(result):
            print(row)
        weak = [result.param_label(j) for j in range(len(result.free)) if result.identified[j] < .5]
        if weak:
            print(f'  基本不可辨识（保持名义值）：{", ".join(weak)}')
        model_dir = out / model
        model_dir.mkdir()
        offsets = np.asarray(compensation.offsets_deg) + np.rad2deg(result.errors[:, 0])
        JointCompensation(compensation.directions, tuple(float(v) for v in offsets)).save(
            model_dir / 'joint_compensation.yaml')
        if model == 'full':
            chain.write_urdf(result.errors, model_dir / f'{config.urdf.stem}_fkcal.urdf')
    (out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n结果：{out}')


if __name__ == '__main__':
    main()
