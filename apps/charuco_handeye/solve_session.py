"""Offline hand-eye calibration from a session recorded by collect_app.py.

  python apps/charuco_handeye/solve_session.py --session apps/charuco_handeye/runtime/sessions/he_<ts> \
      --out apps/charuco_handeye/runtime/calibration_001 [--joint-compensation F] [--reject 3]

Each sample is a static pose: joints averaged over the hold window (raw motor angles, mapped to model
angles with the joint compensation) and one image from the same window. Validation samples are
excluded from the fit and reported separately.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from board import CharucoDetector, MARKER_TO_SQUARE, SQUARES_X, SQUARES_Y
from handeye_session import MODES, load_session
from solve_handeye import inv, refine, residuals, run_linear, summarize


def check_motion(poses):
    if len(poses) < 6:
        raise ValueError(f'Only {len(poses)} valid views; need at least 6, preferably 15–30')
    vectors = Rotation.from_matrix([poses[0][:3, :3].T @ p[:3, :3] for p in poses[1:]]).as_rotvec()
    singular = np.linalg.svd(vectors, compute_uv=False)
    if np.max(np.linalg.norm(vectors, axis=1)) < np.deg2rad(10) or singular[1] < .05 * singular[0]:
        raise ValueError('Insufficient rotation diversity: rotate about at least two independent axes')
    return {'rotation_singular_values': singular.tolist(), 'num_views': len(poses)}


def mode_poses(views, mode):
    """A_i in U B_i = A_i V. Hand-mounted: A_i=inv(T_base_flange_i), U=T_flange_cam, V=T_base_board;
    fixed camera: A_i=T_base_flange_i, U=T_base_cam, V=T_flange_board."""
    if mode not in MODES:
        raise ValueError('Choose eye-to-hand or eye-in-hand explicitly')
    real = [np.asarray(v['T_base_flange']) for v in views]
    return real if mode == 'eye-to-hand' else [inv(T) for T in real]


def view_residuals(U, V, views, K, dist, mode):
    return residuals(U, V, mode_poses(views, mode), [np.asarray(v['T_cam_board']) for v in views],
                     [(np.asarray(v['object_points']), np.asarray(v['corners_px'])) for v in views], K, dist)


def solve_views(views, K, dist, mode, reject=3.):
    poses = mode_poses(views, mode)
    motion = check_motion([np.asarray(v['T_base_flange']) for v in views])
    observed = [np.asarray(v['T_cam_board']) for v in views]
    obs = [(np.asarray(v['object_points']), np.asarray(v['corners_px'])) for v in views]

    def fit(indices):
        A, B, pixels = ([seq[i] for i in indices] for seq in (poses, observed, obs))
        methods, best = run_linear(A, B, pixels, K, dist)
        U, V = refine(methods[best][0], methods[best][1], A, pixels, K, dist)
        return U, V, residuals(U, V, A, B, pixels, K, dist), methods, best

    indices = list(range(len(views)))
    U, V, res, methods, best = fit(indices)
    rejected = []
    if reject > 0:
        threshold = max(.1, reject * float(np.median([r['reproj_px'] for r in res])))
        kept = [i for i, r in enumerate(res) if r['reproj_px'] <= threshold]
        if len(kept) < len(indices):
            check_motion([np.asarray(views[i]['T_base_flange']) for i in kept])
            rejected = [i for i in indices if i not in kept]
            indices = kept
            U, V, res, methods, best = fit(indices)
    if not np.isfinite(U).all() or not np.isfinite(V).all():
        raise ValueError('Optimization returned non-finite extrinsics')
    for i in indices:
        cam_board = inv(U) @ poses[i] @ V
        points = np.asarray(views[i]['object_points'])
        if np.any((points @ cam_board[:3, :3].T + cam_board[:3, 3])[:, 2] <= 0):
            raise ValueError('Optimized board lies behind camera; check views and mounting mode')
    return {'mode': mode, 'U': U, 'V': V, 'indices': indices, 'rejected_indices': rejected,
            'residuals': res, 'method': best + '+refine', 'summary': summarize(res),
            'linear_methods': {k: v[3] for k, v in methods.items()}, 'motion': motion}


def detect_view(detector, sample, image, all_points):
    detection = detector.detect(image)
    report = {'index': sample['index'], 'image_path': sample['image_path'], 'validation': bool(sample['validation']),
              'n_corners': detection.n_corners,
              'reproj_px': detection.reproj_px if np.isfinite(detection.reproj_px) else None,
              'accepted': detection.ok, 'reason': 'accepted' if detection.ok else detection.reason}
    if not detection.ok:
        return report, None, detection
    view = {**sample, 'T_cam_board': detection.T_cam_board.tolist(),
            'object_points': all_points[detection.charuco_ids.ravel()].tolist(),
            'corners_px': detection.charuco_corners.reshape(-1, 2).tolist(),
            'corner_ids': detection.charuco_ids.ravel().tolist(), 'n_corners': detection.n_corners}
    return report, view, detection


def calibrate_session(session_path, output, *, mode=None, urdf=None, base='base_link', tip='Link7',
                      square_length=None, reject=3., allow_synthetic=False, compensation=None):
    import cv2

    if not np.isfinite(reject) or reject < 0:
        raise ValueError('--reject must be finite and nonnegative')
    session = load_session(session_path, urdf=urdf, base=base, tip=tip, allow_synthetic=allow_synthetic,
                           compensation=compensation)
    metadata = session['metadata']
    recorded_mode = metadata.get('mounting_mode')
    if mode is not None and recorded_mode is not None and mode != recorded_mode:
        raise ValueError(f'--mode {mode} differs from the session mounting mode {recorded_mode}')
    mode = mode or recorded_mode
    if mode not in MODES:
        raise ValueError('Session has no mounting mode; pass --mode eye-in-hand or eye-to-hand')
    square_length = square_length or metadata['board']['square_length_m']
    detector = CharucoDetector(session['K'], session['dist'], square_length_m=square_length)
    all_points = np.asarray(detector.board.getChessboardCorners(), dtype=float)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'samples').mkdir()
    reports, train, validation = [], [], []
    for sample in session['samples']:
        image = cv2.imread(sample['image_path'])
        if image is None:
            reports.append({'index': sample['index'], 'image_path': sample['image_path'],
                            'accepted': False, 'reason': 'missing_or_unreadable_image'})
            continue
        if (image.shape[1], image.shape[0]) != session['size']:
            raise ValueError(f"Image size does not match recorded intrinsics: {sample['image_path']}")
        report, view, detection = detect_view(detector, sample, image, all_points)
        reports.append(report)
        if view is None:
            continue
        (validation if sample['validation'] else train).append(view)
        stem = output / 'samples' / f"sample_{sample['index']:04d}"
        stem.with_suffix('.json').write_text(json.dumps(view, indent=2) + '\n')
        if not cv2.imwrite(str(stem.with_suffix('.png')), detector.draw(image, detection)):
            raise OSError('Failed to write detection overlay')
    (output / 'detection_report.json').write_text(json.dumps(reports, indent=2) + '\n')
    print(f"{len(train)} training + {len(validation)} validation views of {len(reports)} samples; "
          f"{dict(Counter(r['reason'] for r in reports))}", flush=True)
    solved = solve_views(train, session['K'], session['dist'], mode, reject)
    U, V = solved['U'], solved['V']
    val_res = view_residuals(U, V, validation, session['K'], session['dist'], mode) if validation else []
    result = {'version': 2, 'mode': mode, 'method': solved['method'],
              'source_session': session['path'],
              'synthetic': bool(metadata.get('mock') or metadata['camera'].get('synthetic')),
              'K': session['K'].tolist(), 'dist': session['dist'].tolist(), 'image_size': session['size'],
              'T_world_base': session['T_world_base'].tolist(), 'urdf': session['urdf'],
              'base_link': base, 'tip_link': tip, 'joint_names': list(metadata['joint_names']),
              'fixed_model_joints': metadata.get('fixed_model_joints', {}),
              'joint_compensation': {'directions': list(session['compensation'].directions),
                                     'offsets_deg': list(session['compensation'].offsets_deg),
                                     'source': session['compensation_source']},
              'joint_positions': 'positions_rad are model angles after compensation; '
                                 'motor_positions_rad are the recorded raw values',
              'board': {'squares_x': SQUARES_X, 'squares_y': SQUARES_Y, 'square_length_m': square_length,
                        'marker_ratio': MARKER_TO_SQUARE, 'dictionary': 'DICT_4X4_50'},
              'samples': [dict(train[i], residual=res) for i, res in zip(solved['indices'], solved['residuals'])],
              'rejected_sample_indices': [train[i]['index'] for i in solved['rejected_indices']],
              'residual_summary': solved['summary'],
              'validation_samples': [dict(v, residual=res) for v, res in zip(validation, val_res)],
              'validation_summary': summarize(val_res) if val_res else None,
              'linear_methods': solved['linear_methods'], 'motion': solved['motion'],
              'units': 'metres/radians', 'transform_convention': 'T_parent_child maps child coordinates into parent',
              'sampling': metadata.get('sampling'), 'camera_metadata': metadata['camera']}
    result['quality_warnings'] = []
    if len(result['samples']) < 10:
        result['quality_warnings'].append('Fewer than 10 independent views; collect 15–30 diverse poses')
    if solved['summary']['reproj_px']['rms'] > 1.:
        result['quality_warnings'].append('Reprojection RMS exceeds 1 px; inspect detections, board rigidity and FK')
    if mode == 'eye-to-hand':
        result.update(T_base_cam=U.tolist(), T_flange_board=V.tolist(),
                      T_world_cam=(session['T_world_base'] @ U).tolist())
        np.save(output / 'T_base_cam.npy', U)
        np.save(output / 'T_RC.npy', U)
        np.save(output / 'T_flange_board.npy', V)
    else:
        result.update(T_flange_cam=U.tolist(), T_base_board=V.tolist(),
                      T_world_board=(session['T_world_base'] @ V).tolist())
        np.save(output / 'T_flange_cam.npy', U)
        np.save(output / 'T_base_board.npy', V)
    (output / 'handeye_result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    s = solved['summary']
    print(f"Calibration saved: {output}; reprojection RMS {s['reproj_px']['rms']:.3f} px, "
          f"{s['trans_mm']['rms']:.2f} mm, {s['rot_deg']['rms']:.3f} deg", flush=True)
    if result['validation_summary']:
        v = result['validation_summary']
        print(f"Validation ({len(val_res)} views): {v['reproj_px']['rms']:.3f} px, "
              f"{v['trans_mm']['rms']:.2f} mm, {v['rot_deg']['rms']:.3f} deg", flush=True)
    for warning in result['quality_warnings']:
        print(f'Quality warning: {warning}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New output directory')
    parser.add_argument('--mode', choices=MODES, help='defaults to the mounting mode recorded with the session')
    parser.add_argument('--urdf', type=Path)
    parser.add_argument('--base-link', default='base_link')
    parser.add_argument('--tip-link', default='Link7')
    parser.add_argument('--square-length-m', type=float, help='measured square length; defaults to the session value')
    parser.add_argument('--reject', type=float, default=3.)
    parser.add_argument('--joint-compensation', type=Path,
                        help='joint_compensation.yaml to apply instead of the one recorded with the session')
    parser.add_argument('--allow-synthetic', action='store_true')
    args = parser.parse_args()
    calibrate_session(args.session, args.out, mode=args.mode, urdf=args.urdf, base=args.base_link,
                      tip=args.tip_link, square_length=args.square_length_m, reject=args.reject,
                      allow_synthetic=args.allow_synthetic, compensation=args.joint_compensation)


if __name__ == '__main__':
    main()
