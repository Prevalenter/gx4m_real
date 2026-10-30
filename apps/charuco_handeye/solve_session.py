"""Offline hand-eye calibration from recorded RGB + aligned GX7 joints."""
import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from board import CharucoDetector, MARKER_TO_SQUARE, SQUARE_LENGTH_M, SQUARES_X, SQUARES_Y
from playback_data import load_session
from solve_handeye import inv, project, refine, residuals, rot_deg, run_linear, summarize


def check_motion(poses):
    if len(poses) < 6:
        raise ValueError(f'Only {len(poses)} valid views; need at least 6, preferably 15–30')
    vectors = Rotation.from_matrix([poses[0][:3, :3].T @ p[:3, :3] for p in poses[1:]]).as_rotvec()
    singular = np.linalg.svd(vectors, compute_uv=False)
    if np.max(np.linalg.norm(vectors, axis=1)) < np.deg2rad(10) or singular[1] < .05 * singular[0]:
        raise ValueError('Insufficient rotation diversity: rotate about at least two independent axes')
    return {'rotation_singular_values': singular.tolist(), 'num_views': len(poses)}


def solve_views(views, K, dist, mode, reject=3.):
    if mode not in ('eye-to-hand', 'eye-in-hand'):
        raise ValueError('Choose eye-to-hand or eye-in-hand explicitly')
    real_poses = [np.asarray(v['T_base_flange']) for v in views]
    motion = check_motion(real_poses)
    # Reuse U B_i = A_i V. In hand-mounted mode A_i=inv(T_base_flange_i),
    # U=T_flange_cam and V=T_base_board; otherwise U=T_base_cam, V=T_flange_board.
    poses = real_poses if mode == 'eye-to-hand' else [inv(T) for T in real_poses]
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
            check_motion([real_poses[i] for i in kept])
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


def calibrate_session(session_path, output, *, mode, urdf=None, base='base_link', tip='Link7',
                      square_length=SQUARE_LENGTH_M, min_interval=.25, min_rotation_deg=5.,
                      min_translation=.01, reject=3., allow_synthetic=False, compensation=None,
                      still_window=.15, max_still_motion_deg=.05):
    if any(not np.isfinite(v) or v < 0 for v in (min_interval, min_rotation_deg, min_translation, reject,
                                                  still_window, max_still_motion_deg)):
        raise ValueError('Sampling and rejection thresholds must be finite and nonnegative')
    session = load_session(session_path, urdf=urdf, base=base, tip=tip, allow_synthetic=allow_synthetic,
                           compensation=compensation, still_window=still_window)
    if max_still_motion_deg > 0 and not session['has_joint_log']:
        raise ValueError('joints.jsonl is missing, so still frames cannot be selected; '
                         'pass --max-still-motion-deg 0 to accept moving frames')
    detector = CharucoDetector(session['K'], session['dist'], square_length_m=square_length)
    all_points = np.asarray(detector.board.getChessboardCorners(), dtype=float)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'samples').mkdir()
    reports, views = [], []
    for row in session['frames']:
        report = {'frame_index': row['frame_index'], 'image_path': row['image_path'], 'accepted': False}
        reports.append(report)
        if row.get('alignment_valid') is not True:
            report['reason'] = 'invalid_alignment'
            continue
        if max_still_motion_deg > 0:
            # Software time alignment is only trustworthy while the arm is not moving.
            motion = row['joint_motion_rad']
            report['joint_motion_deg'] = None if motion is None else float(np.rad2deg(motion))
            if motion is None:
                report['reason'] = 'no_joint_samples'
                continue
            if np.rad2deg(motion) > max_still_motion_deg:
                report['reason'] = 'moving'
                continue
        if views:
            if row['timestamp_monotonic'] - views[-1]['timestamp_monotonic'] < min_interval:
                report['reason'] = 'time_spacing'
                continue
            A = np.asarray(row['T_base_flange'])
            redundant = any(rot_deg(np.asarray(v['T_base_flange'])[:3, :3].T @ A[:3, :3]) < min_rotation_deg
                            and np.linalg.norm(np.asarray(v['T_base_flange'])[:3, 3] - A[:3, 3]) < min_translation
                            for v in views)
            if redundant:
                report['reason'] = 'redundant_pose'
                continue
        image = cv2.imread(row['image_path'])
        if image is None:
            report['reason'] = 'missing_or_unreadable_image'
            continue
        if (image.shape[1], image.shape[0]) != session['size']:
            raise ValueError(f"Image size does not match recorded intrinsics: {row['image_path']}")
        detection = detector.detect(image)
        report.update(n_corners=detection.n_corners,
                      reproj_px=detection.reproj_px if np.isfinite(detection.reproj_px) else None)
        if not detection.ok:
            report['reason'] = detection.reason
            continue
        report.update(accepted=True, reason='accepted')
        view = {**row, 'T_cam_board': detection.T_cam_board.tolist(),
                'object_points': all_points[detection.charuco_ids.ravel()].tolist(),
                'corners_px': detection.charuco_corners.reshape(-1, 2).tolist(),
                'corner_ids': detection.charuco_ids.ravel().tolist(), 'n_corners': detection.n_corners,
                'K': session['K'].tolist(), 'dist': session['dist'].tolist()}
        views.append(view)
        stem = output / 'samples' / f'sample_{len(views) - 1:04d}'
        stem.with_suffix('.json').write_text(json.dumps(view, indent=2) + '\n')
        if not cv2.imwrite(str(stem.with_suffix('.png')), detector.draw(image, detection)):
            raise OSError('Failed to write detection overlay')
    (output / 'detection_report.json').write_text(json.dumps(reports, indent=2) + '\n')
    print(f"Selected {len(views)}/{len(reports)} frames; {dict(Counter(r['reason'] for r in reports))}", flush=True)
    if not views and any(r.get('reason') == 'moving' for r in reports):
        print('No still frames: add pauses to the playback trajectory, or raise --max-still-motion-deg', flush=True)
    solved = solve_views(views, session['K'], session['dist'], mode, reject)
    result = {'version': 1, 'mode': mode, 'method': solved['method'],
              'source_session': session['path'], 'synthetic': bool(session['metadata'].get('mock') or
                                                                  session['metadata']['camera'].get('synthetic')),
              'K': session['K'].tolist(), 'dist': session['dist'].tolist(), 'image_size': session['size'],
              'T_world_base': session['T_world_base'].tolist(), 'urdf': session['urdf'],
              'base_link': base, 'tip_link': tip, 'joint_names': session['metadata']['joint_names'],
              'fixed_model_joints': session['metadata'].get('fixed_model_joints', {}),
              'joint_compensation': {'directions': list(session['compensation'].directions),
                                     'offsets_deg': list(session['compensation'].offsets_deg),
                                     'source': session['compensation_source']},
              'joint_positions': 'positions_rad are model angles after compensation; '
                                 'motor_positions_rad are the recorded raw values',
              'still_frames': {'window_sec': still_window, 'max_motion_deg': max_still_motion_deg},
              'board': {'squares_x': SQUARES_X, 'squares_y': SQUARES_Y, 'square_length_m': square_length,
                        'marker_ratio': MARKER_TO_SQUARE, 'dictionary': 'DICT_4X4_50'},
              'samples': [dict(views[i], residual=res) for i, res in zip(solved['indices'], solved['residuals'])],
              'rejected_frame_indices': [views[i]['frame_index'] for i in solved['rejected_indices']],
              'residual_summary': solved['summary'], 'linear_methods': solved['linear_methods'],
              'motion': solved['motion'], 'units': 'metres/radians',
              'transform_convention': 'T_parent_child maps child coordinates into parent',
              'timing_limit': session['metadata'].get('timing_limit', 'Software time alignment'),
              'camera_metadata': session['metadata']['camera']}
    U, V = solved['U'], solved['V']
    result['quality_warnings'] = []
    if len(result['samples']) < 10:
        result['quality_warnings'].append('Fewer than 10 independent views; collect 15–30 diverse poses')
    if solved['summary']['reproj_px']['rms'] > 1.:
        result['quality_warnings'].append('Reprojection RMS exceeds 1 px; inspect timing, detections and board rigidity')
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
    print(f"Calibration saved: {output}; reprojection RMS {solved['summary']['reproj_px']['rms']:.3f} px")
    for warning in result['quality_warnings']:
        print(f'Quality warning: {warning}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New output directory')
    parser.add_argument('--mode', choices=['eye-to-hand', 'eye-in-hand'], required=True)
    parser.add_argument('--urdf', type=Path)
    parser.add_argument('--base-link', default='base_link')
    parser.add_argument('--tip-link', default='Link7')
    parser.add_argument('--square-length-m', type=float, default=SQUARE_LENGTH_M)
    parser.add_argument('--min-interval', type=float, default=.25)
    parser.add_argument('--min-rotation-deg', type=float, default=5.)
    parser.add_argument('--min-translation-m', type=float, default=.01)
    parser.add_argument('--reject', type=float, default=3.)
    parser.add_argument('--joint-compensation', type=Path,
                        help='joint_compensation.yaml to apply instead of the one recorded with the session '
                             '(required for sessions recorded before it was saved)')
    parser.add_argument('--still-window-sec', type=float, default=.15,
                        help='half-width of the joint window checked around each image time')
    parser.add_argument('--max-still-motion-deg', type=float, default=.05,
                        help='largest joint motion within the window for a still frame; 0 keeps moving frames')
    parser.add_argument('--allow-synthetic', action='store_true')
    args = parser.parse_args()
    calibrate_session(args.session, args.out, mode=args.mode, urdf=args.urdf, base=args.base_link,
                      tip=args.tip_link, square_length=args.square_length_m, min_interval=args.min_interval,
                      min_rotation_deg=args.min_rotation_deg, min_translation=args.min_translation_m,
                      reject=args.reject, allow_synthetic=args.allow_synthetic,
                      compensation=args.joint_compensation, still_window=args.still_window_sec,
                      max_still_motion_deg=args.max_still_motion_deg)


if __name__ == '__main__':
    main()
