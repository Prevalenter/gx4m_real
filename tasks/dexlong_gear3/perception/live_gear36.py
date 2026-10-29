"""Live Gemini + FoundationPose preview: q quit, r reinitialize, s save."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from detect_gear36 import DEFAULT_MESH, HERE, blue_gear_mask
from gx4m_real.camera.cloud import save_cloud
from gx4m_real.camera.orbbec import OrbbecRGBDStream


def label(image, lines, good):
    cv2.rectangle(image, (0, 0), (image.shape[1], 86), (20, 20, 20), -1)
    for i, line in enumerate(lines):
        cv2.putText(image, line, (10, 22 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (80, 255, 80) if good else (0, 210, 255), 1, cv2.LINE_AA)


def main(argv=None, observer=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--mesh', type=Path, default=DEFAULT_MESH)
    parser.add_argument('--register-iterations', type=int, default=5)
    parser.add_argument('--track-iterations', type=int, default=2)
    parser.add_argument('--max-frames', type=int, default=0, help='0 runs until q or Ctrl-C')
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--output', type=Path, help='New directory for logs and saved snapshots')
    parser.add_argument('--record', action='store_true', help='Save preview.avi at a fixed 15 fps playback rate')
    parser.add_argument('--min-iou', type=float, default=.45)
    parser.add_argument('--max-depth-error', type=float, default=.02, help='Median depth residual in metres')
    args = parser.parse_args(argv)
    if min(args.register_iterations, args.track_iterations) < 1 or args.max_frames < 0:
        parser.error('Iterations must be positive and max-frames nonnegative')
    if not 0 <= args.min_iou <= 1 or args.max_depth_error <= 0:
        parser.error('Invalid pose validation thresholds')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    import torch
    import trimesh
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
    from Utils import dr, nvdiffrast_render, draw_xyz_axis, draw_posed_3d_box

    logging.getLogger().setLevel(logging.WARNING)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible CUDA GPU is required')
    output = args.output or HERE / 'outputs' / datetime.now(ZoneInfo('Asia/Shanghai')).strftime('live_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    mesh = trimesh.load(args.mesh, force='mesh')
    if not .10 < float(mesh.extents[:2].max()) < .13:
        raise ValueError('Expected a gear36 mesh in metres (approximately 114 mm diameter)')
    symmetry = np.repeat(np.eye(4)[None], 36, axis=0)
    for i, angle in enumerate(np.arange(36) * 2 * np.pi / 36):
        c, s = np.cos(angle), np.sin(angle)
        symmetry[i, :2, :2] = [[c, -s], [s, c]]
    print(f'Loading FoundationPose on physical GPU {args.gpu}; output: {output}', flush=True)
    scorer, refiner = ScorePredictor(), PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    estimator = None
    writer = None
    frame_count = accepted = registrations = tracks = 0
    bad_frames = 0
    durations = deque(maxlen=30)
    track_times = []
    ious = []
    depth_errors = []
    sequence = -1
    latest = None
    window = 'FoundationPose - gear36 | q quit | r reset | s save'
    if not args.no_display:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 960, 720)

    def save_snapshot(frame, bgr, pose, metrics):
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
        folder = output / f'snapshot_{stamp}'
        folder.mkdir()
        cv2.imwrite(str(folder / 'overlay.png'), bgr)
        cv2.imwrite(str(folder / 'rgb.png'), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
        np.savez_compressed(folder / 'rgbd.npz', rgb=frame.rgb, depth=frame.depth, K=frame.K)
        save_cloud(folder / 'scene.ply', frame.rgb, frame.depth, frame.K)
        record = {**metrics, 'sequence': frame.sequence,
                  'object_in_camera': pose.tolist() if pose is not None else None,
                  'frame': 'camera optical: metres; +X right, +Y down, +Z forward'}
        (folder / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
        if pose is not None:
            np.savetxt(folder / 'object_in_camera.txt', pose)
        print(f'Saved {folder}', flush=True)

    try:
        with OrbbecRGBDStream(args.device_index) as camera, (output / 'frames.jsonl').open('w') as log:
            print(f'Camera: {camera.metadata}; q quit, r reset, s save', flush=True)
            if observer is not None:
                observer.set_camera_metadata(camera.metadata)
            while args.max_frames == 0 or frame_count < args.max_frames:
                frame = camera.read(after=sequence)
                sequence = frame.sequence
                start = time.monotonic()
                pose = None
                metrics = {'status': 'SEARCHING', 'mask_iou': None, 'median_depth_error_m': None}
                bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
                try:
                    mask = blue_gear_mask(frame.rgb, frame.depth)
                except ValueError:
                    mask = None
                    if estimator is not None:
                        estimator.pose_last = None
                    bad_frames = 0

                mode = 'SEARCHING'
                if mask is not None:
                    if estimator is None:
                        color = np.median(frame.rgb[mask], axis=0).astype(np.uint8)
                        mesh.visual.vertex_colors = np.tile(np.r_[color, 255], (len(mesh.vertices), 1))
                        estimator = FoundationPose(mesh.vertices, mesh.vertex_normals, mesh=mesh,
                                                   symmetry_tfs=symmetry, scorer=scorer, refiner=refiner,
                                                   glctx=glctx, debug=0, debug_dir=str(output / 'debug'))
                    if estimator.pose_last is None:
                        label(bgr, ['INITIALIZING - keep the blue gear visible', 'q quit | r reset | s snapshot'], False)
                        if not args.no_display:
                            cv2.imshow(window, bgr)
                            key = cv2.waitKey(1) & 255
                            if key in (ord('q'), 27):
                                break
                        pose = estimator.register(K=frame.K, rgb=frame.rgb, depth=frame.depth,
                                                  ob_mask=mask, iteration=args.register_iterations)
                        mode = 'REGISTER'
                        registrations += 1
                        if estimator.pose_last is None:
                            estimator.pose_last = None
                            pose = None
                    else:
                        pose = estimator.track_one(rgb=frame.rgb, depth=frame.depth, K=frame.K,
                                                   iteration=args.track_iterations)
                        mode = 'TRACK'
                        tracks += 1
                    if pose is not None and np.isfinite(pose).all() and pose[2, 3] > 0:
                        # Render the centered mesh at its matching centered pose.
                        centered = pose @ np.linalg.inv(estimator.get_tf_to_centered_mesh().cpu().numpy())
                        _, rendered, _ = nvdiffrast_render(
                            K=frame.K, H=frame.depth.shape[0], W=frame.depth.shape[1],
                            ob_in_cams=torch.as_tensor(centered[None], dtype=torch.float32, device='cuda'),
                            glctx=glctx, mesh_tensors=estimator.mesh_tensors)
                        rendered = rendered[0].cpu().numpy()
                        silhouette = rendered > 0
                        overlap = silhouette & mask & (frame.depth > 0)
                        iou = float((silhouette & mask).sum() / max(int((silhouette | mask).sum()), 1))
                        error = float(np.median(np.abs(rendered[overlap] - frame.depth[overlap]))) if overlap.any() else None
                        valid = iou >= args.min_iou and error is not None and error <= args.max_depth_error
                        metrics.update(mask_iou=iou, median_depth_error_m=error)
                        if valid:
                            bad_frames = 0
                            accepted += 1
                            ious.append(iou)
                            depth_errors.append(error)
                            rgb_overlay = frame.rgb.copy()
                            contours, _ = cv2.findContours(silhouette.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(rgb_overlay, contours, -1, (0, 255, 0), 1)
                            rgb_overlay = draw_posed_3d_box(frame.K, img=rgb_overlay, ob_in_cam=pose, bbox=mesh.bounds)
                            rgb_overlay = draw_xyz_axis(rgb_overlay, ob_in_cam=pose, scale=.04,
                                                        K=frame.K, thickness=2, is_input_rgb=True)
                            bgr = cv2.cvtColor(rgb_overlay, cv2.COLOR_RGB2BGR)
                            metrics['status'] = mode
                        else:
                            pose = None
                            bad_frames += 1
                            metrics['status'] = 'LOST - reacquiring'
                            if mode == 'REGISTER' or bad_frames >= 3:
                                estimator.pose_last = None
                                bad_frames = 0
                    else:
                        estimator.pose_last = None
                        pose = None
                        metrics['status'] = 'LOST - reacquiring'

                torch.cuda.synchronize()
                processing_ms = (time.monotonic() - start) * 1000
                age_ms = (time.monotonic() - frame.received_at) * 1000
                now = time.monotonic()
                durations.append(now)
                fps = (len(durations) - 1) / (durations[-1] - durations[0]) if len(durations) > 1 else 0
                if mode == 'TRACK':
                    track_times.append(processing_ms)
                xyz = 'XYZ (camera, mm): --' if pose is None else 'XYZ (camera, mm): ' + ', '.join(f'{x * 1000:.1f}' for x in pose[:3, 3])
                info = f'IoU {metrics["mask_iou"]:.2f}' if metrics['mask_iou'] is not None else 'Show one blue gear'
                label(bgr, [f'{metrics["status"]} | {fps:.1f} FPS | {processing_ms:.0f} ms | {info}',
                            xyz, 'q quit | r reset | s snapshot'], pose is not None)
                metrics.update(processing_ms=processing_ms, frame_age_ms=age_ms, fps=fps)
                record = {**metrics, 'sequence': sequence,
                          'timestamp_monotonic': frame.received_at,
                          'timestamp_delta_ms': frame.timestamp_delta_ms,
                          'object_in_camera': pose.tolist() if pose is not None else None}
                log.write(json.dumps(record) + '\n')
                log.flush()
                if observer is not None:
                    observer.update(record, rgb=frame.rgb, K=frame.K)
                if writer is None and args.record:
                    writer = cv2.VideoWriter(str(output / 'preview.avi'), cv2.VideoWriter_fourcc(*'MJPG'),
                                             15, (bgr.shape[1], bgr.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError('Could not open video writer')
                if writer is not None:
                    writer.write(bgr)
                latest = frame, bgr, pose, metrics
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f'frames={frame_count} accepted={accepted} {fps:.1f} FPS {metrics["status"]}', flush=True)
                if not args.no_display:
                    cv2.imshow(window, bgr)
                    key = cv2.waitKey(1) & 255
                    if key in (ord('q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    if key == ord('r') and estimator is not None:
                        estimator.pose_last = None
                    if key == ord('s'):
                        save_snapshot(*latest)
            (output / 'camera.json').write_text(json.dumps(camera.metadata, indent=2) + '\n')
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
        if latest is not None:
            save_snapshot(*latest)
        cv2.destroyAllWindows()
        summary = {
            'frames': frame_count, 'accepted_poses': accepted, 'registrations': registrations, 'track_calls': tracks,
            'median_track_processing_ms': float(np.median(track_times)) if track_times else None,
            'median_mask_iou': float(np.median(ious)) if ious else None,
            'median_depth_error_m': float(np.median(depth_errors)) if depth_errors else None,
            'last_fps': latest[3]['fps'] if latest else None, 'physical_gpu': args.gpu,
            'coordinate_convention': 'Camera optical frame, metres; no robot extrinsics applied',
        }
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
