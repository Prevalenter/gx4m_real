"""Live Gemini + FoundationPose gear tracking (default gear24, step1): q quit, r reinitialize, s save."""
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

from detect_gear36 import HERE
from gear_tracker import GearTracker
from gears import DEFAULT_GEAR, GEAR_COLORS, GEARS, check_gear_mesh, gear_mesh_path, hsv_ranges, yaw_symmetry
from gx4m_real.camera.cloud import save_cloud
from gx4m_real.camera.orbbec import OrbbecRGBDStream


def label(image, lines, good):
    cv2.rectangle(image, (0, 0), (image.shape[1], 86), (20, 20, 20), -1)
    for i, line in enumerate(lines):
        cv2.putText(image, line, (10, 22 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (80, 255, 80) if good else (0, 210, 255), 1, cv2.LINE_AA)


def main(argv=None, observer=None, frame_hook=None, up_in_camera=None):
    """frame_hook(scorer, refiner, glctx, output) returns a callable run on each frame after its gear pose is sent.
    up_in_camera (world +z in the camera frame) keeps registrations hub up (gear_tracker.hub_up)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--gear', choices=sorted(GEARS), default=DEFAULT_GEAR)
    parser.add_argument('--mesh', type=Path, help='Default: the simulation CAD mesh of --gear')
    parser.add_argument('--hsv-low', type=int, nargs=3,
                        help='OpenCV HSV range replacing the gear color (gear24 red, gear36 blue, gear48 yellow)')
    parser.add_argument('--hsv-high', type=int, nargs=3)
    parser.add_argument('--register-iterations', type=int, default=5)
    parser.add_argument('--track-iterations', type=int, default=2)
    parser.add_argument('--max-frames', type=int, default=0, help='0 runs until q or Ctrl-C')
    parser.add_argument('--no-display', action='store_true')
    parser.add_argument('--output', type=Path, help='New directory for logs and saved snapshots')
    parser.add_argument('--record', action='store_true', help='Save preview.avi at a fixed 15 fps playback rate')
    parser.add_argument('--min-iou', type=float, default=.45)
    parser.add_argument('--max-depth-error', type=float, default=.02, help='Median depth residual in metres')
    args = parser.parse_args(argv)
    args.mesh = args.mesh or gear_mesh_path(args.gear)
    ranges = hsv_ranges(args.gear, args.hsv_low, args.hsv_high)
    if min(args.register_iterations, args.track_iterations) < 1 or args.max_frames < 0:
        parser.error('Iterations must be positive and max-frames nonnegative')
    if not 0 <= args.min_iou <= 1 or args.max_depth_error <= 0:
        parser.error('Invalid pose validation thresholds')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    import torch
    import trimesh
    from estimater import ScorePredictor, PoseRefinePredictor
    from Utils import dr

    logging.getLogger().setLevel(logging.WARNING)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible CUDA GPU is required')
    output = args.output or HERE / 'outputs' / datetime.now(ZoneInfo('Asia/Shanghai')).strftime('live_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    mesh = trimesh.load(args.mesh, force='mesh')
    check_gear_mesh(mesh, args.gear)
    symmetry = yaw_symmetry(args.gear)
    print(f'Loading FoundationPose on physical GPU {args.gpu}; output: {output}', flush=True)
    scorer, refiner = ScorePredictor(), PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    on_frame = frame_hook(scorer, refiner, glctx, output) if frame_hook is not None else None
    tracker = GearTracker(mesh, symmetry, ranges, scorer=scorer, refiner=refiner, glctx=glctx,
                          debug_dir=output / 'debug', register_iterations=args.register_iterations,
                          track_iterations=args.track_iterations, min_iou=args.min_iou,
                          max_depth_error=args.max_depth_error, up_in_camera=up_in_camera)
    writer = None
    frame_count = accepted = registrations = tracks = 0
    durations = deque(maxlen=30)
    track_times = []
    ious = []
    depth_errors = []
    sequence = -1
    latest = None
    window = f'FoundationPose - {args.gear} | q quit | r reset | s save'
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
                bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
                mask = tracker.mask(frame)
                if mask is not None and tracker.registering:
                    label(bgr, [f'INITIALIZING - keep the {GEAR_COLORS[args.gear]} {args.gear} visible', 'q quit | r reset | s snapshot'], False)
                    if not args.no_display:
                        cv2.imshow(window, bgr)
                        key = cv2.waitKey(1) & 255
                        if key in (ord('q'), 27):
                            break
                pose, metrics, mode, rgb_overlay = tracker.step(frame, mask)
                registrations += mode == 'REGISTER'
                tracks += mode == 'TRACK'
                if pose is not None:
                    accepted += 1
                    ious.append(metrics['mask_iou'])
                    depth_errors.append(metrics['median_depth_error_m'])
                    bgr = cv2.cvtColor(rgb_overlay, cv2.COLOR_RGB2BGR)

                processing_ms = (time.monotonic() - start) * 1000
                age_ms = (time.monotonic() - frame.received_at) * 1000
                now = time.monotonic()
                durations.append(now)
                fps = (len(durations) - 1) / (durations[-1] - durations[0]) if len(durations) > 1 else 0
                if mode == 'TRACK':
                    track_times.append(processing_ms)
                xyz = 'XYZ (camera, mm): --' if pose is None else 'XYZ (camera, mm): ' + ', '.join(f'{x * 1000:.1f}' for x in pose[:3, 3])
                info = f'IoU {metrics["mask_iou"]:.2f}' if metrics['mask_iou'] is not None else f'Show the {GEAR_COLORS[args.gear]} {args.gear}'
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
                if on_frame is not None:
                    on_frame(frame)
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
                    if key == ord('r'):
                        tracker.reset()
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
