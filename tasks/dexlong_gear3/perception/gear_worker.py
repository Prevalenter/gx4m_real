"""FoundationPose gear tracking for another process (default gear24): no camera of its own.

The parent (apps/charuco_handeye/view_result.py --gear24) owns the camera, picks a free local port and starts
this script in the foundationpose environment; it listens on GEAR_WORKER_PORT (127.0.0.1) with the hex
GEAR_WORKER_AUTHKEY. Messages are gx4m_real.camera.frame_ipc headers plus arrays:
  parent -> worker  {'type': 'frame', 'sequence'} + rgb uint8 HxWx3, depth float32 HxW (m), K 3x3; {'type': 'reset'}
  worker -> parent  {'type': 'ready'} once, then per frame {'type': 'pose', 'sequence', 'status', 'mode',
                    'mask_iou', 'median_depth_error_m', 'processing_ms', 'object_in_camera'} (4x4 list or None)
object_in_camera is the gear in the camera optical frame (metres); gears.py describes the mesh frame.
"""
from __future__ import annotations

import argparse
from multiprocessing.connection import Listener
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

from gears import DEFAULT_GEAR, GEARS, check_gear_mesh, gear_mesh_path, hsv_ranges, yaw_symmetry
from gx4m_real.camera.frame_ipc import receive, send


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gpu', default='0', help='Physical GPU index')
    parser.add_argument('--gear', choices=sorted(GEARS), default=DEFAULT_GEAR)
    parser.add_argument('--mesh', type=Path, help='Default: the simulation CAD mesh of --gear')
    parser.add_argument('--hsv-low', type=int, nargs=3, help='OpenCV HSV range replacing the gear color')
    parser.add_argument('--hsv-high', type=int, nargs=3)
    parser.add_argument('--register-iterations', type=int, default=5)
    parser.add_argument('--track-iterations', type=int, default=2)
    parser.add_argument('--min-iou', type=float, default=.45)
    parser.add_argument('--max-depth-error', type=float, default=.02, help='Median depth residual in metres')
    args = parser.parse_args(argv)
    port, key = int(os.environ['GEAR_WORKER_PORT']), bytes.fromhex(os.environ['GEAR_WORKER_AUTHKEY'])
    ranges = hsv_ranges(args.gear, args.hsv_low, args.hsv_high)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    import logging
    import torch
    import trimesh
    from estimater import PoseRefinePredictor, ScorePredictor
    from Utils import dr
    from gear_tracker import GearTracker

    logging.getLogger().setLevel(logging.WARNING)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible CUDA GPU is required')
    mesh = trimesh.load(args.mesh or gear_mesh_path(args.gear), force='mesh')
    check_gear_mesh(mesh, args.gear)
    print(f'Loading FoundationPose for {args.gear} on physical GPU {args.gpu}', flush=True)
    with tempfile.TemporaryDirectory(prefix='gear_worker_') as debug_dir:
        tracker = GearTracker(mesh, yaw_symmetry(args.gear), ranges, scorer=ScorePredictor(),
                              refiner=PoseRefinePredictor(), glctx=dr.RasterizeCudaContext(), debug_dir=debug_dir,
                              register_iterations=args.register_iterations, track_iterations=args.track_iterations,
                              min_iou=args.min_iou, max_depth_error=args.max_depth_error)
        # Listen only once the models are loaded: until then the parent's connection attempts are refused.
        with Listener(('127.0.0.1', port), authkey=key) as listener, listener.accept() as connection:
            send(connection, {'type': 'ready', 'gear': args.gear})
            while True:
                try:
                    header, arrays = receive(connection)
                except EOFError:
                    break
                if header['type'] == 'reset':
                    tracker.reset()
                    continue
                start = time.monotonic()
                rgb, depth, K = arrays
                frame = SimpleNamespace(rgb=rgb, depth=depth, K=K)
                pose, metrics, mode, _ = tracker.step(frame, tracker.mask(frame))
                send(connection, {'type': 'pose', 'sequence': header['sequence'], **metrics, 'mode': mode,
                                  'processing_ms': (time.monotonic() - start) * 1000,
                                  'object_in_camera': None if pose is None else pose.tolist()})


if __name__ == '__main__':
    main()
