"""Plate pose with FoundationPose (fixed calibrated camera), measured once on request: PlateDetector runs inside the
gear loop of step1_setup.py when its button is pressed, and main() here measures alone and saves plate_result.json.

Several registrations are fused into one pose, which then stays until the next measurement. Registrations are spaced
out so the gear keeps updating between them. The target mask is the black surface just above the table (height from
the hand-eye calibration), minus gear-colored pixels. Those pixels, depth holes (common on black surfaces) and whatever
is in front of the rendered plate are ignored when checking a pose, so a gear or the hand over the plate does not count
as a mismatch.

  bash tasks/dexlong_gear3/perception/run_plate.sh --calibration <handeye_result.json>
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from gears import GEARS, PLATE_MESH, check_plate_mesh, hsv_ranges
from step1_setup import REFERENCE, plate_report
from viser_scene import DEFAULT_INITIAL, load_calibration, rigid

HERE = Path(__file__).resolve().parent
# Height above the table (m) of the plate mask: above depth noise on the table, below the pin tops plus a margin.
BAND = (.006, .045)
# The plate is black: largest HSV value (0-255) of its pixels, above dark noise and mild sheen. Raise it under strong light.
BLACK_VALUE_MAX = 90
MIN_DEPTH_PIXELS = 500
# Largest accepted spread of one frame around the fused pose (m, deg).
MAX_SPREAD = .005, 3.
# Depth this far (m) in front of the rendered plate is an occluder, not a mismatch.
OCCLUDER_MARGIN = .01


def table_height(reference):
    return reference['table_pos'][2] + reference['table_size'][2]/2


def gear_pixels(rgb, grow=5):
    """Pixels of any printed gear color, without the speckle of dark noise, grown over the blurred tooth edges."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    pixels = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for gear in GEARS:
        for low, high in hsv_ranges(gear):
            pixels |= cv2.inRange(hsv, np.array(low), np.array(high))
    pixels = cv2.morphologyEx(pixels, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.dilate(pixels, np.ones((grow, grow), np.uint8)) > 0


def plate_mask(rgb, depth, K, T_world_cam, table_z, band=BAND, value_max=BLACK_VALUE_MAX):
    """The black non-gear region with the most depth inside the height band above the table, and the pixels to
    ignore when checking a pose. Black pixels without depth stay in the region: they are holes in the plate's depth."""
    rows, cols = np.indices(depth.shape)
    # World z of each pixel: the third row of T_world_cam applied to the back-projected camera point.
    a, b, c, t = T_world_cam[2]
    height = a*(cols - K[0, 2])/K[0, 0]*depth + b*(rows - K[1, 2])/K[1, 1]*depth + c*depth + t - table_z
    in_band = (depth > 0) & (height >= band[0]) & (height <= band[1])
    gears = gear_pixels(rgb)
    black = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 2] <= value_max
    candidate = (black & (in_band | (depth <= 0))).astype(np.uint8)
    # Closing bridges small highlights on the plate; opening drops thin dark edges and shadows.
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels = cv2.connectedComponents((candidate.astype(bool) & ~gears).astype(np.uint8))
    votes = np.bincount(labels[in_band], minlength=count)
    votes[0] = 0
    if votes.max() == 0:
        raise ValueError('No black surface found just above the table; check the calibration and that the plate is in view')
    mask = labels == int(np.argmax(votes))
    if int((mask & (depth > 0)).sum()) < MIN_DEPTH_PIXELS:
        raise ValueError('Too little valid depth on the black plate; change the lighting or the camera exposure')
    return mask, gears | (depth <= 0)


def mask_metrics(silhouette, mask, ignore, rendered, depth):
    """IoU and median depth residual of a rendered pose, outside the ignored pixels."""
    silhouette, mask = silhouette & ~ignore, mask & ~ignore
    overlap = silhouette & mask
    iou = float(overlap.sum()/max(int((silhouette | mask).sum()), 1))
    error = float(np.median(np.abs(rendered[overlap] - depth[overlap]))) if overlap.any() else None
    return iou, error


def fuse_poses(poses):
    """Median position and mean rotation of the frames; raises if any frame disagrees (e.g. a flipped registration)."""
    poses = np.asarray(poses)
    fused = np.eye(4)
    fused[:3, 3] = np.median(poses[:, :3, 3], axis=0)
    rotations = Rotation.from_matrix(poses[:, :3, :3])
    mean = rotations.mean()
    fused[:3, :3] = mean.as_matrix()
    position = float(np.linalg.norm(poses[:, :3, 3] - fused[:3, 3], axis=1).max())
    rotation = float(np.degrees((rotations*mean.inv()).magnitude().max()))
    if position > MAX_SPREAD[0] or rotation > MAX_SPREAD[1]:
        raise ValueError(f'Plate poses disagree across frames: {position*1000:.1f} mm, {rotation:.1f} deg')
    return fused, dict(position_m=position, rotation_deg=rotation)


class FoundationPoseBackend:
    """A plate FoundationPose that shares the scorer, refiner and render context of the gear loop."""

    def __init__(self, mesh, scorer, refiner, glctx, debug_dir, *, register_iterations=5):
        self.mesh, self.scorer, self.refiner, self.glctx, self.debug_dir = mesh, scorer, refiner, glctx, str(debug_dir)
        self.register_iterations = register_iterations
        self.estimator = None

    def register(self, frame, mask):
        from estimater import FoundationPose
        if self.estimator is None:
            color = np.median(frame.rgb[mask], axis=0).astype(np.uint8)
            self.mesh.visual.vertex_colors = np.tile(np.r_[color, 255], (len(self.mesh.vertices), 1))
            # The pins and the off-center footprint make the plate asymmetric: no symmetry_tfs.
            self.estimator = FoundationPose(model_pts=self.mesh.vertices, model_normals=self.mesh.vertex_normals,
                                            mesh=self.mesh, scorer=self.scorer, refiner=self.refiner,
                                            glctx=self.glctx, debug=0, debug_dir=self.debug_dir)
        return self.estimator.register(K=frame.K, rgb=frame.rgb, depth=frame.depth, ob_mask=mask,
                                       iteration=self.register_iterations)

    def render(self, frame, pose):
        import torch
        from Utils import nvdiffrast_render
        # Render the centered mesh at its matching centered pose.
        centered = pose @ np.linalg.inv(self.estimator.get_tf_to_centered_mesh().cpu().numpy())
        _, rendered, _ = nvdiffrast_render(
            K=frame.K, H=frame.depth.shape[0], W=frame.depth.shape[1],
            ob_in_cams=torch.as_tensor(centered[None], dtype=torch.float32, device='cuda'),
            glctx=self.glctx, mesh_tensors=self.estimator.mesh_tensors)
        return rendered[0].cpu().numpy()


class PlateDetector:
    """One measurement per start(): a registration every few frames until enough agree. Idle otherwise."""

    def __init__(self, backend, reference, T_world_cam, *, every=5, fuse_frames=5, max_attempts=15, min_iou=.6,
                 max_depth_error=.01, value_max=BLACK_VALUE_MAX):
        self.backend, self.T_world_cam, self.every = backend, T_world_cam, every
        self.table_z = table_height(reference)
        self.min_iou, self.max_depth_error, self.value_max = min_iou, max_depth_error, value_max
        self.fuse_frames, self.max_attempts = fuse_frames, max_attempts
        self.window, self.count, self.attempts = [], 0, 0
        self.active, self.T_camera_plate, self.spread = False, None, None
        self.status, self.reason = 'IDLE', ''
        self.log, self.last = [], None

    def start(self):
        """Measure again. The previous pose stays until the new one is fused."""
        self.window, self.count, self.attempts = [], 0, 0
        self.active, self.status, self.reason = True, 'MEASURING', ''

    def state(self):
        return dict(status=self.status, reason=self.reason, fused=len(self.window), fuse_frames=self.fuse_frames,
                    T_camera_plate=self.T_camera_plate, spread=self.spread)

    def _pose(self, frame):
        """A registered pose that passes the checks, or the reason there is none."""
        try:
            mask, ignore = plate_mask(frame.rgb, frame.depth, frame.K, self.T_world_cam, self.table_z,
                                      value_max=self.value_max)
        except ValueError as exc:
            return None, str(exc)
        pose = self.backend.register(frame, mask)
        if pose is None or not np.isfinite(pose).all() or pose[2, 3] <= 0:
            return None, 'FoundationPose returned no valid pose'
        rendered = self.backend.render(frame, pose)
        occluders = (frame.depth > 0) & (rendered > 0) & (frame.depth < rendered - OCCLUDER_MARGIN)
        iou, error = mask_metrics(rendered > 0, mask, ignore | occluders, rendered, frame.depth)
        self.last = frame, mask, rendered
        if iou < self.min_iou or error is None or error > self.max_depth_error:
            return None, f'IoU {iou:.2f}, depth error {error}'
        self.log.append(dict(sequence=frame.sequence, mask_iou=iou, median_depth_error_m=error,
                             visible_pixels=int(mask.sum()), depth_pixels=int((mask & (frame.depth > 0)).sum()),
                             object_in_camera=pose.tolist()))
        return pose, ''

    def update(self, frame):
        """While measuring, process every `every`-th frame and return the new state; None otherwise."""
        if not self.active:
            return None
        self.count += 1
        if (self.count - 1) % self.every:
            return None
        self.attempts += 1
        pose, self.reason = self._pose(frame)
        if pose is not None:
            self.window.append(pose)
            if len(self.window) == self.fuse_frames:
                try:
                    self.T_camera_plate, self.spread = fuse_poses(self.window)
                    self.active, self.status = False, 'DONE'
                    return self.state()
                except ValueError as exc:
                    # A registration flipped or the plate moved: start over from this pose.
                    self.window, self.reason = [pose], str(exc)
        if self.attempts >= self.max_attempts:
            self.active, self.status = False, 'FAILED'
        return self.state()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--calibration', type=Path, required=True, help='Fixed-camera handeye_result.json')
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--initial-pose', type=Path, default=DEFAULT_INITIAL)
    parser.add_argument('--output', type=Path, help='New output directory')
    parser.add_argument('--frames', type=int, default=5, help='Accepted frames to fuse')
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--min-iou', type=float, default=.6)
    parser.add_argument('--max-depth-error', type=float, default=.01, help='Median depth residual in metres')
    parser.add_argument('--black-value-max', type=int, default=BLACK_VALUE_MAX, help='Largest HSV value of plate pixels')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--device-index', type=int, default=0)
    args = parser.parse_args(argv)
    if args.frames < 1 or args.iterations < 1:
        parser.error('--frames and --iterations must be positive')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('OMP_NUM_THREADS', '4')

    import torch
    import trimesh
    from estimater import ScorePredictor, PoseRefinePredictor
    from Utils import dr
    from gx4m_real.camera.orbbec import OrbbecRGBDStream

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one visible CUDA GPU is required')
    reference = json.loads(args.reference.read_text(encoding='utf-8'))
    calibration = load_calibration(args.calibration, json.loads(args.initial_pose.read_text()))
    T_world_cam = rigid(calibration['T_world_base']) @ rigid(calibration['T_base_cam'])
    now = datetime.now(ZoneInfo('Asia/Shanghai'))
    output = args.output or HERE/'outputs'/now.strftime('plate_%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    backend = load_plate_backend(trimesh, ScorePredictor(), PoseRefinePredictor(), dr.RasterizeCudaContext(), output,
                                 args.iterations)
    detector = PlateDetector(backend, reference, T_world_cam, every=1, fuse_frames=args.frames,
                             max_attempts=3*args.frames, min_iou=args.min_iou,
                             max_depth_error=args.max_depth_error, value_max=args.black_value_max)
    sequence = -1
    with OrbbecRGBDStream(args.device_index) as camera:
        expected = calibration.get('camera_metadata', {}).get('serial')
        if expected and camera.metadata.get('serial') and expected != camera.metadata['serial']:
            raise ValueError('Camera serial differs from the fixed-camera calibration')
        metadata = camera.metadata
        detector.start()
        while detector.active:
            frame = camera.read(after=sequence)
            sequence = frame.sequence
            detector.update(frame)
            print(f'frame {sequence}: {detector.status} {len(detector.window)}/{args.frames} {detector.reason}', flush=True)
    (output/'frames.json').write_text(json.dumps(detector.log, indent=2) + '\n')
    if detector.last is not None:
        frame, mask, rendered = detector.last
        overlay = frame.rgb.copy()
        contours, _ = cv2.findContours((rendered > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
        cv2.imwrite(str(output/'overlay.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output/'mask.png'), mask.astype(np.uint8)*255)
    if detector.status != 'DONE':
        raise RuntimeError(f'Plate not measured: {len(detector.window)}/{args.frames} frames, {detector.reason}; see {output}')
    T_world_plate = T_world_cam @ detector.T_camera_plate
    report = plate_report(T_world_plate, reference)
    result = dict(measured_at=now.isoformat(), camera=metadata, calibration=str(args.calibration.resolve()),
                  T_camera_plate=detector.T_camera_plate.tolist(), T_world_plate=T_world_plate.tolist(),
                  frames_fused=args.frames, spread=detector.spread, report=report,
                  note='Image/depth agreement across frames, not an externally verified accuracy')
    (output/'plate_result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float) + '\n')
    print(json.dumps(report, indent=2, default=float), flush=True)
    print(f'Saved {output/"plate_result.json"}', flush=True)


def load_plate_backend(trimesh, scorer, refiner, glctx, output, register_iterations=5):
    mesh = trimesh.load(PLATE_MESH, force='mesh')
    check_plate_mesh(mesh)
    return FoundationPoseBackend(mesh, scorer, refiner, glctx, output/'plate_debug',
                                 register_iterations=register_iterations)


if __name__ == '__main__':
    main()
