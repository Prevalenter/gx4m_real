"""Estimate a split-task gear pose (default gear24, step1) from calibrated RGB-D and a target mask."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from gears import DEFAULT_GEAR, GEAR_COLORS, GEARS, check_gear_mesh, gear_mesh_path, hsv_ranges, yaw_symmetry

HERE = Path(__file__).resolve().parent
DEFAULT_MESH = gear_mesh_path(DEFAULT_GEAR)


def color_gear_mask(rgb: np.ndarray, depth: np.ndarray, ranges) -> np.ndarray:
    """Select the largest component inside the HSV ranges (gears.hsv_ranges) in this scene, not a general
    object detector: no larger object of the same color may be in view."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    candidate = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in ranges:
        candidate |= cv2.inRange(hsv, np.array(low), np.array(high))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate)
    if count < 2:
        raise ValueError('No target of the gear color found; provide an explicit --mask')
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = labels == label
    if int((mask & (depth > 0)).sum()) < 100:
        raise ValueError('Largest gear-colored component lacks valid depth; provide an explicit --mask')
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True, help='Directory produced by apps/orbbec/capture_rgbd.py')
    masks = parser.add_mutually_exclusive_group(required=True)
    masks.add_argument('--mask', type=Path, help='White target pixels, black background; same size as RGB')
    masks.add_argument('--color-mask', '--blue-mask', dest='color_mask', action='store_true',
                       help="Use the largest region of the gear's color (gear24 red, gear36 blue, gear48 yellow)")
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--gear', choices=sorted(GEARS), default=DEFAULT_GEAR)
    parser.add_argument('--mesh', type=Path, help='Default: the simulation CAD mesh of --gear')
    parser.add_argument('--hsv-low', type=int, nargs=3, help='OpenCV HSV range replacing the gear color (with --hsv-high)')
    parser.add_argument('--hsv-high', type=int, nargs=3)
    parser.add_argument('--gpu', default='0', help='Physical GPU index')
    parser.add_argument('--iterations', type=int, default=5)
    args = parser.parse_args()
    args.mesh = args.mesh or gear_mesh_path(args.gear)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    import torch
    import trimesh
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
    from Utils import dr, make_mesh_tensors, nvdiffrast_render, draw_posed_3d_box, draw_xyz_axis, set_logging_format
    from gx4m_real.camera.cloud import save_cloud

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one visible CUDA GPU')
    set_logging_format()
    with np.load(args.capture / 'rgbd.npz') as data:
        rgb, depth, K = data['rgb'], data['depth'], data['K']
    capture_meta_path = args.capture / 'capture.json'
    capture_meta = json.loads(capture_meta_path.read_text()) if capture_meta_path.exists() else {}
    ranges = hsv_ranges(args.gear, args.hsv_low, args.hsv_high)
    if args.color_mask:
        mask = color_gear_mask(rgb, depth, ranges)
    else:
        mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != depth.shape:
            raise ValueError('Mask must exist and match calibrated RGB-D dimensions')
        mask = mask > 0
    if int((mask & (depth > 0)).sum()) < 100:
        raise ValueError('Insufficient valid target depth for registration')
    args.output.mkdir(parents=True, exist_ok=False)
    mesh = trimesh.load(args.mesh, force='mesh')
    check_gear_mesh(mesh, args.gear)
    # The CAD gear repeats every tooth pitch around its local Z axis.
    symmetry = yaw_symmetry(args.gear)
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    # The simulation asset has no material; use the observed target's median color.
    target_color = np.median(rgb[mask], axis=0).astype(np.uint8)
    mesh.visual.vertex_colors = np.tile(np.r_[target_color, 255], (len(mesh.vertices), 1))
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                         mesh=mesh, symmetry_tfs=symmetry, scorer=scorer, refiner=refiner,
                         glctx=glctx, debug=0, debug_dir=str(args.output / 'debug'))
    start = time.monotonic()
    pose = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=args.iterations)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    if not np.isfinite(pose).all() or pose[2, 3] <= 0 or not hasattr(est, 'scores'):
        raise RuntimeError('FoundationPose did not return a valid scored pose')
    np.savetxt(args.output / 'object_in_camera.txt', pose)
    rendered_rgb, rendered_depth, _ = nvdiffrast_render(
        K=K, H=depth.shape[0], W=depth.shape[1],
        ob_in_cams=torch.as_tensor(pose[None], dtype=torch.float32, device='cuda'),
        glctx=glctx, mesh_tensors=make_mesh_tensors(mesh))
    rendered_rgb = rendered_rgb[0].detach().cpu().numpy()
    rendered_depth = rendered_depth[0].detach().cpu().numpy()
    silhouette = rendered_depth > 0
    overlap = silhouette & mask & (depth > 0)
    union = silhouette | mask
    median_depth_error = float(np.median(np.abs(rendered_depth[overlap] - depth[overlap]))) if overlap.any() else None
    overlay = rgb.copy()
    overlay[silhouette] = np.clip(.5 * overlay[silhouette] + .5 * rendered_rgb[silhouette] * 255, 0, 255)
    contours, _ = cv2.findContours(silhouette.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    bounds = np.asarray(mesh.bounds)
    overlay = draw_posed_3d_box(K, img=overlay, ob_in_cam=pose, bbox=bounds)
    overlay = draw_xyz_axis(overlay, ob_in_cam=pose, scale=.04, K=K, thickness=2, is_input_rgb=True)
    cv2.imwrite(str(args.output / 'overlay.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(args.output / 'mask.png'), mask.astype(np.uint8) * 255)
    cv2.imwrite(str(args.output / 'rendered_mask.png'), silhouette.astype(np.uint8) * 255)
    object_cloud_count = save_cloud(args.output / 'target.ply', rgb, np.where(mask, depth, 0), K)
    mesh.apply_transform(pose)
    mesh.export(args.output / f'posed_{args.gear}.ply')
    report = {
        'capture': str(args.capture.resolve()), 'gear': args.gear, 'mesh': str(args.mesh.resolve()),
        'synthetic_test_only': bool(capture_meta.get('synthetic_test_only', False)),
        'mask_source': f'largest {GEAR_COLORS[args.gear]} component, HSV {ranges}' if args.color_mask else str(args.mask.resolve()),
        'physical_gpu': args.gpu, 'gpu': torch.cuda.get_device_name(),
        'object_in_camera': pose.tolist(), 'translation_m': pose[:3, 3].tolist(),
        'gear_axis_in_camera': pose[:3, 2].tolist(),
        'coordinate_convention': 'p_camera = object_in_camera @ p_mesh; metres; +X right, +Y down, +Z forward',
        'mesh_origin': 'simulation gear body frame; gear disc z=-0.004..0.004 m, hub reaches z=0.034 m',
        'yaw_symmetry_degrees': 360 / GEARS[args.gear][0], 'registration_seconds': elapsed,
        'mask_pixels': int(mask.sum()), 'target_cloud_points': object_cloud_count,
        'render_mask_iou': float((silhouette & mask).sum() / max(int(union.sum()), 1)),
        'median_overlap_depth_error_m': median_depth_error,
        'p90_overlap_depth_error_m': float(np.percentile(np.abs(rendered_depth[overlap] - depth[overlap]), 90)) if overlap.any() else None,
        'overlap_valid_depth_pixels': int(overlap.sum()),
        'score_logit': float(est.scores[0]),
        'score_note': 'Relative ranking logit, not a calibrated confidence or accuracy guarantee',
        'validation_note': 'Image/depth agreement does not verify axis direction or full pose accuracy',
    }
    (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
