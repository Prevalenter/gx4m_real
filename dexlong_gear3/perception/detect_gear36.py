"""Estimate the split task's gear36 pose from calibrated RGB-D and a target mask."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_MESH = HERE.parents[1] / 'asset/mjfit_gears3_split/assembly/gear36_visual.obj'


def blue_gear_mask(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Select the largest blue component in this scene, not a general object detector."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    candidate = cv2.inRange(hsv, np.array([95, 90, 25]), np.array([140, 255, 255]))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate)
    if count < 2:
        raise ValueError('No blue target found; provide an explicit --mask')
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = labels == label
    if int((mask & (depth > 0)).sum()) < 100:
        raise ValueError('Largest blue component lacks valid depth; provide an explicit --mask')
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True, help='Directory produced by capture_gemini.py')
    masks = parser.add_mutually_exclusive_group(required=True)
    masks.add_argument('--mask', type=Path, help='White target pixels, black background; same size as RGB')
    masks.add_argument('--blue-mask', action='store_true', help='Use largest blue region; suitable for the observed blue gear scene')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--mesh', type=Path, default=DEFAULT_MESH)
    parser.add_argument('--gpu', default='0', help='Physical GPU index')
    parser.add_argument('--iterations', type=int, default=5)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    import torch
    import trimesh
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
    from Utils import dr, make_mesh_tensors, nvdiffrast_render, draw_posed_3d_box, draw_xyz_axis, set_logging_format
    from capture_gemini import save_cloud

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one visible CUDA GPU')
    set_logging_format()
    with np.load(args.capture / 'rgbd.npz') as data:
        rgb, depth, K = data['rgb'], data['depth'], data['K']
    capture_meta_path = args.capture / 'capture.json'
    capture_meta = json.loads(capture_meta_path.read_text()) if capture_meta_path.exists() else {}
    if args.blue_mask:
        mask = blue_gear_mask(rgb, depth)
    else:
        mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != depth.shape:
            raise ValueError('Mask must exist and match calibrated RGB-D dimensions')
        mask = mask > 0
    if int((mask & (depth > 0)).sum()) < 100:
        raise ValueError('Insufficient valid target depth for registration')
    args.output.mkdir(parents=True, exist_ok=False)
    mesh = trimesh.load(args.mesh, force='mesh')
    if not (.10 < float(mesh.extents[:2].max()) < .13):
        raise ValueError('Expected gear mesh in metres, approximately 114 mm diameter')
    # The CAD gear repeats every 10 degrees around its local Z axis.
    symmetry = np.repeat(np.eye(4)[None], 36, axis=0)
    for i, angle in enumerate(np.arange(36) * 2 * np.pi / 36):
        c, s = np.cos(angle), np.sin(angle)
        symmetry[i, :2, :2] = [[c, -s], [s, c]]
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
    mesh.export(args.output / 'posed_gear36.ply')
    report = {
        'capture': str(args.capture.resolve()), 'mesh': str(args.mesh.resolve()),
        'synthetic_test_only': bool(capture_meta.get('synthetic_test_only', False)),
        'mask_source': 'largest blue HSV component' if args.blue_mask else str(args.mask.resolve()),
        'physical_gpu': args.gpu, 'gpu': torch.cuda.get_device_name(),
        'object_in_camera': pose.tolist(), 'translation_m': pose[:3, 3].tolist(),
        'gear_axis_in_camera': pose[:3, 2].tolist(),
        'coordinate_convention': 'p_camera = object_in_camera @ p_mesh; metres; +X right, +Y down, +Z forward',
        'mesh_origin': 'simulation CAD origin; gear disc z=-0.004..0.004 m, hub reaches z=0.034 m',
        'yaw_symmetry_degrees': 10, 'registration_seconds': elapsed,
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
