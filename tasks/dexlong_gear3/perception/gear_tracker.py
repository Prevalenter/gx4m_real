"""Per-frame FoundationPose gear registration, tracking and validity check, shared by live_gear36.py and
gear_worker.py. Needs the foundationpose environment and a visible CUDA GPU."""
from __future__ import annotations

import numpy as np

from detect_gear36 import color_gear_mask
from gears import marked_symmetric, marker_point, nearest_symmetric

OCCLUSION_MARGIN = .01      # m: observed depth this much nearer than the rendered gear is something in front of it
OCCLUSION_RING_PX = 15      # around the silhouette, pixels nearer than the gear's nearest point are occluders too
FLIP = np.diag([1., -1., -1., 1.])  # 180 deg about the mesh x axis: the disc (z -4..4 mm) onto itself, the hub over


def occluders(depth, rendered, margin=OCCLUSION_MARGIN, ring_px=OCCLUSION_RING_PX):
    """Pixels in front of the gear rendered at a pose (rendered: its depth, 0 off the gear): inside the silhouette
    nearer than the rendered surface, around it nearer than the gear's nearest point. No depth is no occluder."""
    import cv2

    silhouette = rendered > 0
    if not silhouette.any():
        return np.zeros(rendered.shape, bool)
    size = 2*ring_px + 1
    ring = cv2.dilate(silhouette.astype(np.uint8), np.ones((size, size), np.uint8)).astype(bool) & ~silhouette
    valid = depth > 0
    return (silhouette & valid & (depth < rendered - margin)) | (ring & valid & (depth < rendered[silhouette].min() - margin))


def hub_up(pose, up_in_camera):
    """pose, or pose turned over (FLIP) when its mesh +z (the hub side) points against up_in_camera (world up in
    the camera frame). The turned-over pose covers the same disc; only the hub moves to the other side."""
    if up_in_camera is None or pose[:3, 2] @ np.asarray(up_in_camera, float) >= 0:
        return pose
    return pose @ FLIP


def visible_iou(silhouette, mask, occluded):
    """IoU of the unoccluded silhouette with the gear-colored mask, and the unoccluded fraction of the silhouette."""
    visible = silhouette & ~occluded
    iou = (visible & mask).sum() / max(int((visible | mask).sum()), 1)
    return float(iou), float(visible.sum() / max(int(silhouette.sum()), 1))


class GearTracker:
    """Registers on the gear-colored mask, then tracks. A pose is only returned when its rendered silhouette
    overlaps the mask (IoU >= min_iou) and the median depth residual is at most max_depth_error metres;
    otherwise the tracker reregisters (at once after a failed registration, after 3 bad tracked frames).

    Occlusion (the hand between camera and gear): pixels nearer than the gear rendered at the pose are left out of
    the IoU and the depth residual, as long as at least min_visible of the silhouette stays visible, and while
    tracking, the pixels in front of the last accepted pose are removed from the depth FoundationPose refines on,
    so the hand does not pull the pose toward the camera.

    Upside down: the gear disc is symmetric about its mid-plane and the hub is a small part of the gear (4% of the
    gear48 disc), so a registration with the hub pointing into the table renders and scores almost like the right
    one. With up_in_camera (world +z in the camera frame) a registration whose hub points down is turned over
    (hub_up) before it is checked and tracked; tracking only refines and does not turn the gear over again.

    A beam (reference_in_camera: the trained start orientation in the camera frame) instead keeps, of its box
    symmetries (symmetry), the registration nearest that orientation, so the policy sees the beam the way it lay in
    training whichever end the registration picked. With the end marker (marker_ranges: HSV ranges of the paper on
    the beam's marked end, marker_end: that end in the mesh frame) the registration that puts the marked end at the
    seen paper wins; the reference orientation only decides when the paper is not seen (marker_misses counts those)."""
    reference_in_camera = None
    marker_ranges = marker_end = None
    MARKER_SEARCH_PX = 25       # the paper is looked for this far around the beam's color mask

    def __init__(self, mesh, symmetry, ranges, *, scorer, refiner, glctx, debug_dir,
                 register_iterations=5, track_iterations=2, min_iou=.45, max_depth_error=.02, min_visible=.25,
                 up_in_camera=None, reference_in_camera=None, marker_ranges=None, marker_end=None):
        import torch
        from estimater import FoundationPose
        from Utils import draw_posed_3d_box, draw_xyz_axis, nvdiffrast_render

        self.torch, self.FoundationPose = torch, FoundationPose
        self.draw_box, self.draw_axis, self.render = draw_posed_3d_box, draw_xyz_axis, nvdiffrast_render
        self.mesh, self.symmetry, self.ranges = mesh, symmetry, ranges
        self.scorer, self.refiner, self.glctx, self.debug_dir = scorer, refiner, glctx, str(debug_dir)
        self.register_iterations, self.track_iterations = register_iterations, track_iterations
        self.min_iou, self.max_depth_error, self.min_visible = min_iou, max_depth_error, min_visible
        self.up_in_camera = None if up_in_camera is None else np.asarray(up_in_camera, float)
        self.reference_in_camera = None if reference_in_camera is None else np.asarray(reference_in_camera, float)
        self.marker_ranges, self.marker_end = marker_ranges, None if marker_end is None else np.asarray(marker_end, float)
        self.marker_misses = 0      # beam registrations without the end marker in view
        self.flips = 0              # registrations turned over by hub_up
        self.estimator = None
        self.bad_frames = 0
        self.last_render = None     # rendered depth at the last accepted pose

    @property
    def registering(self):
        return self.estimator is None or self.estimator.pose_last is None

    def reset(self):
        if self.estimator is not None:
            self.estimator.pose_last = None
        self.bad_frames = 0
        self.last_render = None

    def mask(self, frame):
        """The gear-colored mask, or None (and a reset) when the gear is not in view."""
        try:
            return color_gear_mask(frame.rgb, frame.depth, self.ranges)
        except ValueError:
            self.reset()
            return None

    def seen_marker(self, frame, mask):
        """Camera-frame point of the end marker near the beam mask, or None."""
        import cv2
        size = 2*self.MARKER_SEARCH_PX + 1
        near = cv2.dilate(mask.astype(np.uint8), np.ones((size, size), np.uint8)) > 0
        return marker_point(frame.rgb, frame.depth, frame.K, near, self.marker_ranges)

    def keep_hub_up(self, estimator, pose, frame=None, mask=None):
        """hub_up for a registered pose; a turned-over pose also replaces the estimator's pose_last (the centered
        mesh pose that track_one refines from). For a beam: the marked end at the seen marker (frame, mask), else
        the turn nearest reference_in_camera."""
        marked = None
        if self.marker_ranges is not None and frame is not None and mask is not None:
            marker = self.seen_marker(frame, mask)
            marked = None if marker is None else marked_symmetric(pose, self.symmetry, self.marker_end, marker)
            self.marker_misses += marked is None
        if marked is not None:
            upright = marked
            if np.allclose(upright, pose):
                return pose
        elif self.reference_in_camera is not None:
            upright = nearest_symmetric(pose, self.symmetry, self.reference_in_camera)
            if np.allclose(upright, pose):
                return pose
        else:
            upright = hub_up(pose, self.up_in_camera)
        if upright is pose:
            return pose
        to_centered = estimator.get_tf_to_centered_mesh().cpu().numpy()
        last = estimator.pose_last
        estimator.pose_last = self.torch.as_tensor(upright @ np.linalg.inv(to_centered), dtype=last.dtype,
                                                   device=last.device)
        self.flips += 1
        return upright

    def step(self, frame, mask):
        """(pose or None, metrics, mode, overlay RGB or None) for a frame with rgb, depth (m) and K."""
        metrics = {'status': 'SEARCHING', 'mask_iou': None, 'median_depth_error_m': None, 'visible_fraction': None}
        mode, pose, overlay = 'SEARCHING', None, None
        if mask is None:
            return pose, metrics, mode, overlay
        if self.estimator is None:
            color = np.median(frame.rgb[mask], axis=0).astype(np.uint8)
            self.mesh.visual.vertex_colors = np.tile(np.r_[color, 255], (len(self.mesh.vertices), 1))
            self.estimator = self.FoundationPose(self.mesh.vertices, self.mesh.vertex_normals, mesh=self.mesh,
                                                 symmetry_tfs=self.symmetry, scorer=self.scorer,
                                                 refiner=self.refiner, glctx=self.glctx, debug=0,
                                                 debug_dir=self.debug_dir)
        estimator = self.estimator
        if estimator.pose_last is None:
            pose = estimator.register(K=frame.K, rgb=frame.rgb, depth=frame.depth, ob_mask=mask,
                                      iteration=self.register_iterations)
            mode = 'REGISTER'
            if estimator.pose_last is None:
                pose = None
            else:
                pose = self.keep_hub_up(estimator, pose, frame, mask)
        else:
            depth = frame.depth
            if self.last_render is not None and self.last_render.shape == depth.shape:
                depth = np.where(occluders(depth, self.last_render), 0, depth).astype(depth.dtype)
            pose = estimator.track_one(rgb=frame.rgb, depth=depth, K=frame.K, iteration=self.track_iterations)
            mode = 'TRACK'
        if pose is not None and np.isfinite(pose).all() and pose[2, 3] > 0:
            torch = self.torch
            # Render the centered mesh at its matching centered pose.
            centered = pose @ np.linalg.inv(estimator.get_tf_to_centered_mesh().cpu().numpy())
            _, rendered, _ = self.render(
                K=frame.K, H=frame.depth.shape[0], W=frame.depth.shape[1],
                ob_in_cams=torch.as_tensor(centered[None], dtype=torch.float32, device='cuda'),
                glctx=self.glctx, mesh_tensors=estimator.mesh_tensors)
            rendered = rendered[0].cpu().numpy()
            silhouette = rendered > 0
            occluded = occluders(frame.depth, rendered)
            overlap = silhouette & mask & (frame.depth > 0) & ~occluded
            iou, visible = visible_iou(silhouette, mask, occluded)
            error = float(np.median(np.abs(rendered[overlap] - frame.depth[overlap]))) if overlap.any() else None
            metrics.update(mask_iou=iou, median_depth_error_m=error, visible_fraction=visible)
            if (iou >= self.min_iou and visible >= self.min_visible and error is not None
                    and error <= self.max_depth_error):
                import cv2
                self.bad_frames = 0
                self.last_render = rendered
                overlay = frame.rgb.copy()
                contours, _ = cv2.findContours(silhouette.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
                overlay = self.draw_box(frame.K, img=overlay, ob_in_cam=pose, bbox=self.mesh.bounds)
                overlay = self.draw_axis(overlay, ob_in_cam=pose, scale=.04, K=frame.K, thickness=2,
                                         is_input_rgb=True)
                metrics['status'] = mode
            else:
                pose = None
                self.bad_frames += 1
                metrics['status'] = 'LOST - reacquiring'
                if mode == 'REGISTER' or self.bad_frames >= 3:
                    estimator.pose_last = None
                    self.bad_frames = 0
                    self.last_render = None
        else:
            estimator.pose_last = None
            self.last_render = None
            pose = None
            metrics['status'] = 'LOST - reacquiring'
        self.torch.cuda.synchronize()
        return pose, metrics, mode, overlay
