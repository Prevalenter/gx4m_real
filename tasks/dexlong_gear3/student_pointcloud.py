"""Read point clouds from the Orbbec Gemini 335 wrist camera, in the same form as the step0 student.

DexLongGear3SplitStep0StudentEnv._pointcloud samples an even rows x cols pixel grid (512 points)
from distance_to_image_plane depth, unprojects it with the camera intrinsics into the ROS optical
frame (+X right, +Y down, +Z forward, metres) and marks points outside [near, far] invalid
(zeroed). StudentPointCloud gives the same (points, mask) from the real depth stream;
Orbbec Y16 depth is also z-depth, not ray length.

    python tasks/dexlong_gear3/student_pointcloud.py --frames 60 --save
    python tasks/dexlong_gear3/student_pointcloud.py --show
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from gx4m_real.camera.cloud import even_pixel_grid, unproject, write_ply
from gx4m_real.camera.orbbec import OrbbecDepthCamera

SIM_CAMERA = Path(__file__).resolve().parent/'config'/'wrist_camera.json'
OUT_DIR = Path(__file__).resolve().parent/'captures'


def sim_camera():
    return json.loads(SIM_CAMERA.read_text(encoding='utf-8'))


class StudentPointCloud:
    """Student input from a depth image: points (N, 3) zeroed where invalid, mask (N,) float."""

    def __init__(self, camera, num_points=512, near=None, far=None):
        clip = sim_camera()['clipping_range']
        self.near, self.far = near or clip[0], far or clip[1]
        self.K = camera.K
        self.u, self.v = even_pixel_grid(num_points, camera.width, camera.height)

    def __call__(self, depth):
        points = unproject(depth, self.u, self.v, self.K)
        valid = (points[:, 2] >= self.near) & (points[:, 2] <= self.far)
        return np.where(valid[:, None], points, 0.).astype(np.float32), valid.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--width', type=int, default=424, help='Depth width (424x240: closest range, ~8 cm)')
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--serial', default=None)
    parser.add_argument('--preset', default='Medium Density', help='Depth preset: Hand, Default, High Accuracy, High Density, ...')
    parser.add_argument('--frames', type=int, default=60, help='Frames to read (0: until q in --show)')
    parser.add_argument('--save', action='store_true', help='Save the last frame (npz + ply) under tasks/dexlong_gear3/captures')
    parser.add_argument('--show', action='store_true', help='OpenCV window: depth with the 512 sample pixels')
    args = parser.parse_args()

    sim = sim_camera()
    camera = OrbbecDepthCamera(args.width, args.height, args.fps, args.serial, *sim['clipping_range'],
                               preset=args.preset)
    student = StudentPointCloud(camera)
    hfov, vfov = camera.fov_deg
    sim_h = 2*math.degrees(math.atan(sim['horizontal_aperture']/2/sim['focal_length']))
    sim_v = 2*math.degrees(math.atan(sim['horizontal_aperture']*sim['height']/sim['width']/2/sim['focal_length']))
    print(f'{camera.name} {camera.serial} [{camera.preset}]: depth {camera.width}x{camera.height}@{args.fps}, '
          f'fx {camera.k["fx"]:.1f} fy {camera.k["fy"]:.1f} cx {camera.k["cx"]:.1f} cy {camera.k["cy"]:.1f}, '
          f'FOV {hfov:.1f} x {vfov:.1f} deg (sim {sim_h:.1f} x {sim_v:.1f}); range {student.near}-{student.far} m',
          flush=True)
    if args.show:
        import cv2
    count, start, last = 0, time.perf_counter(), None
    try:
        while args.frames <= 0 or count < args.frames:
            depth = camera.read_depth()
            if depth is None:
                print('timeout waiting for depth', flush=True)
                continue
            points, mask = student(depth)
            count += 1
            last = points, mask, depth
            if count % 30 == 0 or count == 1:
                z = depth[depth > 0]
                inside = points[mask > 0]
                print(f'frame {count}: valid depth {z.size/depth.size:.0%} '
                      f'(median {np.median(z) if z.size else 0:.3f} m), student points {int(mask.sum())}/{len(mask)} '
                      f'in range, z {inside[:, 2].min() if len(inside) else 0:.3f}-'
                      f'{inside[:, 2].max() if len(inside) else 0:.3f} m', flush=True)
            if args.show:
                shown = np.clip((depth-student.near)/(student.far-student.near), 0, 1)
                image = cv2.applyColorMap((255*(1-shown)).astype(np.uint8), cv2.COLORMAP_TURBO)
                image[depth == 0] = 0
                for u, v, ok in zip(student.u, student.v, mask):
                    cv2.circle(image, (int(u), int(v)), 2, (255, 255, 255) if ok else (80, 80, 80), -1)
                cv2.imshow('orbbec depth (white: valid student points)', image)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
        elapsed = time.perf_counter()-start
        print(f'{count} frames in {elapsed:.1f} s ({count/elapsed:.1f} fps)', flush=True)
        if args.save and last is not None:
            points, mask, depth = last
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            base = OUT_DIR/time.strftime('orbbec_%Y%m%d_%H%M%S')
            np.savez_compressed(f'{base}.npz', depth=depth, points=points, mask=mask, k=json.dumps(camera.k),
                                u=student.u, v=student.v)
            write_ply(f'{base}_student.ply', points[mask > 0])
            write_ply(f'{base}_full.ply', camera.dense_points(depth, stride=2))
            print(f'saved {base}.npz, {base}_student.ply, {base}_full.ply', flush=True)
    finally:
        camera.close()


if __name__ == '__main__':
    main()
