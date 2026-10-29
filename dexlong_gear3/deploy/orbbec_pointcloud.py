"""Read point clouds from the Orbbec Gemini 335 wrist camera, in the same form as the step0 student.

DexLongGear3SplitStep0StudentEnv._pointcloud samples an even rows x cols pixel grid (512 points)
from distance_to_image_plane depth, unprojects it with the camera intrinsics into the ROS optical
frame (+X right, +Y down, +Z forward, metres) and marks points outside [near, far] invalid
(zeroed). OrbbecDepthCamera.read() returns the same (points, mask) from the real depth stream;
Orbbec Y16 depth is also z-depth, not ray length.

Run with C:/env_isaaclab/python.exe (pyorbbecsdk):
    python dexlong_gear3/deploy/orbbec_pointcloud.py --frames 60 --save
    python dexlong_gear3/deploy/orbbec_pointcloud.py --show
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
from pyorbbecsdk import AlignFilter, Config, Context, OBFormat, OBLogLevel, OBSensorType, OBStreamType, Pipeline

CONFIG_DIR = Path(__file__).resolve().parents[1]/'config'
SIM_CAMERA = CONFIG_DIR/'wrist_camera.json'
OUT_DIR = Path(__file__).resolve().parent/'captures'


def pixel_grid(n, width, height):
    """Even pixel grid (rows x cols >= n, first n), as in the student env's _pointcloud."""
    rows = max(1, int(math.sqrt(n*height/width)))
    cols = math.ceil(n/rows)
    v, u = np.meshgrid(np.linspace(0, height-1, rows).astype(int), np.linspace(0, width-1, cols).astype(int),
                       indexing='ij')
    return u.reshape(-1)[:n], v.reshape(-1)[:n]


def unproject(depth, u, v, k):
    """Depth (m) at pixels (u, v) -> points (..., 3) in the ROS optical frame."""
    z = depth[v, u]
    return np.stack(((u-k['cx'])*z/k['fx'], (v-k['cy'])*z/k['fy'], z), -1)


def load_preset(device, name=None):
    """Switch the depth preset (stereo matching / filtering settings; None keeps the current one)."""
    if name:
        presets = device.get_available_preset_list()
        available = [presets.get_name_by_index(i) for i in range(presets.get_count())]
        if name not in available:
            raise ValueError(f'Unknown preset {name!r}, available: {available}')
        device.load_preset(name)
    return device.get_current_preset_name()


class OrbbecDepthCamera:
    """Depth stream of one Orbbec camera; read() gives the student point cloud.

    color=True also streams RGB and registers it onto the depth pixels (SDK AlignFilter, colour to
    depth), so read_depth() returns (depth, rgb) with rgb (H, W, 3) uint8, black where depth is missing.
    """

    def __init__(self, width=424, height=240, fps=30, serial=None, num_points=512, near=None, far=None,
                 color=False, color_width=640, color_height=360, preset=None):
        sim = json.loads(SIM_CAMERA.read_text(encoding='utf-8'))
        self.near, self.far = near or sim['clipping_range'][0], far or sim['clipping_range'][1]
        self.num_points = num_points
        Context.set_logger_level(OBLogLevel.ERROR)
        devices = Context().query_devices()
        if devices.get_count() == 0:
            raise RuntimeError('No Orbbec camera found')
        self.device = (devices.get_device_by_serial_number(serial) if serial else devices.get_device_by_index(0))
        info = self.device.get_device_info()
        self.name, self.serial = info.get_name(), info.get_serial_number()
        self.preset = load_preset(self.device, preset)
        self.pipeline = Pipeline(self.device)
        profile = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_video_stream_profile(
            width, height, OBFormat.Y16, fps)
        intrinsic = profile.get_intrinsic()
        self.width, self.height = profile.get_width(), profile.get_height()
        self.k = dict(fx=intrinsic.fx, fy=intrinsic.fy, cx=intrinsic.cx, cy=intrinsic.cy)
        self.u, self.v = pixel_grid(num_points, self.width, self.height)
        config = Config()
        config.enable_stream(profile)
        self.align = None
        if color:
            config.enable_stream(self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
                                 .get_video_stream_profile(color_width, color_height, OBFormat.RGB, fps))
            self.pipeline.enable_frame_sync()
            self.align = AlignFilter(align_to_stream=OBStreamType.DEPTH_STREAM)
        self.pipeline.start(config)

    @property
    def fov_deg(self):
        return (2*math.degrees(math.atan(self.width/2/self.k['fx'])),
                2*math.degrees(math.atan(self.height/2/self.k['fy'])))

    def read_depth(self, timeout_ms=1000):
        """Latest depth image in metres (H, W) float32, 0 = no return; None on timeout.
        With color=True: (depth, rgb), None until a frame set holds both streams."""
        frames = self.pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            return None
        if self.align is not None:
            if frames.get_color_frame() is None or frames.get_depth_frame() is None:
                return None
            frames = self.align.process(frames)
            frames = frames.as_frame_set() if frames is not None else None
            if frames is None or frames.get_color_frame() is None:
                return None
        frame = frames.get_depth_frame()
        if frame is None:
            return None
        raw = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape(frame.get_height(), frame.get_width())
        depth = raw.astype(np.float32)*(frame.get_depth_scale()/1000.)
        if self.align is None:
            return depth
        color = frames.get_color_frame()
        return depth, np.frombuffer(color.get_data(), np.uint8).reshape(color.get_height(), color.get_width(), 3)

    def pointcloud(self, depth):
        """Student input from a depth image: points (N, 3) zeroed where invalid, mask (N,) float."""
        points = unproject(depth, self.u, self.v, self.k)
        valid = (points[:, 2] >= self.near) & (points[:, 2] <= self.far)
        return np.where(valid[:, None], points, 0.).astype(np.float32), valid.astype(np.float32)

    def read(self, timeout_ms=1000):
        depth = self.read_depth(timeout_ms)
        if isinstance(depth, tuple):
            depth = depth[0]
        return None if depth is None else (*self.pointcloud(depth), depth)

    def full_pointcloud(self, depth, stride=1):
        """Every (stride-th) valid pixel, for inspection."""
        v, u = np.mgrid[0:self.height:stride, 0:self.width:stride]
        points = unproject(depth, u.reshape(-1), v.reshape(-1), self.k)
        return points[(points[:, 2] >= self.near) & (points[:, 2] <= self.far)]

    def close(self):
        self.pipeline.stop()


def write_ply(path, points):
    with open(path, 'w', encoding='ascii') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\n'
                'property float z\nend_header\n')
        np.savetxt(f, points, fmt='%.5f')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--width', type=int, default=424, help='Depth width (424x240: closest range, ~8 cm)')
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--serial', default=None)
    parser.add_argument('--preset', default='Medium Density', help='Depth preset: Hand, Default, High Accuracy, High Density, ...')
    parser.add_argument('--frames', type=int, default=60, help='Frames to read (0: until q in --show)')
    parser.add_argument('--save', action='store_true', help='Save the last frame (npz + ply) under deploy/captures')
    parser.add_argument('--show', action='store_true', help='OpenCV window: depth with the 512 sample pixels')
    args = parser.parse_args()

    camera = OrbbecDepthCamera(args.width, args.height, args.fps, args.serial, preset=args.preset)
    hfov, vfov = camera.fov_deg
    sim = json.loads(SIM_CAMERA.read_text(encoding='utf-8'))
    sim_h = 2*math.degrees(math.atan(sim['horizontal_aperture']/2/sim['focal_length']))
    sim_v = 2*math.degrees(math.atan(sim['horizontal_aperture']*sim['height']/sim['width']/2/sim['focal_length']))
    print(f'{camera.name} {camera.serial} [{camera.preset}]: depth {camera.width}x{camera.height}@{args.fps}, '
          f'fx {camera.k["fx"]:.1f} fy {camera.k["fy"]:.1f} cx {camera.k["cx"]:.1f} cy {camera.k["cy"]:.1f}, '
          f'FOV {hfov:.1f} x {vfov:.1f} deg (sim {sim_h:.1f} x {sim_v:.1f}); range {camera.near}-{camera.far} m',
          flush=True)
    if args.show:
        import cv2
    count, start, last = 0, time.perf_counter(), None
    try:
        while args.frames <= 0 or count < args.frames:
            result = camera.read()
            if result is None:
                print('timeout waiting for depth', flush=True)
                continue
            points, mask, depth = result
            count += 1
            last = result
            if count % 30 == 0 or count == 1:
                z = depth[depth > 0]
                inside = points[mask > 0]
                print(f'frame {count}: valid depth {z.size/depth.size:.0%} '
                      f'(median {np.median(z) if z.size else 0:.3f} m), student points {int(mask.sum())}/{len(mask)} '
                      f'in range, z {inside[:, 2].min() if len(inside) else 0:.3f}-'
                      f'{inside[:, 2].max() if len(inside) else 0:.3f} m', flush=True)
            if args.show:
                shown = np.clip((depth-camera.near)/(camera.far-camera.near), 0, 1)
                image = cv2.applyColorMap((255*(1-shown)).astype(np.uint8), cv2.COLORMAP_TURBO)
                image[depth == 0] = 0
                for u, v, ok in zip(camera.u, camera.v, mask):
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
                                u=camera.u, v=camera.v)
            write_ply(f'{base}_student.ply', points[mask > 0])
            write_ply(f'{base}_full.ply', camera.full_pointcloud(depth, stride=2))
            print(f'saved {base}.npz, {base}_student.ply, {base}_full.ply', flush=True)
    finally:
        camera.close()


if __name__ == '__main__':
    main()
