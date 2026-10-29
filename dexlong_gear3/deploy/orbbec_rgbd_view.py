"""Live RGB + depth from the Orbbec Gemini 335 in an OpenCV window.

Depth is coloured over --range (default: the sim camera's 0.04-0.6 m, near = red, far = blue);
black = no return or outside the range. Move the mouse over either image to read the depth there.
Keys: q / Esc quit, s save (color png, depth npy in metres, preview png) under deploy/captures.

Run with C:/env_isaaclab/python.exe:
    python dexlong_gear3/deploy/orbbec_rgbd_view.py
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
from pyorbbecsdk import Config, Context, OBFormat, OBLogLevel, OBSensorType, Pipeline

from orbbec_pointcloud import OUT_DIR, OrbbecDepthCamera, load_preset


class OrbbecRGBDCamera(OrbbecDepthCamera):
    """Depth (see OrbbecDepthCamera) plus an MJPG colour stream, unsynchronised: latest of each."""

    def __init__(self, width=424, height=240, fps=30, color_width=640, color_height=360, serial=None, preset=None):
        # Same device lookup and depth profile as the base class, but both streams start together.
        Context.set_logger_level(OBLogLevel.ERROR)
        devices = Context().query_devices()
        if devices.get_count() == 0:
            raise RuntimeError('No Orbbec camera found')
        self.device = devices.get_device_by_serial_number(serial) if serial else devices.get_device_by_index(0)
        info = self.device.get_device_info()
        self.name, self.serial = info.get_name(), info.get_serial_number()
        self.preset = load_preset(self.device, preset)
        self.pipeline = Pipeline(self.device)
        depth = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_video_stream_profile(
            width, height, OBFormat.Y16, fps)
        color = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_video_stream_profile(
            color_width, color_height, OBFormat.MJPG, fps)
        intrinsic = depth.get_intrinsic()
        self.width, self.height = depth.get_width(), depth.get_height()
        self.k = dict(fx=intrinsic.fx, fy=intrinsic.fy, cx=intrinsic.cx, cy=intrinsic.cy)
        config = Config()
        config.enable_stream(depth)
        config.enable_stream(color)
        self.pipeline.disable_frame_sync()
        self.pipeline.start(config)
        self.color, self.depth = None, None

    def poll(self, timeout_ms=1000):
        """Update the latest colour (BGR) and depth (m) images from the next frame set."""
        frames = self.pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            return
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if color is not None:
            image = cv2.imdecode(np.frombuffer(color.get_data(), np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                self.color = image
        if depth is not None:
            raw = np.frombuffer(depth.get_data(), dtype=np.uint16).reshape(depth.get_height(), depth.get_width())
            self.depth = raw.astype(np.float32)*(depth.get_depth_scale()/1000.)


def colorize(depth, near, far):
    scaled = np.clip((depth-near)/(far-near), 0, 1)
    image = cv2.applyColorMap((255*(1-scaled)).astype(np.uint8), cv2.COLORMAP_TURBO)
    image[(depth < near) | (depth > far)] = 0
    return image


def label(image, text, y=24):
    cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--width', type=int, default=424, help='Depth width (424x240: closest range, ~8 cm)')
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--color-width', type=int, default=640)
    parser.add_argument('--color-height', type=int, default=360)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--range', type=float, nargs=2, default=(.04, .6), metavar=('NEAR', 'FAR'),
                        help='Depth colouring range in metres')
    parser.add_argument('--serial', default=None)
    parser.add_argument('--preset', default='Medium Density', help='Depth preset: Hand, Default, High Accuracy, High Density, ...')
    args = parser.parse_args()

    camera = OrbbecRGBDCamera(args.width, args.height, args.fps, args.color_width, args.color_height, args.serial,
                              args.preset)
    print(f'{camera.name} {camera.serial} [{camera.preset}]: depth {camera.width}x{camera.height}, colour '
          f'{args.color_width}x{args.color_height} @ {args.fps} fps. q/Esc quit, s save.', flush=True)
    window = 'Orbbec RGB | depth'
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    mouse = {'xy': None}
    cv2.setMouseCallback(window, lambda event, x, y, flags, param: mouse.update(xy=(x, y)))
    near, far = args.range
    frames, fps, tick = 0, 0., time.perf_counter()
    try:
        while True:
            camera.poll()
            if camera.color is None or camera.depth is None:
                continue
            height = camera.depth.shape[0]
            color = cv2.resize(camera.color, (round(camera.color.shape[1]*height/camera.color.shape[0]), height))
            depth = camera.depth
            preview = np.hstack((color, colorize(depth, near, far)))
            frames += 1
            if time.perf_counter()-tick >= 1.:
                fps, frames, tick = frames/(time.perf_counter()-tick), 0, time.perf_counter()
            valid = (depth >= near) & (depth <= far)
            label(preview, f'{fps:.1f} fps | depth in {near:.2f}-{far:.2f} m: {valid.mean():.0%} of pixels')
            if mouse['xy'] is not None:
                x, y = mouse['xy']
                u = x-color.shape[1] if x >= color.shape[1] else round(x*camera.width/color.shape[1])
                if 0 <= u < camera.width and 0 <= y < height:
                    # The colour and depth sensors are not aligned: on the colour image this is approximate.
                    label(preview, f'depth at ({u}, {y}): {depth[y, u]:.3f} m', 48)
                    cv2.drawMarker(preview, (color.shape[1]+u, y), (255, 255, 255), cv2.MARKER_CROSS, 14, 1)
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('s'):
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                base = OUT_DIR/time.strftime('rgbd_%Y%m%d_%H%M%S')
                cv2.imwrite(f'{base}_color.png', camera.color)
                np.save(f'{base}_depth_m.npy', depth)
                cv2.imwrite(f'{base}_preview.png', preview)
                print(f'saved {base}_*', flush=True)
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
