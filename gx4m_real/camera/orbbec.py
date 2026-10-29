"""Orbbec (Gemini 335) cameras through pyorbbecsdk.

Importing this module loads neither the camera SDK nor OpenCV: pyorbbecsdk, cv2 and PIL are imported
when a camera opens. The control app's environment has no OpenCV, so OrbbecColorCamera decodes with PIL.

Three ways to stream, by what the caller needs:
- OrbbecColorCamera: raw (distorted) colour only, with the SDK intrinsics for offline calibration.
- OrbbecRGBDStream: depth registered to colour and both undistorted, newest pair from a background thread.
- OrbbecDepthCamera: depth at the depth sensor's own resolution, optionally with colour registered to it.
"""
from __future__ import annotations

import io
import math
from functools import lru_cache
import threading
import time

import numpy as np

from .frames import ColorFrame, RGBDFrame

OPTICAL_FRAME = 'camera optical: +X right, +Y down, +Z forward'
DISTORTION_KEYS = ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6')


def sdk():
    try:
        import pyorbbecsdk
    except ImportError as exc:
        raise RuntimeError('pyorbbecsdk is not installed; run the setup.sh of this app first') from exc
    return pyorbbecsdk


@lru_cache(maxsize=1)
def sdk_context():
    """Keep the shared SDK context alive while its devices and streams are in use."""
    ob = sdk()
    ob.Context.set_logger_level(ob.OBLogLevel.ERROR)
    return ob.Context()


def open_device(index: int = 0, serial: str | None = None):
    """Device by serial number if given, else by index. Lowers the SDK log level to errors."""
    devices = sdk_context().query_devices()
    if serial:
        return devices.get_device_by_serial_number(serial)
    if not 0 <= index < devices.get_count():
        raise RuntimeError(f'Orbbec camera {index} not found ({devices.get_count()} devices)')
    return devices.get_device_by_index(index)


def device_info(device) -> dict:
    info = device.get_device_info()
    return {'device': info.get_name(), 'serial': info.get_serial_number(),
            'firmware': info.get_firmware_version(),
            'isp': {key: device.get_extension_info(key) for key in ('IspFwVer', 'IspNeedVer')
                    if device.is_extension_info_exist(key)}}


def load_preset(device, name: str | None = None) -> str:
    """Switch the depth preset (stereo matching / filtering settings; None keeps the current one)."""
    if name:
        presets = device.get_available_preset_list()
        available = [presets.get_name_by_index(i) for i in range(presets.get_count())]
        if name not in available:
            raise ValueError(f'Unknown preset {name!r}, available: {available}')
        device.load_preset(name)
    return device.get_current_preset_name()


def stream_profile(pipeline, sensor: str, width: int, height: int, fmt: str, fps: int):
    """sensor: 'COLOR' or 'DEPTH'; fmt: an OBFormat name such as 'MJPG', 'RGB' or 'Y16'."""
    ob = sdk()
    return pipeline.get_stream_profile_list(getattr(ob.OBSensorType, f'{sensor}_SENSOR')).get_video_stream_profile(
        width, height, getattr(ob.OBFormat, fmt), fps)


def camera_matrix(profile) -> tuple[np.ndarray, np.ndarray]:
    """SDK intrinsics of a video stream profile: K (3, 3) and distortion (k1 k2 p1 p2 k3 k4 k5 k6)."""
    intr, dist = profile.get_intrinsic(), profile.get_distortion()
    K = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]], dtype=np.float64)
    return K, np.array([getattr(dist, key) for key in DISTORTION_KEYS], dtype=np.float64)


def depth_metres(frame) -> np.ndarray:
    """Y16 depth frame -> (H, W) float32 metres, 0 = no return. Orbbec depth is z-depth, not ray length."""
    raw = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape(frame.get_height(), frame.get_width())
    return raw.astype(np.float32) * (float(frame.get_depth_scale()) / 1000)


def decode_color(frame, bgr: bool = False) -> np.ndarray | None:
    """MJPG or RGB colour frame -> (H, W, 3) uint8, RGB unless bgr; None if MJPG decoding fails."""
    import cv2
    ob = sdk()
    data = np.asarray(frame.get_data())
    if frame.get_format() == ob.OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return image if image is None or bgr else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if frame.get_format() == ob.OBFormat.RGB:
        image = data.reshape(frame.get_height(), frame.get_width(), 3).copy()
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if bgr else image
    raise RuntimeError(f'Unsupported color format: {frame.get_format()}')


class OrbbecColorCamera:
    """Raw, distorted colour only; calibration must use the saved intrinsics."""

    def __init__(self, device_index=0, serial=None, width=640, height=480, fps=30, warmup=10):
        self.device_index, self.serial = device_index, serial
        self.width, self.height, self.fps, self.warmup = width, height, fps, warmup
        self.pipeline = None
        self.sequence = 0
        self.metadata = {}

    def open(self):
        ob = self.ob = sdk()
        device = open_device(self.device_index, self.serial)
        self.pipeline = ob.Pipeline(device)
        profile = stream_profile(self.pipeline, 'COLOR', self.width, self.height, 'MJPG', self.fps)
        K, distortion = camera_matrix(profile)
        self.metadata = {
            **device_info(device), 'resolution': [self.width, self.height], 'fps': self.fps,
            'K': K.tolist(), 'distortion': dict(zip(DISTORTION_KEYS, distortion.tolist())),
            'distortion_model': str(getattr(profile.get_distortion(), 'model', 'SDK color distortion')),
            'undistorted': False, 'color_order': 'RGB', 'synthetic': False, 'frame': OPTICAL_FRAME,
        }
        config = ob.Config()
        config.enable_stream(profile)
        self.pipeline.start(config)
        # Warm up before the arm starts moving.
        for _ in range(self.warmup):
            self.read()

    def read(self, timeout=2.) -> ColorFrame:
        from PIL import Image
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frames = self.pipeline.wait_for_frames(200)
            if frames is None:
                continue
            color = frames.get_color_frame()
            if color is None:
                continue
            received_at = time.monotonic()
            if color.get_format() != self.ob.OBFormat.MJPG:
                raise RuntimeError(f'Unsupported color format: {color.get_format()}')
            rgb = np.array(Image.open(io.BytesIO(bytes(color.get_data()))).convert('RGB'))
            self.sequence += 1
            return ColorFrame(self.sequence, rgb, received_at, int(color.get_timestamp_us()),
                              int(color.get_system_timestamp_us()))
        raise TimeoutError('Timed out waiting for an Orbbec colour frame')

    def close(self):
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None


class OrbbecRGBDStream:
    """Newest depth-to-colour aligned, undistorted RGB-D pair, without queuing old frames.

    Use as a context manager, or call open() and close().
    """

    def __init__(self, device_index: int = 0, warmup: int = 30, serial: str | None = None,
                 max_delta_ms: float = 50, depth_range: tuple[float, float] = (.05, 5), pair_timeout: float = 20):
        self.device_index, self.serial, self.warmup = device_index, serial, warmup
        self.max_delta_ms, self.depth_range, self.pair_timeout = max_delta_ms, depth_range, pair_timeout
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._latest = None
        self._error = None
        self._thread = None
        self.pipeline = None
        self.metadata = {}

    def open(self):
        ob = self.ob = sdk()
        self.device = open_device(self.device_index, self.serial)
        self.metadata = {
            **device_info(self.device), 'depth_unit': 'metre', 'frame': OPTICAL_FRAME,
            'alignment': 'SDK depth-to-color, then both images undistorted with color calibration',
            'undistorted': True,
        }
        self.pipeline = ob.Pipeline(self.device)
        config = ob.Config()
        config.enable_stream(stream_profile(self.pipeline, 'COLOR', 640, 480, 'MJPG', 30))
        config.enable_stream(stream_profile(self.pipeline, 'DEPTH', 640, 400, 'Y16', 30))
        config.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        self.pipeline.enable_frame_sync()
        self.align = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
        self.pipeline.start(config)
        self._thread = threading.Thread(target=self._capture, name='orbbec-rgbd', daemon=True)
        self._thread.start()
        return self

    __enter__ = open

    def _capture(self):
        import cv2
        try:
            maps = None
            sequence = 0
            last_pair = time.monotonic()
            while not self._stop.is_set():
                if time.monotonic() - last_pair > self.pair_timeout:
                    raise TimeoutError(f'No synchronized Orbbec RGB-D pair for {self.pair_timeout:g} seconds')
                frames = self.pipeline.wait_for_frames(500)
                if frames is None:
                    continue
                cf, df = frames.get_color_frame(), frames.get_depth_frame()
                if cf is None or df is None:
                    continue
                delta = abs(cf.get_timestamp_us() - df.get_timestamp_us()) / 1000
                if delta > self.max_delta_ms:
                    continue
                received_at = time.monotonic()
                frames = self.align.process(frames)
                if frames is None:
                    continue
                frames = frames.as_frame_set()
                cf, df = frames.get_color_frame(), frames.get_depth_frame()
                if cf is None or df is None:
                    continue
                last_pair = received_at
                sequence += 1
                if sequence < self.warmup:
                    continue
                rgb = decode_color(cf)
                if rgb is None:
                    continue
                depth = depth_metres(df)
                if depth.shape != rgb.shape[:2]:
                    raise RuntimeError('SDK depth-to-color alignment returned inconsistent sizes')
                if maps is None:
                    K, distortion = camera_matrix(cf.get_stream_profile().as_video_stream_profile())
                    maps = cv2.initUndistortRectifyMap(K, distortion, None, K,
                                                     (rgb.shape[1], rgb.shape[0]), cv2.CV_32FC1)
                    self.metadata.update(K=K.tolist(), resolution=[rgb.shape[1], rgb.shape[0]],
                                         original_distortion=distortion.tolist())
                rgb = cv2.remap(rgb, *maps, cv2.INTER_LINEAR)
                depth = cv2.remap(depth, *maps, cv2.INTER_NEAREST)
                near, far = self.depth_range
                depth[(depth < near) | (depth > far)] = 0
                frame = RGBDFrame(sequence, rgb, depth, K, received_at, delta)
                with self._condition:
                    self._latest = frame
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()

    def read(self, after: int = -1, timeout: float = 5) -> RGBDFrame:
        """Newest frame with sequence > after."""
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._error is not None or self._stop.is_set() or
                (self._latest is not None and self._latest.sequence > after), timeout)
            if self._error is not None:
                raise RuntimeError('Orbbec RGB-D capture failed') from self._error
            if not ready or self._stop.is_set():
                raise TimeoutError('Timed out waiting for a new Orbbec RGB-D frame')
            return self._latest

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self.pipeline is not None:
            self.pipeline.stop()
        with self._condition:
            self._condition.notify_all()

    def __exit__(self, *_):
        self.close()


class OrbbecDepthCamera:
    """Depth stream at the depth sensor's resolution.

    color=True also streams RGB and registers it onto the depth pixels (SDK AlignFilter, colour to
    depth), so read_depth() returns (depth, rgb) with rgb (H, W, 3) uint8, black where depth is missing.
    near / far only bound dense_points(); read_depth() returns the raw depth.
    """

    def __init__(self, width=424, height=240, fps=30, serial=None, near=0., far=math.inf,
                 color=False, color_width=640, color_height=360, preset=None, device_index=0):
        ob = sdk()
        self.near, self.far = near, far
        self.device = open_device(device_index, serial)
        info = device_info(self.device)
        self.name, self.serial = info['device'], info['serial']
        self.preset = load_preset(self.device, preset)
        self.pipeline = ob.Pipeline(self.device)
        profile = stream_profile(self.pipeline, 'DEPTH', width, height, 'Y16', fps)
        self.width, self.height = profile.get_width(), profile.get_height()
        self.K, _ = camera_matrix(profile)
        self.k = dict(fx=self.K[0, 0], fy=self.K[1, 1], cx=self.K[0, 2], cy=self.K[1, 2])
        config = ob.Config()
        config.enable_stream(profile)
        self.align = None
        if color:
            config.enable_stream(stream_profile(self.pipeline, 'COLOR', color_width, color_height, 'RGB', fps))
            self.pipeline.enable_frame_sync()
            self.align = ob.AlignFilter(align_to_stream=ob.OBStreamType.DEPTH_STREAM)
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
        depth = depth_metres(frame)
        if self.align is None:
            return depth
        color = frames.get_color_frame()
        return depth, np.frombuffer(color.get_data(), np.uint8).reshape(color.get_height(), color.get_width(), 3)

    def dense_points(self, depth, stride=1):
        """Every stride-th pixel within [near, far], for inspection."""
        from .cloud import dense_points
        return dense_points(depth, self.K, stride, self.near, self.far)

    def close(self):
        self.pipeline.stop()
