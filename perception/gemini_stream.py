"""Read the newest calibrated Gemini RGB-D pair without queuing old frames."""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class RGBDFrame:
    sequence: int
    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    received_at: float
    timestamp_delta_ms: float


class GeminiStream:
    def __init__(self, device_index: int = 0, warmup: int = 30):
        self.device_index = device_index
        self.warmup = warmup
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._latest = None
        self._error = None
        self._thread = None
        self.metadata = {}

    def __enter__(self):
        import pyorbbecsdk as ob

        self.ob = ob
        ob.Context.set_logger_level(ob.OBLogLevel.ERROR)
        self.context = ob.Context()
        devices = self.context.query_devices()
        if not 0 <= self.device_index < devices.get_count():
            raise RuntimeError(f'Camera index {self.device_index} not found ({devices.get_count()} devices)')
        self.device = devices.get_device_by_index(self.device_index)
        info = self.device.get_device_info()
        self.metadata = {
            'device': info.get_name(), 'serial': info.get_serial_number(),
            'firmware': info.get_firmware_version(),
            'isp': {key: self.device.get_extension_info(key) for key in ('IspFwVer', 'IspNeedVer')
                    if self.device.is_extension_info_exist(key)},
            'depth_unit': 'metre', 'frame': 'camera optical: +X right, +Y down, +Z forward',
            'alignment': 'SDK depth-to-color, then both images undistorted with color calibration',
            'undistorted': True,
        }
        self.pipeline = ob.Pipeline(self.device)
        config = ob.Config()
        color = self.pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR).get_video_stream_profile(
            640, 480, ob.OBFormat.MJPG, 30)
        depth = self.pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR).get_video_stream_profile(
            640, 400, ob.OBFormat.Y16, 30)
        config.enable_stream(color)
        config.enable_stream(depth)
        config.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
        self.pipeline.enable_frame_sync()
        self.align = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
        self.pipeline.start(config)
        self._thread = threading.Thread(target=self._capture, name='gemini-capture', daemon=True)
        self._thread.start()
        return self

    def _capture(self):
        try:
            maps = None
            sequence = 0
            last_pair = time.monotonic()
            while not self._stop.is_set():
                if time.monotonic() - last_pair > 20:
                    raise TimeoutError('No synchronized Gemini RGB-D pair for 20 seconds')
                frames = self.pipeline.wait_for_frames(500)
                if frames is None:
                    continue
                cf, df = frames.get_color_frame(), frames.get_depth_frame()
                if cf is None or df is None:
                    continue
                delta = abs(cf.get_timestamp_us() - df.get_timestamp_us()) / 1000
                if delta > 50:
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
                if cf.get_format() != self.ob.OBFormat.MJPG:
                    raise RuntimeError(f'Unexpected color format {cf.get_format()}')
                bgr = cv2.imdecode(np.asarray(cf.get_data()), cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                raw = np.frombuffer(df.get_data(), dtype=np.uint16).reshape(df.get_height(), df.get_width())
                depth = raw.astype(np.float32) * (float(df.get_depth_scale()) / 1000)
                if depth.shape != bgr.shape[:2]:
                    raise RuntimeError('SDK depth-to-color alignment returned inconsistent sizes')
                if maps is None:
                    profile = cf.get_stream_profile().as_video_stream_profile()
                    intr, dist = profile.get_intrinsic(), profile.get_distortion()
                    K = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]], dtype=np.float64)
                    distortion = np.array([dist.k1, dist.k2, dist.p1, dist.p2, dist.k3,
                                           dist.k4, dist.k5, dist.k6])
                    maps = cv2.initUndistortRectifyMap(K, distortion, None, K,
                                                     (bgr.shape[1], bgr.shape[0]), cv2.CV_32FC1)
                    self.metadata.update(K=K.tolist(), resolution=[bgr.shape[1], bgr.shape[0]],
                                         original_distortion=distortion.tolist())
                rgb = cv2.cvtColor(cv2.remap(bgr, *maps, cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
                depth = cv2.remap(depth, *maps, cv2.INTER_NEAREST)
                depth[(depth < .05) | (depth > 5)] = 0
                frame = RGBDFrame(sequence, rgb, depth, K, received_at, delta)
                with self._condition:
                    self._latest = frame
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()

    def read(self, after: int = -1, timeout: float = 5) -> RGBDFrame:
        with self._condition:
            ready = self._condition.wait_for(
                lambda: self._error is not None or self._stop.is_set() or
                (self._latest is not None and self._latest.sequence > after), timeout)
            if self._error is not None:
                raise RuntimeError('Gemini capture failed') from self._error
            if not ready or self._stop.is_set():
                raise TimeoutError('Timed out waiting for a new Gemini frame')
            return self._latest

    def __exit__(self, *_):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.pipeline.stop()
        with self._condition:
            self._condition.notify_all()
