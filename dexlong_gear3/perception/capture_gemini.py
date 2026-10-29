"""Capture calibrated, aligned Gemini RGB-D and a point cloud in metres."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np


def save_cloud(path: Path, rgb: np.ndarray, depth: np.ndarray, K: np.ndarray) -> int:
    v, u = np.nonzero(depth > 0)
    z = depth[v, u]
    data = np.empty(len(z), dtype=[(n, '<f4') for n in ('x', 'y', 'z')] +
                    [(n, 'u1') for n in ('red', 'green', 'blue')])
    data['x'] = (u - K[0, 2]) * z / K[0, 0]
    data['y'] = (v - K[1, 2]) * z / K[1, 1]
    data['z'] = z
    for i, name in enumerate(('red', 'green', 'blue')):
        data[name] = rgb[v, u, i]
    header = ('ply\nformat binary_little_endian 1.0\n'
              'comment coordinates in metres; optical frame +X right +Y down +Z forward\n'
              f'element vertex {len(data)}\nproperty float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with path.open('wb') as stream:
        stream.write(header.encode())
        data.tofile(stream)
    return len(data)


def capture(output: Path, device_index: int = 0, warmup: int = 30) -> dict:
    import pyorbbecsdk as ob

    output.mkdir(parents=True, exist_ok=False)
    ob.Context.set_logger_level(ob.OBLogLevel.ERROR)
    context = ob.Context()
    devices = context.query_devices()
    if devices.get_count() <= device_index:
        raise RuntimeError(f'No Gemini at index {device_index}; found {devices.get_count()} devices')
    device = devices.get_device_by_index(device_index)
    info = device.get_device_info()
    firmware = info.get_firmware_version()
    isp = {key: device.get_extension_info(key) for key in ('IspFwVer', 'IspNeedVer')
           if device.is_extension_info_exist(key)}
    print(f'Device: {info.get_name()}, firmware: {firmware}, ISP: {isp}', flush=True)
    pipeline = ob.Pipeline(device)
    config = ob.Config()
    color_profile = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR).get_video_stream_profile(
        640, 480, ob.OBFormat.MJPG, 30)
    depth_profile = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR).get_video_stream_profile(
        640, 400, ob.OBFormat.Y16, 30)
    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)
    config.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    pipeline.enable_frame_sync()
    align = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
    try:
        pipeline.start(config)
    except Exception as exc:
        diagnostic = {'device': info.get_name(), 'firmware': firmware, 'isp': isp,
                      'captured': False, 'error': str(exc)}
        (output / 'capture_error.json').write_text(json.dumps(diagnostic, indent=2) + '\n')
        raise
    try:
        deadline = time.monotonic() + 30
        accepted = 0
        while time.monotonic() < deadline:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue
            cf, df = frames.get_color_frame(), frames.get_depth_frame()
            if cf is None or df is None:
                continue
            delta_ms = abs(cf.get_timestamp_us() - df.get_timestamp_us()) / 1000
            if delta_ms > 50:
                continue
            aligned = align.process(frames)
            if aligned is None:
                continue
            aligned = aligned.as_frame_set()
            cf, df = aligned.get_color_frame(), aligned.get_depth_frame()
            if cf is None or df is None:
                continue
            accepted += 1
            if accepted < warmup:
                continue
            raw_color = np.asarray(cf.get_data())
            if cf.get_format() == ob.OBFormat.MJPG:
                bgr = cv2.imdecode(raw_color, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError('Could not decode color frame')
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            elif cf.get_format() == ob.OBFormat.RGB:
                rgb = raw_color.reshape(cf.get_height(), cf.get_width(), 3).copy()
            else:
                raise RuntimeError(f'Unsupported color format: {cf.get_format()}')
            raw_depth = np.frombuffer(df.get_data(), dtype=np.uint16).reshape(df.get_height(), df.get_width())
            depth = raw_depth.astype(np.float32) * float(df.get_depth_scale()) / 1000
            if depth.shape != rgb.shape[:2]:
                raise RuntimeError(f'SDK alignment failed: {depth.shape} != {rgb.shape[:2]}')
            profile = cf.get_stream_profile().as_video_stream_profile()
            intr = profile.get_intrinsic()
            dist = profile.get_distortion()
            K = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]], dtype=np.float64)
            distortion = np.array([dist.k1, dist.k2, dist.p1, dist.p2, dist.k3,
                                   dist.k4, dist.k5, dist.k6], dtype=np.float64)
            map_x, map_y = cv2.initUndistortRectifyMap(K, distortion, None, K,
                                                     (rgb.shape[1], rgb.shape[0]), cv2.CV_32FC1)
            rgb = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR)
            depth = cv2.remap(depth, map_x, map_y, cv2.INTER_NEAREST)
            depth[(depth < .05) | (depth > 5)] = 0
            meta = {
                'captured_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                'device': info.get_name(), 'serial': info.get_serial_number(), 'firmware': firmware,
                'resolution': [rgb.shape[1], rgb.shape[0]], 'K': K.tolist(),
                'original_distortion': distortion.tolist(), 'undistorted': True,
                'alignment': 'SDK depth-to-color, then both images undistorted with color calibration',
                'depth_unit': 'metre', 'frame': 'camera optical: +X right, +Y down, +Z forward',
                'timestamp_delta_ms': delta_ms,
                'valid_depth_fraction': float((depth > 0).mean()),
            }
            np.savez_compressed(output / 'rgbd.npz', rgb=rgb, depth=depth, K=K)
            cv2.imwrite(str(output / 'rgb.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(output / 'depth_mm.png'), np.rint(depth * 1000).astype(np.uint16))
            meta['point_count'] = save_cloud(output / 'scene.ply', rgb, depth, K)
            (output / 'capture.json').write_text(json.dumps(meta, indent=2) + '\n')
            print(json.dumps(meta, indent=2), flush=True)
            return meta
        raise TimeoutError('No complete synchronized RGB-D pair after 30 seconds')
    finally:
        pipeline.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory, must not exist')
    parser.add_argument('--device-index', type=int, default=0)
    args = parser.parse_args()
    capture(args.output, args.device_index)
