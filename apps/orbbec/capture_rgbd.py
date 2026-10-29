"""Capture one calibrated, aligned Orbbec RGB-D frame and a point cloud in metres.

Writes rgbd.npz (rgb, depth in metres, K), rgb.png, depth_mm.png, scene.ply and capture.json into a new
directory; tasks/dexlong_gear3/perception/detect_gear36.py --capture reads this layout.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from gx4m_real.camera.cloud import save_cloud
from gx4m_real.camera.orbbec import OrbbecRGBDStream


def capture(output: Path, device_index: int = 0, warmup: int = 30) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    stream = OrbbecRGBDStream(device_index, warmup)
    try:
        stream.open()
    except Exception as exc:
        diagnostic = {**stream.metadata, 'captured': False, 'error': str(exc)}
        (output / 'capture_error.json').write_text(json.dumps(diagnostic, indent=2) + '\n')
        raise
    try:
        print(f"Device: {stream.metadata['device']}, firmware: {stream.metadata['firmware']}, "
              f"ISP: {stream.metadata['isp']}", flush=True)
        frame = stream.read(timeout=30)
    finally:
        stream.close()
    rgb, depth, K = frame.rgb, frame.depth, frame.K
    meta = {
        'captured_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), **stream.metadata,
        'timestamp_delta_ms': frame.timestamp_delta_ms, 'valid_depth_fraction': float((depth > 0).mean()),
    }
    np.savez_compressed(output / 'rgbd.npz', rgb=rgb, depth=depth, K=K)
    cv2.imwrite(str(output / 'rgb.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output / 'depth_mm.png'), np.rint(depth * 1000).astype(np.uint16))
    meta['point_count'] = save_cloud(output / 'scene.ply', rgb, depth, K)
    (output / 'capture.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta, indent=2), flush=True)
    return meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True, help='New directory, must not exist')
    parser.add_argument('--device-index', type=int, default=0)
    args = parser.parse_args()
    capture(args.output, args.device_index)
