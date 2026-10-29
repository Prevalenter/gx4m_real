"""Portable, validated seven-joint recordings (seconds and radians)."""
import json
from pathlib import Path

import numpy as np

from config import ARM_JOINTS, validate_position


def load_trajectory(config, path):
    data = json.loads(Path(path).expanduser().read_text())
    if (data.get('version') != 1 or data.get('units') != 'rad'
            or data.get('joint_names') != list(ARM_JOINTS)):
        raise ValueError('轨迹格式、单位或关节顺序不匹配')
    times = np.asarray(data['times_sec'], dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or times[0] != 0 or np.any(np.diff(times) <= 0)):
        raise ValueError('轨迹至少需要两帧，时间必须从零开始并严格递增')
    positions = np.asarray(data['positions_rad'], dtype=float)
    if positions.shape != (len(times), 7):
        raise ValueError('轨迹每帧必须包含 7 个关节角度')
    for row in positions:
        validate_position(config, row)
    return times, positions


def save_trajectory(path, times, positions):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {'version': 1, 'units': 'rad', 'joint_names': list(ARM_JOINTS),
            'times_sec': list(times), 'positions_rad': [q.tolist() for q in positions]}
    # Exclusive creation protects existing demonstrations from accidental loss.
    with path.open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return path
