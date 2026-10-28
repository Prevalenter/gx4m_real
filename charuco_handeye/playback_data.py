"""Read control/runtime/calibration sessions without importing robot hardware."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robot import UrdfChainFK

HERE = Path(__file__).resolve().parent
DEFAULT_URDF = HERE.parents[2] / 'asset/gx7_right/urdf/gx7_right_gx4m_cam_xl330.urdf'
JOINTS = [f'joint{i}' for i in range(1, 8)]


def transform(value, name='transform'):
    value = np.asarray(value, dtype=float)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(value[:3, :3]), 1., atol=1e-5)):
        raise ValueError(f'Invalid rigid transform: {name}')
    return value


def world_base(metadata):
    position = np.asarray(metadata.get('base_position', [0, 0, 0]), dtype=float)
    wxyz = np.asarray(metadata.get('base_wxyz', [1, 0, 0, 0]), dtype=float)
    if position.shape != (3,) or wxyz.shape != (4,) or not np.isfinite(np.r_[position, wxyz]).all():
        raise ValueError('Invalid recorded robot base pose')
    if np.linalg.norm(wxyz) < 1e-10:
        raise ValueError('Robot base quaternion is zero')
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_matrix()
    result[:3, 3] = position
    return result


def camera_parameters(camera):
    K = np.asarray(camera.get('K'), dtype=float)
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1])):
        raise ValueError('Session camera.K must contain calibrated pinhole intrinsics')
    size = np.asarray(camera.get('resolution'), dtype=float)
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError('Session camera.resolution must be [width, height]')
    if camera.get('undistorted') is True:
        dist = np.zeros(8)
    else:
        raw = camera.get('distortion', camera.get('original_distortion'))
        if isinstance(raw, dict):
            dist = np.array([raw.get(k, 0.) for k in ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6')])
        elif raw is not None:
            dist = np.asarray(raw, dtype=float).ravel()
        else:
            raise ValueError('Raw RGB requires distortion coefficients; do not silently assume zero')
        if dist.size not in (4, 5, 8, 12, 14) or not np.isfinite(dist).all():
            raise ValueError('Invalid OpenCV distortion coefficients')
        model = str(camera.get('distortion_model', '')).lower()
        if 'fisheye' in model or 'kannala' in model:
            raise ValueError('Fisheye camera requires a fisheye-specific calibration pipeline')
    return K, dist.astype(float), tuple(int(v) for v in size)


def load_session(directory, *, urdf=None, base='base_link', tip='Link7', allow_synthetic=False):
    directory = Path(directory).expanduser().resolve()
    metadata = json.loads((directory / 'session.json').read_text())
    if metadata.get('version') != 1 or metadata.get('joint_names') != JOINTS or metadata.get('joint_units') != 'rad':
        raise ValueError('Expected playback session v1 with joint1..joint7 in radians')
    if (metadata.get('mock') or metadata['camera'].get('synthetic')) and not allow_synthetic:
        raise ValueError('Synthetic session: use --allow-synthetic only for synthetic self-tests')
    K, dist, size = camera_parameters(metadata['camera'])
    path = Path(urdf or metadata.get('robot_urdf', DEFAULT_URDF)).expanduser()
    if not path.is_absolute():
        path = directory / path
    if not path.is_file():
        raise ValueError(f'URDF not found: {path}; pass --urdf explicitly')
    fk = UrdfChainFK(str(path), base=base, tip=tip)
    if fk.actuated != JOINTS:
        raise ValueError(f'Expected GX7 joint chain joint1..joint7, got {fk.actuated}')
    rows = []
    last_time = -np.inf
    seen = set()
    for line_number, line in enumerate((directory / 'frames.jsonl').read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        stamp = float(row['timestamp_monotonic'])
        if not np.isfinite(stamp) or stamp < last_time:
            raise ValueError(f'Non-monotonic frame timestamp on line {line_number}')
        last_time = stamp
        frame_id = row['frame_index']
        if frame_id in seen:
            raise ValueError(f'Duplicate frame index: {frame_id}')
        seen.add(frame_id)
        image = (directory / row['rgb_path']).resolve()
        if not image.is_relative_to(directory):
            raise ValueError('RGB image path must stay inside the recording directory')
        row = {**row, 'image_path': str(image)}
        if row.get('alignment_valid') is True:
            q = np.asarray(row.get('positions_rad'), dtype=float)
            if q.shape != (7,) or not np.isfinite(q).all():
                raise ValueError(f'Invalid aligned joints for frame {frame_id}')
            row['T_base_flange'] = fk.fk(q.tolist()).tolist()
        rows.append(row)
    return {'path': str(directory), 'metadata': metadata, 'K': K, 'dist': dist, 'size': size,
            'urdf': str(path.resolve()), 'base_link': base, 'tip_link': tip,
            'T_world_base': world_base(metadata), 'frames': rows}
