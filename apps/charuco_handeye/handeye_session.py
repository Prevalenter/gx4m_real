"""Hand-eye capture sessions written by collect_app.py and read by solve_session.py.

he_<timestamp>/
  session.json        robot/URDF, joint compensation, camera intrinsics, mounting mode, board, sampling
  samples.jsonl       one static pose per line: averaged raw motor angles + the image saved at that pose
  images/000000.png   RGB as received from the camera (no undistortion or overlay)

motor_positions_rad are raw motor angles; FK uses model = direction * motor + offset_deg.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from gx4m_real.robot.compensation import JointCompensation
from robot import UrdfChainFK

VERSION = 2
HERE = Path(__file__).resolve().parent
DEFAULT_URDF = HERE.parents[1] / 'asset/gx7_right/urdf/gx7_right_gx4m_cam_xl330.urdf'
JOINTS = [f'joint{i}' for i in range(1, 8)]
MODES = ('eye-in-hand', 'eye-to-hand')


def atomic_write(path, text):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, prefix=f'.{path.name}.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_session_path(root):
    return Path(root) / f'he_{datetime.now():%Y%m%d_%H%M%S}'


class SessionWriter:
    def __init__(self, path, metadata):
        self.path = Path(path).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / 'images').mkdir()
        self.samples = []
        atomic_write(self.path / 'session.json',
                     json.dumps({'version': VERSION, **metadata}, ensure_ascii=False, indent=2))
        self._flush()

    def append(self, sample, rgb):
        index = self.samples[-1]['index'] + 1 if self.samples else 0
        image = f'images/{index:06d}.png'
        if not cv2.imwrite(str(self.path / image), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise OSError(f'无法写入图像 {image}')
        self.samples.append({'index': index, 'image': image, **sample})
        self._flush()
        return self.samples[-1]

    def remove_last(self):
        if not self.samples:
            raise RuntimeError('没有可删除的样本')
        removed = self.samples.pop()
        self._flush()
        (self.path / removed['image']).unlink(missing_ok=True)
        return removed

    def _flush(self):
        atomic_write(self.path / 'samples.jsonl',
                     ''.join(json.dumps(s, ensure_ascii=False) + '\n' for s in self.samples))


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
        raise ValueError('Camera metadata must contain calibrated pinhole intrinsics K')
    size = np.asarray(camera.get('resolution'), dtype=float)
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError('Camera metadata resolution must be [width, height]')
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


def session_compensation(metadata, override=None):
    """The recorded mapping, or an explicit file (e.g. newer FK-calibrated offsets) with the same directions."""
    c = metadata['joint_compensation']
    recorded = JointCompensation(tuple(c['directions']), tuple(c['offsets_deg']))
    if override is None:
        return recorded, 'session'
    compensation = JointCompensation.load(override)
    if compensation.directions != recorded.directions:
        raise ValueError('Joint directions differ from those recorded with the session; wrong compensation file?')
    return compensation, str(override)


def load_session(directory, *, urdf=None, base='base_link', tip='Link7', allow_synthetic=False,
                 compensation=None):
    directory = Path(directory).expanduser().resolve()
    metadata = json.loads((directory / 'session.json').read_text(encoding='utf-8'))
    if metadata.get('version') != VERSION:
        raise ValueError(f'{directory}: expected a hand-eye session v{VERSION} written by collect_app.py')
    if (metadata.get('mock') or metadata['camera'].get('synthetic')) and not allow_synthetic:
        raise ValueError('Synthetic session: use --allow-synthetic only for synthetic self-tests')
    compensation, source = session_compensation(metadata, compensation)
    K, dist, size = camera_parameters(metadata['camera'])
    path = Path(urdf or metadata['urdf']).expanduser()
    if not path.is_file():
        raise ValueError(f'URDF not found: {path}; pass --urdf explicitly')
    fk = UrdfChainFK(str(path), base=base, tip=tip)
    if fk.actuated != JOINTS:
        raise ValueError(f'Expected GX7 joint chain joint1..joint7, got {fk.actuated}')
    samples = []
    for line in (directory / 'samples.jsonl').read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        image = (directory / s['image']).resolve()
        if not image.is_relative_to(directory):
            raise ValueError('Image path must stay inside the session directory')
        q = compensation.to_model(s['motor_positions_rad'])
        samples.append({**s, 'image_path': str(image), 'positions_rad': q.tolist(),
                        'T_base_flange': fk.fk(q.tolist()).tolist()})
    return {'path': str(directory), 'metadata': metadata, 'K': K, 'dist': dist, 'size': size,
            'urdf': str(path.resolve()), 'base_link': base, 'tip_link': tip,
            'T_world_base': world_base(metadata), 'samples': samples,
            'compensation': compensation, 'compensation_source': source}
