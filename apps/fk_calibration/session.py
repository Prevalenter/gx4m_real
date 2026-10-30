"""Session files shared by the collector and the solver.

fk_<timestamp>/
  session.json    robot config, URDF hash, joint compensation, plate copy, sampling settings
  samples.jsonl   one averaged static pose per line; rewritten atomically on every change
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import yaml

from gx4m_real.robot.compensation import JointCompensation

VERSION = 1


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
    return Path(root) / f'fk_{datetime.now():%Y%m%d_%H%M%S}'


class SessionWriter:
    def __init__(self, path, metadata):
        self.path = Path(path).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=False)
        self.samples = []
        atomic_write(self.path / 'session.json',
                     json.dumps({'version': VERSION, **metadata}, ensure_ascii=False, indent=2))
        self._flush()

    def append(self, sample):
        self.samples.append({'index': len(self.samples), **sample})
        self._flush()
        return self.samples[-1]

    def remove_last(self):
        if not self.samples:
            raise RuntimeError('没有可删除的样本')
        removed = self.samples.pop()
        self._flush()
        return removed

    def _flush(self):
        atomic_write(self.path / 'samples.jsonl',
                     ''.join(json.dumps(s, ensure_ascii=False) + '\n' for s in self.samples))


def load_session(path):
    path = Path(path).expanduser().resolve()
    metadata = json.loads((path / 'session.json').read_text(encoding='utf-8'))
    if metadata.get('version') != VERSION:
        raise ValueError(f'{path}: 不支持的会话版本')
    lines = (path / 'samples.jsonl').read_text(encoding='utf-8').splitlines()
    samples = [json.loads(line) for line in lines if line.strip()]
    return metadata, samples


def session_compensation(metadata):
    c = metadata['joint_compensation']
    return JointCompensation(tuple(c['directions']), tuple(c['offsets_deg']))


def load_plate(path):
    raw = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(raw, dict) or raw.get('version') != 1:
        raise ValueError('不支持的孔板配置版本')
    plate_seats = {name: tuple(map(float, xy)) for name, xy in raw['plate_seats'].items()}
    extra = list(raw.get('extra_seats') or [])
    distances = {}
    for pair, value in (raw.get('distances_mm') or {}).items():
        a, b = pair.split('-')
        if a not in plate_seats or b not in plate_seats:
            raise ValueError(f'孔距 {pair} 引用了未知的孔板座')
        if value is not None:
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'孔距 {pair} 必须为正数')
            distances[(a, b)] = float(value) / 1000
    sigma = raw.get('sigma') or {}
    return {'plate_seats': plate_seats, 'extra_seats': extra, 'distances_m': distances,
            'sigma_pivot_m': float(sigma.get('pivot_mm', .1)) / 1000,
            'sigma_distance_m': float(sigma.get('distance_mm', .05)) / 1000,
            'raw': raw}
