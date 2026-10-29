"""Explicit motor-to-model joint mapping; motor commands and recordings remain raw."""
from dataclasses import dataclass
from pathlib import Path
import os
import tempfile

import numpy as np
import yaml


JOINT_NAMES = tuple(f'joint{index}' for index in range(1, 8))


@dataclass(frozen=True)
class JointCompensation:
    directions: tuple[float, ...] = (1.,) * 7
    offsets_deg: tuple[float, ...] = (0.,) * 7

    def __post_init__(self):
        if len(self.directions) != 7 or len(self.offsets_deg) != 7:
            raise ValueError('补偿必须包含 joint1–joint7')
        if any(isinstance(value, bool) or value not in (-1, 1) for value in self.directions):
            raise ValueError('关节方向只能为 +1 或 -1')
        if not np.isfinite(self.offsets_deg).all():
            raise ValueError('零点偏移必须为有限角度')

    def to_model(self, motor_positions):
        return np.asarray(self.directions) * self._positions(motor_positions) + np.deg2rad(self.offsets_deg)

    def to_motor(self, model_positions):
        return np.asarray(self.directions) * (self._positions(model_positions) - np.deg2rad(self.offsets_deg))

    def model_limits(self, lower, upper):
        first, second = self.to_model(lower), self.to_model(upper)
        return np.minimum(first, second), np.maximum(first, second)

    @staticmethod
    def _positions(positions):
        values = np.asarray(positions, dtype=float)
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError('关节位置必须包含 7 个有限弧度值')
        return values

    @classmethod
    def load(cls, path):
        document = yaml.safe_load(Path(path).read_text())
        if not isinstance(document, dict) or document.get('version') != 1:
            raise ValueError('不支持的关节补偿配置版本')
        joints = document.get('joints')
        if not isinstance(joints, dict) or set(joints) != set(JOINT_NAMES):
            raise ValueError('补偿配置必须完整包含 joint1–joint7')
        try:
            return cls(tuple(joints[name]['direction'] for name in JOINT_NAMES),
                       tuple(float(joints[name]['offset_deg']) for name in JOINT_NAMES))
        except (KeyError, TypeError) as exc:
            raise ValueError('每个关节需要 direction 和 offset_deg') from exc

    def save(self, path):
        path = Path(path)
        document = {'version': 1, 'joints': {
            name: {'direction': int(direction), 'offset_deg': float(offset)}
            for name, direction, offset in zip(JOINT_NAMES, self.directions, self.offsets_deg)}}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=f'.{path.name}.',
                                             suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                yaml.safe_dump(document, stream, sort_keys=False, allow_unicode=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
