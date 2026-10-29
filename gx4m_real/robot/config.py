"""Configuration shared by the seven-joint driver and the viewer."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import yaml

ARM_JOINTS = tuple(f'joint{i}' for i in range(1, 8))


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    urdf: Path
    motor_config: Path
    initial_joints: dict[str, float]
    base_position: tuple[float, ...]
    base_wxyz: tuple[float, ...]
    can_channel: int
    can_device_serial: str
    frequency_hz: int
    control_rate_hz: float
    velocity_limit: float
    torque_limit: float
    max_joint_step_rad: float
    feedback_stale_after_sec: float
    can_response_timeout_sec: float
    can_response_retries: int
    can_retry_delay_sec: float
    lower: np.ndarray
    upper: np.ndarray

    @property
    def initial_position(self):
        return np.array([self.initial_joints[name] for name in ARM_JOINTS], dtype=float)


def load_config(path: Path) -> Config:
    path = path.expanduser().resolve()
    raw = yaml.safe_load(path.read_text())
    arm, model, web = raw['arm'], raw['model'], raw['web']
    resolve = lambda value: (path.parent / value).resolve()
    motor_config = resolve(arm['motor_config'])
    motor = yaml.safe_load(motor_config.read_text())['robot_config']
    entries = motor['motor_configs']
    if motor['num_motors'] != 7 or [item['id'] for item in entries] != list(range(1, 8)):
        raise ValueError('Expected exactly seven motors in ID order 1..7')
    lower = np.deg2rad([item['limits']['position']['lower'] for item in entries])
    upper = np.deg2rad([item['limits']['position']['upper'] for item in entries])
    if not np.isfinite([lower, upper]).all() or np.any(lower >= upper):
        raise ValueError('Motor limits must be finite and ordered')
    initial = json.loads(resolve(model['initial_pose']).read_text())
    cfg = Config(host=str(web['host']), port=int(web['port']), urdf=resolve(model['urdf']),
                 motor_config=motor_config, initial_joints=initial['joint_positions'],
                 base_position=tuple(initial['base_position']), base_wxyz=tuple(initial['base_wxyz']),
                 lower=lower, upper=upper, **arm_without_path(arm))
    if not cfg.urdf.is_file() or not 1 <= cfg.port <= 65535:
        raise ValueError('Invalid URDF path or server port')
    if cfg.can_channel not in (0, 1) or not cfg.can_device_serial.strip():
        raise ValueError('CAN channel must be 0/1 and an explicit adapter serial is required')
    for name in ('frequency_hz', 'control_rate_hz', 'velocity_limit', 'max_joint_step_rad',
                 'feedback_stale_after_sec', 'can_response_timeout_sec'):
        value = getattr(cfg, name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if not np.isfinite(cfg.torque_limit) or not 0 < cfg.torque_limit <= 1:
        raise ValueError('torque_limit must be in (0, 1]')
    if cfg.can_response_retries < 0 or not np.isfinite(cfg.can_retry_delay_sec) or cfg.can_retry_delay_sec < 0:
        raise ValueError('Invalid CAN retry configuration')
    validate_position(cfg, cfg.initial_position)
    return cfg


def arm_without_path(arm):
    return {k: v for k, v in arm.items() if k != 'motor_config'}


def validate_position(config: Config, values) -> np.ndarray:
    q = np.asarray(values, dtype=float)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise ValueError('目标必须包含 7 个有限角度（弧度）')
    outside = np.flatnonzero((q < config.lower - 1e-8) | (q > config.upper + 1e-8))
    if len(outside):
        names = ', '.join(ARM_JOINTS[i] for i in outside)
        raise ValueError(f'目标超出硬件关节限位：{names}')
    return q.copy()
