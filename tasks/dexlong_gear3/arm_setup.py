"""Arm control config and start tolerances shared by the policy runner (run_step1.py, robot_server.py) and the
open-loop trajectory replay (tasks/beam_lift_replay). No torch or kinematics imports, so the replay's control
environment can use it."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from gx4m_real.robot.compensation import JointCompensation
from gx4m_real.robot.config import load_config

ROOT = Path(__file__).resolve().parents[2]
CONTROL_CONFIG = ROOT/'apps'/'control'/'config.yaml'
DEFAULT_HANDEYE = ROOT/'apps/charuco_handeye/runtime/he_20261109_contact_plate/calibration/handeye_result.json'
HAND_START_TOLERANCE = .1   # rad; current-based position control stops a little short of the goal


def arm_config_for_reference(path, reference, handeye=None, velocity=None):
    """Control config at the trained PVT speed limit (or velocity). A hand-eye result that records its joint
    compensation replaces the control one: its calibrated camera is only consistent with that compensation.
    reference: an exported policy reference (velocity_limit, lower, upper in joint_order, arm first)."""
    config = load_config(Path(path))
    config = dataclasses.replace(config, velocity_limit=velocity or float(max(reference['velocity_limit'][:7])))
    recorded = json.loads(Path(handeye).read_text(encoding='utf-8')).get('joint_compensation') if handeye else None
    if recorded:
        compensation = JointCompensation(tuple(recorded['directions']), tuple(recorded['offsets_deg']))
        if compensation != config.compensation:
            print(f"Joint compensation of the hand-eye result ({recorded.get('source', handeye)}), offsets "
                  f"{list(compensation.offsets_deg)} deg, replaces {config.compensation_path} "
                  f"({list(config.compensation.offsets_deg)} deg)")
        config = dataclasses.replace(config, compensation=compensation,
                                     compensation_path=Path(recorded['source']) if 'source' in recorded else None)
    lower, upper = np.asarray(reference['lower'][:7]), np.asarray(reference['upper'][:7])
    model_lower, model_upper = config.compensation.model_limits(config.lower, config.upper)
    gap = max(np.abs(model_lower - lower).max(), np.abs(model_upper - upper).max())
    if gap > np.radians(1.):
        print(f'WARNING: real arm limits differ from the trained ones by up to {np.degrees(gap):.1f} deg')
    return config
