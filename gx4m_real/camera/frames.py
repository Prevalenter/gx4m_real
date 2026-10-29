"""Frames returned by the cameras in this package."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ColorFrame:
    sequence: int
    rgb: np.ndarray
    received_at: float
    device_timestamp_us: int | None = None
    system_timestamp_us: int | None = None


@dataclass(frozen=True)
class RGBDFrame:
    sequence: int
    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    received_at: float
    timestamp_delta_ms: float
