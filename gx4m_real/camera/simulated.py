"""Stand-in colour camera for mock mode and tests; same interface as OrbbecColorCamera."""
import time

import numpy as np

from .frames import ColorFrame


class SimulatedRGBCamera:
    def __init__(self):
        self.sequence = 0
        self.metadata = {'synthetic': True, 'device': 'simulated RGB', 'color_order': 'RGB',
                         'resolution': [320, 240], 'fps': 30, 'K': None,
                         'distortion': None, 'undistorted': False}

    def open(self):
        pass

    def read(self, timeout=2.):
        time.sleep(min(1 / 30, timeout))
        self.sequence += 1
        rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        rgb[..., 0] = np.arange(320, dtype=np.uint16) % 256
        rgb[..., 1] = np.arange(240, dtype=np.uint8)[:, None]
        rgb[..., 2] = self.sequence % 256
        return ColorFrame(self.sequence, rgb, time.monotonic())

    def close(self):
        pass
