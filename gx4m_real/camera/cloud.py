"""Depth images to point clouds in the camera optical frame (+X right, +Y down, +Z forward, metres)."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def unproject(depth: np.ndarray, u: np.ndarray, v: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Z-depth (m) at pixels (u, v) -> points (..., 3). K is the 3x3 camera matrix."""
    z = depth[v, u]
    return np.stack(((u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z), -1)


def even_pixel_grid(n: int, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """First n pixels (u, v) of an even rows x cols grid with rows x cols >= n."""
    rows = max(1, int(math.sqrt(n*height/width)))
    cols = math.ceil(n/rows)
    v, u = np.meshgrid(np.linspace(0, height-1, rows).astype(int), np.linspace(0, width-1, cols).astype(int),
                       indexing='ij')
    return u.reshape(-1)[:n], v.reshape(-1)[:n]


def dense_points(depth: np.ndarray, K: np.ndarray, stride: int = 1, near: float = 0., far: float = math.inf):
    """Every stride-th pixel with near <= depth <= far (missing depth is always dropped)."""
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    points = unproject(depth, u.reshape(-1), v.reshape(-1), K)
    z = points[:, 2]
    return points[(z > 0) & (z >= near) & (z <= far)]


def save_cloud(path: Path, rgb: np.ndarray, depth: np.ndarray, K: np.ndarray) -> int:
    """Binary PLY of every pixel with depth > 0, coloured from the registered RGB image."""
    v, u = np.nonzero(depth > 0)
    points = unproject(depth, u, v, K)
    data = np.empty(len(points), dtype=[(n, '<f4') for n in ('x', 'y', 'z')] +
                    [(n, 'u1') for n in ('red', 'green', 'blue')])
    data['x'], data['y'], data['z'] = points.T
    for i, name in enumerate(('red', 'green', 'blue')):
        data[name] = rgb[v, u, i]
    header = ('ply\nformat binary_little_endian 1.0\n'
              'comment coordinates in metres; optical frame +X right +Y down +Z forward\n'
              f'element vertex {len(data)}\nproperty float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with Path(path).open('wb') as stream:
        stream.write(header.encode())
        data.tofile(stream)
    return len(data)


def write_ply(path, points: np.ndarray):
    """ASCII PLY of xyz points."""
    with open(path, 'w', encoding='ascii') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\n'
                'property float z\nend_header\n')
        np.savetxt(f, points, fmt='%.5f')
