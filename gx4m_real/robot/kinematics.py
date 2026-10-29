"""Forward kinematics of a whole URDF tree (revolute, continuous and fixed joints), numpy only."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def rpy_matrix(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [-sp, cp*sr, cp*cr]])


def axis_angle_matrix(axis, angle):
    x, y, z = axis / np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array([[c + x*x*C, x*y*C - z*s, x*z*C + y*s],
                     [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
                     [z*x*C - y*s, z*y*C + x*s, c + z*z*C]])


def quat_matrix(q):
    """wxyz quaternion -> 3x3 rotation."""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])


def matrix_quat(R):
    """3x3 rotation -> wxyz quaternion with w >= 0."""
    t = np.trace(R)
    if t > 0:
        s = 2*np.sqrt(t + 1)
        q = np.array([s/4, (R[2, 1] - R[1, 2])/s, (R[0, 2] - R[2, 0])/s, (R[1, 0] - R[0, 1])/s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2*np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q = np.empty(4)
        q[0] = (R[k, j] - R[j, k])/s
        q[1 + i], q[1 + j], q[1 + k] = s/4, (R[j, i] + R[i, j])/s, (R[k, i] + R[i, k])/s
    return q if q[0] >= 0 else -q


class UrdfFK:
    """Link poses of the URDF in a world frame where the root link sits at (base_pos, base_quat wxyz)."""

    def __init__(self, urdf: str | Path, base_pos=(0., 0., 0.), base_quat=(1., 0., 0., 0.)):
        root = ET.parse(urdf).getroot()
        self.root_link = ({l.get('name') for l in root.findall('link')} -
                          {j.find('child').get('link') for j in root.findall('joint')}).pop()
        self.joints = []                                    # parents before children
        children = {}
        for j in root.findall('joint'):
            children.setdefault(j.find('parent').get('link'), []).append(j)
        stack = [self.root_link]
        while stack:
            for j in children.get(stack.pop(), []):
                o = j.find('origin')
                xyz = [float(v) for v in (o.get('xyz', '0 0 0') if o is not None else '0 0 0').split()]
                rpy = [float(v) for v in (o.get('rpy', '0 0 0') if o is not None else '0 0 0').split()]
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = rpy_matrix(*rpy), xyz
                ax = j.find('axis')
                axis = np.array([float(v) for v in ax.get('xyz').split()]) if ax is not None else np.array([1., 0., 0.])
                self.joints.append((j.get('name'), j.get('type'), j.find('parent').get('link'),
                                    j.find('child').get('link'), T, axis))
                stack.append(j.find('child').get('link'))
        self.actuated = [n for n, t, *_ in self.joints if t in ('revolute', 'continuous', 'prismatic')]
        self.base = np.eye(4)
        self.base[:3, :3], self.base[:3, 3] = quat_matrix(base_quat), base_pos

    def link_poses(self, q: dict[str, float]) -> dict[str, np.ndarray]:
        """4x4 world pose of every link; q maps actuated joint names to radians (missing: 0)."""
        poses = {self.root_link: self.base}
        for name, jtype, parent, child, T0, axis in self.joints:
            T = poses[parent] @ T0
            if jtype in ('revolute', 'continuous'):
                M = np.eye(4)
                M[:3, :3] = axis_angle_matrix(axis, q.get(name, 0.))
                T = T @ M
            elif jtype == 'prismatic':
                M = np.eye(4)
                M[:3, 3] = axis * q.get(name, 0.)
                T = T @ M
            poses[child] = T
        return poses
