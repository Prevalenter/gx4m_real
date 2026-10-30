"""GX7 J1–J7 serial chain read from the URDF, with per-joint error parameters for FK calibration.

Joint i maps its parent frame to its child as  origin_i · Δ_i · Rot(axis_i, q_i + dθ_i),
where Δ_i = Trans(dx, dy, dz) · Rot(rotvec(drx, dry, drz)). One row of `errors` per joint:
[dθ, dx, dy, dz, drx, dry, drz] in radians / metres. All poses are in the parent link of joint1.
"""
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from gx4m_real.robot.config import ARM_JOINTS
from gx4m_real.robot.kinematics import rpy_matrix

PARAM_NAMES = ('dtheta', 'dx', 'dy', 'dz', 'drx', 'dry', 'drz')


def rotvec_matrix(v):
    v = np.asarray(v, dtype=float)
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + K
    return axis_angle_batch(v / angle, np.array([angle]))[0]


def axis_angle_batch(axis, angles):
    """(N,) angles about one unit axis -> (N, 3, 3)."""
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    s, c = np.sin(angles)[:, None, None], np.cos(angles)[:, None, None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def delta_matrix(e):
    """[dx, dy, dz, drx, dry, drz] -> 4x4."""
    D = np.eye(4)
    D[:3, :3], D[:3, 3] = rotvec_matrix(e[3:6]), e[:3]
    return D


def matrix_rpy(R):
    """Inverse of URDF rpy (R = Rz(yaw) Ry(pitch) Rx(roll))."""
    cp = np.hypot(R[0, 0], R[1, 0])
    if cp < 1e-9:                                   # gimbal lock: put everything into roll
        return np.array([np.arctan2(-R[1, 2], R[1, 1]), np.arctan2(-R[2, 0], cp), 0.])
    return np.array([np.arctan2(R[2, 1], R[2, 2]), np.arctan2(-R[2, 0], cp), np.arctan2(R[1, 0], R[0, 0])])


class ArmChain:
    def __init__(self, urdf):
        self.urdf = Path(urdf).resolve()
        joints = {j.get('name'): j for j in ET.parse(self.urdf).getroot().findall('joint')}
        missing = [name for name in ARM_JOINTS if name not in joints]
        if missing:
            raise ValueError(f'URDF 缺少关节：{missing}')
        self.origins, self.axes, self.links = [], [], []
        for index, name in enumerate(ARM_JOINTS):
            j = joints[name]
            parent, child = j.find('parent').get('link'), j.find('child').get('link')
            if index and parent != self.links[-1]:
                raise ValueError(f'{name} 的父连杆 {parent} 不是 {ARM_JOINTS[index - 1]} 的子连杆')
            o = j.find('origin')
            xyz = [float(v) for v in (o.get('xyz', '0 0 0') if o is not None else '0 0 0').split()]
            rpy = [float(v) for v in (o.get('rpy', '0 0 0') if o is not None else '0 0 0').split()]
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = rpy_matrix(*rpy), xyz
            axis = np.array([float(v) for v in j.find('axis').get('xyz').split()])
            self.origins.append(T)
            self.axes.append(axis / np.linalg.norm(axis))
            self.links.append(child)
        self.base_link = joints[ARM_JOINTS[0]].find('parent').get('link')
        self.tip_link = self.links[-1]

    def fk(self, Q, errors=None):
        """(N, 7) model joint angles -> (N, 4, 4) pose of the last arm link."""
        Q = np.atleast_2d(np.asarray(Q, dtype=float))
        E = np.zeros((7, 7)) if errors is None else np.asarray(errors, dtype=float)
        T = np.broadcast_to(np.eye(4), (len(Q), 4, 4)).copy()
        M = np.zeros((len(Q), 4, 4))
        M[:, 3, 3] = 1
        for i in range(7):
            M[:, :3, :3] = axis_angle_batch(self.axes[i], Q[:, i] + E[i, 0])
            T = T @ (self.origins[i] @ delta_matrix(E[i, 1:])) @ M
        return T

    def tip(self, Q, tip_offset, errors=None):
        """(N, 3) probe ball centre in the base frame; tip_offset is in the last arm link."""
        T = self.fk(Q, errors)
        return T[:, :3, :3] @ np.asarray(tip_offset, dtype=float) + T[:, :3, 3]

    def write_urdf(self, errors, out_path):
        """Copy the URDF with the geometric part of `errors` folded into joint origins.

        Joint zero offsets (dθ) are not written here; they belong in joint_compensation.yaml.
        Relative mesh paths are rewritten so they still resolve from the new location.
        """
        out_path = Path(out_path).resolve()
        tree = ET.parse(self.urdf)
        root = tree.getroot()
        joints = {j.get('name'): j for j in root.findall('joint')}
        for i, name in enumerate(ARM_JOINTS):
            T = self.origins[i] @ delta_matrix(np.asarray(errors[i][1:], dtype=float))
            o = joints[name].find('origin')
            if o is None:
                o = ET.SubElement(joints[name], 'origin')
            o.set('xyz', ' '.join(f'{v:.17g}' for v in T[:3, 3]))
            o.set('rpy', ' '.join(f'{v:.17g}' for v in matrix_rpy(T[:3, :3])))
        for mesh in root.iter('mesh'):
            filename = mesh.get('filename', '')
            if filename and '://' not in filename and not os.path.isabs(filename):
                target = (self.urdf.parent / filename).resolve()
                try:
                    target = os.path.relpath(target, out_path.parent)
                except ValueError:                  # different Windows drive: keep it absolute
                    pass
                mesh.set('filename', Path(target).as_posix())
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(out_path, encoding='utf-8', xml_declaration=True)
        return out_path
