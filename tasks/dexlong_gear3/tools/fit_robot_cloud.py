"""J1 offset error from the fixed camera's point cloud of the robot base and the table.

The hand-eye extrinsic T_base_cam is solved from the board on the flange with the joint compensation (J1 +8.7 deg),
so it fixes the camera relative to the arm after J1. A J1 offset error delta turns the real base_link (column, foot
plate, J1 housing) and the table it stands on by -delta about the J1 axis relative to that. Fitting base_link and the
table to a depth cloud therefore measures delta; the hand is not used (black, too close for depth).

  model      base_link's URDF visual mesh (point-to-plane on sampled points), and the table: base z = TABLE_Z
  modes      J1          delta only
             J1 + shift  delta and a translation (what J1 cannot explain)
             free pose   the full base pose; its rotation about the J1 axis reported as delta

Capture (read-only: an RGB-D burst from the camera, which no other process may hold, and the joints from the step1
robot server's status; nothing is sent to the robot):
  python tasks/dexlong_gear3/tools/fit_robot_cloud.py --capture [--out DIR]
Fit a capture:
  python tasks/dexlong_gear3/tools/fit_robot_cloud.py DIR [DIR ...]
Needs the foundationpose environment (camera, trimesh, yourdfpy).
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from multiprocessing.connection import Client
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tasks/dexlong_gear3/perception'))

URDF = ROOT/'asset/gx7_right/urdf/gx7_right_gx4m_cam_xl330_pad.urdf'
HANDEYE = ROOT/'apps/charuco_handeye/runtime/he_20261102_185319_j1p87/calibration/handeye_result.json'
TABLE_Z = -.676             # base_link: bottom of the column's foot plate (URDF base_link mesh), on the table
SPACING = .004              # m between sampled model surface points
BURST = 15                  # depth frames averaged per capture
MAX_TABLE_POINTS, MAX_ROBOT_POINTS = 2000, 30000


def rigid(x):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
    T[:3, 3] = x[3:6]
    return T


def capture(out, robot=('127.0.0.1', 5702)):
    from gx4m_real.camera.frame_ipc import receive
    from gx4m_real.camera.orbbec import OrbbecRGBDStream
    from robot_panel import KEY_PATH

    out.mkdir(parents=True, exist_ok=False)
    with Client(robot, authkey=bytes.fromhex(KEY_PATH.read_text().strip())) as connection:
        before = receive(connection)[0]
        with OrbbecRGBDStream() as camera:
            frames, sequence = [], -1
            while len(frames) < BURST:
                frame = camera.read(after=sequence)
                sequence = frame.sequence
                frames.append(frame)
        end = time.monotonic() + .5         # the server pushes a status every 0.1 s: read on for 0.5 s
        after = receive(connection)[0]
        while time.monotonic() < end:
            after = receive(connection)[0]
    q0, q1 = before['q'], after['q']
    if not q0 or set(q0) != set(q1):
        raise RuntimeError('No measured joints: connect the arm (and hand) in the viewer first')
    moved = max(abs(q0[k] - q1[k]) for k in q0)
    if moved > 1e-3:
        raise RuntimeError(f'The robot moved during the capture ({np.degrees(moved):.2f} deg)')
    depth = np.stack([f.depth for f in frames])
    valid = depth > 0
    mean = np.where(valid.sum(0) >= BURST//2, np.where(valid, depth, 0).sum(0)/np.maximum(valid.sum(0), 1), 0)
    np.savez_compressed(out/'rgbd.npz', rgb=frames[-1].rgb, depth=mean.astype(np.float32), K=frames[-1].K)
    (out/'joints.json').write_text(json.dumps({'q_model': q1, 'compensation_deg': after.get('compensation_deg'),
                                               'captured_at': datetime.now().isoformat(timespec='seconds')},
                                              indent=2) + '\n')
    print(f'Saved {out}: {BURST} frames, {len(q1)} joints')


class RobotModel:
    """Surface points and normals of every visual mesh at the measured joints, in base_link; flag after J1."""

    def __init__(self, q):
        import trimesh
        import yourdfpy
        model = yourdfpy.URDF.load(str(URDF), load_meshes=True)
        model.update_cfg({name: q.get(name, 0.) for name in model.actuated_joint_names})
        joint1 = model.joint_map['joint1']
        self.pivot = joint1.origin[:3, 3]
        self.axis = joint1.origin[:3, :3] @ np.asarray(joint1.axis, float)
        graph = model.scene.graph
        points, normals, distal, links = [], [], [], []
        for node in graph.nodes_geometry:
            T, name = graph.get(node)
            mesh = model.scene.geometry[name]
            count = int(np.clip(mesh.area/SPACING**2, 50, 60000))
            p, faces = trimesh.sample.sample_surface(mesh, count, seed=0)
            link = graph.transforms.parents.get(node)
            points.append(p @ T[:3, :3].T + T[:3, 3])
            normals.append(mesh.face_normals[faces] @ T[:3, :3].T)
            distal.append(np.full(len(p), link != 'base_link'))
            links += [link]*len(p)
        self.points, self.normals = np.concatenate(points), np.concatenate(normals)
        self.distal, self.links = np.concatenate(distal), np.array(links)

    def base(self):
        """Only base_link (column, foot plate, J1 housing): it does not move with the joints."""
        keep = ~self.distal
        self.points, self.normals, self.links = self.points[keep], self.normals[keep], self.links[keep]
        self.distal = self.distal[keep]
        return self

    def at(self, delta):
        """Points and normals with the model J1 angle increased by delta."""
        R = Rotation.from_rotvec(self.axis*delta).as_matrix()
        p, n = self.points.copy(), self.normals.copy()
        p[self.distal] = (p[self.distal] - self.pivot) @ R.T + self.pivot
        n[self.distal] = n[self.distal] @ R.T
        return p, n


def cloud(rgbd):
    depth, K, rgb = rgbd['depth'], rgbd['K'], rgbd['rgb']
    v, u = np.nonzero(depth > 0)
    z = depth[v, u]
    P = np.c_[(u - K[0, 2])*z/K[0, 0], (v - K[1, 2])*z/K[1, 1], z]
    return P, rgb[v, u]


def table_points(P, rgb, T_base_cam, rng):
    """White pixels within 2 cm of base z = TABLE_Z under the starting extrinsic, then a RANSAC plane."""
    Pb = P @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]
    candidates = P[(rgb.min(axis=1) > 150) & (np.abs(Pb[:, 2] - TABLE_Z) < .03)]
    best = None
    for _ in range(300):
        a, b, c = candidates[rng.choice(len(candidates), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inliers = np.abs((candidates - a) @ n) < .003
        if best is None or inliers.sum() > best.sum():
            best = inliers
    table = candidates[best]
    return table[rng.choice(len(table), min(len(table), MAX_TABLE_POINTS), replace=False)]


def about_j1(model, angle):
    """4x4 rotation by angle about the J1 axis (base_link): what a J1 model angle change does to the arm."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(model.axis*angle).as_matrix()
    T[:3, 3] = model.pivot - T[:3, :3] @ model.pivot
    return T


# Free parameters of the base pose T_cam_base = T_cam_base0 @ about_j1(-delta) @ rigid(rotation, shift):
# the hand-eye extrinsic fixes the camera relative to the arm after J1; a J1 offset error delta turns the real base
# (column, foot plate, table) by -delta about the J1 axis relative to it.
MODES = {'J1': (0,), 'J1 + shift': (0, 4, 5, 6), 'free pose': (1, 2, 3, 4, 5, 6)}


def base_pose(T_cam_base0, model, y):
    return T_cam_base0 @ about_j1(model, -y[0]) @ rigid(y[1:])


def fit(P, table, model, T_cam_base0, mode, radii=(.03, .02, .012, .008, .006, .006)):
    """ICP of the base_link model and the table: associate the measured points with the nearest model point
    (within the radius) at the current estimate, then solve the mode's parameters with a robust point-to-plane cost
    plus the table height (base z = TABLE_Z)."""
    free = list(MODES[mode])
    y = np.zeros(7)
    for radius in radii:
        T = base_pose(T_cam_base0, model, y)
        dist, index = cKDTree(model.points @ T[:3, :3].T + T[:3, 3]).query(P, distance_upper_bound=radius)
        keep = np.isfinite(dist)
        data, idx = P[keep], index[keep]
        if len(data) > MAX_ROBOT_POINTS:
            pick = np.random.default_rng(1).choice(len(data), MAX_ROBOT_POINTS, replace=False)
            data, idx = data[pick], idx[pick]

        def residuals(z):
            yy = y.copy()
            yy[free] = z
            T = base_pose(T_cam_base0, model, yy)
            pc, nc = model.points[idx] @ T[:3, :3].T + T[:3, 3], model.normals[idx] @ T[:3, :3].T
            Tb = np.linalg.inv(T)
            return np.r_[np.sum((data - pc)*nc, axis=1), (table @ Tb[:3, :3].T + Tb[:3, 3])[:, 2] - TABLE_Z]

        y[free] = least_squares(residuals, y[free], loss='soft_l1', f_scale=.003, x_scale=.01).x
    if mode == 'free pose':         # the part of the free rotation about the J1 axis, as a J1 offset
        y[0] = -float(Rotation.from_matrix(rigid(y[1:])[:3, :3]).as_rotvec() @ model.axis)
    T = base_pose(T_cam_base0, model, y) if mode != 'free pose' else T_cam_base0 @ rigid(y[1:])
    return y, T, residual_stats(P, table, model, T, radius=radii[-1])


def table_tilt_deg(table, T_cam_base):
    Tb = np.linalg.inv(T_cam_base)
    Q = table @ Tb[:3, :3].T + Tb[:3, 3]
    n = np.linalg.svd(Q - Q.mean(0), full_matrices=False)[2][-1]
    return float(np.degrees(np.arccos(abs(n[2]))))


def analyse(folder, handeye_path=HANDEYE):
    rgbd = np.load(folder/'rgbd.npz')
    joints = json.loads((folder/'joints.json').read_text()).get('q_model') if (folder/'joints.json').exists() else None
    handeye = json.loads(Path(handeye_path).read_text())
    T_cam_base0 = np.linalg.inv(np.asarray(handeye['T_base_cam']))
    P, rgb = cloud(rgbd)
    P, rgb = P[P[:, 2] < 1.5], rgb[P[:, 2] < 1.5]
    rng = np.random.default_rng(0)
    table = table_points(P, rgb, np.linalg.inv(T_cam_base0), rng)
    model = RobotModel(joints or {}).base()
    offset = np.asarray(handeye['joint_compensation']['offsets_deg'], float)[0]
    print(f'{folder.name}: {len(P)} depth points, {len(table)} table points, {len(model.points)} base_link points; '
          f'hand-eye J1 offset {offset:+.2f} deg')
    rows = {}
    for mode in ('hand-eye', *MODES):
        if mode == 'hand-eye':          # the residuals of the unchanged extrinsic
            y, T = np.zeros(7), T_cam_base0
            stats = residual_stats(P, table, model, T)
        else:
            y, T, stats = fit(P, table, model, T_cam_base0, mode)
        rest = np.linalg.inv(about_j1(model, -y[0])) @ np.linalg.inv(T_cam_base0) @ T
        rows[mode] = dict(delta_deg=float(np.degrees(y[0])), j1_offset_deg=float(offset + np.degrees(y[0])),
                          table_tilt_deg=table_tilt_deg(table, T),
                          other_shift_mm=(rest[:3, 3]*1000).round(1).tolist(),
                          other_rot_deg=np.degrees(Rotation.from_matrix(rest[:3, :3]).as_rotvec()).round(2).tolist(),
                          T_base_cam=np.linalg.inv(T).tolist(), **stats)
        r = rows[mode]
        print(f"  {mode:10s} J1 {r['delta_deg']:+6.2f} deg (offset {r['j1_offset_deg']:+6.2f}) | base rms "
              f"{r['rms_mm']:4.1f} mm ({r['points']} pts) | table {r['table_mm']:+5.1f} mm (rms {r['table_rms_mm']:.1f}), "
              f"tilt {r['table_tilt_deg']:.2f} deg | besides J1: shift {r['other_shift_mm']} mm, "
              f"rot {r['other_rot_deg']} deg")
    (folder/'fit.json').write_text(json.dumps(rows, indent=2) + '\n')
    return rows


def residual_stats(P, table, model, T, radius=.006):
    """Point-to-plane residuals of the measured points within radius of the model, and the table height."""
    pc = model.points @ T[:3, :3].T + T[:3, 3]
    nc = model.normals @ T[:3, :3].T
    dist, index = cKDTree(pc).query(P, distance_upper_bound=radius)
    keep = np.isfinite(dist)
    r = np.sum((P[keep] - pc[index[keep]])*nc[index[keep]], axis=1)
    Tb = np.linalg.inv(T)
    tb = (table @ Tb[:3, :3].T + Tb[:3, 3])[:, 2] - TABLE_Z
    return dict(rms_mm=float(np.sqrt(np.mean(r**2))*1000), points=int(keep.sum()),
                table_mm=float(np.mean(tb)*1000), table_rms_mm=float(np.sqrt(np.mean(tb**2))*1000))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folders', nargs='*', type=Path)
    parser.add_argument('--capture', action='store_true', help='Capture a new burst and the joints, then fit it')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--handeye', type=Path, default=HANDEYE, help='handeye_result.json to start from')
    args = parser.parse_args(argv)
    folders = list(args.folders)
    if args.capture:
        out = args.out or ROOT/'tasks/dexlong_gear3/runs'/datetime.now().strftime('cloud_fit_%Y%m%d_%H%M%S')
        capture(out)
        folders.append(out)
    for folder in folders:
        analyse(folder, args.handeye)


if __name__ == '__main__':
    main()
