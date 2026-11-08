"""Joint solve of the fixed camera's extrinsic and the arm's zero offsets (J1-J7) from three kinds of data.

  hand-eye   ChArUco corners of a calibration session (any compensation: raw motor angles are used): reprojection
             through X^-1 F(q + delta) Y, X = T_base_cam, Y = T_Link7_board
  contacts   collect_contacts.py samples with all four fingertips on the table: each finger's lowest point (its
             distal link and pad meshes, at the recorded hand angles) lies on the table plane fitted to the depth of
             the column frames, taken to the base through X with a depth scale
  column     the column frames' depth points near the URDF column (base_link between the plate and the J1 mount)
             lie on its surface (point to plane); the plate is left out (its mounting is off)
  plate      (--plate) the table plane is the URDF plate's underside: the robot stands on the table

J1 is invisible to the hand-eye and contact terms (it turns the camera with the arm); the column (and the plate)
pin it. Every 5th hand-eye and contact sample is held out of the fit and reported separately.

  python apps/charuco_handeye/solve_joint.py --handeye <handeye_result.json> --contacts <ct_dir> [<ct_dir> ...] \\
      --out <new folder> [--plate] [--finger-bias]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
URDF = ROOT/'asset/gx7_right/urdf/gx7_right_gx4m_cam_xl330_pad.urdf'
ARM_JOINTS = [f'joint{i}' for i in range(1, 8)]
FINGER_LINKS = (('gx4m_link4', 'gx4m_link4_pad'), ('gx4m_link8', 'gx4m_link8_pad'),
                ('gx4m_link12', 'gx4m_link12_pad'), ('gx4m_link16', 'gx4m_link17', 'gx4m_link17_pad'))
COLUMN_Z = (-.62, -.12)      # base_link z of the column: above the plate, below the J1 mount
SIGMA = dict(px=1., contact_mm=2., column_mm=3., plate_deg=.2, plate_mm=2.)
HOLDOUT_EVERY = 5


def rigid(T):
    return np.asarray(T, float)


def se3(xi):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(xi[:3]).as_matrix()
    T[:3, 3] = xi[3:]
    return T


class ArmChain:
    """base_link -> Link7 of the URDF's revolute joints joint1..joint7, numpy only."""

    def __init__(self, model):
        self.joints = [model.joint_map[n] for n in ARM_JOINTS]
        for a, b in zip(self.joints, self.joints[1:]):
            if a.child != b.parent:
                raise ValueError('joint1..joint7 are not one chain')

    def link7(self, q):
        T = np.eye(4)
        for joint, angle in zip(self.joints, q):
            turn = np.eye(4)
            axis = np.asarray(joint.axis, float)
            turn[:3, :3] = Rotation.from_rotvec(axis/np.linalg.norm(axis)*angle).as_matrix()
            T = T @ joint.origin @ turn
        return T


def plane_fit(points, iterations=4):
    """(unit normal, offset d) with n.p = d; normal toward the origin (the camera)."""
    p = points
    for _ in range(iterations):
        c = p.mean(0)
        n = np.linalg.svd(p - c, full_matrices=False)[2][-1]
        r = (p - c) @ n
        p = p[np.abs(r) < 2.5*r.std() + 1e-4]
    c = p.mean(0)
    n = np.linalg.svd(p - c, full_matrices=False)[2][-1]
    if n @ c > 0:
        n = -n
    return n, float(n @ c), float(((p - c) @ n).std())


def table_plane(rgb, depth, K):
    """(normal, offset, rms) of the white table in the lower part of one RGB-D frame, camera frame."""
    H, W = depth.shape
    v, u = np.mgrid[:H, :W]
    P = np.stack([(u - K[0, 2])*depth/K[0, 0], (v - K[1, 2])*depth/K[1, 1], depth], -1)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return plane_fit(P[(depth > 0) & (hsv[..., 2] > 140) & (hsv[..., 1] < 45) & (v > H*.55)][::3])


class Problem:
    def __init__(self, handeye, contact_dirs, *, plate=False, finger_bias=False, free_scale=False,
                 hand_mount=False, handeye_term=True, sample_plane=False):
        import trimesh
        import yourdfpy

        self.he = json.loads(Path(handeye).read_text())
        self.K, self.dist = np.asarray(self.he['K']), np.asarray(self.he['dist'])
        self.directions = np.asarray(self.he['joint_compensation']['directions'], float)
        self.model = yourdfpy.URDF.load(str(URDF), load_meshes=True)
        self.chain = ArmChain(self.model)
        self.plate_term, self.finger_bias, self.free_scale = plate, finger_bias, free_scale
        self.hand_mount, self.handeye_term, self.sample_plane = hand_mount, handeye_term, sample_plane

        # hand-eye corners
        samples = [s for s in self.he['samples'] if not s.get('validation')]
        self.he_samples = [(np.asarray(s['motor_positions_rad']), np.asarray(s['object_points'], float),
                            np.asarray(s['corners_px'], float), i % HOLDOUT_EVERY == HOLDOUT_EVERY - 1)
                           for i, s in enumerate(samples)]

        # contacts: finger mesh points in Link7 (they only depend on the hand angles)
        finger_points = {}
        for node in self.model.scene.graph.nodes_geometry:
            parent = str(self.model.scene.graph.transforms.parents.get(node))
            for f, links in enumerate(FINGER_LINKS):
                if parent in links:
                    T_link_geom = np.linalg.inv(self.model.get_transform(parent, 'base_link')) @ \
                        self.model.scene.graph.get(node)[0]
                    P = trimesh.sample.sample_surface_even(self.model.scene.geometry[self.model.scene.graph.get(node)[1]],
                                                           800, seed=0)[0]
                    finger_points.setdefault(f, []).append((parent, (T_link_geom[:3, :3] @ P.T).T + T_link_geom[:3, 3]))
        self.contacts, self.column_frames = [], []
        for d in contact_dirs:
            d = Path(d)
            for line in (d/'samples.jsonl').read_text().splitlines():
                s = json.loads(line)
                if s['discarded']:
                    continue
                if s['kind'] == 'contact':
                    hand = dict(zip([f'gx4m_joint{i}' for i in range(1, 17)], s['hand_q_rad']))
                    self.model.update_cfg({n: hand.get(n, 0.) for n in self.model.actuated_joint_names})
                    T7inv = np.linalg.inv(self.model.get_transform('Link7', 'base_link'))
                    fingers = []
                    for f in range(4):
                        pts = [(T7inv @ self.model.get_transform(link, 'base_link'))[:3, :3] @ P.T
                               + (T7inv @ self.model.get_transform(link, 'base_link'))[:3, 3:4] for link, P in finger_points[f]]
                        fingers.append(np.hstack(pts).T)
                    held = len(self.contacts) % HOLDOUT_EVERY == HOLDOUT_EVERY - 1
                    plane = None
                    if sample_plane:                    # this sample's own table, in the camera
                        with np.load(d/s['rgbd']) as z:
                            plane = table_plane(z['rgb'], z['depth'], z['K'])
                    self.contacts.append((np.asarray(s['arm_motor_rad']), fingers, held, plane))
                elif s['kind'] in ('column', 'scene'):
                    with np.load(d/s['rgbd']) as z:
                        self.column_frames.append((z['rgb'], z['depth'], z['K']))
        if not self.column_frames:
            raise ValueError('No column frames: the table plane and the column need them')

        # table plane in the camera (white table, lower image) and column candidate points, from the column frames
        table, cloud = [], []
        for rgb, depth, K in self.column_frames:
            H, W = depth.shape
            v, u = np.mgrid[:H, :W]
            P = np.stack([(u - K[0, 2])*depth/K[0, 0], (v - K[1, 2])*depth/K[1, 1], depth], -1)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            ok = depth > 0
            table.append(P[ok & (hsv[..., 2] > 140) & (hsv[..., 1] < 45) & (v > H*.55)])
            cloud.append(P[ok][::3])
        self.table_n, self.table_d, rms = plane_fit(np.vstack(table)[::5])
        self.table_rms_mm = rms*1000
        self.cloud = np.vstack(cloud)

        # base_link column (surface samples + normals) and plate underside
        for node in self.model.scene.graph.nodes_geometry:
            if str(self.model.scene.graph.transforms.parents.get(node)) == 'base_link':
                mesh = self.model.scene.geometry[self.model.scene.graph.get(node)[1]].copy()
                mesh.apply_transform(self.model.scene.graph.get(node)[0])
        pts, faces = trimesh.sample.sample_surface(mesh, 200000, seed=0)
        keep = (pts[:, 2] > COLUMN_Z[0]) & (pts[:, 2] < COLUMN_Z[1])
        self.column_pts, self.column_nrm = pts[keep], mesh.face_normals[faces[keep]]
        self.column_tree = cKDTree(self.column_pts)
        bottom = (mesh.face_normals[:, 2] < -.98) & (mesh.triangles_center[:, 2] < mesh.bounds[0, 2] + .02)
        n = -(mesh.face_normals[bottom]*mesh.area_faces[bottom, None]).sum(0)
        self.plate_n = n/np.linalg.norm(n)
        self.plate_c = (mesh.triangles_center[bottom]*mesh.area_faces[bottom, None]).sum(0)/mesh.area_faces[bottom].sum()

        # start: the handeye result's extrinsic, board and compensation
        self.X0, self.Y0 = rigid(self.he['T_base_cam']), rigid(self.he['T_flange_board'])
        self.delta0 = np.radians(self.he['joint_compensation']['offsets_deg'])
        self.column_sel = None

    # parameters: xiX(6) xiY(6) delta(7, rad, on top of delta0) log_scale(1) [finger bias(4, m)]
    def unpack(self, p):
        X = self.X0 @ se3(p[:6])
        Y = self.Y0 @ se3(p[6:12])
        delta = self.delta0 + p[12:19]
        scale = np.exp(p[19])
        bias = p[20:24] if self.finger_bias else np.zeros(4)
        k = 20 + 4*self.finger_bias
        mount = p[k:k + 3] if self.hand_mount else np.zeros(3)     # hand shift in Link7, m
        return X, Y, delta, scale, bias, mount

    def size(self):
        return 20 + 4*self.finger_bias + 3*self.hand_mount

    def q(self, motor, delta):
        return self.directions*motor + delta

    def table_base(self, X, scale):
        n = X[:3, :3] @ self.table_n
        return n, n @ X[:3, 3] + scale*self.table_d

    def select_column(self, p):
        X, _, _, scale, _, _ = self.unpack(p)
        Pb = (X[:3, :3] @ (scale*self.cloud).T).T + X[:3, 3]
        d, _ = self.column_tree.query(Pb, distance_upper_bound=.025)
        sel = np.flatnonzero(np.isfinite(d) & (Pb[:, 2] > COLUMN_Z[0]) & (Pb[:, 2] < COLUMN_Z[1]))
        self.column_sel = sel[::max(1, len(sel)//400)]

    def residuals(self, p, *, held=False, split=False):
        X, Y, delta, scale, bias, mount = self.unpack(p)
        Xinv = np.linalg.inv(X)
        out = {}
        r = []
        for motor, obj, px, h in self.he_samples if self.handeye_term else ():
            if h != held:
                continue
            C = Xinv @ self.chain.link7(self.q(motor, delta)) @ Y
            uv = cv2.projectPoints(obj, cv2.Rodrigues(C[:3, :3])[0], C[:3, 3], self.K, self.dist)[0].reshape(-1, 2)
            r.append(((uv - px)/SIGMA['px']).ravel())
        out['handeye'] = np.concatenate(r) if r else np.zeros(0)
        n, d = self.table_base(X, scale)
        r = []
        for motor, fingers, h, plane in self.contacts:
            if h != held:
                continue
            ns, ds = n, d                        # the column frames' table, or this sample's own
            if plane is not None:
                ns = X[:3, :3] @ plane[0]
                ds = ns @ X[:3, 3] + scale*plane[1]
            T7 = self.chain.link7(self.q(motor, delta))
            for f, P in enumerate(fingers):
                lowest = ((T7[:3, :3] @ (P + mount).T).T + T7[:3, 3]) @ ns
                r.append((lowest.min() - ds + bias[f])*1000/SIGMA['contact_mm'])
        out['contacts'] = np.asarray(r)
        if not held:
            Pc = scale*self.cloud[self.column_sel]
            Pb = (X[:3, :3] @ Pc.T).T + X[:3, 3]
            _, i = self.column_tree.query(Pb)
            out['column'] = ((Pb - self.column_pts[i])*self.column_nrm[i]).sum(1)*1000/SIGMA['column_mm']
            if self.plate_term:
                ang = np.degrees(np.cross(n, self.plate_n)[:2])/SIGMA['plate_deg']
                out['plate'] = np.r_[ang, (n @ self.plate_c - d)*1000/SIGMA['plate_mm']]
        return out if split else np.concatenate(list(out.values()))

    def solve(self, free_joints):
        mask = np.ones(self.size(), bool)
        mask[12:19] = False
        mask[12 + np.asarray(free_joints, int)] = True
        mask[19] = self.free_scale             # the depth scale is 1 unless asked for
        if not self.handeye_term:
            mask[6:12] = False                  # the board pose has no data without the hand-eye term
        p = np.zeros(self.size())
        for _ in range(3):                      # column correspondences follow the estimate
            self.select_column(p)
            fit = least_squares(lambda x: self.residuals(np.where(mask, self._fill(x, mask, p), p)),
                                p[mask], loss='soft_l1', f_scale=2., x_scale='jac', max_nfev=200)
            p = np.where(mask, self._fill(fit.x, mask, p), p)
        return p

    @staticmethod
    def _fill(x, mask, p):
        full = p.copy()
        full[mask] = x
        return full

    def report(self, p):
        X, Y, delta, scale, bias, mount = self.unpack(p)
        fit, held = self.residuals(p, split=True), self.residuals(p, held=True, split=True)
        units = dict(handeye=('px', SIGMA['px']), contacts=('mm', SIGMA['contact_mm']),
                     column=('mm', SIGMA['column_mm']))
        stats = {}
        for name, (unit, sigma) in units.items():
            for label, res in (('fit', fit), ('held_out', held)):
                v = res.get(name)
                if v is None or not len(v):
                    continue
                v = v*sigma
                if name == 'handeye':
                    v = np.linalg.norm(v.reshape(-1, 2), axis=1)
                stats[f'{name}_{label}'] = dict(unit=unit, n=int(len(v)), rms=float(np.sqrt(np.mean(v**2))),
                                                median_abs=float(np.median(np.abs(v))), max_abs=float(np.abs(v).max()))
        n, d = self.table_base(X, scale)
        planes = [plane for *_, plane in self.contacts if plane is not None]
        if planes:      # each sample's table against the column frames' one, at the plate centre
            heights = [((X[:3, :3] @ pn) @ self.plate_c - ((X[:3, :3] @ pn) @ X[:3, 3] + scale*pd)) - (n @ self.plate_c - d)
                       for pn, pd, _ in planes]
            stats['sample_planes_mm'] = dict(fit_rms=float(np.mean([pr for *_, pr in planes])*1000),
                                             offset_std=float(np.std(heights)*1000),
                                             offset_max=float(np.abs(heights).max()*1000))
        stats['plate_vs_table'] = dict(angle_deg=float(np.degrees(np.arccos(np.clip(n @ self.plate_n, -1, 1)))),
                                       offset_mm=float((n @ self.plate_c - d)*1000))
        per_finger = np.asarray(fit['contacts']).reshape(-1, 4)*SIGMA['contact_mm']
        stats['contacts_per_finger_mm'] = dict(mean=per_finger.mean(0).round(2).tolist(),
                                               rms=np.sqrt((per_finger**2).mean(0)).round(2).tolist())
        return dict(offsets_deg=np.degrees(delta).round(4).tolist(), depth_scale=float(scale),
                    finger_bias_mm=(bias*1000).round(2).tolist(), hand_mount_mm=(mount*1000).round(2).tolist(),
                    stats=stats, T_base_cam=X.tolist(),
                    T_flange_board=Y.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--handeye', type=Path, required=True)
    parser.add_argument('--contacts', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, help='New folder: calibration/handeye_result.json + joint_compensation.yaml')
    parser.add_argument('--plate', action='store_true', help='Also hold the table on the URDF plate underside')
    parser.add_argument('--finger-bias', action='store_true', help='A height offset per finger')
    parser.add_argument('--hand-mount', action='store_true', help='A shift of the whole hand in Link7 (3 values)')
    parser.add_argument('--no-handeye', action='store_true', help='Leave the ChArUco corners out (X from the column)')
    parser.add_argument('--sample-plane', action='store_true',
                        help="Each contact's fingertips on the table fitted to that sample's own depth")
    parser.add_argument('--free', default='1234567', help='Joints whose zero is solved, e.g. 23456')
    parser.add_argument('--free-scale', action='store_true', help='Also solve a depth scale (default 1)')
    args = parser.parse_args()
    problem = Problem(args.handeye, args.contacts, plate=args.plate, finger_bias=args.finger_bias,
                      free_scale=args.free_scale, hand_mount=args.hand_mount, handeye_term=not args.no_handeye,
                      sample_plane=args.sample_plane)
    print(f'{len(problem.he_samples)} hand-eye samples, {len(problem.contacts)} contacts, '
          f'{len(problem.column_frames)} column frames; table plane rms {problem.table_rms_mm:.2f} mm', flush=True)
    p = problem.solve([int(c) - 1 for c in args.free])
    rep = problem.report(p)
    print(json.dumps({k: v for k, v in rep.items() if not k.startswith('T_')}, indent=1))
    if args.out:
        write_result(args, problem, rep)


def write_result(args, problem, rep):
    out = args.out
    (out/'calibration').mkdir(parents=True, exist_ok=False)
    directions = problem.he['joint_compensation']['directions']
    yaml = '# Joint solve (solve_joint.py): hand-eye + fingertip contacts + column' + \
           (' + plate' if args.plate else '') + '\nversion: 1\njoints:\n' + ''.join(
               f'  joint{i}:\n    direction: {d}\n    offset_deg: {o}\n'
               for i, (d, o) in enumerate(zip(directions, rep['offsets_deg']), 1))
    yaml_path = (out/'joint_compensation.yaml').resolve()
    yaml_path.write_text(yaml)
    result = dict(problem.he)
    X, Y = np.asarray(rep['T_base_cam']), np.asarray(rep['T_flange_board'])
    delta = np.radians(rep['offsets_deg'])
    samples = []
    for s in result['samples']:
        s = dict(s)
        q = problem.q(np.asarray(s['motor_positions_rad']), delta)
        s['positions_rad'] = s['model_positions_rad'] = q.tolist()
        s['T_base_flange'] = problem.chain.link7(q).tolist()
        samples.append(s)
    result.update(method='joint (solve_joint.py)', T_base_cam=X.tolist(), T_flange_board=Y.tolist(),
                  T_world_cam=(np.asarray(result['T_world_base']) @ X).tolist(), samples=samples,
                  urdf=str(URDF),
                  joint_compensation=dict(directions=directions, offsets_deg=rep['offsets_deg'],
                                          source=str(yaml_path.relative_to(ROOT) if yaml_path.is_relative_to(ROOT)
                                                     else yaml_path)),
                  joint_solve=dict(contacts=[str(c) for c in args.contacts], plate=args.plate,
                                   finger_bias=args.finger_bias, free=args.free, depth_scale=rep['depth_scale'],
                                   finger_bias_mm=rep['finger_bias_mm'], hand_mount=args.hand_mount,
                                   hand_mount_mm=rep['hand_mount_mm'], stats=rep['stats'], start=str(args.handeye)),
                  quality_warnings=[])
    (out/'calibration'/'handeye_result.json').write_text(json.dumps(result, indent=2) + '\n')
    np.save(out/'calibration'/'T_base_cam.npy', X)
    print(f'Saved {out}', flush=True)


if __name__ == '__main__':
    main()
