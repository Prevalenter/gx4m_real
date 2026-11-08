"""Replay collect_contacts.py samples under a calibration result, in Viser: the sample's saved depth cloud through
the result's extrinsic, the robot at the sample's raw angles mapped with the result's compensation, each finger's
lowest point and its distance to the table plane (with the result's per-finger offsets when it has them).
The result's robot is drawn see-through (opacity slider); --compare draws another result's robot (e.g. the one
before identification) in its own colours, placed by its own extrinsic against the same cloud.

  bash apps/charuco_handeye/run.sh replay --result <handeye_result.json> --contacts <ct_dir> [<ct_dir> ...] \\
      [--compare <handeye_result.json>] [--base-pose tasks/dexlong_gear3/config/initial_pose.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gx4m_real.robot.compensation import JointCompensation
from handeye_session import transform
from solve_joint import ARM_JOINTS, FINGER_LINKS, URDF, plane_fit

FINGERS = ('拇指', '食指', '中指', '无名指')
FINGER_RGB = ((150, 80, 210), (40, 130, 240), (40, 180, 80), (240, 150, 20))


def load_samples(dirs):
    samples = []
    for d in dirs:
        for line in (Path(d)/'samples.jsonl').read_text().splitlines():
            s = json.loads(line)
            if not s['discarded']:
                samples.append((Path(d), s))
    return samples


class ReplayViewer:
    def __init__(self, result_path, contact_dirs, *, compare=None, base_pose=None, host='127.0.0.1', port=8090,
                 stride=2, opacity=.5):
        import viser
        from viser.extras import ViserUrdf
        import yourdfpy
        from view_result import cloud_from_rgbd, set_pose

        self.cloud_from_rgbd, self.set_pose, self.stride = cloud_from_rgbd, set_pose, stride
        self.result = json.loads(Path(result_path).read_text())
        c = self.result['joint_compensation']
        self.compensation = JointCompensation(tuple(c['directions']), tuple(c['offsets_deg']))
        self.bias = np.asarray(self.result.get('joint_solve', {}).get('finger_bias_mm') or [0.]*4)/1000
        self.T_world_base = transform(base_pose if base_pose is not None else self.result['T_world_base'])
        self.T_base_cam = transform(self.result['T_base_cam'])
        self.samples = load_samples(contact_dirs)
        if not self.samples:
            raise ValueError('No kept samples')
        # table plane in the camera, from the column frames (as solve_joint.py does)
        import cv2
        table = []
        for d, s in self.samples:
            if s['kind'] in ('column', 'scene'):
                with np.load(d/s['rgbd']) as z:
                    rgb, depth, K = z['rgb'], z['depth'], z['K']
                H, W = depth.shape
                v, u = np.mgrid[:H, :W]
                P = np.stack([(u - K[0, 2])*depth/K[0, 0], (v - K[1, 2])*depth/K[1, 1], depth], -1)
                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
                table.append(P[(depth > 0) & (hsv[..., 2] > 140) & (hsv[..., 1] < 45) & (v > H*.55)])
        n, d, _ = plane_fit(np.vstack(table)[::5]) if table else (None, None, None)
        if n is not None:
            self.table_n = self.T_base_cam[:3, :3] @ n
            self.table_d = self.table_n @ self.T_base_cam[:3, 3] + d
        else:
            self.table_n = None

        self.model = yourdfpy.URDF.load(str(URDF), load_meshes=True)
        self.finger_nodes = {f: [node for node in self.model.scene.graph.nodes_geometry
                                 if str(self.model.scene.graph.transforms.parents.get(node)) in links]
                             for f, links in enumerate(FINGER_LINKS)}
        self.server = viser.ViserServer(host=host, port=port, label='GX7 · 接触样本回放')
        self.server.scene.set_up_direction('+z')
        self.server.scene.add_grid('/ground', width=2., height=2., plane='xy')
        base = self.server.scene.add_frame('/robot', axes_length=.15, axes_radius=.0006, origin_radius=.0012)
        set_pose(base, self.T_world_base)
        self.visual = ViserUrdf(self.server, self.model, root_node_name='/robot',
                                mesh_color_override=(.25, .6, 1., opacity))
        self.before = None
        if compare is not None:
            other = json.loads(Path(compare).read_text())
            c = other['joint_compensation']
            # The other result's robot where its extrinsic puts it against this cloud: base = cam * X_other^-1.
            T_world_cam = self.T_world_base @ self.T_base_cam
            before_base = self.server.scene.add_frame('/robot_before', axes_length=.1, axes_radius=.0006,
                                                      origin_radius=.0012)
            set_pose(before_base, T_world_cam @ np.linalg.inv(transform(other['T_base_cam'])))
            self.before_model = yourdfpy.URDF.load(str(URDF), load_meshes=True)
            self.before = (JointCompensation(tuple(c['directions']), tuple(c['offsets_deg'])),
                           ViserUrdf(self.server, self.before_model, root_node_name='/robot_before'), before_base,
                           Path(compare).parent.parent.name, list(c['offsets_deg']))
        self.cloud = None
        self.markers = [self.server.scene.add_icosphere(f'/finger_{f}', radius=.005, color=c)
                        for f, c in enumerate(FINGER_RGB)]
        gui = self.server.gui
        gui.add_markdown(f"## 接触样本回放\n标定：`{Path(result_path).parent.parent.name}`，补偿 "
                         f"{list(self.compensation.offsets_deg)}°" +
                         (f"；每指偏差 {list(np.round(self.bias*1000, 1))} mm" if self.bias.any() else '') +
                         '\n\n彩色小球：各手指模型最低点。表里是它到桌面的距离（接触样本应为 0）。')
        self.index = gui.add_slider('样本', min=0, max=len(self.samples) - 1, step=1, initial_value=0)
        alpha = gui.add_slider('辨识后模型（蓝）不透明度', min=0., max=1., step=.05, initial_value=opacity)

        @alpha.on_update
        def _(_):
            for mesh in self.visual._meshes:
                mesh.opacity = float(alpha.value)

        if self.before is not None:
            gui.add_markdown(f'对比（原色）：`{self.before[3]}`，补偿 {self.before[4]}°，按它自己的外参放在同一点云里。')
            shown = gui.add_checkbox('显示对比模型', initial_value=True)
            shown.on_update(lambda e: setattr(self.before[2], 'visible', e.target.value))
        self.text = gui.add_markdown('')
        self.image = gui.add_image(np.zeros((480, 640, 3), np.uint8), label='当时的 RGB')
        self.index.on_update(lambda _: self.show())
        self.show()

    def finger_lowest(self):
        """Per finger: (lowest model point in base_link, its height above the table plane in m)."""
        out = []
        for f in range(4):
            best = None
            for node in self.finger_nodes[f]:
                T, geom = self.model.scene.graph.get(node)
                V = (T[:3, :3] @ self.model.scene.geometry[geom].vertices.T).T + T[:3, 3]
                h = V @ self.table_n - self.table_d
                i = int(np.argmin(h))
                if best is None or h[i] < best[1]:
                    best = (V[i], h[i])
            out.append(best)
        return out

    def show(self):
        d, s = self.samples[int(self.index.value)]
        q = {}
        if s.get('arm_motor_rad') is not None:
            q.update(zip(ARM_JOINTS, self.compensation.to_model(np.asarray(s['arm_motor_rad']))))
        if s.get('hand_q_rad') is not None:
            q.update(zip([f'gx4m_joint{i}' for i in range(1, 17)], s['hand_q_rad']))
        self.model.update_cfg({n: q.get(n, 0.) for n in self.model.actuated_joint_names})
        self.visual.update_cfg(np.array([q.get(n, 0.) for n in self.model.actuated_joint_names]))
        if self.before is not None:
            qb = dict(q)
            if s.get('arm_motor_rad') is not None:
                qb.update(zip(ARM_JOINTS, self.before[0].to_model(np.asarray(s['arm_motor_rad']))))
            self.before[1].update_cfg(np.array([qb.get(n, 0.) for n in self.before_model.actuated_joint_names]))
        with np.load(d/s['rgbd']) as z:
            rgb, depth, K = z['rgb'], z['depth'], z['K']
        points, colors = self.cloud_from_rgbd(rgb, depth, K, self.stride)
        C = self.T_world_base @ self.T_base_cam
        points = ((C[:3, :3] @ points.T).T + C[:3, 3]).astype(np.float32)
        if self.cloud is None:
            self.cloud = self.server.scene.add_point_cloud('/cloud', points=points, colors=colors, point_size=.002,
                                                           point_shape='circle')
        else:
            self.cloud.points, self.cloud.colors = points, colors
        self.image.image = rgb
        rows = [f"**#{s['index']}**（{d.name}，{'接触' if s['kind'] == 'contact' else '立柱/桌面'}）"]
        if self.table_n is not None and s.get('arm_motor_rad') is not None and s.get('hand_q_rad') is not None:
            rows += ['', '| 手指 | 最低点到桌面 | 加每指偏差后 |', '|---|---|---|']
            for f, (point, h) in enumerate(self.finger_lowest()):
                self.markers[f].position = (self.T_world_base[:3, :3] @ point + self.T_world_base[:3, 3])
                self.markers[f].visible = True
                rows.append(f'| {FINGERS[f]} | {h*1000:+.1f} mm | {(h + self.bias[f])*1000:+.1f} mm |')
        else:
            for m in self.markers:
                m.visible = False
        self.text.content = '\n'.join(rows)

    def run(self):
        import time
        try:
            while True:
                time.sleep(1.)
        except KeyboardInterrupt:
            self.server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--contacts', type=Path, nargs='+', required=True)
    parser.add_argument('--compare', type=Path, help='Another result drawn in its own colours, e.g. before identification')
    parser.add_argument('--base-pose', type=Path)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8090)
    args = parser.parse_args()
    from view_result import load_base_pose
    ReplayViewer(args.result, args.contacts, compare=args.compare, base_pose=load_base_pose(args.base_pose) if args.base_pose else None,
                 host=args.host, port=args.port).run()


if __name__ == '__main__':
    main()
