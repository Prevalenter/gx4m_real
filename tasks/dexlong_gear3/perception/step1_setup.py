"""Step1 scene check in Viser before running the policy, on the live gear24 pose (fixed calibrated camera).

The Viser scene adds the training layout to viser_scene.py's robot, camera and detected gear: the plate with its
pins at fixture_pos, the gear's final pose on the left pin (green), the pre-insert waypoint and the start pose
with its +-2 cm randomization square (yellow). The plate itself is not tracked, so it is checked through the gear:
  1. Plate: put gear24 on the left pin. Its pose must match the trained final pose: horizontal offset and height
     within 3 mm, tilt within 3 deg (the policy switches stage at 3 mm). The offset says where to move the plate.
  2. Start: put gear24 flat on the table at the start spot, within the trained +-2 cm (any yaw).
The check uses the median of the last 15 valid poses. step1 only uses the left pin, so a plate yaw about it is
not visible here (and does not matter for step1).

Every pose is also forwarded over UDP (--target, as publish_gear_pose.py), so the same perception process keeps
feeding run_step1.py after the check:

  bash tasks/dexlong_gear3/perception/run_setup.sh --calibration <handeye_result.json>
  python tasks/dexlong_gear3/perception/step1_setup.py --demo     # synthetic gear on the pin, no camera or GPU
"""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from gears import DEFAULT_GEAR, PLATE_MESH, gear_mesh_path
from viser_scene import DEFAULT_INITIAL, SceneViewer, initial_base, load_calibration, rigid, set_pose, world_object

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent/'config'/'step1_reference.json'
# Tolerances (m, deg). The stage switch is 3 mm over the gear keypoints; the policy saw a 1 mm goal-relative bias.
PIN_XY, PIN_Z, PIN_TILT = .003, .003, 3.
START_XY, START_Z, START_TILT = .02, .005, 5.


def pose(pos, wxyz):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(wxyz, dtype=float)[[1, 2, 3, 0]]).as_matrix()
    T[:3, 3] = pos
    return T


def tilt_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b)/np.linalg.norm(a)/np.linalg.norm(b), -1., 1.))))


def median_pose(poses):
    """Median position and mean axis of recent gear poses (yaw is irrelevant for the checks)."""
    poses = np.asarray(poses)
    axis = poses[:, :3, 2].mean(axis=0)
    return np.median(poses[:, :3, 3], axis=0), axis/np.linalg.norm(axis)


def placement_report(position, axis, reference):
    """Gear-on-pin and gear-at-start errors of an env-frame gear pose against the training layout."""
    final, start = np.asarray(reference['final_pos']), np.asarray(reference['object_pos'])
    final_axis = pose(final, reference['final_quat'])[:3, 2]
    start_axis = pose(start, reference['object_quat'])[:3, 2]
    d = position - final
    pin = dict(dx=d[0], dy=d[1], xy=float(np.linalg.norm(d[:2])), dz=d[2], tilt=tilt_deg(axis, final_axis))
    pin['ok'] = pin['xy'] <= PIN_XY and abs(pin['dz']) <= PIN_Z and pin['tilt'] <= PIN_TILT
    s = position - start
    at_start = dict(dx=s[0], dy=s[1], dz=s[2], tilt=tilt_deg(axis, start_axis))
    at_start['ok'] = (max(abs(s[0]), abs(s[1])) <= START_XY and abs(s[2]) <= START_Z
                      and at_start['tilt'] <= START_TILT)
    return dict(on_pin=pin['xy'] < .03 and abs(pin['dz']) < .02, pin=pin, start=at_start)


def report_markdown(report):
    mm = lambda v: f'{v*1000:+.1f} mm'
    p, s = report['pin'], report['start']
    mark = lambda ok: '✅' if ok else '❌'
    pin = (f"**1. 底板（齿轮套在左销钉上）** {mark(p['ok'])}\n\n"
           f"水平偏差 {p['xy']*1000:.1f} mm（x {mm(p['dx'])}，y {mm(p['dy'])}；要求 ≤ {PIN_XY*1000:.0f} mm），"
           f"高度 {mm(p['dz'])}（≤ {PIN_Z*1000:.0f} mm），倾斜 {p['tilt']:.1f}°（≤ {PIN_TILT:.0f}°）\n\n"
           + ('' if p['ok'] else f"底板按 x {mm(-p['dx'])}、y {mm(-p['dy'])} 移动（世界坐标轴）。\n\n"))
    start = (f"**2. 起点（齿轮平放在桌上）** {mark(s['ok'])}\n\n"
             f"x {mm(s['dx'])}，y {mm(s['dy'])}（各 ≤ ±{START_XY*1000:.0f} mm），高度 {mm(s['dz'])}"
             f"（≤ {START_Z*1000:.0f} mm），倾斜 {s['tilt']:.1f}°（≤ {START_TILT:.0f}°）\n\n")
    head = '当前判断：齿轮在销钉上，检查底板。\n\n' if report['on_pin'] else '当前判断：齿轮不在销钉上，检查起点。\n\n'
    return head + (pin + start if report['on_pin'] else start + pin)


def mesh_handle(server, name, mesh, color, opacity, T):
    handle = server.scene.add_mesh_simple(name, np.asarray(mesh.vertices, dtype=np.float32),
                                          np.asarray(mesh.faces, dtype=np.uint32), color=color, opacity=opacity)
    set_pose(handle, T)
    return handle


class Step1Setup(SceneViewer):
    def __init__(self, calibration, *, reference, history=15, **kwargs):
        super().__init__(calibration, gear=DEFAULT_GEAR, **kwargs)
        self.ref, self.history = reference, deque(maxlen=history)
        scene = self.server.scene
        plate = trimesh.load(PLATE_MESH, force='mesh')
        gear = trimesh.load(gear_mesh_path(DEFAULT_GEAR), force='mesh')
        r = reference
        mesh_handle(self.server, '/sim/plate', plate, (190, 195, 200), .8, pose(r['fixture_pos'], r['fixture_quat']))
        mesh_handle(self.server, '/sim/goal_gear', gear, (40, 200, 90), .35, pose(r['final_pos'], r['final_quat']))
        mesh_handle(self.server, '/sim/start_gear', gear, (240, 200, 40), .35, pose(r['object_pos'], r['object_quat']))
        scene.add_box('/sim/start_region', color=(240, 200, 40), dimensions=(.04, .04, .001), opacity=.3,
                      position=np.asarray(r['object_pos']) - [0, 0, .004])
        scene.add_frame('/sim/preinsert', axes_length=.03, position=np.asarray(r['preinsert_pos']),
                        wxyz=np.asarray(r['preinsert_quat']))
        scene.add_label('/sim/goal_label', '目标：左销钉', position=np.asarray(r['final_pos']) + [0, 0, .05])
        scene.add_label('/sim/start_label', '起点 ±2 cm', position=np.asarray(r['object_pos']) + [0, 0, .05])
        gui = self.server.gui
        gui.add_markdown('## step1 摆放检查\n灰色：仿真底板；绿色：齿轮插好的目标；黄色：起点和 ±2 cm 范围。'
                         '先把齿轮套在左销钉上对齐底板，再把齿轮放到起点。')
        self.check = gui.add_markdown('等待齿轮位姿。')
        self.report = None

    def update(self, record, *, rgb=None, K=None):
        super().update(record, rgb=rgb, K=K)
        with self.lock:
            if not self.part.visible:
                self.history.clear()
                self.report = None
                self.check.content = '没有有效的齿轮位姿，检查暂停。'
                return
            self.history.append(world_object(self.calibration, record['object_in_camera']))
            self.report = placement_report(*median_pose(self.history), self.ref)
            self.check.content = report_markdown(self.report) + f'（最近 {len(self.history)} 帧中位数）'

    def expire(self, now=None):
        super().expire(now)
        with self.lock:
            if not self.part.visible and self.history:
                self.history.clear()
                self.report = None
                self.check.content = '齿轮位姿已过期，检查暂停。'


class Observers:
    """Forward live_gear36 observer calls to several observers."""

    def __init__(self, *observers):
        self.observers = observers

    def set_camera_metadata(self, metadata):
        for o in self.observers:
            o.set_camera_metadata(metadata)

    def update(self, record, rgb=None, K=None):
        for o in self.observers:
            o.update(record, rgb=rgb, K=K)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--calibration', type=Path, help='Fixed-camera handeye_result.json')
    parser.add_argument('--demo', action='store_true', help='Synthetic gear on the pin, then at the start; no camera')
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--initial-pose', type=Path, default=DEFAULT_INITIAL)
    parser.add_argument('--target', default='127.0.0.1:5701', help="Forward poses to run_step1.py ('' to disable)")
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8088)
    args, inference_args = parser.parse_known_args(argv)
    if not args.demo and args.calibration is None:
        parser.error('--calibration is required (or --demo)')
    if inference_args and args.demo:
        parser.error(f'Unknown options: {inference_args}')
    reference = json.loads(args.reference.read_text(encoding='utf-8'))
    initial = json.loads(args.initial_pose.read_text())
    if args.demo:
        C = np.eye(4)
        C[:3, 3] = [.35, -.25, .45]
        C[:3, :3] = Rotation.from_euler('xyz', [140, 0, 45], degrees=True).as_matrix()
        calibration = {'T_world_base': initial_base(initial), 'T_base_cam': C, 'synthetic': True,
                       'base_link': 'base_link'}
    else:
        calibration = load_calibration(args.calibration, initial)
    viewer = Step1Setup(calibration, reference=reference, initial=initial, host=args.host, port=args.port,
                        live=not args.demo)
    print(f'Viser: http://{args.host}:{viewer.server.get_port()}', flush=True)
    try:
        if args.demo:
            T_cam_world = np.linalg.inv(rigid(calibration['T_world_base']) @ rigid(calibration['T_base_cam']))
            on_pin = pose(np.asarray(reference['final_pos']) + [.002, -.001, 0], reference['final_quat'])
            for _ in range(15):
                viewer.update({'status': 'TRACK', 'object_in_camera': T_cam_world @ on_pin})
            print(json.dumps(viewer.report, indent=2, default=lambda v: v.item()), flush=True)
            while not viewer.stop_event.wait(.2):
                pass
        else:
            from live_gear36 import main as detect_live
            observers = [viewer]
            if args.target:
                from publish_gear_pose import UdpPosePublisher
                host, port = args.target.rsplit(':', 1)
                observers.append(UdpPosePublisher(host, int(port)))
            detect_live(['--gear', DEFAULT_GEAR, '--no-display', *inference_args], observer=Observers(*observers))
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == '__main__':
    main()
