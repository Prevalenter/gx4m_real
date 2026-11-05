"""Step1 / step2 scene check in Viser before running the policy, on the live gear pose (fixed calibrated camera).

--step 1 (default) tracks gear24 with the left pin as its target, --step 2 gear48 with the right pin (gear24 already
mounted on the left one); the pin, start and plate below are that step's reference. Start run_step1.py /
robot_server.py with the same --step.

The Viser scene adds the training layout to viser_scene.py's robot, camera and detected gear: the plate with its
pins at fixture_pos, the gear's final pose on the left pin (green), the pre-insert waypoint and the start pose
with its +-2 cm randomization square (yellow). The gear updates live. The plate (black) is measured once each time its
button is pressed, checked at the left pin. Measuring stalls the gear loop for short moments, so do not press it while
the policy runs. --plate shows a saved detect_plate.py result until then. It can also be checked through the gear:
  1. Plate: put gear24 on the left pin. Its pose must match the trained final pose: horizontal offset and height
     within 3 mm, tilt within 3 deg (the policy switches stage at 3 mm). The offset says where to move the plate.
  2. Start: put gear24 flat on the table at the start spot, within the trained +-2 cm (any yaw).
The check uses the median of the last 15 valid poses. step1 only uses the left pin, so a plate yaw about it is
not visible through the gear (and does not matter for step1).
Both checks are always shown: distance to the training pose cannot tell whether the gear is physically on a pin.

Every pose is also forwarded over UDP (--target, as publish_gear_pose.py), so the same perception process keeps
feeding run_step1.py after the check. The robot buttons (robot_panel.py: connect / disconnect arm and hand, hand
guiding, start pose, policy, stop) drive tasks/dexlong_gear3/robot_server.py, which then takes those poses instead:

  bash tasks/dexlong_gear3/perception/run_setup.sh --calibration <handeye_result.json>
  bash tasks/dexlong_gear3/run_robot_server.sh --handeye <handeye_result.json> --hand-port /dev/ttyUSB0
  python tasks/dexlong_gear3/perception/step1_setup.py --demo     # synthetic gear on the pin, no camera or GPU
  bash tasks/dexlong_gear3/perception/run_setup.sh --step 2 --calibration <handeye_result.json>
"""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from gears import DEFAULT_GEAR, PLATE_MESH, gear_mesh_path
from viser_scene import DEFAULT_INITIAL, SceneViewer, initial_base, load_calibration, rigid, set_pose, world_object

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent/'config'/'step1_reference.json'
# Split step -> (reference, gear it inserts); as step1.REFERENCES (that module needs the robot environment).
STEPS = {1: (REFERENCE, 'gear24'), 2: (HERE.parent/'config'/'step2_reference.json', 'gear48')}
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
    return dict(pin=pin, start=at_start)


def plate_report(T_world_plate, reference):
    """Error of a measured env-frame plate pose, at the left pin (the only part of the plate step1 uses)."""
    trained = pose(reference['fixture_pos'], reference['fixture_quat'])
    delta = T_world_plate @ np.linalg.inv(trained)
    pin = np.asarray(reference['final_pos'])
    d = delta[:3, :3] @ pin + delta[:3, 3] - pin
    relative = trained[:3, :3].T @ T_world_plate[:3, :3]
    report = dict(dx=d[0], dy=d[1], xy=float(np.linalg.norm(d[:2])), dz=d[2],
                  tilt=tilt_deg(T_world_plate[:3, 2], trained[:3, 2]),
                  yaw=float(np.degrees(np.arctan2(relative[1, 0], relative[0, 0]))))
    report['ok'] = bool(report['xy'] <= PIN_XY and abs(report['dz']) <= PIN_Z and report['tilt'] <= PIN_TILT)
    return report


def plate_markdown(report, note=''):
    mm = lambda v: f'{v*1000:+.1f} mm'
    r = report
    return (f"**底板已检测（{note}）**\n\n"
            f"与训练位置对齐：{'✅' if r['ok'] else '❌'}\n\n"
            f"目标销钉处水平偏差 {r['xy']*1000:.1f} mm（x {mm(r['dx'])}，y {mm(r['dy'])}；要求 ≤ {PIN_XY*1000:.0f} mm），"
            f"高度 {mm(r['dz'])}（≤ {PIN_Z*1000:.0f} mm），倾斜 {r['tilt']:.1f}°（≤ {PIN_TILT:.0f}°），"
            f"偏航 {r['yaw']:+.1f}°（不要求）\n\n"
            + ('' if r['ok'] else f"目标销钉处按 x {mm(-r['dx'])}、y {mm(-r['dy'])} 移动底板。\n\n"))


def report_markdown(report):
    mm = lambda v: f'{v*1000:+.1f} mm'
    p, s = report['pin'], report['start']
    mark = lambda ok: '✅' if ok else '❌'
    pin = (f"**1. 底板（齿轮套在目标销钉上）** {mark(p['ok'])}\n\n"
           f"水平偏差 {p['xy']*1000:.1f} mm（x {mm(p['dx'])}，y {mm(p['dy'])}；要求 ≤ {PIN_XY*1000:.0f} mm），"
           f"高度 {mm(p['dz'])}（≤ {PIN_Z*1000:.0f} mm），倾斜 {p['tilt']:.1f}°（≤ {PIN_TILT:.0f}°）\n\n"
           + ('' if p['ok'] else f"底板按 x {mm(-p['dx'])}、y {mm(-p['dy'])} 移动（世界坐标轴）。\n\n"))
    start = (f"**2. 起点（齿轮平放在桌上）** {mark(s['ok'])}\n\n"
             f"x {mm(s['dx'])}，y {mm(s['dy'])}（各 ≤ ±{START_XY*1000:.0f} mm），高度 {mm(s['dz'])}"
             f"（≤ {START_Z*1000:.0f} mm），倾斜 {s['tilt']:.1f}°（≤ {START_TILT:.0f}°）\n\n")
    head = '齿轮套在目标销钉上时查看底板偏差；放在起点时查看起点偏差。\n\n'
    return head + pin + start


def mesh_handle(server, name, mesh, color, opacity, T):
    handle = server.scene.add_mesh_simple(name, np.asarray(mesh.vertices, dtype=np.float32),
                                          np.asarray(mesh.faces, dtype=np.uint32), color=color, opacity=opacity)
    set_pose(handle, T)
    return handle


class Step1Setup(SceneViewer):
    def __init__(self, calibration, *, reference, history=15, plate=None, gear=DEFAULT_GEAR, **kwargs):
        super().__init__(calibration, gear=gear, **kwargs)
        self.ref, self.history = reference, deque(maxlen=history)
        scene = self.server.scene
        plate_mesh = trimesh.load(PLATE_MESH, force='mesh')
        gear = trimesh.load(gear_mesh_path(gear), force='mesh')
        r = reference
        mesh_handle(self.server, '/sim/plate', plate_mesh, (190, 195, 200), .3, pose(r['fixture_pos'], r['fixture_quat']))
        mesh_handle(self.server, '/sim/goal_gear', gear, (40, 200, 90), .35, pose(r['final_pos'], r['final_quat']))
        mesh_handle(self.server, '/sim/start_gear', gear, (240, 200, 40), .35, pose(r['object_pos'], r['object_quat']))
        scene.add_box('/sim/start_region', color=(240, 200, 40), dimensions=(.04, .04, .001), opacity=.3,
                      position=np.asarray(r['object_pos']) - [0, 0, .004])
        scene.add_frame('/sim/preinsert', axes_length=.08, axes_radius=.0006, origin_radius=.0012,
                        position=np.asarray(r['preinsert_pos']),
                        wxyz=np.asarray(r['preinsert_quat']))
        scene.add_label('/sim/goal_label', '目标：目标销钉', position=np.asarray(r['final_pos']) + [0, 0, .05])
        scene.add_label('/sim/start_label', '起点 ±2 cm', position=np.asarray(r['object_pos']) + [0, 0, .05])
        gui = self.server.gui
        gui.add_markdown('## step1 摆放检查\n半透明灰色：底板的训练位置；绿色：齿轮插好的目标；黄色：起点和 ±2 cm 范围。'
                         '点击「检测底板」显示实际底板模型和坐标。')
        self.plate_report = None
        self.T_world_plate = None
        self.detected_plate = mesh_handle(self.server, '/detected/plate', plate_mesh, (35, 35, 40), 1., np.eye(4))
        self.detected_plate.visible = False
        self.plate_axes = scene.add_frame('/detected/plate_axes', axes_length=.15, axes_radius=.0006,
                                          origin_radius=.0012, visible=False)
        self.plate_label = scene.add_label('/detected/plate_label', '检测到的底板', visible=False)
        self.plate_check = gui.add_markdown('底板还没有检测，按下面的按钮检测一次。')
        self.plate_status = gui.add_markdown('')
        self.remeasure = threading.Event()
        gui.add_button('检测底板').on_click(lambda _: self.remeasure.set())
        if plate is not None:
            self.show_plate(plate['T_camera_plate'], f"已保存的结果，{plate.get('measured_at', '')}")
        with gui.add_folder('通过齿轮校验摆放（可选）'):
            self.check = gui.add_markdown('等待齿轮位姿。')
        self.report = None

    def configure_client_camera(self, client):
        if getattr(self, 'T_world_plate', None) is None:
            super().configure_client_camera(client)
        else:
            center = self.T_world_plate[:3, 3]
            client.camera.position = center + [.55, -.70, .55]
            client.camera.look_at = center

    def show_plate(self, T_camera_plate, note):
        with self.lock:
            T_world_plate = world_object(self.calibration, T_camera_plate)
            self.T_world_plate = T_world_plate
            set_pose(self.detected_plate, T_world_plate)
            self.detected_plate.visible = True
            set_pose(self.plate_axes, T_world_plate)
            self.plate_axes.visible = True
            self.plate_label.position = T_world_plate[:3, 3] + [0, 0, .05]
            self.plate_label.visible = True
            self.plate_status.content = ''
            self.plate_report = plate_report(T_world_plate, self.ref)
            self.plate_check.content = ('黑色：检测到的底板；细坐标轴：底板原点。\n\n'
                f"底板世界坐标 XYZ：{T_world_plate[:3, 3].round(4).tolist()} m\n\n"
                f"底板相机坐标 XYZ：{np.asarray(T_camera_plate)[:3, 3].round(4).tolist()} m\n\n"
                + plate_markdown(self.plate_report, note))
            for client in self.server.get_clients().values():
                self.configure_client_camera(client)

    def update_plate(self, state):
        """Show a PlateDetector state. Until a new pose is fused the last one stays on screen."""
        status, pose = state['status'], state['T_camera_plate']
        if status == 'DONE':
            self.show_plate(pose, f"{time.strftime('%H:%M:%S')} 检测")
            return
        message = (f"检测中 {state['fused']}/{state['fuse_frames']}。" if status == 'MEASURING'
                   else f"检测失败：{state['reason']}。")
        if pose is not None:
            message += '黑色底板仍是上次结果。'
        self.plate_status.content = message

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


def plate_frame_hook(viewer, reference):
    """live_gear36 frame_hook measuring the plate when the viewer's button was pressed."""
    def build(scorer, refiner, glctx, output):
        from detect_plate import PlateDetector, load_plate_backend
        backend = load_plate_backend(trimesh, scorer, refiner, glctx, output)
        detector = PlateDetector(backend, reference, viewer.T_world_cam)

        def on_frame(frame):
            if viewer.remeasure.is_set():
                viewer.remeasure.clear()
                if not detector.active:
                    detector.start()
                    viewer.update_plate(detector.state())
            state = detector.update(frame)
            if state is not None:
                viewer.update_plate(state)
                if state['status'] == 'DONE':
                    result = dict(measured_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                                  T_camera_plate=np.asarray(state['T_camera_plate']).tolist(),
                                  T_world_plate=viewer.T_world_plate.tolist(),
                                  spread=state['spread'], report=viewer.plate_report)
                    (output / 'plate_result.json').write_text(json.dumps(result, indent=2, default=float) + '\n')
                    print(f"Plate measured: {viewer.T_world_plate[:3, 3].round(4).tolist()} m; "
                          f"saved {output / 'plate_result.json'}", flush=True)
                elif state['status'] == 'FAILED':
                    print(f"Plate detection failed: {state['reason']}", flush=True)
        return on_frame
    return build


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
    parser.add_argument('--plate', type=Path, help='Show this plate_result.json from detect_plate.py until the button is pressed')
    parser.add_argument('--step', type=int, choices=sorted(STEPS), default=1, help='Split step: its reference and gear')
    parser.add_argument('--reference', type=Path, help="Default: the step's reference")
    parser.add_argument('--initial-pose', type=Path, default=DEFAULT_INITIAL)
    parser.add_argument('--target', default='127.0.0.1:5701', help="Forward poses to run_step1.py ('' to disable)")
    parser.add_argument('--robot', default='127.0.0.1:5702',
                        help="robot_server.py behind the robot buttons ('' to hide them)")
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8088)
    args, inference_args = parser.parse_known_args(argv)
    if not args.demo and args.calibration is None:
        parser.error('--calibration is required (or --demo)')
    if inference_args and args.demo:
        parser.error(f'Unknown options: {inference_args}')
    reference_path, gear = STEPS[args.step]
    reference = json.loads((args.reference or reference_path).read_text(encoding='utf-8'))
    initial = json.loads(args.initial_pose.read_text())
    if args.demo:
        C = np.eye(4)
        C[:3, 3] = [.35, -.25, .45]
        C[:3, :3] = Rotation.from_euler('xyz', [140, 0, 45], degrees=True).as_matrix()
        calibration = {'T_world_base': initial_base(initial), 'T_base_cam': C, 'synthetic': True,
                       'base_link': 'base_link'}
    else:
        calibration = load_calibration(args.calibration, initial)
    plate = json.loads(args.plate.read_text(encoding='utf-8')) if args.plate else None
    viewer = Step1Setup(calibration, reference=reference, plate=plate, gear=gear, initial=initial, host=args.host,
                        port=args.port, live=not args.demo)
    if args.robot:
        from robot_panel import RobotPanel
        host, port = args.robot.rsplit(':', 1)
        RobotPanel(viewer, (host, int(port)))
    print(f'Viser: http://{args.host}:{viewer.server.get_port()}', flush=True)
    try:
        if args.demo:
            T_cam_world = np.linalg.inv(rigid(calibration['T_world_base']) @ rigid(calibration['T_base_cam']))
            on_pin = pose(np.asarray(reference['final_pos']) + [.002, -.001, 0], reference['final_quat'])
            for _ in range(15):
                viewer.update({'status': 'TRACK', 'object_in_camera': T_cam_world @ on_pin})
            print(json.dumps(viewer.report, indent=2, default=lambda v: v.item()), flush=True)
            shifted = pose(np.asarray(reference['fixture_pos']) + [.002, -.001, 0], reference['fixture_quat'])
            viewer.show_plate(T_cam_world @ shifted, '合成')
            while not viewer.stop_event.wait(.2):
                pass
        else:
            from live_gear36 import main as detect_live
            observers = [viewer]
            if args.target:
                from publish_gear_pose import UdpPosePublisher
                host, port = args.target.rsplit(':', 1)
                observers.append(UdpPosePublisher(host, int(port)))
            up_in_camera = viewer.T_world_cam[:3, :3].T @ [0., 0., 1.]     # gears lie hub up (gear_tracker.hub_up)
            detect_live(['--gear', gear, '--no-display', *inference_args], observer=Observers(*observers),
                        frame_hook=plate_frame_hook(viewer, reference), up_in_camera=up_in_camera)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == '__main__':
    main()
