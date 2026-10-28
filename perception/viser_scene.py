"""Viser scene for the GX7 robot, a fixed calibrated camera and gear36 poses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import viser
from viser.extras import ViserUrdf
import yourdfpy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_URDF = ROOT / 'asset/gx7_right/urdf/gx7_right_gx4m_cam_xl330.urdf'
DEFAULT_MESH = ROOT / 'asset/mjfit_gears3_split/assembly/gear36_visual.obj'
DEFAULT_INITIAL = HERE.parent / 'step0/initial_pose.json'
JOINTS = [f'joint{i}' for i in range(1, 8)]


def rigid(value):
    value = np.asarray(value, dtype=float)
    if (value.shape != (4, 4) or not np.isfinite(value).all()
            or not np.allclose(value[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(value[:3, :3]), 1., atol=1e-4)):
        raise ValueError('Expected a finite rigid 4x4 transform in metres')
    return value


def initial_base(initial):
    W = np.eye(4)
    W[:3, :3] = Rotation.from_quat(np.asarray(initial['base_wxyz'])[[1, 2, 3, 0]]).as_matrix()
    W[:3, 3] = initial['base_position']
    return rigid(W)


def load_calibration(path, initial):
    if path.suffix == '.npy':
        return {'T_base_cam': rigid(np.load(path)), 'T_world_base': initial_base(initial),
                'base_link': 'base_link', 'synthetic': False}
    data = json.loads(path.read_text())
    if data.get('mode', 'eye-to-hand') != 'eye-to-hand':
        raise ValueError('This viewer expects a fixed-camera (eye-to-hand) calibration')
    return {**data, 'T_base_cam': rigid(data['T_base_cam']),
            'T_world_base': rigid(data.get('T_world_base', initial_base(initial))),
            'base_link': data.get('base_link', 'base_link')}


def world_object(calibration, object_in_camera):
    C = rigid(calibration['T_world_base']) @ rigid(calibration['T_base_cam'])
    return C @ rigid(object_in_camera)


def set_pose(handle, T):
    handle.position = T[:3, 3]
    handle.wxyz = Rotation.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]]


def read_detection(path):
    """Load existing single-frame detection/snapshot output without running inference."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / 'result.json'
    record = json.loads(path.read_text())
    candidates = [path.parent]
    if record.get('capture'):
        candidates.append(Path(record['capture']).expanduser())
    rgb = K = None
    from PIL import Image
    for folder in candidates:
        if rgb is None and (folder / 'rgb.png').is_file():
            rgb = np.array(Image.open(folder / 'rgb.png').convert('RGB'))
        if K is None and (folder / 'rgbd.npz').is_file():
            with np.load(folder / 'rgbd.npz') as data:
                K = data['K']
    return record, rgb, K


class SceneViewer:
    def __init__(self, calibration, *, initial, urdf=DEFAULT_URDF, mesh=DEFAULT_MESH,
                 joints=None, host='127.0.0.1', port=8088, live=False, stale_after=2.):
        self.calibration = calibration
        self.live = live
        self.stale_after = stale_after
        self.lock = threading.RLock()
        self.last_update = None
        self.stop_event = threading.Event()
        self.T_world_cam = rigid(calibration['T_world_base']) @ rigid(calibration['T_base_cam'])
        self.model = yourdfpy.URDF.load(str(urdf), load_meshes=True)
        q = dict(initial['joint_positions'])
        if joints is not None:
            data = json.loads(Path(joints).read_text())
            values = data.get('positions_rad', data.get('joint_positions', data)) if isinstance(data, dict) else data
            if isinstance(values, dict):
                q.update(values)
            else:
                values = np.asarray(values, dtype=float)
                if values.shape != (7,):
                    raise ValueError('Joint pose needs exactly seven radians')
                q.update(zip(JOINTS, values))
        self.q = np.array([q[name] for name in self.model.actuated_joint_names], dtype=float)
        if not np.isfinite(self.q).all():
            raise ValueError('Joint pose contains non-finite values')
        self.model.update_cfg(self.q)
        base_link = calibration.get('base_link', 'base_link')
        root_in_base = self.model.get_transform(self.model.base_link, base_link)
        gear = trimesh.load(mesh, force='mesh')
        if not .10 < float(gear.extents[:2].max()) < .13:
            raise ValueError('Expected gear36 mesh in metres, about 114 mm in diameter')
        gear.visual.vertex_colors = np.tile([60, 120, 230, 255], (len(gear.vertices), 1))
        self.server = viser.ViserServer(host=host, port=port, label='GX7 · 固定相机 · 零件位姿')
        self.server.scene.set_up_direction('+z')
        self.server.scene.add_grid('/ground', width=2, height=2, plane='xy')
        self.server.scene.add_frame('/world', axes_length=.15)
        base = self.server.scene.add_frame('/robot', axes_length=.1)
        set_pose(base, calibration['T_world_base'] @ root_in_base)
        self.robot = ViserUrdf(self.server, self.model, root_node_name='/robot')
        self.robot.update_cfg(self.q)
        camera = self.server.scene.add_frame('/camera', axes_length=.1)
        set_pose(camera, self.T_world_cam)
        self.frustum = self.server.scene.add_camera_frustum('/camera/frustum',
            fov=np.deg2rad(55), aspect=4/3, scale=.15, color=(255, 180, 40))
        self.part = self.server.scene.add_frame('/part', axes_length=.07, visible=False)
        self.server.scene.add_mesh_trimesh('/part/gear36', gear)
        gui = self.server.gui
        gui.add_markdown('## 固定相机与零件位姿\n' +
                         ('**合成演示/标定数据，非真机测量。**\n\n' if calibration.get('synthetic') else '') +
                         '机器人显示参考关节姿态，不读取或控制真机。相机光轴为 +Z。')
        gui.add_markdown('读取图像内参后更新视锥；没有内参时视锥仅作示意。')
        self.status = gui.add_text('检测状态', initial_value='等待检测', disabled=True)
        self.details = gui.add_markdown('')
        self.image = gui.add_image(np.zeros((480, 640, 3), dtype=np.uint8), label='相机 RGB')
        with gui.add_folder('机器人参考姿态（度）'):
            for name in JOINTS:
                index = self.model.actuated_joint_names.index(name)
                limit = self.model.joint_map[name].limit
                lower = -180 if limit is None or limit.lower is None else float(np.rad2deg(limit.lower))
                upper = 180 if limit is None or limit.upper is None else float(np.rad2deg(limit.upper))
                slider = gui.add_slider(name, min=min(lower, float(np.rad2deg(self.q[index]))),
                                        max=max(upper, float(np.rad2deg(self.q[index]))), step=.1,
                                        initial_value=float(np.rad2deg(self.q[index])))
                def change(event, i=index):
                    with self.lock:
                        self.q[i] = np.deg2rad(event.target.value)
                        self.robot.update_cfg(self.q)
                slider.on_update(change)
        @self.server.on_client_connect
        def connected(client):
            base_pos = calibration['T_world_base'][:3, 3]
            client.camera.position = base_pos + [1., -1.3, .9]
            client.camera.look_at = (base_pos + self.T_world_cam[:3, 3]) / 2
        self.monitor = threading.Thread(target=self._monitor, daemon=True, name='viser-pose-expiry')
        self.monitor.start()

    def set_camera_metadata(self, metadata):
        expected = self.calibration.get('camera_metadata', {}).get('serial')
        if expected and metadata.get('serial') and expected != metadata['serial']:
            raise ValueError('Camera serial differs from the fixed-camera calibration')

    def update(self, record, *, rgb=None, K=None):
        with self.lock:
            self.last_update = time.monotonic()
            self.part.visible = False
            self.details.content = ''
            status = record.get('status', '已载入检测结果')
            self.status.value = status
            if rgb is not None:
                self.image.image = rgb
                self.frustum.image = rgb
            if K is not None and rgb is not None:
                K = np.asarray(K, dtype=float)
                if K.shape != (3, 3) or not np.isfinite(K).all() or K[1, 1] <= 0:
                    raise ValueError('Invalid image intrinsics')
                self.frustum.fov = 2 * np.arctan(rgb.shape[0] / (2 * K[1, 1]))
                self.frustum.aspect = rgb.shape[1] / rgb.shape[0]
            pose = record.get('object_in_camera')
            if pose is None or status.startswith(('SEARCHING', 'LOST')):
                self.details.content = '当前无有效零件位姿。'
                return
            try:
                pose = rigid(pose)
                if pose[2, 3] <= 0:
                    raise ValueError('Object origin is behind the camera')
                world = world_object(self.calibration, pose)
            except ValueError as exc:
                self.status.value = f'无效位姿：{exc}'
                return
            set_pose(self.part, world)
            self.part.visible = True
            self.details.content = (
                f"相机世界位置：{self.T_world_cam[:3, 3].round(4).tolist()} m\n\n"
                f"零件世界位置：{world[:3, 3].round(4).tolist()} m\n\n"
                f"零件世界四元数 wxyz：{Rotation.from_matrix(world[:3, :3]).as_quat()[[3, 0, 1, 2]].round(5).tolist()}\n\n"
                '齿轮模型使用检测输出的 CAD 原点；10° 对称偏航可能等价。')

    def expire(self, now=None):
        with self.lock:
            if self.live and self.last_update is not None and (now or time.monotonic()) - self.last_update > self.stale_after:
                self.part.visible = False
                self.status.value = '检测结果已过期'
                self.details.content = '等待新的检测结果。'

    def _monitor(self):
        while not self.stop_event.wait(.1):
            self.expire()

    def close(self):
        self.stop_event.set()
        self.monitor.join(timeout=1.)
        self.server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--live', action='store_true', help='Run existing Gemini/FoundationPose detector')
    mode.add_argument('--detection', type=Path, help='Existing result.json or snapshot/detection directory')
    mode.add_argument('--demo', action='store_true', help='Explicit synthetic scene without camera/GPU')
    parser.add_argument('--calibration', type=Path, help='Fixed-camera handeye_result.json or T_base_cam.npy')
    parser.add_argument('--initial-pose', type=Path, default=DEFAULT_INITIAL)
    parser.add_argument('--joints', type=Path, help='Reference joint pose JSON (radians); never commands hardware')
    parser.add_argument('--urdf', type=Path, default=DEFAULT_URDF)
    parser.add_argument('--mesh', type=Path, default=DEFAULT_MESH)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8088)
    parser.add_argument('--stale-after', type=float, default=2.)
    args, inference_args = parser.parse_known_args()
    if not args.demo and args.calibration is None:
        parser.error('--calibration is required for actual detection poses')
    if inference_args and not args.live:
        parser.error(f'Unknown options: {inference_args}')
    if not np.isfinite(args.stale_after) or args.stale_after <= 0:
        parser.error('--stale-after must be positive')
    initial = json.loads(args.initial_pose.read_text())
    if args.demo:
        C = np.eye(4)
        C[:3, 3] = [.35, -.25, .45]
        C[:3, :3] = Rotation.from_euler('xyz', [140, 0, 45], degrees=True).as_matrix()
        calibration = {'T_world_base': initial_base(initial), 'T_base_cam': C,
                       'synthetic': True, 'base_link': 'base_link'}
    else:
        calibration = load_calibration(args.calibration, initial)
    viewer = SceneViewer(calibration, initial=initial, urdf=args.urdf, mesh=args.mesh,
                         joints=args.joints, host=args.host, port=args.port,
                         live=args.live, stale_after=args.stale_after)
    try:
        if args.live:
            from live_gear36 import main as detect_live
            detect_live(['--mesh', str(args.mesh), '--no-display', *inference_args], observer=viewer)
        else:
            if args.demo:
                pose = np.eye(4)
                pose[2, 3] = .5
                viewer.update({'status': '合成演示', 'object_in_camera': pose})
            else:
                record, rgb, K = read_detection(args.detection)
                viewer.update(record, rgb=rgb, K=K)
            while not viewer.stop_event.wait(.2):
                pass
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == '__main__':
    main()
