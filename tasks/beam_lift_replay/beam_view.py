"""Beam placement reference, recorded positions and optional live fixed-camera poses."""
import json
from pathlib import Path
import socket
import time

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


def rigid(value):
    transform = np.asarray(value, dtype=float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0., 0., 0., 1.])
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(transform[:3, :3]), 1., atol=1e-4)):
        raise ValueError('Expected a finite rigid beam transform')
    return transform


def world_camera(config, handeye):
    document = json.loads(Path(handeye).read_text(encoding='utf-8'))
    if document.get('mode') != 'eye-to-hand' or document.get('synthetic'):
        raise ValueError('Live beam display requires a real fixed-camera hand-eye calibration')
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(np.asarray(config.base_wxyz)[[1, 2, 3, 0]]).as_matrix()
    transform[:3, 3] = config.base_position
    return transform @ rigid(document['T_base_cam'])


class BeamView:
    def __init__(self, server, data, trajectory, config, *, handeye=None, pose_port=5711, max_age=1.):
        self.server, self.max_age = server, max_age
        self.sock, self.camera, self.latest = None, None, None
        self.initial = np.asarray(data.get('initial_object_pose_xyz_wxyz', []), dtype=float)
        if (self.initial.shape != (7,) or not np.isfinite(self.initial).all()
                or np.linalg.norm(self.initial[3:]) < 1e-8):
            raise ValueError('Trajectory must contain initial_object_pose_xyz_wxyz')
        self.initial[3:] /= np.linalg.norm(self.initial[3:])
        mesh_path = Path(__file__).resolve().parents[2]/'asset/fabrica_beam_3x/beam2_visual.obj'
        mesh = trimesh.load(mesh_path, force='mesh')
        self.start = server.scene.add_frame('/beam/start', position=self.initial[:3], wxyz=self.initial[3:],
                                             axes_length=.08, axes_radius=.0006)
        server.scene.add_mesh_simple('/beam/start/mesh', vertices=np.asarray(mesh.vertices, dtype=np.float32),
                                      faces=np.asarray(mesh.faces, dtype=np.uint32), color=(240, 200, 40), opacity=.35)
        server.scene.add_label('/beam/start/label', '轨迹初始位姿', position=(0., 0., .06))
        self.current = server.scene.add_frame('/beam/current', visible=False, axes_length=.1, axes_radius=.0006)
        server.scene.add_mesh_simple('/beam/current/mesh', vertices=np.asarray(mesh.vertices, dtype=np.float32),
                                      faces=np.asarray(mesh.faces, dtype=np.uint32), color=(40, 160, 240), opacity=.8)
        server.scene.add_label('/beam/current/label', '实时追踪位姿', position=(0., 0., .06))
        self.recorded = server.scene.add_icosphere('/beam/recorded_position', radius=.008,
                                                   color=(200, 80, 230), visible=False)
        self.recorded_label = server.scene.add_label('/beam/recorded_label', '仿真记录位置（无朝向）', visible=False)
        self.recorded_times = self.recorded_positions = None
        recorded_path = Path(trajectory).with_name('contact_force.npz')
        if recorded_path.is_file():
            with np.load(recorded_path, allow_pickle=False) as archive:
                times = np.asarray(archive['time_s'], dtype=float)
                positions = np.asarray(archive['object_pos_m'], dtype=float)
            if (times.ndim != 1 or len(times) == 0 or positions.shape != (len(times), 3)
                    or not np.isfinite(times).all() or not np.isfinite(positions).all()
                    or np.any(np.diff(times) <= 0)):
                raise ValueError('Invalid recorded beam position timeline')
            self.recorded_times, self.recorded_positions = times, positions
        with server.gui.add_folder('Beam 位姿 / 摆放检查'):
            server.gui.add_markdown('黄色半透明：轨迹初始位姿；蓝色：相机实时追踪的当前完整位姿。'
                                     '紫色点：当前回放时间对应的仿真位置，不是真机测量，也未记录朝向。'
                                     '显示不参与开环运动控制。坐标为世界坐标，单位 m。')
            self.show_start = server.gui.add_checkbox('显示初始位姿', initial_value=True)
            self.show_recorded = server.gui.add_checkbox('显示仿真记录位置', initial_value=not bool(handeye))
            self.start_text = server.gui.add_text('初始位置', initial_value=self.position_text(self.initial[:3]), disabled=True)
            self.current_text = server.gui.add_text('当前追踪位置', initial_value='等待实时感知', disabled=True)
            self.delta_text = server.gui.add_text('相对起点偏差', initial_value='—', disabled=True)
            self.recorded_text = server.gui.add_text('仿真记录位置', initial_value='没有配套位置记录', disabled=True)
            self.source = server.gui.add_markdown('')
        if handeye is not None:
            self.camera = world_camera(config, handeye)
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                self.sock.bind(('127.0.0.1', pose_port))
                self.sock.setblocking(False)
            except BaseException:
                self.sock.close()
                self.sock = None
                raise
            self.source.content = (f'实时位姿监听 `127.0.0.1:{self.sock.getsockname()[1]}`；'
                                   '请启动 beam1 感知并将 `--target` 指向此地址。超过 1 秒未更新即隐藏。')
        else:
            self.source.content = '未提供 --handeye：仅显示初始位姿和仿真记录位置，不显示实时追踪。'

    @staticmethod
    def position_text(position):
        return f'x {position[0]:+.4f} · y {position[1]:+.4f} · z {position[2]:+.4f}'

    def update(self, elapsed, now=None):
        now = time.monotonic() if now is None else now
        self.start.visible = self.show_start.value
        if self.recorded_times is not None:
            position = np.array([np.interp(elapsed, self.recorded_times, self.recorded_positions[:, axis])
                                 for axis in range(3)])
            self.recorded.position = position
            self.recorded_label.position = position + [0., 0., .04]
            self.recorded.visible = self.recorded_label.visible = self.show_recorded.value
            text = self.position_text(position)
            if self.recorded_text.value != text:
                self.recorded_text.value = text
        if self.sock is not None:
            for _ in range(64):
                try:
                    payload, _ = self.sock.recvfrom(65536)
                except BlockingIOError:
                    break
                try:
                    record = json.loads(payload)
                    stamp = float(record['timestamp_monotonic'])
                    transform = rigid(record['object_in_camera'])
                    if not np.isfinite(stamp) or stamp > now + .1 or now - stamp > self.max_age:
                        continue
                    if self.latest is None or stamp >= self.latest[0]:
                        self.latest = (stamp, self.camera @ transform)
                except (ValueError, TypeError, KeyError):
                    continue
        if self.latest is None or now - self.latest[0] > self.max_age:
            self.current.visible = False
            self.current_text.value = '追踪已过期，等待感知更新' if self.latest is not None else '等待实时感知'
            self.delta_text.value = '—'
            return
        transform = self.latest[1]
        self.current.position = transform[:3, 3]
        self.current.wxyz = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
        self.current.visible = True
        self.current_text.value = self.position_text(transform[:3, 3])
        delta = transform[:3, 3] - self.initial[:3]
        initial_rotation = Rotation.from_quat(self.initial[3:][[1, 2, 3, 0]])
        angle = (initial_rotation.inv() * Rotation.from_matrix(transform[:3, :3])).magnitude()
        self.delta_text.value = (f'Δxyz [{delta[0]*1000:+.1f}, {delta[1]*1000:+.1f}, {delta[2]*1000:+.1f}] mm · '
                                f'朝向差 {np.degrees(angle):.1f}°')

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None
