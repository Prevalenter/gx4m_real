"""View calibrated GX7, camera and ChArUco board poses in Viser."""
import argparse
import dataclasses
from datetime import datetime
import json
from multiprocessing.connection import Client
import os
from pathlib import Path
import socket
import subprocess
import threading
import time

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import trimesh
import viser
from viser.extras import ViserUrdf
import yourdfpy

from board import make_board
from gx4m_real.camera.cloud import unproject
from gx4m_real.camera.frame_ipc import receive, send
from gx4m_real.robot.compensation import JointCompensation
from gx4m_real.robot.hand import HAND_JOINTS, HandFeedbackReader
from handeye_session import transform
from solve_handeye import project

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GEAR_WORKER = ROOT / 'tasks/dexlong_gear3/perception/run_gear_worker.sh'
# Mesh origin and axes are the simulated gear's (tasks/dexlong_gear3/perception/gears.py).
GEAR24_MESH = ROOT / 'asset/mjfit_gears3_split/assembly/gear24_visual.obj'
JOINT_PERIOD_SEC = .1
HAND_PERIOD_SEC = .05  # A 16-motor sync read takes about 12 ms.
TACTILE_PERIOD_SEC = .1  # A 135-point frame plus the total force takes about 7 ms per sensor.
RETRY_SEC = 2.


def poses_for_sample(result, sample):
    W = transform(result['T_world_base'])
    E = transform(sample['T_base_flange'])
    if result['mode'] == 'eye-to-hand':
        C = transform(result['T_base_cam'])
        B = E @ transform(result['T_flange_board'])
    elif result['mode'] == 'eye-in-hand':
        C = E @ transform(result['T_flange_cam'])
        B = transform(result['T_base_board'])
    else:
        raise ValueError('Unknown calibration mode')
    return {'camera': W @ C, 'board': W @ B, 'flange': W @ E,
            'observed_board': W @ C @ transform(sample['T_cam_board'])}


def set_pose(handle, T):
    handle.position = T[:3, 3]
    handle.wxyz = Rotation.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]]


def cloud_from_rgbd(rgb, depth, K, stride=2):
    """Every stride-th pixel with depth 0.1–2 m, as camera-frame points and their colours."""
    if stride < 1:
        raise ValueError('Point cloud stride must be positive')
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    v, u = v.ravel(), u.ravel()
    valid = np.isfinite(depth[v, u]) & (depth[v, u] >= .1) & (depth[v, u] <= 2.)
    v, u = v[valid], u[valid]
    return unproject(depth, u, v, K).astype(np.float32), rgb[v, u]


def load_pointcloud_capture(path, *, stride=2):
    """Read an aligned, undistorted RGB-D capture in the camera optical frame."""
    path = Path(path)
    metadata = json.loads((path / 'capture.json').read_text())
    if metadata.get('undistorted') is not True or metadata.get('depth_unit') != 'metre':
        raise ValueError('Point cloud capture must contain undistorted RGB-D with depth in metres')
    if stride < 1:
        raise ValueError('Point cloud stride must be positive')
    with np.load(path / 'rgbd.npz') as data:
        rgb, depth, K = data['rgb'], data['depth'], data['K']
    if (depth.ndim != 2 or rgb.shape != (*depth.shape, 3) or K.shape != (3, 3)
            or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0):
        raise ValueError('Point cloud capture has inconsistent RGB, depth or intrinsics')
    points, colors = cloud_from_rgbd(rgb, depth, K, stride)
    if not len(points):
        raise ValueError('Point cloud capture has no valid depth between 0.1 and 2 metres')
    return points, colors, metadata, rgb


def without_links_below(model, link, mesh_dir):
    """URDF copy without everything mounted below `link`, e.g. the GX4M hand below Link7."""
    drop, stack = set(), [link]
    while stack:
        parent = stack.pop()
        children = [j.child for j in model.robot.joints if j.parent == parent]
        drop.update(children)
        stack += children
    robot = dataclasses.replace(model.robot, links=[l for l in model.robot.links if l.name not in drop],
                                joints=[j for j in model.robot.joints if j.child not in drop])
    return yourdfpy.URDF(robot=robot, mesh_dir=str(mesh_dir))


def orbbec_stream(device_index):
    from gx4m_real.camera.orbbec import OrbbecRGBDStream
    return OrbbecRGBDStream(device_index)


def paxini_labels(serial_numbers=None):
    """{label: port} of the given USB serial numbers, or of every PaXini port found; finger order, thumb first."""
    from gx4m_real.robot.tactile import FINGERS, find_ports
    order = list(FINGERS)
    ports = sorted(find_ports(serial_numbers).items(),
                   key=lambda item: (order.index(item[0]) if item[0] in order else len(order), item[0]))
    return {f"**{FINGERS.get(serial_number, '未知手指')}**（{serial_number}，{port}）": port
            for serial_number, port in ports}


def paxini_sensors(serial_numbers=None):
    """{label: open PaXiniL3530} of the given USB serial numbers, or of every PaXini port found."""
    from gx4m_real.robot.tactile import PaXiniL3530
    sensors = {}
    try:
        for label, port in paxini_labels(serial_numbers).items():
            sensors[label] = PaXiniL3530(port)
    except BaseException:
        for sensor in sensors.values():
            sensor.close()
        raise
    if not sensors:
        raise RuntimeError('没有找到 PaXini 触觉传感器')
    return sensors


def joint_reader(robot_config):
    from gx4m_real.robot.config import load_config
    from gx4m_real.robot.feedback import JointFeedbackReader
    return JointFeedbackReader(load_config(robot_config))


class GearWorker:
    """FoundationPose gear worker process (see tasks/dexlong_gear3/perception/gear_worker.py), a context manager.

    It runs in its own conda environment, so frames go out and poses come back over a local connection. Startup
    waits while the worker loads its models, until cancel is set or startup_timeout passes.
    """

    def __init__(self, command, *, cancel=None, startup_timeout=300., reply_timeout=20.):
        self.command, self.cancel = command, cancel or threading.Event()
        self.startup_timeout, self.reply_timeout = startup_timeout, reply_timeout
        self.process = self.connection = None

    def __enter__(self):
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            port = s.getsockname()[1]
        key = os.urandom(16)
        self.process = subprocess.Popen(self.command, env={**os.environ, 'GEAR_WORKER_PORT': str(port),
                                                           'GEAR_WORKER_AUTHKEY': key.hex()})
        try:
            deadline = time.monotonic() + self.startup_timeout
            while self.connection is None:
                if self.process.poll() is not None:
                    raise RuntimeError(f'FoundationPose 进程已退出（代码 {self.process.returncode}），见终端输出')
                if self.cancel.is_set() or time.monotonic() > deadline:
                    raise TimeoutError('FoundationPose 进程未就绪')
                try:
                    self.connection = Client(('127.0.0.1', port), authkey=key)
                except ConnectionRefusedError:
                    self.cancel.wait(.5)
            if self._receive()['type'] != 'ready':
                raise RuntimeError('FoundationPose 进程没有报告就绪')
        except BaseException:
            self.__exit__()
            raise
        return self

    def _receive(self):
        if not self.connection.poll(self.reply_timeout):
            raise TimeoutError(f'FoundationPose {self.reply_timeout:g} 秒内没有回应')
        return receive(self.connection)[0]

    def estimate(self, frame):
        """The worker's pose message for one RGB-D frame (object_in_camera in the camera optical frame)."""
        send(self.connection, {'type': 'frame', 'sequence': int(frame.sequence)}, frame.rgb,
             np.asarray(frame.depth, np.float32), np.asarray(frame.K, float))
        return self._receive()

    def reset(self):
        send(self.connection, {'type': 'reset'})

    def __exit__(self, *_):
        connection, self.connection = self.connection, None
        if connection is not None:
            connection.close()  # The worker exits on end of input.
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()


class ResultViewer:
    """Calibration samples, plus either one RGB-D capture or (live=True) the streaming cloud and
    read-only joint feedback. live needs reader_factory() and optionally camera_factory() (no camera: no
    cloud), returning unopened context managers: a joint reader with read() and an RGB-D stream with
    read(after, timeout). In live mode gear_factory(stop) adds the gear24 pose (an unopened GearWorker-like
    context manager, needs the camera), hand_factory() the dexterous hand with its read-only angles
    (a reader with read() -> 16 URDF radians; without it the hand is not drawn) and tactile_factory()
    the fingertip values ({label: open sensor with read_frame(), read_total() and close()}); every 2 s
    tactile_labels(), when given, lists the labels now available, and a change reopens the sensors."""

    def __init__(self, result_path, *, host='127.0.0.1', port=8087, urdf=None,
                 pointcloud=None, cloud_stride=2, live=False, camera_factory=None, reader_factory=None,
                 cloud_hz=5., gear_factory=None, hand_factory=None, tactile_factory=None, tactile_labels=None):
        self.result = json.loads(Path(result_path).read_text())
        self.samples = self.result['samples']
        if not self.samples:
            raise ValueError('Result contains no calibration samples')
        capture = None
        expected = self.result.get('camera_metadata', {}).get('serial')
        if (pointcloud is not None or live) and self.result['mode'] != 'eye-to-hand':
            raise ValueError('A current point cloud requires fixed-camera (eye-to-hand) calibration')
        if pointcloud is not None and live:
            raise ValueError('Choose either a captured point cloud or live mode')
        if live and (reader_factory is None or cloud_hz <= 0):
            raise ValueError('Live mode needs a joint reader and a positive cloud rate')
        if gear_factory is not None and (not live or camera_factory is None):
            raise ValueError('Gear detection needs live mode with the camera')
        if (hand_factory is not None or tactile_factory is not None) and not live:
            raise ValueError('The live hand and tactile values need live mode')
        if pointcloud is not None:
            capture = load_pointcloud_capture(pointcloud, stride=cloud_stride)
            if expected and capture[2].get('serial') != expected:
                raise ValueError('Point cloud camera serial differs from the calibration camera')
        if cloud_stride < 1:
            raise ValueError('Point cloud stride must be positive')
        self.live, self.cloud_stride, self.cloud_hz = live, cloud_stride, cloud_hz
        self.camera_factory, self.reader_factory, self.expected_serial = camera_factory, reader_factory, expected
        self.compensation = None
        if live:
            c = self.result['joint_compensation']
            # The extrinsic was solved with this mapping, so live motor angles must use it too.
            self.compensation = JointCompensation(tuple(c['directions']), tuple(c['offsets_deg']))
        self.live_q = None
        self.arm_shown = None
        self.hand_q = {}
        self.cloud = None
        self.lock = threading.RLock()
        self.gear_factory, self.hand_factory, self.tactile_factory = gear_factory, hand_factory, tactile_factory
        self.tactile_labels = tactile_labels
        self.latest_frame = None
        self.frame_ready = threading.Condition()
        self.gear_reset = threading.Event()
        self.server = viser.ViserServer(host=host, port=port, label='GX7 · ChArUco 标定结果')
        self.server.scene.set_up_direction('+z')
        self.server.scene.add_grid('/ground', width=2., height=2., plane='xy')
        self.server.scene.add_frame('/world', axes_length=.22, axes_radius=.0006, origin_radius=.0012)
        base = self.server.scene.add_frame('/robot', axes_length=.18, axes_radius=.0006, origin_radius=.0012)
        set_pose(base, transform(self.result['T_world_base']))
        urdf = Path(urdf or self.result['urdf'])
        if hand_factory is not None:
            self.model = yourdfpy.URDF.load(str(urdf), load_meshes=True)
        else:
            # The dexterous hand is not part of the calibrated chain and hides the board and the cloud.
            self.model = without_links_below(yourdfpy.URDF.load(str(urdf), load_meshes=False),
                                             self.result['tip_link'], urdf.parent)
        if self.model.base_link != self.result['base_link']:
            # Robot mesh root relative to the calibrated base frame.
            root_in_base = self.model.get_transform(self.model.base_link, self.result['base_link'])
            set_pose(base, transform(self.result['T_world_base']) @ root_in_base)
        self.visual = ViserUrdf(self.server, self.model, root_node_name='/robot')
        self.camera = self.server.scene.add_frame('/calibrated_camera', axes_length=.16,
                                                  axes_radius=.0006, origin_radius=.0012)
        if capture is not None:
            self.cloud = self.server.scene.add_point_cloud('/calibrated_camera/point_cloud',
                points=capture[0], colors=capture[1], point_size=.002, point_shape='circle', precision='float32')
        w, h = self.result['image_size']
        K = np.asarray(self.result['K'])
        self.frustum = self.server.scene.add_camera_frustum('/calibrated_camera/frustum',
            fov=2 * np.arctan(h / (2 * K[1, 1])), aspect=w / h, scale=.05, thickness=2,
            thickness_units='screen', color=(70, 170, 255))
        self.flange = self.server.scene.add_frame('/flange', axes_length=.1, axes_radius=.0006, origin_radius=.0012)
        self.board = self.server.scene.add_frame('/predicted_board', axes_length=.12, axes_radius=.0006, origin_radius=.0012)
        self.observed = self.server.scene.add_frame('/observed_board', axes_length=.1, axes_radius=.0006, origin_radius=.0012)
        spec = self.result['board']
        if (spec['squares_x'], spec['squares_y'], spec['dictionary']) != (4, 5, 'DICT_4X4_50'):
            raise ValueError('Viewer board definition differs from calibration')
        bw, bh = spec['squares_x'] * spec['square_length_m'], spec['squares_y'] * spec['square_length_m']
        texture = make_board(spec['square_length_m']).generateImage((720, 900))
        mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0]],
                               faces=[[0, 1, 2], [0, 2, 3]], process=False)
        mesh.visual = trimesh.visual.TextureVisuals(uv=[[0, 1], [1, 1], [1, 0], [0, 0]],
                                                   image=Image.fromarray(texture).convert('RGB'))
        self.server.scene.add_mesh_trimesh('/predicted_board/texture', mesh)
        outline = np.array([[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0], [0, 0, 0]])
        self.server.scene.add_line_segments('/observed_board/outline',
            points=np.stack([outline[:-1], outline[1:]], axis=1), colors=(255, 90, 190),
            thickness=3, thickness_units='screen')
        gui = self.server.gui
        gui.add_markdown('## 标定结果\n' + self.result['mode'] +
                         ('\n\n**合成数据测试结果，不是真机标定。**' if self.result['synthetic'] else '') +
                         '\n\n实体板：关节与标定结果预测；粉色轮廓：图像检测。蓝色相机沿 +Z 看向前方。')
        if capture is not None:
            with gui.add_folder('当前场景点云'):
                gui.add_markdown(f"采集时间：{capture[2]['captured_at']}\n\n"
                                 f"显示 {len(capture[0]):,} 个点，深度范围 0.1–2 米。\n\n"
                                 '点云为此次拍摄的静态场景；机器人姿态来自下方选中的标定样本。')
                show_cloud = gui.add_checkbox('显示点云', initial_value=True)
                show_cloud.on_update(lambda e: setattr(self.cloud, 'visible', e.target.value))
                show_robot = gui.add_checkbox('显示机器人模型', initial_value=True)
                show_robot.on_update(lambda e: setattr(base, 'visible', e.target.value))
                gui.add_image(capture[3], label='当前场景 RGB')
        if live:
            offsets = ', '.join(f'J{i} {v:+g}°' for i, v in enumerate(self.compensation.offsets_deg, 1) if v)
            with gui.add_folder('实时场景'):
                gui.add_markdown('关节为只读查询，不使能电机；控制或采集程序占用 CAN 时读不到，会每 2 秒重试。\n\n'
                                 f'电机角按标定结果的补偿换算为模型角（{offsets or "零位偏置为 0"}）。')
                self.joint_status = gui.add_markdown('关节：连接中…')
                if hand_factory is not None:
                    gui.add_markdown('灵巧手只读同步读取位置，不写任何寄存器（扭矩保持原状）；'
                                     '不要和驱动灵巧手的程序同时运行。')
                    self.hand_status = gui.add_markdown('灵巧手：连接中…')
                if tactile_factory is not None:
                    gui.add_markdown('触觉（PaXini L3530，只读）：135 个测点的法向原始值（0–255，未标定成牛顿）'
                                     '求和、最大值和大于 5 的点数；总力寄存器为原始值，部署代码按 0.1 N 换算。')
                    self.tactile_status = gui.add_markdown('触觉：连接中…')
                show_robot = gui.add_checkbox('显示机器人模型', initial_value=True)
                show_robot.on_update(lambda e: setattr(base, 'visible', e.target.value))
                if camera_factory is not None:
                    self.cloud_status = gui.add_markdown('点云：相机打开中…')
                    self.show_cloud = gui.add_checkbox('显示点云', initial_value=True)
                    @self.show_cloud.on_update
                    def _(event):
                        with self.lock:
                            if self.cloud is not None:
                                self.cloud.visible = event.target.value
                    w, h = self.result['image_size']
                    self.live_image = gui.add_image(np.zeros((h, w, 3), dtype=np.uint8), label='实时 RGB')
        if gear_factory is not None:
            self.T_world_cam = transform(self.result['T_world_base']) @ transform(self.result['T_base_cam'])
            gear = trimesh.load(GEAR24_MESH, force='mesh')
            gear.visual.vertex_colors = np.tile([220, 50, 40, 255], (len(gear.vertices), 1))
            # A child of the camera, so it lands where the cloud is.
            self.gear = self.server.scene.add_frame('/calibrated_camera/gear24', axes_length=.06,
                axes_radius=.0006, origin_radius=.0012, visible=False)
            self.server.scene.add_mesh_trimesh('/calibrated_camera/gear24/mesh', gear)
            with gui.add_folder('24 齿齿轮（FoundationPose）'):
                gui.add_markdown('红色齿轮：按红色区域配准后逐帧跟踪；位姿通过重叠率和深度残差检查才显示。'
                                 '原点和坐标轴与仿真齿轮相同，偏航每 15° 等价。')
                self.gear_status = gui.add_markdown('齿轮：FoundationPose 启动中…')
                gui.add_button('重新检测齿轮').on_click(lambda _: self.gear_reset.set())
        self.index = gui.add_slider('样本', min=0, max=max(1, len(self.samples) - 1), step=1,
                                   initial_value=0 if capture is None and not live else len(self.samples) - 1)
        self.play = gui.add_checkbox('自动切换样本', initial_value=False)
        self.show_boards = gui.add_checkbox('显示标定样本的板与末端', initial_value=capture is None and not live)
        self.show_observed = gui.add_checkbox('显示图像检测板位姿', initial_value=True)
        self.details = gui.add_markdown('')
        self.image = gui.add_image(np.zeros((h, w, 3), dtype=np.uint8), label='绿点：检测；蓝圈：预测')
        gui.add_markdown('视锥按内参近似显示，未表达畸变；下方重投影使用完整内参与畸变。')
        self.index.on_update(lambda _: self.refresh())
        self.show_boards.on_update(lambda _: self.refresh())
        self.show_observed.on_update(lambda _: self.refresh())
        @self.server.on_client_connect
        def on_connect(client):
            if self.cloud is not None:
                center = np.median(self.cloud.points, axis=0)
                C = transform(self.result['T_world_base']) @ transform(self.result['T_base_cam'])
                center = C[:3, :3] @ center + C[:3, 3]
                client.camera.position = center + [.75, -.95, .65]
                client.camera.look_at = center
                return
            p = np.asarray(self.result['T_world_base'])[:3, 3]
            client.camera.position = p + [1., -1.3, .9]
            client.camera.look_at = p + [0, 0, .25]
        self.refresh()

    def refresh(self):
        with self.lock:
            sample = self.samples[min(int(self.index.value), len(self.samples) - 1)]
            poses = poses_for_sample(self.result, sample)
            for name in ('camera', 'board', 'flange'):
                set_pose(getattr(self, name), poses[name])
            set_pose(self.observed, poses['observed_board'])
            self.board.visible = self.flange.visible = self.show_boards.value
            self.observed.visible = self.show_boards.value and self.show_observed.value
            self.set_robot(sample['positions_rad'] if self.live_q is None else self.live_q)
            bgr = cv2.imread(sample['image_path'])
            if bgr is None:
                w, h = self.result['image_size']
                bgr = np.zeros((h, w, 3), dtype=np.uint8)
            C_B = np.linalg.inv(poses['camera']) @ poses['board']
            pixels = project(C_B, np.asarray(sample['object_points']), np.asarray(self.result['K']),
                             np.asarray(self.result['dist']))
            for actual, predicted in zip(sample['corners_px'], pixels):
                if np.isfinite(predicted).all():
                    cv2.circle(bgr, tuple(np.rint(predicted).astype(int)), 6, (255, 120, 0), 1)
                cv2.circle(bgr, tuple(np.rint(actual).astype(int)), 2, (0, 255, 0), -1)
            self.image.image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.frustum.image = self.image.image
            r = sample['residual']
            self.details.content = (f"样本 {sample['index']} · `{Path(sample['image_path']).name}`\n\n"
                f"重投影 {r['reproj_px']:.3f} px · 位移残差 {r['trans_mm']:.2f} mm · 旋转残差 {r['rot_deg']:.3f}°\n\n"
                f"相机世界位置：{poses['camera'][:3, 3].round(4).tolist()} m\n\n"
                f"标定板世界位置：{poses['board'][:3, 3].round(4).tolist()} m")

    def set_robot(self, positions):
        """Arm model angles (7,), plus the live hand angles when there are any."""
        q = dict(self.result.get('fixed_model_joints', {}))
        q.update(zip(self.result['joint_names'], positions))
        with self.lock:
            q.update(self.hand_q)
            self.arm_shown = positions
            self.visual.update_cfg(np.array([q.get(name, 0.) for name in self.model.actuated_joint_names]))

    def show_hand(self, q):
        """Show live hand angles (16,) in URDF radians, gx4m_joint1..16."""
        q = np.asarray(q, float)
        with self.lock:
            self.hand_q = dict(zip(HAND_JOINTS, q))
            self.set_robot(self.arm_shown)
            self.hand_status.content = (f"灵巧手（URDF 角，{datetime.now():%H:%M:%S}）：" +
                                        ' · '.join(f'{i} {v:+.1f}°' for i, v in enumerate(np.degrees(q), 1)))

    def show_tactile(self, readings):
        """readings: {label: (frame (135, 3) or None, total (3,) or None, error or None)}."""
        rows = ['| 手指（序列号，端口） | 法向和 | 最大点 | >5 点数 | 总力 x / y / z（≈0.1 N） |', '|---|---|---|---|---|']
        for label, (frame, total, error) in readings.items():
            if error is not None:
                rows.append(f'| {label} | 读取失败：{error} | | | |')
                continue
            normal = frame[:, 2]
            rows.append(f'| {label} | {normal.sum():.0f} | {normal.max():.0f} | {int((normal > 5).sum())} | '
                        f'{total[0]:+.0f} / {total[1]:+.0f} / {total[2]:.0f} |')
        self.tactile_status.content = f'触觉（{datetime.now():%H:%M:%S}）\n\n' + '\n'.join(rows)

    def _tactile_loop(self, stop):
        while not stop.is_set():
            sensors = {}
            try:
                sensors = self.tactile_factory()
                rescan = time.monotonic() + RETRY_SEC
                while not stop.wait(TACTILE_PERIOD_SEC):
                    if self.tactile_labels is not None and time.monotonic() > rescan:
                        rescan = time.monotonic() + RETRY_SEC
                        try:
                            if set(self.tactile_labels()) != set(sensors):
                                break  # A sensor was plugged in or out: reopen them all.
                        except Exception:
                            pass
                    readings = {}
                    for label, sensor in sensors.items():
                        try:
                            readings[label] = (sensor.read_frame(), sensor.read_total(), None)
                        except (RuntimeError, TimeoutError) as exc:  # This sensor only; the port is fine.
                            readings[label] = (None, None, str(exc))
                    self.show_tactile(readings)
            except Exception as exc:
                self.tactile_status.content = f'触觉读取失败：{exc}（{RETRY_SEC:g} 秒后重试）'
                stop.wait(RETRY_SEC)
            finally:
                for sensor in sensors.values():
                    try:
                        sensor.close()
                    except Exception:
                        pass

    def _hand_loop(self, stop):
        while not stop.is_set():
            try:
                with self.hand_factory() as reader:
                    while not stop.wait(HAND_PERIOD_SEC):
                        self.show_hand(reader.read())
            except Exception as exc:
                self.hand_status.content = f'灵巧手读取失败：{exc}（{RETRY_SEC:g} 秒后重试；模型保持上次姿态）'
                stop.wait(RETRY_SEC)

    def show_joints(self, motor_positions):
        """Show live raw motor angles, mapped with the calibration's compensation."""
        q = np.asarray(self.compensation.to_model(motor_positions), float)
        with self.lock:
            self.live_q = q
            self.set_robot(q)
            self.joint_status.content = (f"关节（模型角，{datetime.now():%H:%M:%S}）：" +
                                         ' · '.join(f'J{i} {v:+.1f}°' for i, v in enumerate(np.degrees(q), 1)))

    def show_frame(self, frame):
        points, colors = cloud_from_rgbd(frame.rgb, frame.depth, frame.K, self.cloud_stride)
        with self.lock:
            if self.cloud is None:
                self.cloud = self.server.scene.add_point_cloud('/calibrated_camera/point_cloud',
                    points=points, colors=colors, point_size=.002, point_shape='circle', precision='float32',
                    visible=self.show_cloud.value)
            else:
                self.cloud.points, self.cloud.colors = points, colors
            self.live_image.image = frame.rgb
            self.cloud_status.content = f'点云：{len(points):,} 个点 · 第 {frame.sequence} 帧 · {datetime.now():%H:%M:%S}'

    def show_gear(self, result):
        """Show a GearWorker pose message; the gear hides when the pose is missing or rejected."""
        pose = result.get('object_in_camera')
        timing = f"{result['processing_ms']:.0f} ms · {datetime.now():%H:%M:%S}"
        with self.lock:
            if pose is None:
                self.gear.visible = False
                self.gear_status.content = f"齿轮：{result['status']} · {timing}"
                return
            pose = transform(pose)
            set_pose(self.gear, pose)
            self.gear.visible = True
            world = self.T_world_cam @ pose
            tilt = np.degrees(np.arccos(np.clip(world[2, 2], -1., 1.)))
            self.gear_status.content = (
                f"齿轮：{result['status']} · 重叠率 {result['mask_iou']:.2f} · "
                f"深度残差 {result['median_depth_error_m'] * 1000:.1f} mm · {timing}\n\n"
                f"世界坐标 XYZ：{(world[:3, 3] * 1000).round(1).tolist()} mm · 轴线相对世界 +z 倾斜 {tilt:.1f}°\n\n"
                f"相机坐标 XYZ：{(pose[:3, 3] * 1000).round(1).tolist()} mm")

    def _gear_loop(self, stop):
        while not stop.is_set():
            try:
                self.gear_status.content = '齿轮：FoundationPose 加载中（约半分钟）…'
                with self.gear_factory(stop) as worker:
                    self.gear_status.content = '齿轮：等待画面…'
                    last = None
                    while not stop.is_set():
                        if self.gear_reset.is_set():
                            self.gear_reset.clear()
                            worker.reset()
                        with self.frame_ready:
                            self.frame_ready.wait_for(lambda: stop.is_set() or self.latest_frame is not last, 1.)
                            frame = self.latest_frame
                        if frame is not None and frame is not last:
                            last = frame
                            self.show_gear(worker.estimate(frame))
            except Exception as exc:
                with self.lock:
                    self.gear.visible = False
                    self.gear_status.content = f'齿轮检测失败：{exc}（{RETRY_SEC:g} 秒后重启）'
                stop.wait(RETRY_SEC)

    def _joint_loop(self, stop):
        while not stop.is_set():
            try:
                with self.reader_factory() as reader:
                    while not stop.wait(JOINT_PERIOD_SEC):
                        self.show_joints(reader.read())
            except Exception as exc:
                self.joint_status.content = f'关节读取失败：{exc}（{RETRY_SEC:g} 秒后重试；模型保持上次姿态）'
                stop.wait(RETRY_SEC)

    def _cloud_loop(self, stop):
        while not stop.is_set():
            try:
                with self.camera_factory() as camera:
                    serial = getattr(camera, 'metadata', {}).get('serial')
                    if self.expected_serial and serial != self.expected_serial:
                        raise ValueError(f'相机序列号 {serial} 与标定相机 {self.expected_serial} 不同')
                    sequence, shown = -1, 0.
                    while not stop.is_set():
                        frame = camera.read(after=sequence, timeout=5)
                        sequence = frame.sequence
                        # Every frame goes to the gear tracker; the cloud display is throttled.
                        with self.frame_ready:
                            self.latest_frame = frame
                            self.frame_ready.notify_all()
                        if self.gear_factory is None:
                            stop.wait(1 / self.cloud_hz)
                        elif time.monotonic() - shown < 1 / self.cloud_hz:
                            continue
                        shown = time.monotonic()
                        self.show_frame(frame)
            except Exception as exc:
                self.cloud_status.content = f'点云失败：{exc}（{RETRY_SEC:g} 秒后重试）'
                stop.wait(RETRY_SEC)

    def run(self):
        stop = threading.Event()
        loops = (self._joint_loop,) if self.live else ()
        loops += (self._cloud_loop,) if self.live and self.camera_factory is not None else ()
        loops += (self._gear_loop,) if self.gear_factory is not None else ()
        loops += (self._hand_loop,) if self.hand_factory is not None else ()
        loops += (self._tactile_loop,) if self.tactile_factory is not None else ()
        threads = [threading.Thread(target=loop, args=(stop,), daemon=True) for loop in loops]
        for thread in threads:
            thread.start()
        try:
            while not stop.wait(.5):
                if self.play.value:
                    self.index.value = (int(self.index.value) + 1) % len(self.samples)
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            for thread in threads:
                thread.join(timeout=7)
            self.server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8087)
    parser.add_argument('--urdf', type=Path, help='Override moved URDF path')
    scene = parser.add_mutually_exclusive_group()
    scene.add_argument('--pointcloud', type=Path, help='Current capture_rgbd.py output directory (fixed camera only)')
    scene.add_argument('--live', action='store_true',
                       help='Stream the Orbbec cloud and read-only joint feedback (fixed camera only)')
    parser.add_argument('--cloud-stride', type=int, default=2, help='Show every Nth depth pixel per dimension')
    parser.add_argument('--cloud-hz', type=float, default=5., help='Live point cloud update rate')
    parser.add_argument('--camera-device-index', type=int, default=0)
    parser.add_argument('--robot-config', type=Path, default=HERE.parent / 'control' / 'config.yaml')
    parser.add_argument('--gear24', action='store_true',
                        help='With --live: track the red gear24 with FoundationPose (foundationpose env, one GPU)')
    parser.add_argument('--gear-gpu', default='0', help='Physical GPU index for FoundationPose')
    parser.add_argument('--no-cloud', action='store_true', help='With --live: do not open the camera')
    parser.add_argument('--hand-port', help='With --live: draw the hand and read its angles here, e.g. /dev/ttyUSB0')
    parser.add_argument('--tactile', nargs='*', metavar='SERIAL',
                        help='With --live: show PaXini values of these USB serial numbers (none given: all found)')
    args = parser.parse_args()
    if (args.gear24 or args.no_cloud or args.hand_port or args.tactile is not None) and not args.live:
        parser.error('--gear24, --no-cloud, --hand-port and --tactile need --live')
    if args.gear24 and args.no_cloud:
        parser.error('--gear24 needs the camera')
    command = ['bash', str(GEAR_WORKER), '--gear', 'gear24', '--gpu', args.gear_gpu]
    ResultViewer(args.result, host=args.host, port=args.port, urdf=args.urdf,
                 pointcloud=args.pointcloud, cloud_stride=args.cloud_stride, live=args.live,
                 camera_factory=None if args.no_cloud else lambda: orbbec_stream(args.camera_device_index),
                 reader_factory=lambda: joint_reader(args.robot_config), cloud_hz=args.cloud_hz,
                 gear_factory=(lambda stop: GearWorker(command, cancel=stop)) if args.gear24 else None,
                 hand_factory=(lambda: HandFeedbackReader(args.hand_port)) if args.hand_port else None,
                 tactile_factory=None if args.tactile is None else lambda: paxini_sensors(args.tactile or None),
                 tactile_labels=lambda: paxini_labels(args.tactile or None)).run()


if __name__ == '__main__':
    main()
