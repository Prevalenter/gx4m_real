"""Viser collector for ChArUco hand-eye calibration: static poses with averaged joints and one image each.

Workflow per pose: 失能拖动 → 保持当前位置（电机出力）→ 松手 → 采集. A sample averages the joint
feedback over the hold window, rejects it if any joint moved, and saves one camera image from the same
window. Defaults to the real arm and camera; nothing is enabled until "连接真机（使能）" is clicked.
--simulate uses the simulated arm and image (no board, so samples are rejected) to try the interface.
Do not run apps/control or the FK collector at the same time: they open the same CAN adapter and camera.
"""
import argparse
from collections import deque
import dataclasses
from datetime import datetime
from pathlib import Path
import signal
import sys
import threading
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import viser
from viser.extras import ViserUrdf
import yourdfpy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from board import CharucoDetector, MARKER_TO_SQUARE, SQUARE_LENGTH_M, SQUARES_X, SQUARES_Y
from handeye_session import JOINTS, MODES, SessionWriter, camera_parameters, file_sha256, new_session_path
from robot import UrdfChainFK
from solve_session import solve_views, view_residuals
from gx4m_real.camera.process import ProcessColorCamera
from gx4m_real.camera.simulated import SimulatedRGBCamera
from gx4m_real.robot.arm import ArmDriver
from gx4m_real.robot.config import ARM_JOINTS, load_config
from gx4m_real.robot.kinematics import quat_matrix

MODE_LABELS = {'disconnected': '未连接', 'hold': '保持位置', 'manual': '关节控制', 'fault': '故障',
               'recording': '无力录制中', 'released': '已失能（需手动支撑）', 'playback': '位置回放中'}
MODE_NAMES = {'eye-in-hand': '腕部相机（相机在末端，板固定）', 'eye-to-hand': '固定相机（板在末端）'}


class FrameBuffer:
    """Reads the camera on its own thread and keeps the last few seconds of frames."""

    def __init__(self, camera_factory, keep=120):
        self.camera_factory = camera_factory
        self.frames = deque(maxlen=keep)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.metadata, self.error = None, None
        self.thread = threading.Thread(target=self._run, name='handeye-camera', daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        camera = None
        try:
            camera = self.camera_factory()
            camera.open()
            with self.lock:
                self.metadata = dict(camera.metadata)
            while not self.stop_event.is_set():
                frame = camera.read(timeout=1.)
                with self.lock:
                    self.frames.append(frame)
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
        finally:
            if camera is not None:
                try:
                    camera.close()
                except Exception:
                    pass

    def latest(self):
        with self.lock:
            return self.frames[-1] if self.frames else None

    def between(self, start, end):
        with self.lock:
            return [f for f in self.frames if start <= f.received_at <= end]

    def snapshot(self):
        with self.lock:
            return {'metadata': self.metadata, 'error': self.error, 'frames': len(self.frames)}

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=5.)


def corner_motion(a, b):
    """Largest pixel displacement of ChArUco corners seen in both detections."""
    if not (a.ok and b.ok):
        return None
    ids_a, ids_b = a.charuco_ids.ravel().tolist(), b.charuco_ids.ravel().tolist()
    common = sorted(set(ids_a) & set(ids_b))
    if not common:
        return None
    pa = a.charuco_corners.reshape(-1, 2)[[ids_a.index(i) for i in common]]
    pb = b.charuco_corners.reshape(-1, 2)[[ids_b.index(i) for i in common]]
    return float(np.linalg.norm(pa - pb, axis=1).max())


class CollectApp:
    def __init__(self, config, *, real=True, driver=None, camera_factory=None, mode='eye-in-hand',
                 sample_sec=1., max_spread_deg=.05, max_pixel_motion=.5, square_length=SQUARE_LENGTH_M,
                 camera_device_index=0, session_root=HERE / 'runtime' / 'sessions'):
        if mode not in MODES:
            raise ValueError('Choose eye-in-hand or eye-to-hand')
        self.config = config
        self.driver = driver if driver is not None else ArmDriver(config, mock=not real)
        if camera_factory is None:
            camera_factory = (SimulatedRGBCamera if self.driver.mock
                              else lambda: ProcessColorCamera(camera_device_index))
        self.camera = FrameBuffer(camera_factory)
        self.fk = UrdfChainFK(str(config.urdf), base='base_link', tip='Link7')
        self.base = np.eye(4)
        self.base[:3, :3], self.base[:3, 3] = quat_matrix(config.base_wxyz), config.base_position
        self.sample_sec, self.max_spread = sample_sec, np.deg2rad(max_spread_deg)
        self.max_pixel_motion, self.square_length = max_pixel_motion, square_length
        self.session_root = Path(session_root)
        self.stop = threading.Event()
        self.ui_lock = threading.RLock()
        self.closed = False
        self.session = self.pending = self.solution = None
        self.detector = self.K = self.dist = None
        self.solution_note = '至少 6 个训练样本后自动求解。'
        self.message = '尚未采集。'
        self.last_preview = 0.

        model = yourdfpy.URDF.load(str(config.urdf), build_scene_graph=True, load_meshes=True)
        self.model_names = tuple(model.actuated_joint_names)
        self.model_positions = np.array([config.initial_joints.get(n, 0.) for n in self.model_names])
        self.arm_indices = [self.model_names.index(n) for n in ARM_JOINTS]
        self.server = viser.ViserServer(host=config.host, port=config.port, label='GX7 · 手眼标定采集')
        self.server.scene.set_up_direction('+z')
        self.server.scene.add_grid('/ground', width=1.5, height=1.5, plane='xy', cell_size=.05)
        self.server.scene.add_frame('/world', axes_length=.15, axes_radius=.003)
        self.robot_frame = self.server.scene.add_frame('/robot', position=config.base_position,
                                                       wxyz=config.base_wxyz, show_axes=False)
        self.visual = ViserUrdf(self.server, model, root_node_name='/robot')
        self.camera_axes = self.server.scene.add_frame('/calibrated_camera', axes_length=.06, visible=False)
        self.board_axes = self.server.scene.add_frame('/calibrated_board', axes_length=.06, visible=False)

        gui = self.server.gui
        gui.add_markdown('## GX7 手眼标定采集\n**' + ('真机模式' if not self.driver.mock else
                                                  '模拟模式（数据不能用于真机标定）') + '**')
        self.status = gui.add_text('状态', initial_value='未连接', disabled=True)
        self.error = gui.add_text('错误', initial_value='', disabled=True)
        with gui.add_folder('连接'):
            self.connect_button = gui.add_button('连接模拟机械臂' if self.driver.mock else '连接真机（使能）')
            self.release_button = gui.add_button('失能拖动（先托住机械臂）', disabled=True)
            self.hold_button = gui.add_button('保持当前位置（恢复出力）', disabled=True)
            self.disconnect_button = gui.add_button('断开（会失能）', disabled=True)
        self.sliders = []
        with gui.add_folder('关节微调（度）', expand_by_default=self.driver.mock):
            gui.add_markdown('模型角度；发送后限速运动。')
            lower, upper = config.compensation.model_limits(config.lower, config.upper)
            for i in range(7):
                self.sliders.append(gui.add_slider(f'J{i + 1}', min=float(np.rad2deg(lower[i])),
                                                   max=float(np.rad2deg(upper[i])), step=.1, initial_value=0.))
            self.send_button = gui.add_button('发送目标', disabled=True)
        with gui.add_folder('采集'):
            gui.add_markdown('「失能拖动」到新姿态 →「保持当前位置」→ 松手等 1 秒 → 采集。'
                             '标定板要完整清晰地出现在画面中；采 15–30 个姿态，绕至少两个不同方向转动。')
            self.session_path = gui.add_text('会话目录', initial_value='（第一次采集时创建）', disabled=True)
            self.mode = gui.add_dropdown('安装方式', options=list(MODES), initial_value=mode)
            self.mode_hint = gui.add_markdown(MODE_NAMES[mode])
            self.new_session_button = gui.add_button('新建会话（改变安装后）')
            self.validation = gui.add_checkbox('作为验证样本（不参与拟合）', initial_value=False)
            self.sample_button = gui.add_button(f'采集（静止 {sample_sec:.1f} 秒）', disabled=True)
            self.undo_button = gui.add_button('删除最后一个样本', disabled=True)
            self.sample_status = gui.add_markdown('')
        with gui.add_folder('相机'):
            self.image = gui.add_image(np.zeros((240, 320, 3), dtype=np.uint8), label='实时画面（检测叠加）',
                                       format='jpeg')
            self.camera_status = gui.add_markdown('相机启动中…')
        with gui.add_folder('实时求解'):
            self.analysis = gui.add_markdown('')
        self.feedback = gui.add_markdown('')

        self.connect_button.on_click(lambda _: self.action(self.connect, sync_sliders=True))
        self.release_button.on_click(lambda _: self.action(self.driver.release))
        self.hold_button.on_click(lambda _: self.action(self.driver.hold, sync_sliders=True))
        self.disconnect_button.on_click(lambda _: self.action(self.disconnect))
        self.send_button.on_click(lambda _: self.action(self.send_target))
        self.new_session_button.on_click(lambda _: self.action(self.new_session))
        self.sample_button.on_click(lambda _: self.action(self.start_sample))
        self.undo_button.on_click(lambda _: self.action(self.undo))
        self.mode.on_update(lambda _: self.action(self.change_mode))

        @self.server.on_client_connect
        def on_client(client):
            client.camera.position = (.9, -1.2, 1.1)
            client.camera.look_at = (.1, -.25, .45)

        self.camera.start()
        self.refresh()

    # ------------------------------------------------------------------ actions
    def action(self, fn, *, sync_sliders=False):
        with self.ui_lock:
            if self.closed:
                return
            try:
                value = fn()
                self.error.value = ''
                if sync_sliders and value is not None:
                    for slider, angle in zip(self.sliders, self.config.compensation.to_model(value)):
                        slider.value = float(np.rad2deg(angle))
            except Exception as exc:
                self.error.value = str(exc)
                print(f'Hand-eye collect action failed: {exc}', flush=True)
            self.refresh()

    def connect(self):
        return self.driver.connect()

    def disconnect(self):
        self.pending = None
        self.driver.disconnect()

    def send_target(self):
        target = np.deg2rad([slider.value for slider in self.sliders])
        self.driver.command(self.config.compensation.to_motor(target))

    def change_mode(self):
        self.mode_hint.content = MODE_NAMES[self.mode.value]
        if self.session is not None and self.session.samples:
            # One session is one physical mounting; a different mode starts a new session.
            self.session = None
            self.session_path.value = '（第一次采集时创建）'
            self.message = '安装方式已改变，下次采集会新建会话。'
        self.update_solution()

    def ensure_detector(self):
        if self.detector is not None:
            return
        state = self.camera.snapshot()
        if state['error']:
            raise RuntimeError(f"相机错误：{state['error']}")
        if state['metadata'] is None:
            raise RuntimeError('相机尚未就绪')
        if state['metadata'].get('K') is None:
            raise RuntimeError('相机没有内参（模拟画面不能用于标定）')
        self.K, self.dist, _ = camera_parameters(state['metadata'])
        self.detector = CharucoDetector(self.K, self.dist, square_length_m=self.square_length)

    def new_session(self):
        if self.pending is not None:
            raise RuntimeError('正在采集')
        self.ensure_detector()
        c = self.config.compensation
        self.session = SessionWriter(new_session_path(self.session_root), {
            'created': datetime.now().isoformat(timespec='seconds'), 'mock': self.driver.mock,
            'mounting_mode': self.mode.value, 'joint_names': JOINTS, 'joint_units': 'rad',
            'urdf': str(self.config.urdf), 'urdf_sha256': file_sha256(self.config.urdf),
            'motor_config': str(self.config.motor_config),
            'joint_compensation': {'directions': list(c.directions), 'offsets_deg': list(c.offsets_deg)},
            'base_position': list(self.config.base_position), 'base_wxyz': list(self.config.base_wxyz),
            'fixed_model_joints': self.config.initial_joints,
            'camera': self.camera.snapshot()['metadata'],
            'board': {'squares_x': SQUARES_X, 'squares_y': SQUARES_Y, 'square_length_m': self.square_length,
                      'marker_ratio': MARKER_TO_SQUARE, 'dictionary': 'DICT_4X4_50'},
            'sampling': {'duration_sec': self.sample_sec, 'max_spread_deg': float(np.rad2deg(self.max_spread)),
                         'max_pixel_motion': self.max_pixel_motion},
            'positions': 'motor_positions_rad are raw motor angles; model angles = compensation.to_model(...)',
        })
        self.session_path.value = str(self.session.path)
        self.update_solution()
        self.message = '新会话，尚无样本。'

    def start_sample(self):
        state = self.driver.snapshot()
        if self.pending is not None:
            raise RuntimeError('正在采集')
        if state['mode'] not in ('hold', 'manual'):
            raise RuntimeError('请先「保持当前位置」让电机出力，松手后再采集')
        self.ensure_detector()
        self.driver.feedback_sample()
        if self.session is None:
            self.new_session()
        start = time.monotonic()
        self.pending = {'validation': bool(self.validation.value), 'mode': state['mode'], 'start': start,
                        'until': start + self.sample_sec, 'stamps': set(), 'positions': []}
        self.message = '正在采集…请勿触碰机械臂和标定板'

    def undo(self):
        if self.session is None:
            raise RuntimeError('没有会话')
        removed = self.session.remove_last()
        self.update_solution()
        self.message = f"已删除样本 {removed['index']}"

    # ------------------------------------------------------------------ sampling
    def advance_sample(self):
        p = self.pending
        try:
            sample = self.driver.feedback_sample()
        except Exception as exc:
            self.pending = None
            raise RuntimeError(f'采集中断：{exc}') from exc
        if sample['timestamp_monotonic'] not in p['stamps']:
            p['stamps'].add(sample['timestamp_monotonic'])
            p['positions'].append(sample['positions_rad'])
        if time.monotonic() < p['until']:
            return
        self.pending = None
        positions = np.asarray(p['positions'])
        if len(positions) < 5:
            raise RuntimeError(f'只收到 {len(positions)} 帧关节反馈，样本作废')
        spread = np.ptp(positions, axis=0)
        if np.any(spread > self.max_spread):
            worst = int(np.argmax(spread))
            raise RuntimeError(f'机械臂未静止（J{worst + 1} 波动 {np.rad2deg(spread[worst]):.3f}°），样本作废')
        frames = self.camera.between(p['start'], p['until'])
        if len(frames) < 3:
            raise RuntimeError(f'采集窗口内只收到 {len(frames)} 张图像，样本作废')
        # Detect on BGR, exactly as the offline solver sees the saved PNG.
        first, middle, last = (self.detector.detect(cv2.cvtColor(frames[i].rgb, cv2.COLOR_RGB2BGR))
                               for i in (0, len(frames) // 2, -1))
        if not middle.ok:
            raise RuntimeError(f'未检测到合格的标定板（{middle.reason}），样本作废')
        motion = corner_motion(first, last)
        if motion is None or motion > self.max_pixel_motion:
            raise RuntimeError('画面未静止（标定板角点移动 '
                               f"{'无法比较' if motion is None else f'{motion:.2f} px'}），样本作废")
        mean = positions.mean(axis=0)
        model = self.config.compensation.to_model(mean)
        record = self.session.append({
            'validation': p['validation'], 'mode': p['mode'],
            'time': datetime.now().isoformat(timespec='milliseconds'), 'joint_frames': len(positions),
            'motor_positions_rad': mean.tolist(), 'spread_rad': spread.tolist(),
            'model_positions_rad': model.tolist(),
            'image_frames': len(frames), 'image_received_at': frames[len(frames) // 2].received_at,
            'image_motion_px': motion,
            'detection': {'n_corners': middle.n_corners, 'reproj_px': middle.reproj_px,
                          'corner_ids': middle.charuco_ids.ravel().tolist(),
                          'corners_px': middle.charuco_corners.reshape(-1, 2).tolist(),
                          'T_cam_board': middle.T_cam_board.tolist()}},
            frames[len(frames) // 2].rgb)
        self.update_solution()
        self.message = (f"已保存样本 {record['index']}{'（验证）' if p['validation'] else ''}："
                        f"{len(positions)} 帧关节，最大波动 {np.rad2deg(spread.max()):.4f}°；"
                        f"{len(frames)} 张图像，角点 {middle.n_corners}，画面移动 {motion:.2f} px")

    def views(self, validation=False):
        points = np.asarray(self.detector.board.getChessboardCorners(), dtype=float)
        out = []
        for s in self.session.samples if self.session else []:
            if bool(s['validation']) != validation:
                continue
            d = s['detection']
            out.append({'index': s['index'], 'T_base_flange': self.fk.fk(s['model_positions_rad']).tolist(),
                        'T_cam_board': d['T_cam_board'], 'object_points': points[d['corner_ids']].tolist(),
                        'corners_px': d['corners_px']})
        return out

    def update_solution(self):
        self.solution = None
        train = self.views() if self.detector is not None else []
        if len(train) < 6:
            self.solution_note = f'训练样本 {len(train)} 个，至少 6 个后自动求解。'
            return
        try:
            self.solution = solve_views(train, self.K, self.dist, self.mode.value, reject=0)
            self.solution_note = ''
        except Exception as exc:
            self.solution_note = f'暂时无法求解：{exc}'

    # ------------------------------------------------------------------ display
    def analysis_markdown(self):
        samples = self.session.samples if self.session else []
        n_val = sum(bool(s['validation']) for s in samples)
        lines = [f'样本：训练 {len(samples) - n_val}，验证 {n_val}']
        if self.solution_note:
            lines.append(self.solution_note)
        if self.solution is not None:
            s, sv = self.solution['summary'], self.solution['motion']['rotation_singular_values']
            U = self.solution['U']
            name = 'T_flange_cam' if self.mode.value == 'eye-in-hand' else 'T_base_cam'
            lines.append(f"重投影 RMS {s['reproj_px']['rms']:.3f} px（最大 {s['reproj_px']['max']:.3f}），"
                         f"位移 {s['trans_mm']['rms']:.2f} mm，旋转 {s['rot_deg']['rms']:.3f}°")
            lines.append(f"{name} 平移：{np.round(U[:3, 3] * 1000, 1).tolist()} mm")
            lines.append('旋转激励奇异值：' + ', '.join(f'{v:.2f}' for v in sv) + '（第二个太小说明转轴单一）')
            rows = ['| 样本 | 重投影 px | 位移 mm |', '|---:|---:|---:|']
            for view, r in list(zip(self.views(), self.solution['residuals']))[-8:]:
                rows.append(f"| {view['index']} | {r['reproj_px']:.3f} | {r['trans_mm']:.2f} |")
            lines.append('\n'.join(rows))
            val = self.views(validation=True)
            if val:
                res =view_residuals(self.solution['U'], self.solution['V'], val, self.K, self.dist, self.mode.value)
                lines.append(f"验证样本重投影 RMS {np.sqrt(np.mean([r['reproj_px'] ** 2 for r in res])):.3f} px")
            lines.append('实时结果只用于检查数据；最终标定用 `run.sh calibrate`。')
        return '\n\n'.join(lines)

    def update_preview(self):
        frame = self.camera.latest()
        state = self.camera.snapshot()
        if state['error']:
            self.camera_status.content = f"相机错误：{state['error']}"
            return
        if frame is None:
            return
        try:
            self.ensure_detector()
        except RuntimeError as exc:
            self.image.image = frame.rgb
            self.camera_status.content = str(exc)
            return
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        detection = self.detector.detect(bgr)
        self.image.image = cv2.cvtColor(self.detector.draw(bgr, detection), cv2.COLOR_BGR2RGB)
        self.camera_status.content = ('标定板合格' if detection.ok else f'标定板不合格：{detection.reason}') + \
            f'（角点 {detection.n_corners}）'

    def refresh(self):
        with self.ui_lock:
            state = self.driver.snapshot()
            self.status.value = MODE_LABELS[state['mode']] + ('（采集中）' if self.pending else '')
            if state['error']:
                self.error.value = state['error']
            enabled = state['connected'] and state['mode'] != 'fault'
            busy = self.pending is not None
            self.connect_button.disabled = state['connected']
            self.disconnect_button.disabled = not state['connected']
            self.release_button.disabled = not enabled or state['mode'] == 'released' or busy
            self.hold_button.disabled = not enabled or busy
            self.send_button.disabled = not enabled or state['mode'] == 'released' or busy
            self.sample_button.disabled = not enabled or state['mode'] not in ('hold', 'manual') or busy
            self.undo_button.disabled = self.session is None or not self.session.samples or busy
            self.new_session_button.disabled = self.mode.disabled = self.validation.disabled = busy
            self.sample_status.content = self.message
            self.analysis.content = self.analysis_markdown()
            motor = state['positions']
            model = self.config.compensation.to_model(motor)
            self.robot_frame.visible = state['connected'] or self.driver.mock
            self.model_positions[self.arm_indices] = model
            self.visual.update_cfg(self.model_positions)
            self.camera_axes.visible = self.board_axes.visible = self.solution is not None
            if self.solution is not None:
                E, U, V = self.base @ self.fk.fk(model.tolist()), self.solution['U'], self.solution['V']
                cam, board = ((E @ U, self.base @ V) if self.mode.value == 'eye-in-hand'
                              else (self.base @ U, E @ V))
                for handle, T in ((self.camera_axes, cam), (self.board_axes, board)):
                    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
                    handle.position, handle.wxyz = T[:3, 3], (w, x, y, z)
            rows = ['| 关节 | 电机 ° | 模型 ° |', '|---|---:|---:|']
            rows += [f'| J{i + 1} | {np.rad2deg(m):.3f} | {np.rad2deg(v):.3f} |'
                     for i, (m, v) in enumerate(zip(motor, model))]
            if state['feedback_age'] is not None:
                rows.append(f"\n反馈年龄：{state['feedback_age'] * 1000:.0f} ms")
            self.feedback.content = '\n'.join(rows)

    # ------------------------------------------------------------------ loop
    def update(self):
        self.driver.update()
        if self.pending is not None:
            try:
                self.advance_sample()
            except Exception as exc:
                self.message = str(exc)
                self.error.value = str(exc)
        if self.pending is None and time.monotonic() - self.last_preview > .2:
            self.last_preview = time.monotonic()
            self.update_preview()
        self.refresh()

    def run(self):
        print(f'Open http://{self.config.host}:{self.config.port} · '
              f'{"模拟" if self.driver.mock else "真机"}模式，当前未连接', flush=True)
        period = 1 / self.config.control_rate_hz
        try:
            while not self.stop.is_set():
                start = time.monotonic()
                with self.ui_lock:
                    self.update()
                self.stop.wait(max(0., period - (time.monotonic() - start)))
        finally:
            self.close()

    def close(self):
        with self.ui_lock:
            if self.closed:
                return
            self.closed = True
            self.stop.set()
            try:
                self.driver.disconnect()
            finally:
                self.camera.stop()
                self.server.stop()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--robot-config', type=Path, default=HERE.parent / 'control' / 'config.yaml')
    parser.add_argument('--port', type=int, default=8088)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--real', dest='real', action='store_true', help='Real arm and camera (default)')
    mode.add_argument('--simulate', dest='real', action='store_false', help='Simulated arm and image')
    parser.set_defaults(real=True)
    parser.add_argument('--mode', choices=MODES, default='eye-in-hand', help='initial mounting mode in the UI')
    parser.add_argument('--camera-device-index', type=int, default=0)
    parser.add_argument('--square-length-m', type=float, default=SQUARE_LENGTH_M, help='measured square length')
    parser.add_argument('--sample-sec', type=float, default=1., help='hold window per sample')
    parser.add_argument('--max-spread-deg', type=float, default=.05, help='reject a sample if any joint moves more')
    parser.add_argument('--max-pixel-motion', type=float, default=.5,
                        help='reject a sample if board corners move more (px) between first and last image')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = dataclasses.replace(load_config(args.robot_config), port=args.port)
    app = CollectApp(config, real=args.real, mode=args.mode, sample_sec=args.sample_sec,
                     max_spread_deg=args.max_spread_deg, max_pixel_motion=args.max_pixel_motion,
                     square_length=args.square_length_m, camera_device_index=args.camera_device_index)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: app.stop.set())
    app.run()


if __name__ == '__main__':
    main()
