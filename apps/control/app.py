"""GX7 joint control and GX4M preview. The CLI defaults to real, read-only startup data."""
import argparse
from datetime import datetime
from pathlib import Path
import signal
import threading
import time

import numpy as np
import viser
from viser.extras import ViserUrdf
import yourdfpy

from gx4m_real.camera.process import ProcessColorCamera
from gx4m_real.camera.preview import CameraPreview
from gx4m_real.camera.simulated import SimulatedRGBCamera
from gx4m_real.robot.arm import ArmDriver
from gx4m_real.robot.config import ARM_JOINTS, load_config
from gx4m_real.robot.compensation import JointCompensation
from gx4m_real.robot.feedback import read_joint_positions
from gx4m_real.robot.recorder import PlaybackRecorder
from gx4m_real.robot.trajectory import load_trajectory

HERE = Path(__file__).resolve().parent
MODE_LABELS = {'disconnected': '未连接', 'hold': '保持位置', 'manual': '关节控制', 'fault': '故障',
               'recording': '无力录制中', 'released': '已失能（需手动支撑）', 'playback': '位置回放中'}


def load_model(config):
    model = yourdfpy.URDF.load(str(config.urdf), build_scene_graph=True, load_meshes=True)
    names = tuple(model.actuated_joint_names)
    if any(name not in names for name in ARM_JOINTS):
        raise ValueError('模型缺少 joint1–joint7')
    missing = set(names) - config.initial_joints.keys()
    if missing:
        raise ValueError(f'初始姿态缺少模型关节：{sorted(missing)}')
    positions = np.array([config.initial_joints[name] for name in names], dtype=float)
    return model, names, positions


class ControlApp:
    def __init__(self, config, *, real=False, driver=None, camera_device_index=0):
        self.config = config
        self.driver = driver if driver is not None else ArmDriver(config, mock=not real)
        self.stop = threading.Event()
        self.ui_lock = threading.RLock()
        self.closed = False
        self.capture = None
        self.capture_mode = None
        self.camera_device_index = camera_device_index
        self.camera_preview = CameraPreview(self.make_preview_camera)
        self.last_image_at = 0.
        self.last_image_key = None
        self.last_recorded_key = None
        self.preview_compensation = self.saved_compensation = config.compensation
        self.readonly_positions = None
        self.readonly_sample_at = None
        self.readonly_error = None
        model, self.model_names, self.model_positions = load_model(config)
        self.arm_indices = [self.model_names.index(name) for name in ARM_JOINTS]
        self.server = viser.ViserServer(host=config.host, port=config.port,
                                       label='GX7 + GX4M · 关节控制')
        self.server.scene.set_up_direction('+z')
        self.server.scene.add_grid('/ground', width=1.5, height=1.5, plane='xy', cell_size=.05)
        self.server.scene.add_frame('/world', axes_length=.15, axes_radius=.003)
        self.robot_frame = self.server.scene.add_frame('/robot', position=config.base_position,
                                                       wxyz=config.base_wxyz, show_axes=False)
        self.visual = ViserUrdf(self.server, model, root_node_name='/robot')
        self.visual.update_cfg(self.model_positions)
        gui = self.server.gui
        gui.add_markdown('## GX7＋GX4M\n**' + ('真机控制模式' if not self.driver.mock else '模拟控制模式（不会驱动真机）') +
                         '**\n\n仅控制 J1–J7；GX4M 手固定显示。')
        self.status = gui.add_text('状态', initial_value='未连接', disabled=True)
        self.error = gui.add_text('错误', initial_value='', disabled=True)
        self.feedback_source = gui.add_text('关节数据来源', initial_value='未读取真机关节', disabled=True)
        with gui.add_folder('相机 RGB 预览'):
            gui.add_markdown('仅预览相机，不连接机械臂。录制或回放同步采集时显示采集相机的实时画面。')
            self.camera_source = gui.add_dropdown('预览画面来源', options=('真实相机', '模拟画面'),
                                                  initial_value='真实相机')
            self.camera_start_button = gui.add_button('打开相机预览')
            self.camera_stop_button = gui.add_button('关闭相机预览', disabled=True)
            self.camera_status = gui.add_markdown('相机未打开。')
            self.camera_image = gui.add_image(np.zeros((240, 320, 3), dtype=np.uint8), label='相机 RGB')
            self.camera_image.visible = False
        with gui.add_folder('连接与控制'):
            self.connect_button = gui.add_button('连接模拟机械臂' if self.driver.mock else '连接真机（使能）')
            self.send_button = gui.add_button('发送目标', disabled=True)
            self.hold_button = gui.add_button('保持当前位置', disabled=True)
            self.disconnect_button = gui.add_button('断开', disabled=True)
        self.sliders = []
        model_lower, model_upper = config.compensation.model_limits(config.lower, config.upper)
        model_initial = config.compensation.to_model(config.initial_position)
        with gui.add_folder('模型目标关节角（度）'):
            gui.add_markdown('目标采用补偿后的模型角度，发送时转换回电机角度；硬件限位保持不变。')
            for i, name in enumerate(ARM_JOINTS):
                self.sliders.append(gui.add_slider(
                    f'J{i+1} · {name}', min=float(np.rad2deg(model_lower[i])),
                    max=float(np.rad2deg(model_upper[i])), step=.1,
                    initial_value=float(np.rad2deg(model_initial[i]))))
        self.feedback = gui.add_markdown('')
        self.direction_fields, self.offset_fields = [], []
        with gui.add_folder('关节方向与零点补偿'):
            gui.add_markdown('模型角度 = 方向 × 电机角度 + 零点偏移。\n\n'
                             '先断开，再只读采样真机（独立于模拟/真机控制模式）；编辑仅预览，不使能电机、不发送运动目标。'
                             '只有点击保存才写文件，重启后生效。')
            self.read_button = gui.add_button('只读采样真机关节（不使能）')
            for index, name in enumerate(ARM_JOINTS):
                with gui.add_folder(f'J{index + 1} · {name}'):
                    self.direction_fields.append(gui.add_dropdown(
                        '方向', options=('+1', '-1'),
                        initial_value=f'{int(config.compensation.directions[index]):+d}'))
                    self.offset_fields.append(gui.add_number(
                        '零点偏移（度）', initial_value=config.compensation.offsets_deg[index], step=.1))
            self.save_compensation_button = gui.add_button('保存补偿（重启后生效）')
            self.restore_compensation_button = gui.add_button('恢复启动时补偿')
            self.compensation_status = gui.add_markdown('')
        with gui.add_folder('无力录制与位置回放'):
            gui.add_markdown('录制会失能 J1–J7，请先托住机械臂。停止录制后保持失能；'
                             '点击“保持当前位置”或“开始位置回放”恢复出力。')
            self.record_path = gui.add_text('录制保存路径', initial_value=self.new_recording_path())
            self.record_button = gui.add_button('开始无力录制', disabled=True)
            self.stop_record_button = gui.add_button('停止并保存录制', disabled=True)
            self.play_path = gui.add_text('回放文件路径', initial_value='')
            self.play_speed = gui.add_slider('回放倍率', min=.1, max=2., step=.1, initial_value=1.)
            self.play_button = gui.add_button('开始位置回放', disabled=True)
            self.stop_play_button = gui.add_button('停止回放并保持', disabled=True)
            self.trajectory_status = gui.add_markdown('尚未录制。')
        with gui.add_folder('录制与回放同步采集 RGB 与关节位置'):
            self.record_capture_enabled = gui.add_checkbox('无力录制时同步采集', initial_value=True)
            self.capture_enabled = gui.add_checkbox('回放时同步采集', initial_value=True)
            self.capture_path = gui.add_text('采集目录', initial_value=self.new_capture_path())
            gui.add_markdown('保存原始 RGB PNG、相机内参、实测关节及时间对齐索引。'
                             '模拟模式保存明确标记的合成图片；真机模式使用 Gemini 相机。')
            self.capture_status = gui.add_markdown('尚未采集。')
            self.recorded_image = gui.add_image(np.zeros((240, 320, 3), dtype=np.uint8), label='已保存的录制帧')
            self.recorded_image.visible = False
            self.recorded_status = gui.add_markdown('开始同步采集后显示已保存图像和同一时间戳的关节位置。')

        self.connect_button.on_click(lambda _: self.action(self.connect_driver, sync_sliders=True))
        self.hold_button.on_click(lambda _: self.action(self.stop_playback, sync_sliders=True))
        self.send_button.on_click(lambda _: self.send_target())
        self.disconnect_button.on_click(lambda _: self.action(self.disconnect))
        self.record_button.on_click(lambda _: self.action(self.start_recording))
        self.stop_record_button.on_click(lambda _: self.action(self.save_recording))
        self.play_button.on_click(lambda _: self.action(self.start_playback))
        self.stop_play_button.on_click(lambda _: self.action(self.stop_playback, sync_sliders=True))
        self.read_button.on_click(lambda _: self.action(self.read_positions))
        self.camera_start_button.on_click(lambda _: self.action(self.start_camera_preview))
        self.camera_stop_button.on_click(lambda _: self.action(self.camera_preview.stop))
        self.save_compensation_button.on_click(lambda _: self.action(self.save_compensation))
        self.restore_compensation_button.on_click(lambda _: self.action(self.restore_compensation))
        for field in self.direction_fields + self.offset_fields:
            field.on_update(lambda _: self.action(self.preview_mapping))

        @self.server.on_client_connect
        def on_client(client):
            client.camera.position = (.9, -1.2, 1.1)
            client.camera.look_at = (.1, -.25, .45)

        self.refresh()

    @staticmethod
    def new_recording_path():
        return str(HERE / 'runtime' / 'recordings' / f'joints_{datetime.now():%Y%m%d_%H%M%S_%f}.json')

    @staticmethod
    def new_capture_path():
        return str(HERE / 'runtime' / 'calibration' / f'playback_{datetime.now():%Y%m%d_%H%M%S_%f}')

    def make_camera(self):
        return SimulatedRGBCamera() if self.driver.mock else ProcessColorCamera(self.camera_device_index)

    def make_preview_camera(self):
        return (SimulatedRGBCamera() if self.camera_source.value == '模拟画面'
                else ProcessColorCamera(self.camera_device_index))

    def start_camera_preview(self):
        if self.capture is not None and self.capture.active:
            raise RuntimeError('正在同步采集相机，预览已复用采集画面')
        self.camera_preview.start()

    def refresh_camera_preview(self):
        capturing = self.capture is not None and self.capture.active
        preview = self.camera_preview.snapshot()
        if capturing:
            frame = self.capture.latest_frame()
            metadata = self.capture.camera.metadata
            error = self.capture.snapshot()['error']
            running = True
            status = '无力录制同步采集中' if self.capture_mode == 'recording' else '回放同步采集中'
        else:
            frame, metadata, error, running = (preview[key] for key in ('frame', 'metadata', 'error', 'running'))
            status = '相机预览中' if frame is not None else '正在打开相机'
        self.camera_start_button.disabled = capturing or preview['running']
        self.camera_stop_button.disabled = not preview['running']
        self.camera_source.disabled = capturing or preview['running']
        now = time.monotonic()
        fresh = frame is not None and now - frame.received_at < 2.
        self.camera_image.visible = running and fresh and not error
        if error:
            self.camera_status.content = f'相机错误：{error}'
        elif not running:
            self.camera_status.content = '相机未打开。'
        else:
            source = '模拟画面（非真实相机）' if metadata.get('synthetic') else '真实相机'
            device = metadata.get('serial') or metadata.get('device', '')
            details = f'{frame.rgb.shape[1]}×{frame.rgb.shape[0]}' if frame is not None else '等待图像'
            if frame is not None and not fresh:
                details = '图像已超过 2 秒未更新'
            self.camera_status.content = f'{status} · {source} · {device} · {details}'
        if running and fresh and not error and now - self.last_image_at >= .1:
            key = (frame.sequence, frame.received_at)
            if key != self.last_image_key:
                self.camera_image.image = frame.rgb
                self.last_image_key, self.last_image_at = key, now

    def refresh_recorded_preview(self):
        saved = self.capture.latest_recorded() if self.capture is not None else None
        self.recorded_image.visible = saved is not None
        if saved is None:
            self.recorded_status.content = '等待已保存的录制帧。' if self.capture is not None else '尚未同步采集。'
            return
        frame, record = saved
        key = (str(self.capture.path), record['frame_index'])
        if key == self.last_recorded_key:
            return
        # Display the actual saved frame, never combine a newer image with older joints.
        self.recorded_image.image = frame.rgb
        self.last_recorded_key = key
        rows = [f"帧 {record['frame_index']} · `{record['rgb_path']}`",
                f"图像时间戳：{record['timestamp_monotonic']:.6f} s · 会话时间：{record['time_sec']:.3f} s"]
        if record['alignment_valid']:
            offsets = record['joint_time_offsets_sec']
            rows += [f'对齐有效 · 前后反馈间隔：{offsets[0] * 1000:.2f} / {offsets[1] * 1000:.2f} ms',
                     '| 关节 | 对齐的实测角度（rad） |', '|---|---:|']
            rows += [f'| J{i + 1} | {q:.6f} |' for i, q in enumerate(record['positions_rad'])]
        else:
            rows.append(f"对齐无效：{record['alignment_reason']}；该图没有可用的配对关节位置。")
        self.recorded_status.content = '\n\n'.join(rows[:3]) + '\n' + '\n'.join(rows[3:])

    def begin_capture(self, mode, trajectory_path=None):
        if self.capture is not None and self.capture.active:
            raise RuntimeError('上次采集尚未结束')
        self.camera_preview.stop()
        self.capture_mode = mode
        self.capture = PlaybackRecorder(self.driver.feedback_sample, camera_factory=self.make_camera,
                                        sample_hz=self.config.frequency_hz)
        self.last_recorded_key = None
        try:
            self.capture.start(self.capture_path.value, trajectory_path, {
                'mock': self.driver.mock, 'capture_mode': mode,
                'requested_playback_speed': self.play_speed.value if mode == 'playback' else None,
                'joint_recording_path': str(Path(self.record_path.value).expanduser().resolve())
                if mode == 'recording' else None,
                'velocity_limit_rad_sec': self.config.velocity_limit,
                'robot_urdf': str(self.config.urdf),
                'base_position': self.config.base_position, 'base_wxyz': self.config.base_wxyz,
                'fixed_model_joints': self.config.initial_joints,
            })
        except Exception:
            if self.capture.error:
                self.capture_path.value = self.new_capture_path()
            raise

    def start_recording(self):
        self.require_active_compensation()
        self.driver.feedback_sample()
        state = self.driver.snapshot()
        if state['recording'] or state['mode'] == 'playback':
            raise RuntimeError('请先停止当前录制或回放')
        if Path(self.record_path.value).expanduser().exists():
            raise ValueError('录制文件已存在，请更换文件名')
        if self.record_capture_enabled.value:
            self.begin_capture('recording')
        try:
            self.driver.start_recording(self.record_path.value)
        except Exception:
            self.finish_capture('recording_start_failed')
            raise

    def start_playback(self):
        self.require_active_compensation()
        path = Path(self.play_path.value).expanduser().resolve()
        load_trajectory(self.config, path)
        self.driver.feedback_sample()  # Reject disconnected/faulted arms before opening a camera.
        state = self.driver.snapshot()
        if state['recording'] or state['mode'] == 'playback':
            raise RuntimeError('请先停止当前录制或回放')
        if self.capture is not None and self.capture.active:
            raise RuntimeError('上次采集尚未结束')
        if self.capture_enabled.value:
            if state['mode'] != 'released':
                self.driver.hold()
            self.begin_capture('playback', path)
        try:
            self.driver.play(path, self.play_speed.value)
        except Exception:
            self.finish_capture('playback_start_failed')
            raise

    def finish_capture(self, reason):
        if self.capture is not None and self.capture.active:
            self.capture.stop(reason)
            self.capture_path.value = self.new_capture_path()

    def stop_playback(self):
        try:
            return self.driver.hold()
        finally:
            self.finish_capture('stopped')

    def disconnect(self):
        try:
            state = self.driver.snapshot()
            if state['recording']:
                self.save_recording()
            if state['mode'] == 'playback':
                self.driver.hold()
            self.finish_capture('disconnected')
        finally:
            self.driver.disconnect()

    def update(self):
        self.driver.update()
        if self.capture is not None and self.capture.active:
            state = self.driver.snapshot()
            if self.capture.snapshot()['error']:
                if state['recording']:
                    self.save_recording(reason='capture_failed')
                elif state['connected'] and state['mode'] != 'fault' and self.capture_mode == 'playback':
                    self.driver.hold()
                self.finish_capture('capture_failed')
            elif state['mode'] != self.capture_mode:
                self.finish_capture('fault' if state['mode'] == 'fault' else 'completed')
        self.refresh()

    def save_recording(self, reason='recording_saved'):
        path = self.driver.stop_recording()
        if self.capture_mode == 'recording':
            self.finish_capture(reason)
        self.play_path.value = str(path)
        self.record_path.value = self.new_recording_path()

    def action(self, fn, *, sync_sliders=False):
        with self.ui_lock:
            if self.closed:
                return
            try:
                value = fn()
                self.error.value = ''
                if sync_sliders:
                    for slider, angle in zip(self.sliders, self.config.compensation.to_model(value)):
                        slider.value = float(np.rad2deg(angle))
            except Exception as exc:
                self.error.value = str(exc)
                print(f'Control action failed: {exc}', flush=True)
            self.refresh()

    def send_target(self):
        with self.ui_lock:
            target = np.deg2rad([slider.value for slider in self.sliders])
            self.action(lambda: self.command_model(target))

    def require_active_compensation(self):
        if self.draft_compensation() != self.config.compensation:
            raise RuntimeError('补偿预览尚未应用；请保存并重启，或恢复启动时补偿')

    def connect_driver(self):
        self.require_active_compensation()
        positions = self.driver.connect()
        self.readonly_positions = None
        self.readonly_error = None
        return positions

    def command_model(self, positions):
        self.require_active_compensation()
        return self.driver.command(self.config.compensation.to_motor(positions))

    def require_disconnected(self):
        if self.driver.snapshot()['connected']:
            raise RuntimeError('请先断开机械臂，再采样或编辑补偿；断开会失能，请先支撑机械臂')

    def read_positions(self):
        self.require_disconnected()
        try:
            positions = read_joint_positions(self.config)
        except Exception as exc:
            self.readonly_error = str(exc)
            raise
        self.readonly_positions = positions
        self.readonly_sample_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.readonly_error = None
        for slider, angle in zip(self.sliders, self.config.compensation.to_model(positions)):
            slider.value = float(np.rad2deg(angle))
        print(f'Read-only real joints at {self.readonly_sample_at}: '
              f'{np.rad2deg(positions).round(3).tolist()} deg', flush=True)

    def draft_compensation(self):
        return JointCompensation(
            tuple(int(field.value) for field in self.direction_fields),
            tuple(float(field.value) for field in self.offset_fields))

    def preview_mapping(self):
        self.require_disconnected()
        self.preview_compensation = self.draft_compensation()

    def save_compensation(self):
        self.preview_mapping()
        self.preview_compensation.save(self.config.compensation_path)
        self.saved_compensation = self.preview_compensation

    def restore_compensation(self):
        self.require_disconnected()
        for index, (direction, offset) in enumerate(zip(self.direction_fields, self.offset_fields)):
            direction.value = f'{int(self.config.compensation.directions[index]):+d}'
            offset.value = self.config.compensation.offsets_deg[index]
        self.preview_compensation = self.config.compensation

    def refresh(self):
        with self.ui_lock:
            state = self.driver.snapshot()
            self.status.value = MODE_LABELS[state['mode']]
            if state['error']:
                self.error.value = state['error']
            enabled = state['connected'] and state['mode'] != 'fault'
            pending = self.preview_compensation != self.config.compensation
            self.connect_button.disabled = state['connected'] or pending
            self.disconnect_button.disabled = not state['connected']
            recording = state['recording']
            self.send_button.disabled = not enabled or pending or state['mode'] in ('recording', 'released', 'playback')
            self.hold_button.disabled = not enabled or recording
            self.record_button.disabled = not enabled or pending or recording or state['mode'] == 'playback'
            self.stop_record_button.disabled = not recording
            self.play_button.disabled = not enabled or pending or recording or state['mode'] == 'playback'
            self.stop_play_button.disabled = not enabled or state['mode'] != 'playback'
            self.record_path.disabled = recording
            self.play_path.disabled = self.play_speed.disabled = state['mode'] == 'playback'
            capturing = self.capture is not None and self.capture.active
            self.capture_enabled.disabled = self.capture_path.disabled = capturing or state['mode'] == 'playback'
            self.record_capture_enabled.disabled = capturing or recording
            self.capture_path.disabled = capturing or recording or state['mode'] == 'playback'
            if self.capture is not None:
                capture = self.capture.snapshot()
                self.capture_status.content = (
                    f"{'采集中' if capture['active'] else '已结束'}：{capture['frames']} 张 RGB，"
                    f"{capture['aligned_frames']} 张有效对齐，{capture['joint_samples']} 条关节反馈\n\n"
                    f"目录：{capture['path']}")
                if capture['error']:
                    self.error.value = f"采集失败：{capture['error']}"
            if recording:
                self.trajectory_status.content = (f"已录制 {state['recording_frames']} 帧，"
                                                  f"{state['recording_seconds']:.1f} 秒")
            elif state['mode'] == 'playback':
                self.trajectory_status.content = f"回放进度：{state['playback_progress']:.0%}"
            elif state['last_recording']:
                self.trajectory_status.content = f"已保存：{state['last_recording']}"
            for field in (self.direction_fields + self.offset_fields +
                          [self.read_button, self.save_compensation_button, self.restore_compensation_button]):
                field.disabled = state['connected']
            unsaved = self.preview_compensation != self.saved_compensation
            message = ('预览未保存，不改变控制映射。' if unsaved else
                       '已保存；重启后应用，当前仍为预览。' if pending else '当前补偿已生效。')
            if self.readonly_positions is not None and not state['connected']:
                message += '\n\n当前显示单次采样，不是连续反馈；移动实物后请再次采样。'
            self.compensation_status.content = f'{message}\n\n文件：`{self.config.compensation_path}`'
            motor_positions = (self.readonly_positions if self.readonly_positions is not None
                               and not state['connected'] else state['positions'])
            if state['connected']:
                source = '模拟反馈（非真机）' if self.driver.mock else '真机实时反馈'
            elif self.readonly_positions is not None:
                source = f'真机只读采样：{self.readonly_sample_at}（非实时）'
            else:
                source = ('模拟参考姿态（非真机）' if self.driver.mock
                          else '未收到真机关节数据；模型已隐藏')
            if self.readonly_error and not state['connected']:
                source = f'采样失败；{source}'
            self.feedback_source.value = source
            has_positions = state['connected'] or self.readonly_positions is not None or self.driver.mock
            self.robot_frame.visible = has_positions
            model_positions = self.preview_compensation.to_model(motor_positions)
            self.model_positions[self.arm_indices] = model_positions
            self.visual.update_cfg(self.model_positions)
            targets = self.config.compensation.to_model(state['target'])
            rows = ['| 关节 | 电机角度 ° | 模型角度 ° | 已提交目标（模型）° |', '|---|---:|---:|---:|']
            for i, (motor, model, target) in enumerate(zip(motor_positions, model_positions, targets)):
                target_text = f'{np.rad2deg(target):.2f}' if state['connected'] else '—'
                motor_text = f'{np.rad2deg(motor):.2f}' if has_positions else '—'
                model_text = f'{np.rad2deg(model):.2f}' if has_positions else '—'
                rows.append(f'| J{i+1} | {motor_text} | {model_text} | {target_text} |')
            age = state['feedback_age']
            if age is not None:
                rows.append(f'\n反馈年龄：{age * 1000:.0f} ms')
            self.feedback.content = '\n'.join(rows)
            self.refresh_camera_preview()
            self.refresh_recorded_preview()

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
                self.camera_preview.stop()
            finally:
                try:
                    self.disconnect()
                finally:
                    self.server.stop()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'config.yaml')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--real', dest='real', action='store_true', help='Use real data (default); startup only reads joints')
    mode.add_argument('--simulate', dest='real', action='store_false', help='Use simulated robot and camera data')
    parser.set_defaults(real=True)
    parser.add_argument('--camera-device-index', type=int, default=0, help='Gemini camera index for RGB capture')
    args = parser.parse_args(argv)
    if args.camera_device_index < 0:
        parser.error('--camera-device-index must be nonnegative')
    return args


def main(argv=None):
    args = parse_args(argv)
    app = ControlApp(load_config(args.config), real=args.real, camera_device_index=args.camera_device_index)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: app.stop.set())
    if args.real:
        app.action(app.read_positions)
    else:
        app.camera_source.value = '模拟画面'
    app.run()


if __name__ == '__main__':
    main()
