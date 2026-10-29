"""GX7 joint control and GX4M model preview. Mock mode is the default."""
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

from gx4m_real.camera.orbbec import OrbbecColorCamera
from gx4m_real.camera.simulated import SimulatedRGBCamera
from gx4m_real.robot.arm import ArmDriver
from gx4m_real.robot.config import ARM_JOINTS, load_config
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
        self.camera_device_index = camera_device_index
        model, self.model_names, self.model_positions = load_model(config)
        self.arm_indices = [self.model_names.index(name) for name in ARM_JOINTS]
        self.server = viser.ViserServer(host=config.host, port=config.port,
                                       label='GX7 + GX4M · 关节控制')
        self.server.scene.set_up_direction('+z')
        self.server.scene.add_grid('/ground', width=1.5, height=1.5, plane='xy', cell_size=.05)
        self.server.scene.add_frame('/world', axes_length=.15, axes_radius=.003)
        self.server.scene.add_frame('/robot', position=config.base_position, wxyz=config.base_wxyz,
                                    show_axes=False)
        self.visual = ViserUrdf(self.server, model, root_node_name='/robot')
        self.visual.update_cfg(self.model_positions)
        gui = self.server.gui
        gui.add_markdown('## GX7＋GX4M\n**' + ('真机模式' if real else '模拟模式（不访问 CAN）') +
                         '**\n\n仅控制 J1–J7；GX4M 手固定显示。')
        self.status = gui.add_text('状态', initial_value='未连接', disabled=True)
        self.error = gui.add_text('错误', initial_value='', disabled=True)
        with gui.add_folder('连接与控制'):
            self.connect_button = gui.add_button('连接')
            self.send_button = gui.add_button('发送目标', disabled=True)
            self.hold_button = gui.add_button('保持当前位置', disabled=True)
            self.disconnect_button = gui.add_button('断开', disabled=True)
        self.sliders = []
        with gui.add_folder('目标关节角（度）'):
            gui.add_markdown('滑块只编辑目标，点击“发送目标”后执行。')
            for i, name in enumerate(ARM_JOINTS):
                self.sliders.append(gui.add_slider(
                    f'J{i+1} · {name}', min=float(np.rad2deg(config.lower[i])),
                    max=float(np.rad2deg(config.upper[i])), step=.1,
                    initial_value=float(np.rad2deg(config.initial_position[i]))))
        self.feedback = gui.add_markdown('')
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
        with gui.add_folder('回放同步采集 RGB 与关节位置'):
            self.capture_enabled = gui.add_checkbox('回放时同步采集', initial_value=True)
            self.capture_path = gui.add_text('采集目录', initial_value=self.new_capture_path())
            gui.add_markdown('保存原始 RGB PNG、相机内参、实测关节及时间对齐索引。'
                             '模拟模式保存明确标记的合成图片；真机模式使用 Gemini 相机。')
            self.capture_status = gui.add_markdown('尚未采集。')

        self.connect_button.on_click(lambda _: self.action(self.driver.connect, sync_sliders=True))
        self.hold_button.on_click(lambda _: self.action(self.stop_playback, sync_sliders=True))
        self.send_button.on_click(lambda _: self.send_target())
        self.disconnect_button.on_click(lambda _: self.action(self.disconnect))
        self.record_button.on_click(lambda _: self.action(
            lambda: self.driver.start_recording(self.record_path.value)))
        self.stop_record_button.on_click(lambda _: self.action(self.save_recording))
        self.play_button.on_click(lambda _: self.action(self.start_playback))
        self.stop_play_button.on_click(lambda _: self.action(self.stop_playback, sync_sliders=True))

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

    def start_playback(self):
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
            camera_factory = (SimulatedRGBCamera if self.driver.mock else
                              lambda: OrbbecColorCamera(self.camera_device_index))
            self.capture = PlaybackRecorder(self.driver.feedback_sample, camera_factory=camera_factory,
                                            sample_hz=self.config.frequency_hz)
            try:
                self.capture.start(self.capture_path.value, path, {
                    'mock': self.driver.mock, 'requested_playback_speed': self.play_speed.value,
                    'velocity_limit_rad_sec': self.config.velocity_limit,
                    'robot_urdf': str(self.config.urdf),
                    'base_position': self.config.base_position, 'base_wxyz': self.config.base_wxyz,
                    'fixed_model_joints': self.config.initial_joints,
                })
            except Exception:
                if self.capture.error:
                    self.capture_path.value = self.new_capture_path()
                raise
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
                if state['connected'] and state['mode'] != 'fault':
                    self.driver.hold()
                self.finish_capture('capture_failed')
            elif state['mode'] != 'playback':
                self.finish_capture('fault' if state['mode'] == 'fault' else 'completed')
        self.refresh()

    def save_recording(self):
        path = self.driver.stop_recording()
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
                    for slider, angle in zip(self.sliders, value):
                        slider.value = float(np.rad2deg(angle))
            except Exception as exc:
                self.error.value = str(exc)
            self.refresh()

    def send_target(self):
        with self.ui_lock:
            target = np.deg2rad([slider.value for slider in self.sliders])
            self.action(lambda: self.driver.command(target))

    def refresh(self):
        with self.ui_lock:
            state = self.driver.snapshot()
            self.status.value = MODE_LABELS[state['mode']]
            if state['error']:
                self.error.value = state['error']
            enabled = state['connected'] and state['mode'] != 'fault'
            self.connect_button.disabled = state['connected']
            self.disconnect_button.disabled = not state['connected']
            recording = state['recording']
            self.send_button.disabled = not enabled or state['mode'] in ('recording', 'released', 'playback')
            self.hold_button.disabled = not enabled or recording
            self.record_button.disabled = not enabled or recording or state['mode'] == 'playback'
            self.stop_record_button.disabled = not recording
            self.play_button.disabled = not enabled or recording or state['mode'] == 'playback'
            self.stop_play_button.disabled = not enabled or state['mode'] != 'playback'
            self.record_path.disabled = recording
            self.play_path.disabled = self.play_speed.disabled = state['mode'] == 'playback'
            capturing = self.capture is not None and self.capture.active
            self.capture_enabled.disabled = self.capture_path.disabled = capturing or state['mode'] == 'playback'
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
            self.model_positions[self.arm_indices] = state['positions']
            self.visual.update_cfg(self.model_positions)
            rows = ['| 关节 | 实际角度 ° | 已提交目标 ° |', '|---|---:|---:|']
            for i, (q, target) in enumerate(zip(state['positions'], state['target'])):
                rows.append(f'| J{i+1} | {np.rad2deg(q):.2f} | {np.rad2deg(target):.2f} |')
            age = state['feedback_age']
            if age is not None:
                rows.append(f'\n反馈年龄：{age * 1000:.0f} ms')
            self.feedback.content = '\n'.join(rows)

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
                self.disconnect()
            finally:
                self.server.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'config.yaml')
    parser.add_argument('--real', action='store_true', help='Enable real hardware connection button')
    parser.add_argument('--camera-device-index', type=int, default=0, help='Gemini camera index for RGB capture')
    args = parser.parse_args()
    if args.camera_device_index < 0:
        parser.error('--camera-device-index must be nonnegative')
    app = ControlApp(load_config(args.config), real=args.real, camera_device_index=args.camera_device_index)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: app.stop.set())
    app.run()


if __name__ == '__main__':
    main()
