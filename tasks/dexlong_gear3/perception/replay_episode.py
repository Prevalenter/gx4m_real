"""Offline Viser replay of recorded episodes; never connects to robot hardware or cameras."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from viser_scene import DEFAULT_INITIAL, SceneViewer, initial_base, load_calibration, reference_urdf

TASK_DIR = Path(__file__).resolve().parents[1]
OUTCOMES = {'stopped': '人工停止', 'drop': '掉落终止', 'success': '成功', 'timeout': '超时',
            'pose_lost': '位姿丢失', 'tracking_error': '关节跟踪误差', 'tactile_lost': '触觉丢失'}


def find_episodes(paths):
    episodes = set()
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if path.is_file() and path.suffix == '.npz':
            episodes.add(path)
        elif path.is_dir():
            episodes.update(path.rglob('episode_*.npz'))
        else:
            raise ValueError(f'找不到回放文件或目录：{path}')
    if not episodes:
        raise ValueError('没有找到 episode_*.npz 回合记录')
    return sorted(episodes)


def axis_quaternion(axes):
    axes = np.asarray(axes, dtype=float)
    if axes.shape not in ((3,), (6,)) or not np.isfinite(axes).all():
        return None
    normal = axes[-3:]
    if np.linalg.norm(normal) < 1e-8:
        return None
    normal = normal / np.linalg.norm(normal)
    if axes.size == 6:
        tangent = axes[:3] - np.dot(axes[:3], normal) * normal
        if np.linalg.norm(tangent) < 1e-8:
            return None
        tangent /= np.linalg.norm(tangent)
        rotation = Rotation.from_matrix(np.stack((tangent, np.cross(normal, tangent), normal), axis=1))
        return rotation.as_quat()[[3, 0, 1, 2]]
    quaternion = np.array([1. + normal[2], -normal[1], normal[0], 0.])
    if quaternion[0] < 1e-9:
        return np.array([0., 1., 0., 0.])
    return quaternion / np.linalg.norm(quaternion)


class Episode:
    def __init__(self, path, joint_count):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as archive:
            self.data = {name: archive[name] for name in archive.files}
        for name in ('t', 'q', 'object_pos', 'object_quat'):
            if name not in self.data:
                raise ValueError(f'{self.path.name} 缺少 {name}')
        self.times = np.asarray(self.data['t'], dtype=float)
        if self.times.ndim != 1 or not self.times.size or not np.isfinite(self.times).all():
            raise ValueError(f'{self.path.name} 没有有效采样时间')
        if np.any(np.diff(self.times) < 0):
            raise ValueError(f'{self.path.name} 采样时间不递增')
        for name, width in (('q', joint_count), ('object_pos', 3), ('object_quat', 4)):
            values = self.data[name]
            if values.shape != (len(self.times), width) or not np.isfinite(values).all():
                raise ValueError(f'{self.path.name} 的 {name} 形状或数值无效')
        if np.any(np.linalg.norm(self.data['object_quat'], axis=1) < 1e-8):
            raise ValueError(f'{self.path.name} 包含无效物体四元数')
        self.elapsed = self.times - self.times[0]
        self.duration = float(self.elapsed[-1])
        self.outcome = str(self.data.get('outcome', np.array('unknown')).item())

    def index_at(self, elapsed):
        return int(np.clip(np.searchsorted(self.elapsed, elapsed, side='right') - 1, 0, len(self.times) - 1))

    def estimate(self, index):
        positions, axes = self.data.get('estimated_pos'), self.data.get('estimated_axis')
        if positions is None or axes is None or index >= len(positions) or index >= len(axes):
            return None
        position = positions[index]
        quaternion = axis_quaternion(axes[index])
        if position.shape != (3,) or not np.isfinite(position).all() or quaternion is None:
            return None
        return {'position': position.tolist(), 'wxyz': quaternion.tolist()}


class EpisodeVideo:
    def __init__(self, episode):
        self.capture = None
        self.times = None
        self.index = -1
        self.rgb = None
        self.note = '没有配套视频'
        path = episode.path.with_suffix('.avi')
        if not path.is_file():
            return
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            self.close()
            self.note = '视频无法打开'
            return
        count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        metadata = episode.path.with_name(episode.path.stem + '_video.json')
        try:
            times = np.asarray(json.loads(metadata.read_text())['timestamp_monotonic'], dtype=float)
            if times.ndim != 1 or len(times) != count or not len(times) or not np.isfinite(times).all():
                raise ValueError('Invalid video timestamps')
            if np.any(np.diff(times) < 0):
                raise ValueError('Unordered video timestamps')
            self.times = times
            self.note = '按相机单调时间戳同步'
        except (OSError, ValueError, KeyError, TypeError):
            self.times = episode.times[0] + np.arange(count) / (fps if fps > 0 else 30.)
            self.note = '缺少有效视频时间戳，按帧率估算同步'

    def frame(self, timestamp):
        if self.capture is None or self.times is None or not len(self.times):
            return None
        index = int(np.searchsorted(self.times, timestamp, side='right') - 1)
        if index < 0:
            return None
        index = min(index, len(self.times) - 1)
        if self.rgb is not None and index == self.index:
            return self.rgb
        if index != self.index + 1:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.capture.read()
        if not ok:
            self.note = '视频帧读取失败'
            return None
        self.index, self.rgb = index, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.rgb

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class ReplayViewer:
    def __init__(self, paths, reference, initial, *, calibration=None, host='127.0.0.1', port=8089):
        self.paths, self.skipped, self.reference = [], [], reference
        for path in find_episodes(paths):
            try:
                Episode(path, len(reference['joint_order']))
            except (OSError, ValueError, KeyError) as error:
                self.skipped.append((path, str(error)))
            else:
                self.paths.append(path)
        if not self.paths:
            raise ValueError('所选记录均为空或无效，没有可回放的数据')
        self.lock = threading.RLock()
        self.playing, self.sample = False, -1
        self.video = None
        self.episode = None
        calibrated = calibration is not None
        calibration = calibration or {'T_world_base': initial_base(initial), 'T_base_cam': np.eye(4)}
        self.viewer = SceneViewer(calibration, initial=initial, urdf=reference_urdf(reference), gear='beam2',
                                  host=host, port=port, live=False, stale_after=float('inf'))
        self.viewer.intro.content = ('## 离线回放 · 不连接真机\n'
                                     '机器人、原色横梁和紫色估计模型均来自记录。视频与状态使用保存的时间戳同步。')
        self.viewer.configure_client_camera = self.configure_camera
        self.viewer.image.label = '录制的相机视频（非实时）'
        if not calibrated:
            self.viewer.camera_frame.visible = False
        self.joint_indices = [reference['joint_order'].index(name) if name in reference['joint_order'] else None
                              for name in self.viewer.model.actuated_joint_names]
        self.base_q = self.viewer.q.copy()
        gui = self.viewer.server.gui
        labels = [f'{index + 1:02d} · {path.parent.name}/{path.stem}' for index, path in enumerate(self.paths)]
        with gui.add_folder('回放控制', order=-1):
            gui.add_markdown(f'已找到 **{len(self.paths)}** 个回合。不会发送控制指令，也不会加载策略或估计器权重。')
            if self.skipped:
                gui.add_markdown(f'已跳过 **{len(self.skipped)}** 份空记录或无效记录；详情见启动终端。')
                for path, reason in self.skipped:
                    print(f'Skipped {path}: {reason}', flush=True)
            self.selection = gui.add_dropdown('回合', labels, initial_value=labels[-1])
            self.play_button = gui.add_button('播放', color='green')
            self.previous = gui.add_button('上一帧')
            self.next = gui.add_button('下一帧')
            self.speed = gui.add_dropdown('速度', ('0.25×', '0.5×', '1×', '2×'), initial_value='1×')
            self.loop = gui.add_checkbox('循环当前回合', initial_value=False)
            self.slider = gui.add_slider('采样帧', min=0, max=1, step=1, initial_value=0)
            self.summary = gui.add_markdown('')
            self.metrics = gui.add_markdown('')
            self.notice = gui.add_markdown('未指定标定：相机坐标轴隐藏，记录的世界坐标不受影响。' if not calibrated else '')

        @self.selection.on_update
        def select(event):
            if event.client is not None:
                try:
                    self.load(labels.index(event.target.value))
                    self.notice.content = ''
                except (OSError, ValueError, KeyError) as error:
                    self.notice.content = f'加载失败：{error}'

        @self.play_button.on_click
        def play(_):
            with self.lock:
                self.set_playing(not self.playing)

        @self.previous.on_click
        def previous(_):
            self.seek(self.sample - 1)

        @self.next.on_click
        def next_frame(_):
            self.seek(self.sample + 1)

        @self.slider.on_update
        def scrub(event):
            if event.client is not None:
                self.seek(int(event.target.value))

        @self.speed.on_update
        def speed(event):
            if event.client is not None:
                with self.lock:
                    self.anchor_wall = time.monotonic()
                    self.anchor_time = float(self.episode.elapsed[self.sample])

        self.load(len(self.paths) - 1)

    def configure_camera(self, client):
        with self.lock:
            base = self.viewer.calibration['T_world_base'][:3, 3]
            target = (base + self.episode.data['object_pos'][0]) / 2
            client.camera.position = target + [1., -1.3, .9]
            client.camera.look_at = target

    def load(self, index):
        episode = Episode(self.paths[index], len(self.reference['joint_order']))
        video = EpisodeVideo(episode)
        with self.lock:
            if self.video is not None:
                self.video.close()
            self.episode, self.video = episode, video
            self.sample = 0
            self.set_playing(False)
            self.slider.value = 0
            self.slider.max = max(1, len(episode.times) - 1)
            self.slider.disabled = len(episode.times) == 1
            self.viewer.image.image = np.zeros((480, 640, 3), dtype=np.uint8)
            self.summary.content = (f'**{episode.path.parent.name}/{episode.path.name}**\n\n'
                                    f'结束原因：**{OUTCOMES.get(episode.outcome, episode.outcome)}** · '
                                    f'{len(episode.times)} 帧 · {episode.duration:.2f} s\n\n'
                                    '人工停止仅表示停止方式，并不代表任务成功。')
            self.render(0)

    def set_playing(self, playing):
        self.playing = bool(playing)
        if self.episode is not None:
            if self.playing and self.sample >= len(self.episode.times) - 1:
                self.render(0)
            self.anchor_time = float(self.episode.elapsed[max(0, self.sample)])
        self.anchor_wall = time.monotonic()
        self.play_button.label = '暂停' if self.playing else '播放'

    def seek(self, index):
        with self.lock:
            self.set_playing(False)
            self.render(int(np.clip(index, 0, len(self.episode.times) - 1)))

    def render(self, index):
        episode, viewer = self.episode, self.viewer
        self.sample = index
        self.slider.value = index
        with viewer.lock:
            joints = self.base_q.copy()
            for target, source in enumerate(self.joint_indices):
                if source is not None:
                    joints[target] = episode.data['q'][index, source]
            viewer.q = joints
            viewer.robot.update_cfg(joints)
            viewer.part.position = episode.data['object_pos'][index]
            quaternion = episode.data['object_quat'][index]
            viewer.part.wxyz = quaternion / np.linalg.norm(quaternion)
            viewer.part.visible = True
            estimate = episode.estimate(index)
            viewer.update_estimate(estimate, enabled='estimated_pos' in episode.data)
            viewer.status.value = '历史记录回放（非实时检测）'
            viewer.rate.value = f'{episode.elapsed[index]:.3f} / {episode.duration:.3f} s'
            rgb = self.video.frame(episode.times[index])
            viewer.image.image = rgb if rgb is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            details = [f'**采样帧**：{index + 1}/{len(episode.times)}', self.video.note]
            for name, label, scale, unit in (('pose_age', '追踪位姿年龄', 1000., 'ms'),
                                            ('tracking_error', '关节跟踪误差', 1., 'rad'),
                                            ('distance', '目标关键点误差', 1000., 'mm')):
                if name in episode.data:
                    value = float(episode.data[name][index])
                    if np.isfinite(value):
                        details.append(f'**{label}**：{value * scale:.2f} {unit}')
            if 'pose_valid' in episode.data:
                details.append('**相机追踪有效**：' + ('是' if episode.data['pose_valid'][index] else '否'))
            if estimate is not None:
                error = np.linalg.norm(np.asarray(estimate['position']) - episode.data['object_pos'][index])
                details.append(f'**估计/追踪位置差**：{error * 1000:.1f} mm（不是相对真值的误差）')
            self.metrics.content = '\n\n'.join(details)

    def tick(self, now=None):
        with self.lock:
            if not self.playing:
                return
            now = time.monotonic() if now is None else now
            elapsed = self.anchor_time + (now - self.anchor_wall) * float(self.speed.value[:-1])
            index = self.episode.index_at(elapsed)
            if index != self.sample:
                self.render(index)
            if elapsed >= self.episode.duration:
                self.set_playing(bool(self.loop.value))

    def close(self):
        with self.lock:
            self.playing = False
            if self.video is not None:
                self.video.close()
            self.viewer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='*', type=Path, help='Episode .npz files or run directories')
    parser.add_argument('--reference', type=Path, default=TASK_DIR/'config/beam1_reference.json')
    parser.add_argument('--calibration', type=Path, help='Optional recorded fixed-camera calibration')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8089)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    initial = json.loads(DEFAULT_INITIAL.read_text())
    initial.update(base_position=reference['base_pos'], base_wxyz=reference['base_quat'])
    initial['joint_positions'].update(zip(reference['joint_order'], reference['initial_joint_pos']))
    calibration = load_calibration(args.calibration, initial) if args.calibration else None
    paths = args.paths or sorted((TASK_DIR/'runs').glob('beam1_*'))
    viewer = ReplayViewer(paths, reference, initial, calibration=calibration, host=args.host, port=args.port)
    print(f'Offline replay: http://{args.host}:{viewer.viewer.server.get_port()} ({len(viewer.paths)} episodes)', flush=True)
    try:
        while not viewer.viewer.stop_event.wait(.02):
            viewer.tick()
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()


if __name__ == '__main__':
    main()
