"""RGB and measured-joint capture during playback; no camera opens on import."""
from bisect import bisect_left
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import shutil
import threading
import time
from zoneinfo import ZoneInfo

import numpy as np

from ..camera.orbbec import OrbbecColorCamera

from .config import ARM_JOINTS


class JointHistory:
    def __init__(self, max_gap=.05):
        self.max_gap = max_gap
        self.samples = deque(maxlen=2000)
        self.condition = threading.Condition()

    def append(self, sample):
        with self.condition:
            if self.samples and sample['timestamp_monotonic'] <= self.samples[-1]['timestamp_monotonic']:
                return False
            self.samples.append(sample)
            self.condition.notify_all()
            return True

    def align(self, timestamp, timeout=.15):
        with self.condition:
            self.condition.wait_for(lambda: self.samples and
                                    self.samples[-1]['timestamp_monotonic'] >= timestamp, timeout)
            samples = list(self.samples)
        stamps = [sample['timestamp_monotonic'] for sample in samples]
        i = bisect_left(stamps, timestamp)
        if not samples or i == len(samples) or (i == 0 and stamps[0] != timestamp):
            return {'alignment_valid': False, 'alignment_reason': 'no bracketing feedback',
                    'positions_rad': None}
        before, after = samples[max(0, i - 1)], samples[i]
        t0, t1 = before['timestamp_monotonic'], after['timestamp_monotonic']
        offsets = [timestamp - t0, t1 - timestamp]
        valid = max(offsets) <= self.max_gap
        alpha = (timestamp - t0) / (t1 - t0) if t1 > t0 else 0.
        q = ((1 - alpha) * np.asarray(before['positions_rad']) +
             alpha * np.asarray(after['positions_rad']))
        return {'alignment_valid': valid, 'alignment_reason': None if valid else 'feedback gap too large',
                'positions_rad': q.tolist() if valid else None,
                'joint_timestamps_monotonic': [t0, t1], 'joint_time_offsets_sec': offsets,
                'interpolation_alpha': alpha, 'mode': before['mode'],
                'playback_phase': before['playback_phase'],
                'trajectory_time_sec': before['trajectory_time_sec']}


class PlaybackRecorder:
    """Camera/disk and joint sampling run outside the arm command loop."""

    def __init__(self, joint_reader, *, camera_factory=OrbbecColorCamera, sample_hz=100, max_gap=.05):
        self.joint_reader = joint_reader
        self.camera_factory = camera_factory
        self.sample_hz = sample_hz
        self.history = JointHistory(max_gap)
        self.lock = threading.Lock()
        self.stop_frames = threading.Event()
        self.stop_samples = threading.Event()
        self.frame_thread = self.sample_thread = None
        self.camera = None
        self.path = None
        self.active = False
        self.error = None
        self.frames = self.aligned_frames = self.joint_samples = 0
        self.started_at = None

    def start(self, directory, trajectory_path, metadata):
        from PIL import Image  # Check the image dependency before moving the arm.
        if self.active or self.path is not None:
            raise RuntimeError('每次回放需要新的采集会话')
        self.path = Path(directory).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / 'rgb').mkdir()
        self.camera = self.camera_factory()
        try:
            self.camera.open()
            # Verify that actual images can be read before playback begins.
            self.camera.read()
            shutil.copyfile(trajectory_path, self.path / 'source_trajectory.json')
            self.started_at = time.monotonic()
            self.session = {
                'version': 1, 'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                'started_at_monotonic': self.started_at, 'camera': self.camera.metadata,
                'joint_names': list(ARM_JOINTS), 'joint_units': 'rad',
                'image_format': 'PNG, decoded RGB, no undistortion or overlay',
                'alignment': 'linear interpolation of measured joints at host RGB receipt time',
                'clock': 'host time.monotonic seconds; camera device microseconds stored separately',
                'max_joint_offset_sec': self.history.max_gap,
                'timing_limit': 'Software receipt alignment, not hardware exposure synchronization; '
                                'USB latency and sequential CAN joint reads remain.',
                **metadata,
            }
            (self.path / 'session.json').write_text(json.dumps(self.session, indent=2) + '\n')
            self.joint_log = (self.path / 'joints.jsonl').open('x')
            self.frame_log = (self.path / 'frames.jsonl').open('x')
            self.active = True
            self.sample_thread = threading.Thread(target=self._sample, name='calibration-joints', daemon=True)
            self.frame_thread = threading.Thread(target=self._capture, name='calibration-rgb', daemon=True)
            self.sample_thread.start()
            self.frame_thread.start()
        except Exception as exc:
            self.error = str(exc)
            self.stop('startup_failed')
            raise

    def _fail(self, exc):
        with self.lock:
            self.error = self.error or str(exc)
        self.stop_frames.set()

    def _sample(self):
        try:
            while not self.stop_samples.is_set():
                start = time.monotonic()
                sample = self.joint_reader()
                if self.history.append(sample):
                    self.joint_log.write(json.dumps(sample) + '\n')
                    self.joint_log.flush()
                    with self.lock:
                        self.joint_samples += 1
                self.stop_samples.wait(max(0, 1 / self.sample_hz - (time.monotonic() - start)))
        except Exception as exc:
            self._fail(exc)

    def _capture(self):
        from PIL import Image
        try:
            while not self.stop_frames.is_set():
                frame = self.camera.read(timeout=1.)
                if self.stop_frames.is_set():
                    break
                aligned = self.history.align(frame.received_at)
                filename = f'rgb/{self.frames:06d}.png'
                Image.fromarray(frame.rgb).save(self.path / filename)
                record = {'frame_index': self.frames, 'camera_sequence': frame.sequence,
                          'rgb_path': filename, 'timestamp_monotonic': frame.received_at,
                          'time_sec': frame.received_at - self.started_at,
                          'camera_device_timestamp_us': frame.device_timestamp_us,
                          'camera_system_timestamp_us': frame.system_timestamp_us, **aligned}
                self.frame_log.write(json.dumps(record) + '\n')
                self.frame_log.flush()
                with self.lock:
                    self.frames += 1
                    self.aligned_frames += int(aligned['alignment_valid'])
        except Exception as exc:
            self._fail(exc)
        finally:
            try:
                self.camera.close()
            except Exception as exc:
                self._fail(exc)

    def snapshot(self):
        with self.lock:
            return {'active': self.active, 'error': self.error, 'frames': self.frames,
                    'aligned_frames': self.aligned_frames, 'joint_samples': self.joint_samples,
                    'path': str(self.path) if self.path else None}

    def stop(self, reason='stopped'):
        self.stop_frames.set()
        # Keep joint sampling alive until the final image has its right-hand sample.
        if self.frame_thread is not None:
            self.frame_thread.join(timeout=4.)
            if self.frame_thread.is_alive():
                raise RuntimeError('RGB 采集线程尚未退出')
        elif self.camera is not None:
            try:
                self.camera.close()
            except Exception as exc:
                self._fail(exc)
        self.stop_samples.set()
        if self.sample_thread is not None:
            self.sample_thread.join(timeout=2.)
            if self.sample_thread.is_alive():
                raise RuntimeError('关节采样线程尚未退出')
        for name in ('frame_log', 'joint_log'):
            stream = getattr(self, name, None)
            if stream is not None and not stream.closed:
                stream.close()
        with self.lock:
            self.active = False
        if self.path is not None:
            summary = {**self.snapshot(), 'stop_reason': reason,
                       'finished_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}
            (self.path / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        return self.path
