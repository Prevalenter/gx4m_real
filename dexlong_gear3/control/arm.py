"""Minimal PVT driver, extracted from tracker_app.arm.RightArmDriver.

Hardware imports are lazy: constructing a driver or running mock mode never opens CAN.
"""
import threading
import time
from pathlib import Path

import numpy as np

from config import Config, validate_position
from trajectory import load_trajectory, save_trajectory


class SimulatedGX7:
    def __init__(self, q):
        self.q = q.copy()
        self.velocity = np.zeros(7)
        self.running = True
        self.last_feedback_at = time.monotonic()
        self.last_error = None
        self.released = False

    def getJP(self):
        return self.q.copy()

    def getJV(self):
        return self.velocity.copy()

    def setJPVTs(self, positions, velocities, torques):
        now = time.monotonic()
        dt = max(now - self.last_feedback_at, 1e-6)
        step = np.clip(np.asarray(positions) - self.q,
                       -np.asarray(velocities) * dt, np.asarray(velocities) * dt)
        if self.released:
            step[:] = 0
        self.q += step
        self.velocity = step / dt
        self.last_feedback_at = now

    def run(self):
        self.running = True

    def stop(self):
        self.running = False

    def set_released(self, released):
        self.released = released
        self.last_feedback_at = time.monotonic()


def hardware_factory(config):
    from libgx7 import GX7, VCICAN

    can = VCICAN(device_serial_number=config.can_device_serial,
                 response_timeout_sec=config.can_response_timeout_sec,
                 response_retries=config.can_response_retries,
                 retry_delay_sec=config.can_retry_delay_sec,
                 enforce_single_io_thread=True)
    robot = None
    try:
        can.init_can()
        robot = GX7(can, can_channel=config.can_channel, freq=config.frequency_hz,
                    control_mode='pvt', soft_limit=True, config=str(config.motor_config))
        robot.setup()
        return robot, can
    except Exception:
        try:
            if robot is not None:
                robot.stop()
                robot.disable()  # setup may have enabled motors before run() existed.
        finally:
            can.close()
        raise


class ArmDriver:
    def __init__(self, config: Config, *, mock=True, factory=None):
        self.config, self.mock = config, mock
        self._factory = factory or hardware_factory
        self.lock = threading.RLock()
        self.robot = self.can = None
        self.mode = 'disconnected'
        self.error = None
        self.q = config.initial_position
        self.requested = self.q.copy()
        self.recording = None
        self.last_recording = None
        self.playback = None
        self.playback_progress = 0.

    def connect(self):
        with self.lock:
            if self.robot is not None:
                self._require_healthy()
                return self.q.copy()
            robot = can = None
            try:
                if self.mock:
                    robot = SimulatedGX7(self.q)
                else:
                    robot, can = self._factory(self.config)
                q = validate_position(self.config, robot.getJP())
                # Seed measured hold BEFORE starting the SDK's command thread.
                self._send(robot, q)
                robot.run()
                self.robot, self.can = robot, can
                self._require_healthy()
                self.q = self.requested = q.copy()
                self.mode, self.error = 'hold', None
                return q.copy()
            except Exception as exc:
                try:
                    if robot is not None:
                        robot.stop()
                        if not self.mock:
                            robot.disable()
                finally:
                    self.robot = self.can = None
                    self.mode, self.error = 'disconnected', str(exc)
                    if can is not None:
                        can.close()
                raise

    def command(self, values):
        with self.lock:
            q = validate_position(self.config, values)
            self._require_healthy()
            if self.mode in ('recording', 'released'):
                raise RuntimeError('请先停止录制并点击“保持当前位置”恢复出力')
            self.playback = None
            self.requested = q
            self.mode = 'manual'

    def hold(self):
        with self.lock:
            self._require_healthy()
            if self.recording is not None:
                raise RuntimeError('请先停止并保存录制')
            self.playback = None
            self.q = validate_position(self.config, self.robot.getJP())
            self.requested = self.q.copy()
            self._send(self.robot, self.q)
            if self.mode == 'released':
                try:
                    self.robot.set_released(False)
                except Exception as exc:
                    self._fault(exc)
                    raise
                self.q = validate_position(self.config, self.robot.getJP())
                self.requested = self.q.copy()
            self.mode = 'hold'
            return self.q.copy()

    def start_recording(self, path):
        with self.lock:
            self._require_healthy()
            if self.recording is not None:
                raise RuntimeError('正在录制')
            path = Path(path).expanduser().resolve()
            if path.exists():
                raise ValueError('录制文件已存在，请更换文件名')
            self.playback = None
            try:
                self.robot.set_released(True)
                self.q = validate_position(self.config, self.robot.getJP())
            except Exception as exc:
                self._fault(exc)
                raise
            self.requested = self.q.copy()
            self.recording = {'path': path, 'start': time.monotonic(),
                              'times': [0.], 'positions': [self.q.copy()]}
            self.mode = 'recording'

    def stop_recording(self):
        with self.lock:
            if self.recording is None:
                raise RuntimeError('当前没有录制')
            record = self.recording
            if self.mode == 'recording':
                self._require_healthy()
                q = validate_position(self.config, self.robot.getJP())
                elapsed = time.monotonic() - record['start']
                if elapsed > record['times'][-1]:
                    record['times'].append(elapsed)
                    record['positions'].append(q)
            path = save_trajectory(record['path'], record['times'], record['positions'])
            self.last_recording = str(path)
            self.recording = None
            if self.mode == 'recording':
                self.mode = 'released'
            return path

    def play(self, path, speed=1.):
        with self.lock:
            self._require_healthy()
            if self.recording is not None:
                raise RuntimeError('请先停止并保存录制')
            if not np.isfinite(speed) or not 0 < speed <= 2:
                raise ValueError('回放倍率必须在 (0, 2] 范围内')
            times, positions = load_trajectory(self.config, path)
            peak = float(np.max(np.abs(np.diff(positions, axis=0)) / np.diff(times)[:, None]))
            speed = min(speed, self.config.velocity_limit / peak) if peak else speed
            self.hold()
            self.playback = {'times': times, 'positions': positions, 'speed': speed,
                             'time': 0., 'last': time.monotonic(), 'at_start': False}
            self.playback_progress = 0.
            self.requested = positions[0].copy()
            self.mode = 'playback'

    def _advance_playback(self, now):
        play = self.playback
        dt = min(max(0., now - play['last']), .1)
        play['last'] = now
        if not play['at_start']:
            if np.max(np.abs(self.q - play['positions'][0])) > .01:
                return
            play['at_start'] = True
            dt = 0.
        # Pause the clock when measured motion falls behind its reference.
        if np.max(np.abs(self.q - self.requested)) <= .03:
            play['time'] = min(play['time'] + dt * play['speed'], play['times'][-1])
        self.requested = np.array([np.interp(play['time'], play['times'], play['positions'][:, i])
                                   for i in range(7)])
        self.playback_progress = float(play['time'] / play['times'][-1])
        if play['time'] >= play['times'][-1] and np.max(np.abs(self.q - self.requested)) <= .01:
            self.playback = None
            self.mode = 'hold'

    def update(self):
        with self.lock:
            if self.robot is None or self.mode == 'fault':
                return
            try:
                self._require_healthy()
                self.q = validate_position(self.config, self.robot.getJP())
                if self.mode in ('recording', 'released'):
                    if self.mock:
                        self.robot.last_feedback_at = time.monotonic()
                    self.requested = self.q.copy()
                    if self.recording is not None:
                        elapsed = time.monotonic() - self.recording['start']
                        if elapsed > self.recording['times'][-1]:
                            self.recording['times'].append(elapsed)
                            self.recording['positions'].append(self.q.copy())
                    return
                if self.playback is not None:
                    self._advance_playback(time.monotonic())
                limited = self.q + np.clip(self.requested - self.q,
                                          -self.config.max_joint_step_rad, self.config.max_joint_step_rad)
                self._send(self.robot, limited)
                self.q = validate_position(self.config, self.robot.getJP())
            except Exception as exc:
                self._fault(exc)

    def snapshot(self):
        with self.lock:
            age = None
            if self.robot is not None:
                age = max(0., time.monotonic() - self.robot.last_feedback_at)
            return {'mode': self.mode, 'mock': self.mock, 'error': self.error,
                    'connected': self.robot is not None, 'positions': self.q.copy(),
                    'target': self.requested.copy(), 'feedback_age': age,
                    'recording_frames': len(self.recording['times']) if self.recording else 0,
                    'recording_seconds': self.recording['times'][-1] if self.recording else 0.,
                    'recording': self.recording is not None,
                    'last_recording': self.last_recording,
                    'playback_progress': self.playback_progress}

    def feedback_sample(self):
        """Read cached SDK feedback; never perform CAN I/O on the caller thread."""
        with self.lock:
            self._require_healthy()
            if self.mock:
                timestamp, q = time.monotonic(), self.robot.getJP()
            else:
                timestamp, q = self.robot.get_joint_sample()
            q = validate_position(self.config, q)
            play = self.playback
            return {'timestamp_monotonic': float(timestamp), 'positions_rad': q.tolist(),
                    'target_rad': self.requested.tolist(), 'mode': self.mode,
                    'trajectory_time_sec': float(play['time']) if play else None,
                    'playback_phase': ('trajectory' if play['at_start'] else 'return_to_start')
                    if play else self.mode}

    def disconnect(self):
        with self.lock:
            robot, can = self.robot, self.can
            self.robot = self.can = None
            self.mode = 'disconnected'
            self.requested = self.q.copy()
            self.playback = None
            try:
                if robot is not None:
                    robot.stop()  # SDK's loop finally block disables motors.
            finally:
                if can is not None:
                    can.close()
                if self.recording is not None:
                    self.stop_recording()

    def _require_healthy(self):
        if self.robot is None:
            raise RuntimeError('请先连接机械臂')
        if self.mode == 'fault':
            raise RuntimeError('机械臂已故障，请断开后重新连接')
        age = time.monotonic() - self.robot.last_feedback_at
        if not self.robot.running or (not self.mock and age > self.config.feedback_stale_after_sec):
            exc = RuntimeError(f'机械臂反馈异常：{self.robot.last_error or f"反馈年龄 {age:.3f}s"}')
            self._fault(exc)
            raise exc

    def _fault(self, exc):
        self.mode, self.error = 'fault', str(exc)
        self.playback = None
        if self.robot is not None and self.robot.running:
            self.robot.stop()
        if self.recording is not None:
            try:
                self.stop_recording()
            except Exception as save_error:
                self.error += f'；录制保存失败：{save_error}'

    def _send(self, robot, q):
        q = validate_position(self.config, q)
        robot.setJPVTs(q.tolist(), [self.config.velocity_limit] * 7, [self.config.torque_limit] * 7)
