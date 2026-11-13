"""30 Hz synchronized arm/hand trajectory viewer and optional hardware replay."""
from pathlib import Path
from collections import deque
import argparse
import dataclasses
import json
import queue
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gx4m_real.robot.config import ARM_JOINTS, load_config, validate_position
from gx4m_real.robot.hand import HAND_JOINTS, LOWER, UPPER, HandDriver
from gx4m_real.robot.arm import ArmDriver
from gx4m_real.robot.compensation import JointCompensation
from tasks.beam_lift_replay.beam_view import BeamView
from tasks.beam_lift_replay.recording import ReplayRecorder
from tasks.dexlong_gear3.arm_setup import (CONTROL_CONFIG, DEFAULT_HANDEYE, HAND_START_TOLERANCE,
                                          arm_config_for_reference)

ARM_START_TOLERANCE = .08
REFERENCE = ROOT/'tasks/dexlong_gear3/config/beam1_reference.json'


class CommandRate:
    def __init__(self, window=1.):
        self.window = window
        self.samples = deque()

    def mark(self, now=None):
        now = time.monotonic() if now is None else now
        self.samples.append(now)
        self.prune(now)

    def prune(self, now):
        while self.samples and now - self.samples[0] > self.window:
            self.samples.popleft()

    def hz(self, now=None):
        now = time.monotonic() if now is None else now
        self.prune(now)
        if len(self.samples) < 2:
            return 0.
        elapsed = now - self.samples[0]
        return (len(self.samples) - 1) / elapsed if elapsed > 0 else 0.


def replay_config(path, handeye=None, reference=None, velocity=None):
    config = (load_config(path) if reference is None else
              arm_config_for_reference(path, reference, handeye, velocity))
    if handeye is None:
        return config
    recorded = json.loads(Path(handeye).read_text(encoding='utf-8')).get('joint_compensation')
    if not isinstance(recorded, dict) or not all(key in recorded for key in ('directions', 'offsets_deg')):
        raise ValueError('Hand-eye result must contain joint_compensation directions and offsets_deg')
    compensation = JointCompensation(tuple(recorded['directions']), tuple(recorded['offsets_deg']))
    print(f'Joint compensation from {handeye}: directions {list(compensation.directions)}, '
          f'offsets {list(compensation.offsets_deg)} deg', flush=True)
    return dataclasses.replace(config, compensation=compensation, compensation_path=Path(handeye))


def load_targets(path, config):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    names = list(ARM_JOINTS + HAND_JOINTS)
    t = np.asarray(data['times_sec'], dtype=float)
    q = np.asarray(data['positions_rad'], dtype=float)
    if data.get('version') != 1 or data.get('units') != 'rad' or data.get('joint_names') != names:
        raise ValueError('Expected version 1, radians, joint1..7 then gx4m_joint1..16')
    if t.ndim != 1 or len(t) < 2 or t[0] != 0 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Invalid timestamps')
    if q.shape != (len(t), 23) or not np.isfinite(q).all():
        raise ValueError('Expected finite N x 23 positions')
    for row in q:
        validate_position(config, config.compensation.to_motor(row[:7]))
    clipped = np.clip(q[:, 7:], LOWER, UPPER)
    excess = float(np.max(np.abs(clipped-q[:, 7:])))
    if excess > .011:
        raise ValueError('Hand trajectory exceeds hardware limits by more than joint-limit DR (0.01 rad)')
    data['hand_limit_clip_max_rad'] = excess
    q[:, 7:] = clipped
    # Resample the 120 Hz source at 30 Hz, preserving the final endpoint.
    times = np.arange(int(np.ceil(t[-1] * 30)) + 1) / 30
    source_times = np.minimum(times, t[-1])
    positions = np.stack([np.interp(source_times, t, q[:, j]) for j in range(23)], axis=1)
    return data, times, positions


class ReplayApp:
    def __init__(self, args):
        import viser
        import yourdfpy
        from viser.extras import ViserUrdf
        self.args = args
        reference = json.loads(Path(args.reference).read_text(encoding='utf-8'))
        self.config = replay_config(args.config, args.handeye, reference, args.arm_velocity)
        self.data, self.times, self.q = load_targets(args.trajectory, self.config)
        self.arm = self.hand = None
        self.mode = 'idle'
        self.play_hz = 30
        self.index = 0
        self.events = queue.Queue()
        self.closed = False
        self.last_target = self.q[0].copy()
        self.arm_rate, self.hand_rate = CommandRate(), CommandRate()
        self.rates_shown = -float('inf')
        self.recorder = (ReplayRecorder(args.record_dir, args, self.config, self.times, self.q)
                         if args.record_dir is not None else None)
        self.tail_deadline = None
        self.server = viser.ViserServer(host=args.host, port=args.port)
        self.server.scene.set_up_direction('+z')
        self.server.scene.add_grid('/ground', plane='xy', width=2, height=2)
        self.server.scene.add_frame('/robot', position=self.config.base_position,
                                    wxyz=self.config.base_wxyz, show_axes=False)
        self.model = yourdfpy.URDF.load(str(self.config.urdf))
        self.visual = ViserUrdf(self.server, self.model, root_node_name='/robot')
        self.names = list(self.model.actuated_joint_names)
        self.mapping = dict(zip(self.data['joint_names'], range(23)))
        self.gui = self.server.gui
        self.gui.add_markdown('## Beam lift · 30 / 10 Hz\n' +
                             ('**Real hardware mode**' if args.real else '**Offline preview (no hardware)**') +
                             '\n\n23 joint targets, radians. Robot shows commanded positions. '
                             'Connect → Move to start → Play. Stop holds measured positions. '
                             f"Hand limit clipping: {self.data['hand_limit_clip_max_rad']:.6f} rad.")
        offsets = ', '.join(f'J{index + 1} {value:+.4f}°'
                            for index, value in enumerate(self.config.compensation.offsets_deg))
        self.gui.add_markdown(f'**关节补偿来源**：{args.handeye or self.config.compensation_path}\n\n'
                              f'**零点偏移**：{offsets}\n\n'
                              '模型角用于显示，发送电机目标时应用方向和零点补偿；不会重复叠加偏移。')
        self.status = self.gui.add_text('Status', initial_value='Ready', disabled=True)
        self.record_status = self.gui.add_text('采集进度', initial_value=(
            self.recorder.status if self.recorder else '未启用；使用 --record-dir 开启'), disabled=True)
        self.frame = self.gui.add_slider('Frame', min=0, max=len(self.q)-1, step=1, initial_value=0)
        self.clock = self.gui.add_text('Trajectory time', initial_value='0.000 s', disabled=True)
        with self.gui.add_folder('实时控制频率'):
            self.gui.add_markdown('最近 1 秒成功下发的轨迹目标频率，每 0.25 秒刷新，目标由重放按钮选择。'
                                  '10 Hz 逐帧慢放，耗时约为 30 Hz 的三倍，不跳帧。'
                                  '不是电机内部伺服频率或 CAN 总线帧率；保持阶段无新轨迹目标时显示 0 Hz，'
                                  '不表示电机已失能。离线模拟会明确标注。')
            self.arm_hz = self.gui.add_text('机械臂目标下发', initial_value='未连接', disabled=True)
            self.hand_hz = self.gui.add_text('灵巧手目标下发', initial_value='未连接', disabled=True)
            self.target_hz = self.gui.add_text('重放目标频率', initial_value='30 Hz', disabled=True)
        self.beam = BeamView(self.server, self.data, args.trajectory, self.config,
                             handeye=args.handeye, pose_port=args.pose_port)
        for label, action in [('Connect hardware' if args.real else 'Connect mock robot', 'connect'),
                              ('Move to start (0.2 rad/s)', 'home'), ('Play once · 30 Hz', 'play'),
                              ('Play once · 10 Hz', 'play10'),
                              ('Stop / hold', 'stop')]:
            button = self.gui.add_button(label)
            button.on_click(lambda _, action=action: self.events.put(action))
        self.frame.on_update(lambda _: self.events.put('seek'))
        self.draw(self.q[0])

    def draw(self, q):
        values = [q[self.mapping[n]] if n in self.mapping else self.config.initial_joints[n] for n in self.names]
        self.visual.update_cfg(np.array(values))
        self.clock.value = f'{self.times[self.index]:.3f} s'

    def at_start(self, q):
        """Measured q (23,) at the first target: arm within 0.08 rad, fingers within HAND_START_TOLERANCE
        (current-based position control stops a little short of the goal, as in run_step1.py)."""
        error = np.abs(q-self.q[0])
        return error[:7].max() <= ARM_START_TOLERANCE and error[7:].max() <= HAND_START_TOLERANCE

    def measured(self):
        self.arm.update()
        state = self.arm.snapshot()
        if state['mode'] == 'fault':
            raise RuntimeError(state['error'])
        return np.r_[self.config.compensation.to_model(state['positions']), self.hand.read()]

    def send(self, q):
        self.arm.command(self.config.compensation.to_motor(q[:7]))
        self.hand.command(q[7:])
        self.hand_rate.mark()
        self.arm.update()
        state = self.arm.snapshot()
        if state['mode'] == 'fault':
            raise RuntimeError(state['error'])
        self.arm_rate.mark()
        self.last_target = q.copy()

    def refresh_rates(self, now=None):
        now = time.monotonic() if now is None else now
        if now - self.rates_shown < .25:
            return
        self.rates_shown = now
        for device, counter, field in ((self.arm, self.arm_rate, self.arm_hz),
                                       (self.hand, self.hand_rate, self.hand_hz)):
            if device is None:
                text = '未连接（离线预览）' if not self.args.real else '未连接'
            else:
                prefix = '实机' if self.args.real else '模拟'
                text = f'{prefix} · {counter.hz(now):.1f} Hz'
            if field.value != text:
                field.value = text

    def hold(self):
        self.mode = 'idle'
        self.frame.disabled = False
        if self.arm is not None:
            # Try each device independently, including after communication errors.
            errors = []
            for fn in (self.arm.hold, lambda: self.hand.command(self.hand.read())):
                try:
                    fn()
                except Exception as exc:
                    errors.append(str(exc))
            if errors:
                raise RuntimeError('; '.join(errors))

    def finish_recording(self, outcome):
        if self.recorder is not None:
            self.recorder.finish(outcome)
            self.record_status.value = self.recorder.status

    def action(self, action):
        if action == 'stop':
            try:
                self.hold()
            finally:
                self.finish_recording('stopped')
            self.status.value = 'Stopped; holding measured positions'
        elif action == 'connect':
            if self.arm is not None:
                return
            arm = ArmDriver(self.config, mock=not self.args.real)
            # Current limit and goal current as the policy runner's HandDriver (defaults 1000 / 600 mA).
            hand = HandDriver(port=self.args.hand_port, serial_number=self.args.hand_serial,
                              mock=not self.args.real, curr_limit=max(1000, self.args.hand_current),
                              goal_current=self.args.hand_current)
            try:
                arm.connect()
                hand.connect()
            except BaseException:
                if arm.robot is not None:
                    arm.disconnect()
                hand.close()
                raise
            self.arm, self.hand = arm, hand
            self.arm_rate, self.hand_rate = CommandRate(), CommandRate()
            self.status.value = 'Connected; holding current pose'
        elif action == 'home':
            if self.mode != 'idle':
                raise RuntimeError('Stop the current operation first')
            if self.arm is None:
                raise RuntimeError('Connect first')
            self.home_q = self.measured()
            self.home_deadline = time.monotonic() + float(np.max(np.abs(self.home_q-self.q[0])))/.2 + 10
            self.mode = 'home'
            self.index = 0
            self.frame.disabled = True
            self.status.value = 'Moving to first target at up to 0.2 rad/s'
        elif action in ('play', 'play10'):
            if self.mode != 'idle':
                raise RuntimeError('Stop the current operation first')
            if self.args.real and self.arm is None:
                raise RuntimeError('Connect hardware first')
            if self.recorder is not None and self.arm is None:
                raise RuntimeError('Connect hardware or mock robot before recording')
            if self.arm is not None:
                q = self.measured()
                if not self.at_start(q):
                    raise RuntimeError('Robot is not at the first target; Move to start first')
            play_hz = 10 if action == 'play10' else 30
            if self.recorder is not None:
                if self.args.real and (self.beam.latest is None or
                                       time.monotonic() - self.beam.latest[0] > .2):
                    raise RuntimeError('Recording needs a fresh beam pose (within 200 ms); check perception')
                self.recorder.start(play_hz)
                self.last_target = self.q[0].copy()
                self.recorder.sample(self, 0.)
                self.record_status.value = self.recorder.status
            self.index = 0
            self.frame.disabled = True
            self.mode = 'play'
            self.play_hz = play_hz
            self.target_hz.value = f'{play_hz} Hz'
            self.status.value = f'Playing once at {play_hz} Hz'
        elif action == 'seek' and self.mode == 'idle':
            self.index = int(self.frame.value)
            self.draw(self.q[self.index])  # Preview only; never sends hardware commands.

    def tick(self):
        self.beam.update(self.times[self.index])
        while not self.events.empty():
            self.action(self.events.get_nowait())
        if self.mode == 'home':
            if time.monotonic() > self.home_deadline:
                raise RuntimeError('Move to start timed out; check tracking')
            self.home_q += np.clip(self.q[0]-self.home_q, -.2/30, .2/30)
            self.send(self.home_q)
            self.draw(self.home_q)
            if np.max(np.abs(self.home_q-self.q[0])) < 1e-6:
                if self.at_start(self.measured()):
                    self.mode = 'idle'
                    self.frame.disabled = False
                    self.status.value = 'At start; ready to play'
        elif self.mode == 'play':
            q = self.q[self.index]
            command_time = time.monotonic()
            if self.arm is not None:
                self.send(q)
            if self.recorder is not None:
                self.recorder.sample(self, self.times[self.index], command_time)
            self.draw(q)
            self.frame.value = self.index
            if self.index == len(self.q)-1:
                if self.recorder is not None:
                    self.mode = 'settle'
                    self.tail_deadline = time.monotonic() + self.args.record_tail
                    self.status.value = 'Recording final-target settling; holding final target'
                else:
                    self.mode = 'idle'
                    self.frame.disabled = False
                    self.status.value = 'Completed; holding final target'
            else:
                self.index += 1
        elif self.mode == 'settle':
            self.arm.update()
            if self.arm.snapshot()['mode'] == 'fault':
                raise RuntimeError(self.arm.snapshot()['error'])
            self.recorder.sample(self, self.times[-1])
            if time.monotonic() >= self.tail_deadline:
                self.mode = 'idle'
                self.frame.disabled = False
                self.finish_recording('completed')
                self.status.value = 'Saved; manually reset beam and Move to start before next trial'
        elif self.arm is not None:
            self.measured()

    def tick_period(self):
        return 1 / (self.play_hz if self.mode == 'play' else 30)

    def run(self):
        deadline = time.monotonic()
        try:
            while True:
                try:
                    self.tick()
                except Exception as exc:
                    self.mode = 'idle'
                    try:
                        self.hold()
                    except Exception as hold_error:
                        exc = RuntimeError(f'{exc}; hold: {hold_error}')
                    self.finish_recording('error: ' + str(exc))
                    self.status.value = f'Error: {exc}'
                self.refresh_rates()
                period = self.tick_period()
                deadline += period
                now = time.monotonic()
                if now > deadline:
                    # Do not burst commands or skip targets when hardware is late.
                    if now-deadline > period and self.mode == 'play':
                        self.status.value = (f'Playing; hardware loop late, replay is running slower than '
                                             f'{self.play_hz} Hz')
                    deadline = now
                time.sleep(max(0., deadline-time.monotonic()))
        except KeyboardInterrupt:
            try:
                self.hold()
            finally:
                self.finish_recording('interrupted')
                self.beam.close()
                self.server.stop()
                # Keep arm enabled and held on exit, as in the existing robot server.
                # Hardware is disabled only through the user's existing control tools.


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trajectory', type=Path, default=Path(__file__).parent/'trajectory/joint_targets.json')
    p.add_argument('--pose-port', type=int, default=5711, help='Local UDP port for live beam tracking (display only)')
    p.add_argument('--config', type=Path, default=CONTROL_CONFIG)
    p.add_argument('--reference', type=Path, default=REFERENCE,
                   help='Policy reference whose trained arm PVT speed limit is used')
    p.add_argument('--handeye', type=Path, default=DEFAULT_HANDEYE,
                   help='Hand-eye result (default: he_20261109_contact_plate, J1 +11.1842 deg), as run_step1.py')
    p.add_argument('--arm-velocity', type=float, help='PVT speed limit, rad/s (default: the trained limit)')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8090)
    p.add_argument('--real', action='store_true')
    p.add_argument('--hand-port')
    p.add_argument('--hand-serial')
    p.add_argument('--hand-current', type=int, default=600,
                   help='Hand goal current, mA (current limit max(1000, this)); default as run_step1.py')
    p.add_argument('--record-dir', type=Path, help='Create a timestamped manually triggered acquisition session')
    p.add_argument('--record-trials', type=int, default=10, help='Maximum completed trials per session')
    p.add_argument('--record-tail', type=float, default=1., help='Record final-target settling for this many seconds')
    args = p.parse_args(argv)
    if args.real and args.handeye is None:
        p.error('--handeye is required in real hardware mode; use the same calibration as step1 lift')
    if args.record_trials < 1 or not np.isfinite(args.record_tail) or args.record_tail < 0:
        p.error('--record-trials must be positive and --record-tail must be finite and nonnegative')
    return args


def main():
    ReplayApp(parse_args()).run()


if __name__ == '__main__':
    main()
