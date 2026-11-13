"""30 Hz synchronized arm/hand trajectory viewer and optional hardware replay."""
from pathlib import Path
import argparse
import json
import queue
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gx4m_real.robot.config import ARM_JOINTS, validate_position
from gx4m_real.robot.hand import HAND_JOINTS, LOWER, UPPER, HandDriver
from gx4m_real.robot.arm import ArmDriver
from tasks.dexlong_gear3.arm_setup import CONTROL_CONFIG, HAND_START_TOLERANCE, arm_config_for_reference

ARM_START_TOLERANCE = .08   # rad
REFERENCE = ROOT/'tasks/dexlong_gear3/config/beam1_reference.json'


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
        # Same arm setup as the policy runner (run_step1.py): the trained PVT speed limit and, with --handeye,
        # the joint compensation recorded with that hand-eye calibration.
        reference = json.loads(Path(args.reference).read_text(encoding='utf-8'))
        self.config = arm_config_for_reference(args.config, reference, args.handeye, args.arm_velocity)
        self.data, self.times, self.q = load_targets(args.trajectory, self.config)
        self.arm = self.hand = None
        self.mode = 'idle'
        self.index = 0
        self.events = queue.Queue()
        self.closed = False
        self.last_target = self.q[0].copy()
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
        self.gui.add_markdown('## Beam lift · 30 Hz\n' +
                             ('**Real hardware mode**' if args.real else '**Offline preview (no hardware)**') +
                             '\n\n23 joint targets, radians. Robot shows commanded positions. '
                             'Connect → Move to start → Play. Stop holds measured positions. '
                             f"Hand limit clipping: {self.data['hand_limit_clip_max_rad']:.6f} rad.")
        self.status = self.gui.add_text('Status', initial_value='Ready', disabled=True)
        self.frame = self.gui.add_slider('Frame', min=0, max=len(self.q)-1, step=1, initial_value=0)
        self.clock = self.gui.add_text('Trajectory time', initial_value='0.000 s', disabled=True)
        for label, action in [('Connect hardware' if args.real else 'Connect mock robot', 'connect'),
                              ('Move to start (0.2 rad/s)', 'home'), ('Play once · 30 Hz', 'play'),
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
        self.arm.update()
        state = self.arm.snapshot()
        if state['mode'] == 'fault':
            raise RuntimeError(state['error'])
        self.last_target = q.copy()

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

    def action(self, action):
        if action == 'stop':
            self.hold()
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
        elif action == 'play':
            if self.mode != 'idle':
                raise RuntimeError('Stop the current operation first')
            if self.args.real and self.arm is None:
                raise RuntimeError('Connect hardware first')
            if self.arm is not None:
                q = self.measured()
                if not self.at_start(q):
                    raise RuntimeError('Robot is not at the first target; Move to start first')
            self.index = 0
            self.frame.disabled = True
            self.mode = 'play'
            self.status.value = 'Playing once at 30 Hz'
        elif action == 'seek' and self.mode == 'idle':
            self.index = int(self.frame.value)
            self.draw(self.q[self.index])  # Preview only; never sends hardware commands.

    def tick(self):
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
            if self.arm is not None:
                self.send(q)
            self.draw(q)
            self.frame.value = self.index
            if self.index == len(self.q)-1:
                self.mode = 'idle'
                self.frame.disabled = False
                self.status.value = 'Completed; holding final target'
            else:
                self.index += 1
        elif self.arm is not None:
            self.measured()

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
                    self.status.value = f'Error: {exc}'
                deadline += 1/30
                now = time.monotonic()
                if now > deadline:
                    # Do not burst commands or skip targets when hardware is late.
                    if now-deadline > 1/30 and self.mode == 'play':
                        self.status.value = 'Playing; hardware loop late, replay is running slower than 30 Hz'
                    deadline = now
                time.sleep(max(0., deadline-time.monotonic()))
        except KeyboardInterrupt:
            try:
                self.hold()
            finally:
                self.server.stop()
                # Keep arm enabled and held on exit, as in the existing robot server.
                # Hardware is disabled only through the user's existing control tools.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trajectory', type=Path, default=Path(__file__).parent/'trajectory/joint_targets.json')
    p.add_argument('--config', type=Path, default=CONTROL_CONFIG)
    p.add_argument('--reference', type=Path, default=REFERENCE,
                   help='Policy reference whose trained arm PVT speed limit is used')
    p.add_argument('--handeye', type=Path,
                   help='handeye_result.json: its recorded joint compensation, as run_step1.py (required with --real)')
    p.add_argument('--arm-velocity', type=float, help='PVT speed limit, rad/s (default: the trained limit)')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8090)
    p.add_argument('--real', action='store_true')
    p.add_argument('--hand-port')
    p.add_argument('--hand-serial')
    p.add_argument('--hand-current', type=int, default=600,
                   help='Hand goal current, mA (current limit max(1000, this)); default as run_step1.py')
    args = p.parse_args()
    if args.real and args.handeye is None:
        p.error('--handeye is required with --real (the joint compensation must match the policy deployment)')
    ReplayApp(args).run()


if __name__ == '__main__':
    main()
