"""Arm, hand and step1 policy behind the buttons of perception/step1_setup.py.

The viewer runs in the foundationpose environment, the robot in gear3_control (libgx7, gex16, torch), so the robot
side is this separate process in its own terminal. Closing the viewer stops a running move or policy and leaves the
arm and hand as they are; only this process connects or disconnects them.

  bash tasks/dexlong_gear3/run_robot_server.sh --handeye <handeye_result.json> [--hand-port /dev/ttyUSB0]

It listens on 127.0.0.1:5702 (--listen) with a random authkey written to KEY_PATH (mode 0600), where the viewer
reads it. Messages are gx4m_real.camera.frame_ipc headers without arrays:
  viewer -> server  {'type': 'command', 'action': one of ACTIONS}
  server -> viewer  {'type': 'status', 'arm', 'hand', 'busy', 'message', 'q', 'stage', ...} at 10 Hz
Gear poses arrive over UDP from step1_setup.py (its --target, default 127.0.0.1:5701), as for run_step1.py, and are
converted with the same hand-eye result. Arm, hand, joint compensation and policy setup are run_step1.py's.

Actions run one at a time on one thread that owns the hardware; between actions it keeps the arm at its hold target
(or follows it while released) and reads the hand. Arm and hand move to their start poses separately ('arm_home',
'hand_home'). 'stop' is not queued: it interrupts those moves or the policy, which then hold the measured pose.
'release' disables J1-J7 for hand guiding (the arm drops unless supported), as does 'disconnect_arm'. The policy
starts only with arm and hand at the start pose and a fresh gear pose inside the trained start range. Ctrl-C here holds the arm, then asks before disabling it.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from multiprocessing.connection import AuthenticationError, Listener
import os
from pathlib import Path
import queue
import sys
import tempfile
import threading
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from gx4m_real.camera.frame_ipc import receive, send            # noqa: E402
from gx4m_real.robot.arm import ArmDriver                       # noqa: E402
from gx4m_real.robot.hand import HandDriver                     # noqa: E402
from tasks.dexlong_gear3.run_step1 import (CONTROL_CONFIG, HAND_START_TOLERANCE, MODEL, START_RANGE,  # noqa: E402
                                           CameraToEnv, Step1Runner, UdpGearPose, arm_config, load_actor,
                                           save_episode)
from tasks.dexlong_gear3.step1 import Step1Interface            # noqa: E402

# Same expression in perception/robot_panel.py (other environment, no shared import).
KEY_PATH = Path(os.environ.get('XDG_RUNTIME_DIR') or tempfile.gettempdir())/'gx4m_step1_robot.key'
ACTIONS = {'connect_arm': '连接机械臂', 'disconnect_arm': '断开机械臂', 'connect_hand': '连接灵巧手',
           'disconnect_hand': '断开灵巧手', 'release': '进入拖动模式', 'hold': '退出拖动模式',
           'arm_home': '机械臂回初始位置', 'hand_home': '灵巧手回初始位置', 'run_policy': '执行策略', 'stop': '停止'}
START_POSE_AGE = .3     # s: the policy starts only on a pose this fresh; once running, --max-pose-age applies
OUTCOMES = {'success': '成功', 'stopped': '已停止', 'pose_lost': '齿轮位姿丢失', 'tracking_error': '机械臂跟不上目标',
            'max_steps': '达到最大步数', 'interrupted': '中断'}


class RobotController:
    """Runs viewer actions on the runner's arm and hand; the hand is opened by connect_hand (hand_factory())."""

    def __init__(self, runner, hand_factory, out, *, log=print):
        self.runner, self.arm, self.hand_factory, self.out, self.log = runner, runner.arm, hand_factory, out, log
        self.iface = runner.iface
        runner.hand = None
        self.lock = threading.Lock()
        self.queue = queue.Queue()
        self.busy, self.message, self.hand_error, self.episodes = None, '就绪', None, 0
        self.closing = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name='robot-controller')
        self.thread.start()

    def submit(self, action):
        if action not in ACTIONS:
            raise ValueError(f'Unknown action {action!r}')
        if action == 'stop':
            self.runner.stop_request.set()
            return
        with self.lock:
            if self.busy is not None:
                self.message = f'正在{ACTIONS[self.busy]}，先停止或等它结束'
                return
            self.busy, self.message = action, f'{ACTIONS[action]}…'
        self.queue.put(action)

    def _run(self):
        while not self.closing.is_set():
            try:
                action = self.queue.get(timeout=self.iface.dt)
            except queue.Empty:
                self._idle()
                continue
            self.runner.stop_request.clear()
            try:
                message = getattr(self, action)()
            except Exception as exc:
                message = f'{ACTIONS[action]}失败：{exc}'
            with self.lock:
                self.busy, self.message = None, message
            self.log(message)

    def _idle(self):
        self.arm.update()       # sends the hold target, or follows the arm while released; faults set arm.mode
        hand = self.runner.hand
        if hand is not None:
            try:
                hand.read()
                self.hand_error = None
            except Exception as exc:
                self.hand_error = str(exc)

    def _require(self, hand=False):
        if self.arm.robot is None or self.arm.mode == 'fault':
            raise RuntimeError('机械臂没有连接或已故障')
        if hand and self.runner.hand is None:
            raise RuntimeError('先连接灵巧手')

    def connect_arm(self):
        self.arm.connect()
        return '机械臂已连接，保持当前位置'

    def disconnect_arm(self):
        self.arm.disconnect()
        return '机械臂已断开，电机失能'

    def connect_hand(self):
        if self.runner.hand is None:
            hand = self.hand_factory()
            hand.connect()
            self.runner.hand, self.hand_error = hand, None
        return '灵巧手已连接，保持当前位置'

    def disconnect_hand(self):
        hand, self.runner.hand = self.runner.hand, None
        if hand is not None:
            hand.close()
        return '灵巧手已断开，扭矩关闭'

    def release(self):
        self._require()
        self.arm.release()
        return '拖动模式：J1–J7 已失能，可以手动拖动'

    def hold(self):
        self._require()
        self.arm.hold()
        return '已退出拖动模式，保持当前位置'

    def arm_home(self):
        self._require()
        if self.arm.mode == 'released':
            self.arm.hold()
        q = self.runner.arm_to_start()
        return '已停止，机械臂保持当前位置' if q is None else '机械臂已回到初始位置'

    def hand_home(self):
        if self.runner.hand is None:
            raise RuntimeError('先连接灵巧手')
        q = self.runner.hand_to_start()
        if q is None:
            return '已停止，灵巧手保持当前位置'
        error = np.abs(q - self.iface.ref['initial_joint_pos'][7:])
        if error.max() >= HAND_START_TOLERANCE:
            return (f'灵巧手没有完全到位：gx4m_joint{int(error.argmax()) + 1} 差 {np.degrees(error.max()):.1f}°'
                    '（目标已下发，可能被挡住）')
        return '灵巧手已回到初始位置'

    def run_policy(self):
        self._require(hand=True)
        if self.arm.mode == 'released':
            raise RuntimeError('机械臂在拖动模式，先回初始位置')
        q = self.runner.joints()
        goal = np.asarray(self.iface.ref['initial_joint_pos'])
        error = np.abs(q[:7] - goal[:7]).max()
        if error > .02:
            raise RuntimeError(f'机械臂离初始位置 {np.degrees(error):.1f}°，先点“机械臂回初始位置”')
        error = np.abs(q[7:] - goal[7:]).max()
        if error >= HAND_START_TOLERANCE:
            raise RuntimeError(f'灵巧手离初始位置 {np.degrees(error):.1f}°，先点“灵巧手回初始位置”')
        T, age = self.runner.gear_pose(q)
        if T is None or age > START_POSE_AGE:
            raise RuntimeError('没有齿轮位姿' if np.isinf(age) else f'齿轮位姿已过期（{age:.2f} s）')
        offset = T[:3, 3] - self.iface.ref['object_pos']
        if np.abs(offset[:2]).max() > START_RANGE or abs(offset[2]) > .005:
            raise RuntimeError(f'齿轮离训练起点 x {offset[0]*1000:+.0f}、y {offset[1]*1000:+.0f}、'
                               f'z {offset[2]*1000:+.0f} mm（x、y 各 ±{START_RANGE*1000:.0f} mm，z ±5 mm）')
        self.log(f'Policy episode {self.episodes}: gear at {np.round(T[:3, 3], 3)}')
        outcome, rows = self.runner.episode()
        self.out.mkdir(parents=True, exist_ok=True)
        path = self.out/f'episode_{self.episodes:03d}.npz'
        save_episode(path, rows, outcome)
        self.episodes += 1
        return f'策略结束：{OUTCOMES.get(outcome, outcome)}，{len(rows)} 步，记录 {path}'

    def status(self):
        arm = self.arm.snapshot()
        hand = self.runner.hand
        names = self.iface.joint_order
        q = {}
        if arm['connected']:
            q.update(zip(names[:7], self.arm.config.compensation.to_model(arm['positions']).tolist()))
        if hand is not None and hand.q is not None:
            q.update(zip(names[7:], np.asarray(hand.q, float).tolist()))
        with self.lock:
            busy, message = self.busy, self.message
        return {'type': 'status', 'busy': busy, 'message': message, 'q': q,
                'arm': {'connected': arm['connected'], 'mode': arm['mode'], 'error': arm['error'], 'mock': arm['mock']},
                'hand': {'connected': hand is not None, 'error': self.hand_error},
                'stage': self.iface.stage if busy == 'run_policy' else None,
                'compensation_deg': list(self.arm.config.compensation.offsets_deg)}

    def close(self):
        """Stop a running action and the thread; the arm and hand stay as they are."""
        self.runner.stop_request.set()
        self.closing.set()
        self.thread.join(timeout=5.)


def serve(controller, address, key, *, status_period=.1, stop=None):
    """Serve one viewer at a time until stop is set (or Ctrl-C). A viewer that goes away stops a running action."""
    stop = stop or threading.Event()
    with Listener(address, authkey=key) as listener:
        listener._listener._socket.settimeout(.5)    # Listener has no accept timeout; wake up to check stop
        while not stop.is_set():
            try:
                connection = listener.accept()
            except (TimeoutError, OSError, AuthenticationError, EOFError):
                continue
            controller.log('Viewer connected')
            try:
                with connection:
                    next_status = 0.
                    while not stop.is_set():
                        if connection.poll(.02):
                            header, _ = receive(connection)
                            if header.get('type') == 'command':
                                controller.submit(header['action'])
                        if time.monotonic() >= next_status:
                            send(connection, controller.status())
                            next_status = time.monotonic() + status_period
            except (EOFError, OSError, ValueError):
                pass
            controller.submit('stop')
            controller.log('Viewer disconnected: a running move or policy was stopped; arm and hand keep their state')


def write_key(path=KEY_PATH):
    key = os.urandom(16)
    path.unlink(missing_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write(key.hex() + '\n')
    return key


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--handeye', type=Path, help='handeye_result.json (required without --mock)')
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--config', type=Path, default=CONTROL_CONFIG, help='Arm config (CAN, limits)')
    parser.add_argument('--hand-port')
    parser.add_argument('--hand-serial')
    parser.add_argument('--mock', action='store_true', help='Simulated arm and hand; gear poses still from UDP')
    parser.add_argument('--listen', default='127.0.0.1:5702')
    parser.add_argument('--pose-port', type=int, default=5701)
    parser.add_argument('--arm-velocity', type=float, help='PVT speed limit, rad/s (default: the trained limit)')
    parser.add_argument('--arm-speed-scale', type=float, help='Arm rad/s per unit action (default: trained)')
    parser.add_argument('--approach-speed', type=float, default=.2, help='Arm to the start pose, rad/s')
    parser.add_argument('--hand-approach-speed', type=float, default=1., help='Hand to the start pose, rad/s')
    parser.add_argument('--max-pose-age', type=float, default=1.,
                        help='s: tracking lost this long stops the policy (a start needs a pose under 0.3 s)')
    parser.add_argument('--max-tracking-error', type=float, default=.3)
    parser.add_argument('--out', type=Path, help='Directory for episode logs (default: runs/step1_<time>)')
    args = parser.parse_args(argv)
    if not args.mock and args.handeye is None:
        parser.error('--handeye is required on the real robot')

    iface = Step1Interface()
    actor = load_actor(args.model, iface)
    config = arm_config(args.config, iface, args.handeye, args.arm_velocity)
    arm = ArmDriver(config, mock=args.mock)
    gear = UdpGearPose(args.pose_port)
    camera = CameraToEnv(args.handeye, iface) if args.handeye else CameraToEnv()
    runner = Step1Runner(iface, actor, arm, None, gear, camera, arm_speed_scale=args.arm_speed_scale,
                         max_pose_age=args.max_pose_age, max_tracking_error=args.max_tracking_error,
                         approach_speed=args.approach_speed, hand_approach_speed=args.hand_approach_speed)
    out = args.out or HERE/'runs'/datetime.now().strftime('step1_%Y%m%d_%H%M%S')
    controller = RobotController(
        runner, lambda: HandDriver(args.hand_port, serial_number=args.hand_serial, mock=args.mock), out)
    host, port = args.listen.rsplit(':', 1)
    key = write_key()
    print(f'Robot server on {host}:{port} (key {KEY_PATH}); arm PVT limit {config.velocity_limit:.2f} rad/s, '
          f'joint offsets {list(config.compensation.offsets_deg)} deg; episode logs: {out}', flush=True)
    try:
        serve(controller, (host, int(port)), key)
    except KeyboardInterrupt:
        pass
    finally:
        controller.close()
        KEY_PATH.unlink(missing_ok=True)
        gear.close()
        if arm.robot is not None:
            try:
                if arm.mode != 'fault':
                    arm.hold()
                input('Arm is holding. Support it or move it to a safe pose, then Enter to disable the motors: ')
            except (KeyboardInterrupt, EOFError, RuntimeError):
                pass
            arm.disconnect()
        if runner.hand is not None:
            runner.hand.close()


if __name__ == '__main__':
    main()
