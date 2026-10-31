"""Step1 policy on the real robot: a 30 Hz loop of measure -> observation -> LSTM -> joint targets.

Per control step (1/30 s, as the policy was trained):
  1. read the arm (GX7, PVT) and hand (GX16) joints, take the newest gear pose from the perception process;
  2. after the first step, run the stage logic (Step1Interface.update) on that state and stop on an outcome;
  3. build the 78-wide observation, run the policy, map the action to joint targets;
  4. send the arm targets (model -> motor frame through the joint compensation) and the finger targets.

Gear pose: tasks/dexlong_gear3/perception/publish_gear_pose.py (or step1_setup.py, the placement check) sends
FoundationPose's gear24 object_in_camera per frame over UDP; it becomes env-frame through the hand-eye result
(eye-to-hand: T_world_base @ T_base_cam; eye-in-hand: FK(Link7) @ T_flange_cam). The gear24 mesh is the
simulation's gear body frame, so no mesh-to-gear transform is needed (--mesh-to-object only for another mesh).
The world frame of the hand-eye result is the env frame of the reference (same robot base pose), checked at
start. Place the plate with perception/step1_setup.py first.

Tactile: not wired yet (PaXini mapping and calibration pending), the policy sees no contact.

Safety: every episode starts at the reference initial pose (moved there slowly, after a keypress), the arm
speed is capped by the PVT velocity limit (default: the trained 1.5 rad/s), a stale gear pose or an arm that
falls behind its target stops the episode and holds the measured pose. Ctrl-C holds too. The arm stays
powered until the operator confirms; disconnecting disables the motors (the arm can drop).

  python tasks/dexlong_gear3/run_step1.py --mock --yes --max-steps 60          # no hardware
  python tasks/dexlong_gear3/run_step1.py --handeye <handeye_result.json> --hand-port COM5
"""
from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime
import json
from pathlib import Path
import socket
import sys
import threading
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from gx4m_real.policy import RslRlActor                         # noqa: E402
from gx4m_real.robot.arm import ArmDriver                       # noqa: E402
from gx4m_real.robot.config import load_config                  # noqa: E402
from gx4m_real.robot.hand import HandDriver                     # noqa: E402
from gx4m_real.robot.kinematics import matrix_quat, quat_matrix  # noqa: E402
from tasks.dexlong_gear3.step1 import Step1Interface            # noqa: E402

MODEL = HERE/'models'/'step1_lstm_20261031_iter500.pt'
CONTROL_CONFIG = ROOT/'apps'/'control'/'config.yaml'
START_RANGE = .02           # trained gear start: +-2 cm in x and y (reset_position_noise), any yaw


def pose_matrix(pos, quat):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = quat_matrix(np.asarray(quat, dtype=np.float64)), pos
    return T


class UdpGearPose:
    """Newest gear pose from publish_gear_pose.py: JSON datagrams {"timestamp_monotonic", "object_in_camera"}
    (4x4 or null when tracking is lost). Age is measured on the sender's frame time (same host, shared
    monotonic clock) or, with same_host=False, on the receive time."""

    def __init__(self, port, host='127.0.0.1', same_host=True):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.settimeout(.1)
        self.same_host, self.lock, self.latest = same_host, threading.Lock(), None
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(65536)
            except (socket.timeout, OSError):
                continue
            try:
                record = json.loads(data)
                pose = record.get('object_in_camera')
                stamp = float(record['timestamp_monotonic']) if self.same_host else time.monotonic()
            except (ValueError, KeyError, TypeError):
                continue
            with self.lock:
                self.latest = (stamp, None if pose is None else np.asarray(pose, dtype=np.float64))

    def get(self):
        """(T_cam_object 4x4 or None, age in s); (None, inf) before the first message."""
        with self.lock:
            latest = self.latest
        if latest is None:
            return None, float('inf')
        return latest[1], time.monotonic() - latest[0]

    def close(self):
        self.running = False
        self.thread.join(timeout=1.)
        self.sock.close()


class FixedGearPose:
    """Constant env-frame gear pose (dry runs, mock mode); camera frame = env frame."""

    def __init__(self, pos, quat):
        self.T = pose_matrix(pos, quat)

    def get(self):
        return self.T.copy(), 0.

    def close(self):
        pass


class CameraToEnv:
    """T_env_object from T_cam_object: hand-eye result (eye-to-hand or eye-in-hand) and mesh-to-gear offset."""

    def __init__(self, handeye=None, iface=None, mesh_to_object=None):
        self.mesh_to_object = np.eye(4) if mesh_to_object is None else np.asarray(mesh_to_object, dtype=np.float64)
        self.mode, self.iface = 'identity', iface
        if handeye is None:
            return
        d = json.loads(Path(handeye).read_text(encoding='utf-8'))
        if d.get('synthetic'):
            raise ValueError(f'{handeye} is a synthetic calibration')
        T_env_base = pose_matrix(iface.ref['base_pos'], iface.ref['base_quat'])
        gap = np.abs(np.asarray(d['T_world_base']) - T_env_base).max()
        if gap > 2e-3:
            raise ValueError(f'Hand-eye base pose differs from the reference base pose by {gap:.4f}')
        self.mode = d['mode']
        if self.mode == 'eye-to-hand':
            self.T_env_cam = T_env_base @ np.asarray(d['T_base_cam'])
        elif self.mode == 'eye-in-hand':
            self.T_flange_cam, self.tip = np.asarray(d['T_flange_cam']), d.get('tip_link', 'Link7')
        else:
            raise ValueError(f"Unsupported hand-eye mode {self.mode!r}")

    def __call__(self, T_cam_object, q):
        if self.mode == 'identity':
            T_env_cam = np.eye(4)
        elif self.mode == 'eye-to-hand':
            T_env_cam = self.T_env_cam
        else:
            poses = self.iface.fk.link_poses(dict(zip(self.iface.joint_order, q)))
            T_env_cam = poses[self.tip] @ self.T_flange_cam
        return T_env_cam @ T_cam_object @ self.mesh_to_object


class Rate:
    def __init__(self, hz):
        self.period, self.next = 1./hz, None

    def sleep(self):
        """Wait for the next tick; returns how late this tick started (s)."""
        now = time.perf_counter()
        if self.next is None or now - self.next > self.period:   # first tick, or a whole period behind
            self.next = now
        while (left := self.next - time.perf_counter()) > 0:
            time.sleep(left - .002 if left > .003 else 0)
        late = time.perf_counter() - self.next
        self.next += self.period
        return late


class Step1Runner:
    """Robot side of the loop; arm is an ArmDriver (motor frame), hand a HandDriver, gear a pose source."""

    def __init__(self, iface, actor, arm, hand, gear, camera, *, arm_speed_scale=None, max_pose_age=.3,
                 max_tracking_error=.3, approach_speed=.2, log=print):
        self.iface, self.actor, self.arm, self.hand, self.gear, self.camera = iface, actor, arm, hand, gear, camera
        self.comp = arm.config.compensation
        self.arm_speed_scale, self.max_pose_age = arm_speed_scale, max_pose_age
        self.max_tracking_error, self.approach_speed, self.log = max_tracking_error, approach_speed, log
        self.rate = Rate(1./iface.dt)

    def joints(self):
        """Measured joints (23,) in the reference order, model frame."""
        self.arm.update()
        if self.arm.mode == 'fault':
            raise RuntimeError(f'Arm fault: {self.arm.error}')
        return np.concatenate((self.comp.to_model(self.arm.q), self.hand.read()))

    def send(self, targets):
        motor = np.clip(self.comp.to_motor(targets[:7]), self.arm.config.lower, self.arm.config.upper)
        self.arm.command(motor)
        self.arm.update()
        self.hand.command(targets[7:])

    def hold(self):
        try:
            self.arm.hold()
        finally:
            self.hand.command(self.hand.read())

    def gear_pose(self, q):
        T_cam, age = self.gear.get()
        if T_cam is None or age > self.max_pose_age:
            return None, age
        return self.camera(T_cam, q), age

    def move_to_start(self, timeout=30.):
        """Straight joint-space move to the reference initial pose at approach_speed (rad/s)."""
        goal = np.array(self.iface.ref['initial_joint_pos'])
        q = self.joints()
        start, distance = q.copy(), np.abs(goal - q)
        duration = max(float(distance.max())/self.approach_speed, self.iface.dt)
        t0 = time.monotonic()
        while True:
            s = min((time.monotonic() - t0)/duration, 1.)
            self.send(start + s*(goal - start))
            self.rate.sleep()
            q = self.joints()
            if s >= 1 and np.abs(q[:7] - goal[:7]).max() < .02:
                return q
            if time.monotonic() - t0 > duration + timeout:
                raise RuntimeError(f'Did not reach the start pose: arm error {np.abs(q[:7]-goal[:7]).max():.3f} rad')

    def episode(self, max_steps=None):
        """One policy episode from the start pose; returns (outcome, per-step log)."""
        iface = self.iface
        iface.reset()
        self.actor.reset()
        rows, result, outcome = [], None, None
        self.rate.next = None
        k = 0
        try:
            while max_steps is None or k < max_steps:
                late = self.rate.sleep()
                q = self.joints()
                T, age = self.gear_pose(q)
                if T is None:
                    outcome = 'pose_lost'
                    break
                pos, quat = T[:3, 3], matrix_quat(T[:3, :3])
                if k > 0:
                    result = iface.update(q, pos, quat)
                    rows[-1].update(stage_after=result['stage'], distance=result['distance'], away=result['away'])
                    if result['outcome']:
                        outcome = result['outcome']
                        break
                error = np.abs(iface.targets[:7] - q[:7]).max()
                if k > 0 and error > self.max_tracking_error:
                    outcome = 'tracking_error'
                    break
                obs = iface.observation(q, pos, T[:3, 2], np.zeros(4))
                action = self.actor(obs)
                targets = iface.apply_action(action, self.arm_speed_scale)
                self.send(targets)
                rows.append(dict(t=time.monotonic(), late=late, q=q, object_pos=pos, object_quat=quat, pose_age=age,
                                 obs=obs, action=action, targets=targets, stage=iface.stage, tracking_error=error))
                if k % 30 == 0:
                    self.log(f'step {k:4d}: stage {iface.stage}, keypoint error '
                             f'{iface.keypoint_distance(pos, quat)*1000:6.1f} mm, pose age {age*1000:4.0f} ms, '
                             f'arm lag {error:.3f} rad, late {late*1000:.0f} ms')
                k += 1
            else:
                outcome = 'max_steps'
        except KeyboardInterrupt:
            outcome = 'interrupted'
        finally:
            self.hold()
        return outcome or 'max_steps', rows


def save_episode(path, rows, outcome):
    keys = ('t', 'late', 'q', 'object_pos', 'object_quat', 'pose_age', 'obs', 'action', 'targets', 'stage',
            'tracking_error')
    data = {k: np.array([r[k] for r in rows]) for k in keys}
    data['distance'] = np.array([r.get('distance', np.nan) for r in rows])
    np.savez_compressed(path, outcome=outcome, **data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--config', type=Path, default=CONTROL_CONFIG, help='Arm config (CAN, limits, compensation)')
    parser.add_argument('--handeye', type=Path, help='handeye_result.json (required without --mock)')
    parser.add_argument('--mesh-to-object', type=Path, help='4x4 (np.loadtxt) mesh frame -> Isaac gear frame')
    parser.add_argument('--pose-port', type=int, default=5701)
    parser.add_argument('--pose-remote', action='store_true', help='Perception on another host: age from receive time')
    parser.add_argument('--hand-port')
    parser.add_argument('--hand-serial')
    parser.add_argument('--mock', action='store_true', help='Simulated arm and hand, gear fixed at the reference start')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--max-steps', type=int, help='Stop an episode after this many steps (default: stage timeouts)')
    parser.add_argument('--arm-velocity', type=float, help='PVT speed limit, rad/s (default: the trained limit)')
    parser.add_argument('--arm-speed-scale', type=float, help='Arm rad/s per unit action (default: trained)')
    parser.add_argument('--approach-speed', type=float, default=.2, help='Move to the start pose, rad/s')
    parser.add_argument('--max-pose-age', type=float, default=.3, help='s; trained with up to 0.13 s')
    parser.add_argument('--max-tracking-error', type=float, default=.3, help='Arm target - measured, rad')
    parser.add_argument('--yes', action='store_true', help='No keypress prompts (mock runs)')
    parser.add_argument('--out', type=Path, help='New directory for logs (default: runs/step1_<time>)')
    args = parser.parse_args(argv)
    if not args.mock and args.handeye is None:
        parser.error('--handeye is required on the real robot')

    iface = Step1Interface()
    actor = RslRlActor(args.model)
    if (actor.obs_dim, actor.action_dim) != (iface.obs_dim, len(iface.joint_order)):
        raise ValueError(f'{args.model} does not match the reference: {actor.obs_dim} -> {actor.action_dim}')
    trained_velocity = float(max(iface.ref['velocity_limit'][:7]))
    config = load_config(args.config)
    config = dataclasses.replace(config, velocity_limit=args.arm_velocity or trained_velocity)
    model_lower, model_upper = config.compensation.model_limits(config.lower, config.upper)
    gap = max(np.abs(model_lower - iface.lower[:7]).max(), np.abs(model_upper - iface.upper[:7]).max())
    if gap > np.radians(1.):
        print(f'WARNING: real arm limits differ from the trained ones by up to {np.degrees(gap):.1f} deg')
    mesh_to_object = np.loadtxt(args.mesh_to_object) if args.mesh_to_object else None
    print('WARNING: tactile is not wired, the policy sees no finger contact')
    out = args.out or HERE/'runs'/datetime.now().strftime('step1_%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    (out/'run.json').write_text(json.dumps(dict(
        argv=sys.argv if argv is None else argv, model=str(args.model.resolve()), velocity_limit=config.velocity_limit,
        arm_speed_scale=args.arm_speed_scale or iface.arm_speed_scale, torque_limit=config.torque_limit,
        max_joint_step_rad=config.max_joint_step_rad, handeye=str(args.handeye) if args.handeye else None,
        mesh_to_object=None if mesh_to_object is None else mesh_to_object.tolist(), tactile=None,
        mock=args.mock), indent=2) + '\n')

    ask = (lambda message: None) if args.yes else (lambda message: input(message))
    arm = ArmDriver(config, mock=args.mock)
    hand = HandDriver(args.hand_port, serial_number=args.hand_serial, mock=args.mock)
    if args.mock:
        gear = FixedGearPose(iface.ref['object_pos'], iface.ref['object_quat'])
        camera = CameraToEnv()
    else:
        gear = UdpGearPose(args.pose_port, same_host=not args.pose_remote)
        camera = CameraToEnv(args.handeye, iface, mesh_to_object)
    runner = Step1Runner(iface, actor, arm, hand, gear, camera, arm_speed_scale=args.arm_speed_scale,
                         max_pose_age=args.max_pose_age, max_tracking_error=args.max_tracking_error,
                         approach_speed=args.approach_speed)
    print(f'Arm PVT limit {config.velocity_limit:.2f} rad/s, torque ratio {config.torque_limit}, '
          f'action scale {args.arm_speed_scale or iface.arm_speed_scale} rad/s; logs: {out}')
    outcomes = []
    try:
        arm.connect()
        hand.connect()
        for episode in range(args.episodes):
            ask(f'[episode {episode}] Enter: move slowly to the start pose ({args.approach_speed} rad/s)')
            runner.move_to_start()
            T, age = runner.gear_pose(runner.joints())
            while T is None:
                ask(f'No fresh gear pose (age {age:.2f} s). Fix perception, then Enter to retry')
                T, age = runner.gear_pose(runner.joints())
            d = iface.keypoint_distance(T[:3, 3], matrix_quat(T[:3, :3]))
            offset = T[:3, 3] - iface.ref['object_pos']
            if np.abs(offset[:2]).max() > START_RANGE or abs(offset[2]) > .005:
                print(f'WARNING: gear {np.round(offset*1000)} mm from the trained start (x, y within '
                      f'+-{START_RANGE*1000:.0f} mm, flat on the table)')
            ask(f'Gear at {np.round(T[:3, 3], 3)} (start {np.round(iface.ref["object_pos"], 3)}), '
                f'{d*1000:.0f} mm from the waypoint. Enter: run the policy (Ctrl-C holds)')
            outcome, rows = runner.episode(args.max_steps)
            save_episode(out/f'episode_{episode:03d}.npz', rows, outcome)
            outcomes.append(outcome)
            print(f'[episode {episode}] {outcome} after {len(rows)} steps')
    except KeyboardInterrupt:
        print('Interrupted: holding')
        runner.hold()
    finally:
        gear.close()
        (out/'outcomes.json').write_text(json.dumps(outcomes) + '\n')
        if arm.robot is not None:
            try:
                ask('Arm is holding. Support it or move it to a safe pose, then Enter to disable the motors')
            except KeyboardInterrupt:
                pass
            arm.disconnect()
        hand.close()
    return outcomes


if __name__ == '__main__':
    main()
