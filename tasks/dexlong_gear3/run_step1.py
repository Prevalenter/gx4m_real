"""Step1 / step2 policy on the real robot: a 30 Hz loop of measure -> observation -> LSTM -> joint targets.

--step 1 (default) inserts gear24 on the left pin, --step 2 gear48 on the right pin with gear24 already mounted;
each step has its own reference and model (step1.REFERENCES / MODELS). The perception process must track that
step's gear (step1_setup.py --step 2, or publish_gear_pose.py --gear gear48).

Per control step (1/30 s, as the policy was trained):
  1. read the arm (GX7, PVT) and hand (GX16) joints, take the newest gear pose from the perception process;
  2. after the first step, run the stage logic (Step1Interface.update) on that state and stop on an outcome;
  3. build the 78-wide observation, run the policy, map the action to joint targets;
  4. send the arm targets (model -> motor frame through the joint compensation) and the finger targets.

Gear pose: tasks/dexlong_gear3/perception/publish_gear_pose.py (or step1_setup.py, the placement check) sends
FoundationPose's gear24 object_in_camera per frame over UDP; it becomes env-frame through the hand-eye result
(eye-to-hand: T_world_base @ T_base_cam; eye-in-hand: FK(Link7) @ T_flange_cam). The gear24 mesh is the
simulation's gear body frame, so no mesh-to-gear transform is needed (--mesh-to-object only for another mesh).
The calibrated camera (T_base_cam / T_flange_cam) and the joint compensation it was solved with are taken from the
hand-eye result; the env frame comes from the reference robot base pose, not from the base pose recorded with the
hand-eye session. Place the plate with
perception/step1_setup.py first (it uses the same base pose, from config/initial_pose.json).

Tactile: the four PaXini fingertip sensors (gx4m_real.robot.tactile.FingertipContacts, thumb first as the simulation's
fingertips) are tared at the start of each episode, with the hand open at the start pose, and a tared total normal
force above --tactile-threshold (default: the trained binary_tactile_threshold_n, 0.3 N) is a contact. The raw
count is scaled by 0.1 N as in the hora deployment, not calibrated here. --no-tactile feeds zeros (the policies fail
without contact); a sensor whose reading is older than --max-tactile-age stops the episode.

Safety: every episode starts at the reference initial pose (moved there slowly, after a keypress), the arm
speed is capped by the PVT velocity limit (default: the trained PhysX limit, 1.5 rad/s; the policy targets move at
most arm_moving_average * control.arm_speed_scale = 0.15 rad/s), a stale gear pose or an arm that
falls behind its target stops the episode and holds the measured pose. Ctrl-C holds too. The arm stays
powered until the operator confirms; disconnecting disables the motors (the arm can drop).

  python tasks/dexlong_gear3/run_step1.py --mock --yes --max-steps 60          # no hardware
  python tasks/dexlong_gear3/run_step1.py --handeye <handeye_result.json> --hand-port COM5
  python tasks/dexlong_gear3/run_step1.py --step 2 --handeye <handeye_result.json> --hand-port COM5
  python tasks/dexlong_gear3/run_step1.py --step beam1 --handeye <handeye_result.json> --hand-port COM5

--step beam1 grasps Beam part 2 and lifts it 20 cm (gx4m_learning dexlong_beam_lift step 1, the perception tracking
beam2): the goal is set above the beam's measured pose when the episode starts, the policy sees the beam's full
orientation, and holding the goal for success_steps ends the episode as a success ('drop' if it falls out of the hand).
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
from gx4m_real.robot.compensation import JointCompensation      # noqa: E402
from gx4m_real.robot.config import load_config                  # noqa: E402
from gx4m_real.robot.hand import HandDriver                     # noqa: E402
from gx4m_real.robot.kinematics import matrix_quat, quat_matrix  # noqa: E402
from tasks.dexlong_gear3.step1 import (ESTIMATORS, MODELS, REFERENCES, Step1Interface, load_reference,  # noqa: E402
                                       start_check, step_key)

MODEL = MODELS[1]
CONTROL_CONFIG = ROOT/'apps'/'control'/'config.yaml'
START_RANGE = .02           # trained gear start: +-2 cm in x and y (reset_position_noise), any yaw
HAND_START_TOLERANCE = .1   # rad; current-based position control stops a little short of the goal


def pose_matrix(pos, quat):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = quat_matrix(np.asarray(quat, dtype=np.float64)), pos
    return T


class UdpGearPose:
    """Newest valid gear pose from publish_gear_pose.py: JSON datagrams {"timestamp_monotonic", "object_in_camera"}
    (4x4 or null when tracking is lost). A lost frame keeps the last valid pose, which ages until the caller's
    max_pose_age rejects it, so a short occlusion does not stop the policy. Age is measured on the sender's frame
    time (same host, shared monotonic clock) or, with same_host=False, on the receive time."""

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
            if pose is None:
                continue
            with self.lock:
                self.latest = (stamp, np.asarray(pose, dtype=np.float64))

    def get(self):
        """(last valid T_cam_object 4x4, its age in s); (None, inf) before the first valid pose."""
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
        # T_base_cam / T_flange_cam are the calibration; its T_world_base is only the robot base pose recorded
        # with the session. The env frame is the reference's, whose base may have moved since.
        T_env_base = pose_matrix(iface.ref['base_pos'], iface.ref['base_quat'])
        gap = np.abs(np.asarray(d['T_world_base']) - T_env_base).max()
        if gap > 2e-3:
            print(f'Hand-eye session recorded another robot base pose (up to {gap:.3f} off); '
                  'using the reference base pose')
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


class TactileLost(RuntimeError):
    """A fingertip sensor stopped answering (or the tare failed): the episode stops."""


class Step1Runner:
    """Robot side of the loop; arm is an ArmDriver (motor frame), hand a HandDriver, gear a pose source, tactile a
    FingertipContacts (None: no contact)."""

    def __init__(self, iface, actor, arm, hand, gear, camera, *, tactile=None, arm_speed_scale=None, max_pose_age=1.,
                 max_tactile_age=.2, max_tracking_error=.3, approach_speed=.2, hand_approach_speed=1., valid_pose_age=.1,
                 max_lost_age=10., log=print):
        """With the pose estimator (iface.estimator): a gear pose up to valid_pose_age s old counts as tracked; an older
        one is held and flagged, as in training, and the episode stops only once it is max_lost_age s old."""
        self.iface, self.actor, self.arm, self.hand, self.gear, self.camera = iface, actor, arm, hand, gear, camera
        self.tactile, self.max_tactile_age = tactile, max_tactile_age
        self.comp = arm.config.compensation
        self.arm_speed_scale, self.max_pose_age = arm_speed_scale, max_pose_age
        self.valid_pose_age, self.max_lost_age = valid_pose_age, max_lost_age
        self.max_tracking_error, self.approach_speed, self.log = max_tracking_error, approach_speed, log
        self.hand_approach_speed = hand_approach_speed     # hand_to_start alone; the trained finger limit is 3 rad/s
        self.rate = Rate(1./iface.dt)
        self.stop_request = threading.Event()     # set from another thread: move_to_start / episode hold and return

    def joints(self):
        """Measured joints (23,) in the reference order, model frame."""
        return np.concatenate((self.arm_joints(), self.hand.read()))

    def send(self, targets):
        self.send_arm(targets[:7])
        self.hand.command(targets[7:])

    def hold(self):
        try:
            self.arm.hold()
        finally:
            self.hand.command(self.hand.read())

    def gear_pose(self, q):
        """(gear pose in the env frame, age s); None once older than the allowed age. With the pose estimator the
        last tracked pose is kept for max_lost_age s (the fixed camera's pose does not depend on q)."""
        T_cam, age = self.gear.get()
        limit = self.max_lost_age if self.iface.estimator is not None else self.max_pose_age
        if T_cam is None or age > limit:
            return None, age
        return self.camera(T_cam, q), age

    def arm_joints(self):
        self.arm.update()
        if self.arm.mode == 'fault':
            raise RuntimeError(f'Arm fault: {self.arm.error}')
        return self.comp.to_model(self.arm.q)

    def send_arm(self, targets):
        self.arm.command(np.clip(self.comp.to_motor(targets), self.arm.config.lower, self.arm.config.upper))
        self.arm.update()

    def _ramp(self, read, send, goal, settled, timeout, speed=None):
        """Straight joint-space move from the measured joints to goal at speed (rad/s, default approach_speed), then
        wait until settled(q); the measured joints, or None if stop_request was set."""
        q = read()
        start = q.copy()
        duration = max(float(np.abs(goal - start).max())/(speed or self.approach_speed), self.iface.dt)
        t0 = time.monotonic()
        while True:
            if self.stop_request.is_set():
                return None
            s = min((time.monotonic() - t0)/duration, 1.)
            send(start + s*(goal - start))
            self.rate.sleep()
            q = read()
            if s >= 1 and (settled(q) or time.monotonic() - t0 > duration + timeout):
                return q

    def move_to_start(self, timeout=30.):
        """Arm and hand to the reference initial pose together; None if stopped (then holding)."""
        goal = np.array(self.iface.ref['initial_joint_pos'])
        q = self._ramp(self.joints, self.send, goal, lambda q: np.abs(q[:7] - goal[:7]).max() < .02, timeout)
        if q is None:
            self.hold()
        elif np.abs(q[:7] - goal[:7]).max() >= .02:
            raise RuntimeError(f'Did not reach the start pose: arm error {np.abs(q[:7]-goal[:7]).max():.3f} rad')
        return q

    def arm_to_start(self, timeout=30.):
        """The arm alone to its initial pose (the hand is not touched); None if stopped (then holding)."""
        goal = np.array(self.iface.ref['initial_joint_pos'][:7])
        q = self._ramp(self.arm_joints, self.send_arm, goal, lambda q: np.abs(q - goal).max() < .02, timeout)
        if q is None:
            self.arm.hold()
        elif np.abs(q - goal).max() >= .02:
            raise RuntimeError(f'Did not reach the start pose: arm error {np.abs(q - goal).max():.3f} rad')
        return q

    def hand_to_start(self, timeout=3.):
        """The hand alone to its initial pose at hand_approach_speed (the arm is not touched); None if stopped (then
        holding). A finger that is blocked is left commanded to the goal, its error shows in the returned joints."""
        goal = np.array(self.iface.ref['initial_joint_pos'][7:])
        q = self._ramp(self.hand.read, self.hand.command, goal, lambda q: np.abs(q - goal).max() < HAND_START_TOLERANCE,
                       timeout, self.hand_approach_speed)
        if q is None:
            self.hand.command(self.hand.read())
        return q

    def tare(self):
        """Zero the fingertip forces; the hand must be open at the start pose, touching nothing."""
        if self.tactile is not None:
            try:
                self.log(f'Tactile tare (N): {np.round(self.tactile.tare(), 2).tolist()}')
            except RuntimeError as exc:
                raise TactileLost(exc) from exc

    def contacts(self):
        """(0/1 contact per fingertip (4,), tared normal force N (4,)); zeros without tactile."""
        if self.tactile is None:
            return np.zeros(4), np.zeros(4)
        try:
            return self.tactile.contacts(self.max_tactile_age)
        except RuntimeError as exc:
            raise TactileLost(exc) from exc

    def pad_forces(self):
        """Tared pad forces (4, 3) N as (normal, distal, lateral); zeros without tactile."""
        if self.tactile is None:
            return np.zeros((4, 3))
        try:
            return self.tactile.pad_forces(self.max_tactile_age)
        except RuntimeError as exc:
            raise TactileLost(exc) from exc

    def episode(self, max_steps=None):
        """One policy episode from the start pose; returns (outcome, per-step log)."""
        iface = self.iface
        if iface.lift is not None:
            # The lift goal is above where the part rests now (as in training, from the pose at the reset).
            T, _ = self.gear_pose(self.joints())
            if T is None:
                return 'pose_lost', []
            iface.reset(object_pose=(T[:3, 3], matrix_quat(T[:3, :3])))
        else:
            iface.reset()
        self.actor.reset()
        rows, result, outcome = [], None, None
        self.rate.next = None
        k = 0
        try:
            self.tare()
            while max_steps is None or k < max_steps:
                if self.stop_request.is_set():
                    outcome = 'stopped'
                    break
                late = self.rate.sleep()
                q = self.joints()
                T, age = self.gear_pose(q)
                if T is None:
                    outcome = 'pose_lost'
                    break
                pos, quat = T[:3, 3], matrix_quat(T[:3, :3])
                valid = age <= self.valid_pose_age
                stage_pos, stage_quat = pos, quat
                if iface.estimator is not None and not valid and iface.estimated_pose() is not None:
                    stage_pos, stage_quat = iface.estimated_pose()     # the camera has lost the gear: the estimate
                if k > 0:
                    result = iface.update(q, stage_pos, stage_quat)
                    rows[-1].update(stage_after=result['stage'], distance=result['distance'], away=result['away'])
                    if result['outcome']:
                        outcome = result['outcome']
                        break
                error = np.abs(iface.targets[:7] - q[:7]).max()
                if k > 0 and error > self.max_tracking_error:
                    outcome = 'tracking_error'
                    break
                contact, force = self.contacts()
                pad = None
                if iface.estimator is not None:
                    pad = self.pad_forces()
                    contact = (pad[:, 0] > iface.ref['binary_tactile_threshold_n']).astype(np.float64)
                obs = iface.observation(q, pos, T[:3, :3], contact, valid=valid, age=age, pad_force=pad)
                action = self.actor(obs)
                targets = iface.apply_action(action, self.arm_speed_scale)
                self.send(targets)
                rows.append(dict(t=time.monotonic(), late=late, q=q, object_pos=pos, object_quat=quat, pose_age=age,
                                 pose_valid=valid, contact=contact, tactile_force=force, obs=obs, action=action,
                                 targets=targets, stage=iface.stage, tracking_error=error,
                                 **({} if iface.estimate is None else dict(
                                     pad_force=pad, estimated_pos=iface.estimate[0], estimated_axis=iface.estimate[1],
                                     estimate_log_std=iface.estimate[2]))))
                if k % 30 == 0:
                    self.log(f'step {k:4d}: stage {iface.stage}, keypoint error '
                             f'{iface.keypoint_distance(pos, quat)*1000:6.1f} mm, pose age {age*1000:4.0f} ms, '
                             f'arm lag {error:.3f} rad, late {late*1000:.0f} ms, '
                             f'contact {"".join(str(int(c)) for c in contact)}')
                k += 1
            else:
                outcome = 'max_steps'
        except KeyboardInterrupt:
            outcome = 'interrupted'
        except TactileLost as exc:
            self.log(f'Tactile lost: {exc}')
            outcome = 'tactile_lost'
        finally:
            self.hold()
        return outcome or 'max_steps', rows


def save_episode(path, rows, outcome):
    keys = ('t', 'late', 'q', 'object_pos', 'object_quat', 'pose_age', 'contact', 'tactile_force', 'obs', 'action',
            'targets', 'stage', 'tracking_error')
    data = {k: np.array([r[k] for r in rows]) for k in keys}
    data['distance'] = np.array([r.get('distance', np.nan) for r in rows])
    for key in ('pose_valid', 'pad_force', 'estimated_pos', 'estimated_axis', 'estimate_log_std'):   # pose estimator
        if rows and key in rows[0]:
            data[key] = np.array([r[key] for r in rows])
    np.savez_compressed(path, outcome=outcome, **data)


def load_actor(path, iface):
    actor = RslRlActor(path)
    if (actor.obs_dim, actor.action_dim) != (iface.obs_dim, len(iface.joint_order)):
        raise ValueError(f'{path} does not match the reference: {actor.obs_dim} -> {actor.action_dim}')
    return actor


def arm_config(path, iface, handeye=None, velocity=None):
    """Control config at the trained PVT speed limit (or velocity). A hand-eye result that records its joint
    compensation replaces the control one: its calibrated camera is only consistent with that compensation."""
    config = load_config(path)
    config = dataclasses.replace(config, velocity_limit=velocity or float(max(iface.ref['velocity_limit'][:7])))
    recorded = json.loads(Path(handeye).read_text(encoding='utf-8')).get('joint_compensation') if handeye else None
    if recorded:
        compensation = JointCompensation(tuple(recorded['directions']), tuple(recorded['offsets_deg']))
        if compensation != config.compensation:
            print(f"Joint compensation of the hand-eye result ({recorded.get('source', handeye)}), offsets "
                  f"{list(compensation.offsets_deg)} deg, replaces {config.compensation_path} "
                  f"({list(config.compensation.offsets_deg)} deg)")
        config = dataclasses.replace(config, compensation=compensation,
                                     compensation_path=Path(recorded['source']) if 'source' in recorded else None)
    model_lower, model_upper = config.compensation.model_limits(config.lower, config.upper)
    gap = max(np.abs(model_lower - iface.lower[:7]).max(), np.abs(model_upper - iface.upper[:7]).max())
    if gap > np.radians(1.):
        print(f'WARNING: real arm limits differ from the trained ones by up to {np.degrees(gap):.1f} deg')
    return config


def add_tactile_args(parser):
    parser.add_argument('--no-tactile', action='store_true', help='Feed zero contact (no PaXini sensors)')
    parser.add_argument('--tactile-threshold', type=float,
                        help='Tared normal force (N) counted as contact (default: the trained threshold)')
    parser.add_argument('--max-tactile-age', type=float, default=.2, help='s: an older sensor reading stops the episode')


def open_tactile(args, iface):
    """FingertipContacts of the four PaXini fingertips (thumb first), or None with --mock / --no-tactile."""
    if args.mock or args.no_tactile:
        print('WARNING: no tactile, the policy sees no finger contact')
        return None
    from gx4m_real.robot.tactile import FingertipContacts
    threshold = args.tactile_threshold or iface.ref['binary_tactile_threshold_n']
    tactile = FingertipContacts.open(threshold_n=threshold)
    try:       # a first zero for the displayed forces; each episode tares again with the hand open at the start
        print(f'PaXini fingertips (thumb, index, middle, ring) tared at {np.round(tactile.tare(.5), 2).tolist()} N, '
              f'contact above {threshold} N after the tare at each episode start')
    except Exception:
        tactile.close()
        raise
    return tactile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--step', type=step_key, choices=list(REFERENCES), default=1,
                        help='Split step 1 or 2, or beam1 (Beam part 2 lift)')
    parser.add_argument('--model', type=Path, help="Default: the step's deployed model")
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
    parser.add_argument('--max-pose-age', type=float, default=1.,
                        help='s: tracking lost this long stops the episode (trained with poses up to 0.13 s old)')
    parser.add_argument('--max-tracking-error', type=float, default=.3, help='Arm target - measured, rad')
    add_tactile_args(parser)
    parser.add_argument('--yes', action='store_true', help='No keypress prompts (mock runs)')
    parser.add_argument('--out', type=Path, help='New directory for logs (default: runs/step<n>_<time>)')
    args = parser.parse_args(argv)
    if not args.mock and args.handeye is None:
        parser.error('--handeye is required on the real robot')

    iface = Step1Interface(load_reference(REFERENCES[args.step]), estimator=ESTIMATORS.get(args.step))
    args.model = args.model or MODELS[args.step]
    actor = load_actor(args.model, iface)
    config = arm_config(args.config, iface, args.handeye, args.arm_velocity)
    mesh_to_object = np.loadtxt(args.mesh_to_object) if args.mesh_to_object else None
    tactile = open_tactile(args, iface)
    out =args.out or HERE/'runs'/datetime.now().strftime(f'step{args.step}_%Y%m%d_%H%M%S'.replace('stepbeam', 'beam'))
    out.mkdir(parents=True, exist_ok=False)
    (out/'run.json').write_text(json.dumps(dict(
        argv=sys.argv if argv is None else argv, step=args.step, model=str(args.model.resolve()), velocity_limit=config.velocity_limit,
        arm_speed_scale=args.arm_speed_scale or iface.arm_speed_scale, torque_limit=config.torque_limit,
        max_joint_step_rad=config.max_joint_step_rad, handeye=str(args.handeye) if args.handeye else None,
        mesh_to_object=None if mesh_to_object is None else mesh_to_object.tolist(),
        tactile=None if tactile is None else dict(threshold_n=tactile.threshold_n, max_age=args.max_tactile_age,
                                                  newton_per_count=tactile.newton_per_count),
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
    runner = Step1Runner(iface, actor, arm, hand, gear, camera, tactile=tactile, arm_speed_scale=args.arm_speed_scale,
                         max_pose_age=args.max_pose_age, max_tactile_age=args.max_tactile_age,
                         max_tracking_error=args.max_tracking_error, approach_speed=args.approach_speed)
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
            ok, message = start_check(iface.ref, T[:3, 3], matrix_quat(T[:3, :3]))
            if not ok:
                print(f'WARNING: {message}')
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
        if tactile is not None:
            tactile.close()
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
