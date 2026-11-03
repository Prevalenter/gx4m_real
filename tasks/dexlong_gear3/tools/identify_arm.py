"""Joint response of the GX7 arm for system identification, through the policy's command path: model-frame targets
-> joint compensation of the hand-eye result -> ArmDriver.command at 30 Hz (PVT at the trained 1.5 rad/s limit),
feedback recorded at 100 Hz.

Plan (model frame, one joint at a time, the others held): at each base pose (the step1 start pose, and the deepest
pose of a recorded episode that keeps the hand 8 cm above the table), steps of +-2 and +-5 deg, ramps of 8 deg at
0.05/0.1/0.15/0.3 rad/s (the other way round when one way is too close) and a chirp (0.1 -> 2 Hz, amplitude limited
to 0.3 rad/s); with --replay also open-loop replays of recorded episode targets. Segments that fail the clearance
checks are left out and listed. Moves between parts go at 0.1 rad/s. All targets stay 5 deg inside the model joint
limits.

Collisions: every third target sample of the whole plan (with the hand at its start pose) is checked before
anything moves, on points sampled from the URDF collision meshes: --table-clearance above the table, clear of the
plate at its trained pose and of a keep-out sphere around the calibrated camera, and --self-clearance of the
forearm and hand from the column (base_link) and upper arm (Link1, Link2), and of Link3/Link4 from the column.
While running, the measured pose gets the table and camera checks every sample, and a joint more than
--max-tracking-error from its target stops the run (the arm holds).

The real arm only moves from the Viser page of --hardware, by the operator's buttons: 1 connect arm and hand
(both hold their measured poses), 2 arm and hand to the start pose (the hand first, then the arm along a straight
joint move at 0.1 rad/s that is clearance-checked first), 3 start collecting (only from the start pose), stop
(holds), 4 disconnect (arm motors off: support the arm; hand torque off). Nothing on the command line moves it.

  python tasks/dexlong_gear3/tools/identify_arm.py --check            # print the plan and its clearance checks
  python tasks/dexlong_gear3/tools/identify_arm.py --preview          # Viser animation of the plan, no CAN
  python tasks/dexlong_gear3/tools/identify_arm.py --mock-run         # the plan through a simulated arm and hand
  python tasks/dexlong_gear3/tools/identify_arm.py --hardware --hand-port /dev/ttyUSB0     # buttons, real robot
  python tasks/dexlong_gear3/tools/identify_arm.py --hardware --mock  # the same buttons on a simulated arm and hand

Output of a run (tasks/dexlong_gear3/runs/arm_ident_<time>/): plan.json (segments, checks, settings), run.npz
(t, target, q in the model frame, motor, torque, segment, written) and result.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from gx4m_real.robot.kinematics import rpy_matrix                        # noqa: E402

COMMAND_HZ, FEEDBACK_HZ = 30., 100.
LIMIT_MARGIN = np.radians(5.)
STEPS_DEG = (2., 5.)
HOLD, SETTLE = 2., 1.5
RAMP_DEG, RAMP_SPEEDS = 8., (.05, .1, .15, .3)
CHIRP = dict(f0=.1, f1=2., seconds=15., amplitude_deg=3., max_speed=.3)
TRANSFER_SPEED = .1
TABLE_Z = .21                   # env frame (fixture_pos z of the reference)
CAMERA_RADIUS = .12
POINTS_PER_M2 = 8000            # collision-mesh surface samples (30..600 per link)
UPPER_LINKS, MID_LINKS = ('Link1', 'Link2'), ('Link3', 'Link4')
EPISODES = ROOT/'tasks/dexlong_gear3/runs'
HANDEYE = ROOT/'apps/charuco_handeye/runtime/he_20261102_185319_j1p87/calibration/handeye_result.json'


def segment_targets(base, j, lower, upper):
    """[(kind, params, f(t) -> joint j target, duration)] around base (7,) for joint j inside [lower, upper]."""
    b = base[j]
    out = []
    for deg in STEPS_DEG:
        for sign in (1, -1):
            goal = b + sign*np.radians(deg)
            if lower[j] <= goal <= upper[j]:
                out.append(('step', dict(start=b, goal=goal), lambda t, g=goal: g if t < HOLD else b, HOLD + SETTLE))
    for v in RAMP_SPEEDS:
        ramp = ramp_segment(b, j, v, lower, upper)
        if ramp is not None:
            out.append(ramp)
    c = CHIRP
    amp = np.radians(c['amplitude_deg'])
    center = float(np.clip(b, lower[j] + amp, upper[j] - amp))
    k = (c['f1'] - c['f0'])/c['seconds']

    def wave(t):
        f = c['f0'] + k*t
        return center + min(amp, c['max_speed']/(2*np.pi*f))*np.sin(2*np.pi*(c['f0']*t + k*t*t/2))
    end = wave(c['seconds'])

    def chirp(t):
        if t < c['seconds']:
            return wave(t)
        # From where the wave ends back to base at the transfer speed (the wave does not end at a zero).
        return end + (b - end)*min((t - c['seconds'])*TRANSFER_SPEED/max(abs(b - end), 1e-9), 1.)
    out.append(('chirp', dict(center=center, amplitude=amp, **c), chirp, c['seconds'] + SETTLE))
    return out


def ramp_segment(b, j, v, lower, upper, sign=1):
    """RAMP_DEG out from b at v (rad/s), half a second there, and back; toward sign first, the other way when that
    leaves the limits. None if neither fits."""
    span = np.radians(RAMP_DEG)
    for top in (b + sign*span, b - sign*span):
        if lower[j] <= top <= upper[j]:
            T = span/v
            f = (lambda t, top=top: b + (top - b)*min(t/T, 1.) if t < T + .5
                 else top + (b - top)*min((t - T - .5)/T, 1.))
            return 'ramp', dict(start=b, top=top, speed=v), f, 2*T + .5 + SETTLE
    return None


def transfer(a, b, speed=TRANSFER_SPEED):
    """Samples of a straight joint-space move a -> b (7,) at speed (rad/s, fastest joint), ending with 1 s at b."""
    n = max(int(np.ceil(np.abs(b - a).max()/speed*COMMAND_HZ)), 1)
    s = np.linspace(0., 1., n + 1)[1:, None]
    return np.vstack([a + s*(b - a), np.repeat(b[None], int(COMMAND_HZ), 0)])


def build_plan(poses, lower, upper, replays=(), check=None):
    """Targets (n, 7) at COMMAND_HZ, segment index per sample, segment descriptions, per-segment clearances and the
    skipped segments, starting at poses[0]. check(samples) -> ({key: (metres, link)} smallest, [violations]):
    excitation and replay segments that fail it are left out (they start and end at their base pose); a failing
    move between poses fails the plan."""
    rows, index, segments, clear, skipped = [], [], [], [], []

    def add(samples, description, essential=False):
        result = check(samples) if check else ({}, [])
        if result[1]:
            if essential:
                raise SystemExit(f"{description['kind']} to {description.get('pose')}: {'; '.join(result[1])}")
            skipped.append(dict(description, why=result[1]))
            return False
        segments.append(description)
        clear.append(result[0])
        rows.extend(samples)
        index.extend([len(segments) - 1]*len(samples))
        return True

    current = np.asarray(poses[0][1], float)
    add([current], dict(kind='start', pose=poses[0][0]), essential=True)
    for name, base in poses:
        base = np.asarray(base, float)
        if np.abs(base - current).max() > 1e-6:
            add(transfer(current, base), dict(kind='transfer', pose=name), essential=True)
        def sampled(j, f, duration):
            samples = []
            for t in np.arange(0., duration, 1/COMMAND_HZ):
                q = base.copy()
                q[j] = f(t)
                samples.append(q)
            return samples

        for j in range(7):
            for kind, params, f, duration in segment_targets(base, j, lower, upper):
                if add(sampled(j, f, duration), dict(kind=kind, pose=name, joint=j + 1, **params)) or kind != 'ramp':
                    continue
                # A ramp that comes too close one way may fit the other way round.
                sign = -np.sign(params['top'] - params['start'])
                flipped = ramp_segment(base[j], j, params['speed'], lower, upper, sign)
                if flipped is not None and flipped[1]['top'] != params['top']:
                    if add(sampled(j, flipped[2], flipped[3]), dict(kind='ramp', pose=name, joint=j + 1, **flipped[1])):
                        skipped.pop()
        current = base
    for name, targets in replays:
        targets = np.asarray(targets, float)
        samples = list(transfer(current, targets[0])) + list(targets) + [targets[-1]]*int(2*COMMAND_HZ)
        if add(samples, dict(kind='replay', pose=name)):
            current = targets[-1]
    add(transfer(current, np.asarray(poses[0][1], float)), dict(kind='transfer', pose=poses[0][0]), essential=True)
    return np.array(rows), np.array(index), segments, clear, skipped


class RobotGeometry:
    """Surface points of the URDF collision meshes per link, for clearance checks in the env frame."""

    def __init__(self, iface, urdf, handeye=HANDEYE, table_z=TABLE_Z):
        import trimesh
        from tasks.dexlong_gear3.run_step1 import pose_matrix

        self.iface, self.table_z = iface, table_z
        root = ET.parse(urdf).getroot()
        rng = np.random.default_rng(0)
        self.points = {}
        for link in root.findall('link'):
            for c in link.findall('collision'):
                m = c.find('geometry/mesh')
                if m is None:
                    continue
                mesh = trimesh.load(Path(urdf).parent/m.get('filename'), force='mesh')
                n = int(np.clip(mesh.area*POINTS_PER_M2, 30, 600))
                p = mesh.vertices[mesh.faces[rng.choice(len(mesh.faces), n, p=mesh.area_faces/mesh.area)]]
                w = rng.dirichlet(np.ones(3), n)
                local = np.einsum('nk,nkd->nd', w, p)
                o = c.find('origin')
                if o is not None:
                    R = rpy_matrix(*[float(v) for v in o.get('rpy', '0 0 0').split()])
                    local = local @ R.T + [float(v) for v in o.get('xyz', '0 0 0').split()]
                self.points[link.get('name')] = np.vstack([self.points.get(link.get('name'), np.empty((0, 3))), local])
        from scipy.spatial import cKDTree

        self.distal = [l for l in self.points if l not in ('base_link', *UPPER_LINKS, *MID_LINKS)]
        base = iface.fk.link_poses({})['base_link']
        self.column = cKDTree(self.points['base_link'] @ base[:3, :3].T + base[:3, 3])     # base_link never moves
        d = json.loads(Path(handeye).read_text())
        T_env_base = pose_matrix(iface.ref['base_pos'], iface.ref['base_quat'])
        self.camera = (T_env_base @ np.asarray(d['T_base_cam']))[:3, 3]
        plate = trimesh.load(ROOT/'asset/mjfit_gears3_split/assembly/fixture_visual.obj', force='mesh')
        T_plate = pose_matrix(iface.ref['fixture_pos'], iface.ref['fixture_quat'])
        self.plate_points = plate.sample(4000) @ T_plate[:3, :3].T + T_plate[:3, 3]
        self.plate_tree = cKDTree(self.plate_points)

    def world(self, q23):
        poses = self.iface.fk.link_poses(dict(zip(self.iface.joint_order, q23)))
        return {l: p @ poses[l][:3, :3].T + poses[l][:3, 3] for l, p in self.points.items()}

    def clearance(self, q23, self_check=True):
        """{table, plate, camera, self: (metres, link)}; self only with self_check."""
        from scipy.spatial import cKDTree

        pts = self.world(q23)
        moving = {l: p for l, p in pts.items() if l != 'base_link'}
        table = min(((p[:, 2].min() - self.table_z, l) for l, p in moving.items()))
        plate = min(((self.plate_tree.query(p)[0].min(), l) for l, p in moving.items()))
        camera = min(((np.linalg.norm(p - self.camera, axis=1).min() - CAMERA_RADIUS, l) for l, p in pts.items()))
        out = dict(table=table, plate=plate, camera=camera)
        if self_check:
            upper = cKDTree(np.vstack([pts[l] for l in UPPER_LINKS]))
            best = min((float(self.column.query(pts[l])[0].min()), f'base_link/{l}') for l in (*MID_LINKS, *self.distal))
            best = min(best, min((float(upper.query(pts[l])[0].min()), f'Link1-2/{l}') for l in self.distal))
            out['self'] = best
        return out


def segment_checker(geometry, hand_q, limits, stride=3):
    """check(samples) for build_plan: the smallest clearances over every stride-th sample (and the last) and the
    limits they break."""
    def check(samples):
        worst = {}
        for k in list(range(0, len(samples), stride)) + [len(samples) - 1]:
            for key, (d, link) in geometry.clearance(np.r_[samples[k], hand_q]).items():
                if key not in worst or d < worst[key][0]:
                    worst[key] = (float(d), link)
        return worst, [f'{key} {d*1000:.0f} mm ({link})' for key, (d, link) in worst.items() if d < limits[key]]
    return check


def load_replays(paths):
    out = []
    for path in paths:
        d = np.load(path)
        if d['targets'].shape[0] >= 30:
            out.append((Path(path).parent.name, d['targets'][:, :7]))
    return out


def descent_pose(paths, geometry, hand_q, min_table=.08):
    """The deepest measured arm pose of the first recorded episode on the way down to the gear that keeps the hand
    min_table above the table."""
    for path in paths:
        q = np.load(path)['q'][:, :7]
        if len(q) < 30:
            continue
        best = None
        for k in range(0, len(q), 3):
            if geometry.clearance(np.r_[q[k], hand_q], self_check=False)['table'][0] < min_table:
                break
            best = q[k]
        return best
    return None


def plan_from_args(args):
    from tasks.dexlong_gear3.run_step1 import CONTROL_CONFIG, arm_config
    from tasks.dexlong_gear3.step1 import URDF, Step1Interface

    iface = Step1Interface()
    config = arm_config(CONTROL_CONFIG, iface, args.handeye, None)
    lower, upper = config.compensation.model_limits(config.lower, config.upper)
    lower, upper = lower + LIMIT_MARGIN, upper - LIMIT_MARGIN
    episodes = sorted(EPISODES.glob('step1_*/episode_*.npz'))
    geometry = RobotGeometry(iface, URDF, args.handeye)
    hand_q = np.asarray(iface.ref['initial_joint_pos'][7:], float)
    limits = dict(table=args.table_clearance, plate=args.plate_clearance, camera=0., self=args.self_clearance)
    poses = [('start', np.asarray(iface.ref['initial_joint_pos'][:7], float))]
    descent = None if args.start_only else descent_pose(episodes, geometry, hand_q)
    if descent is not None and np.abs(descent - poses[0][1]).max() > np.radians(5.):
        poses.append(('descent', descent))
    replays = load_replays(episodes) if args.replay else []
    targets, index, segments, clear, skipped = build_plan(poses, lower, upper, replays,
                                                          segment_checker(geometry, hand_q, limits))
    if (targets < lower - 1e-9).any() or (targets > upper + 1e-9).any():
        bad = np.unique(np.nonzero((targets < lower - 1e-9) | (targets > upper + 1e-9))[1]) + 1
        raise SystemExit(f'Plan leaves the joint limits (less {np.degrees(LIMIT_MARGIN):.0f} deg) at J{bad.tolist()}')
    return dict(iface=iface, config=config, targets=targets, index=index, segments=segments, geometry=geometry,
                hand_q=hand_q, per=dict(enumerate(clear)), skipped=skipped, problems=[], limits=limits,
                poses=poses, urdf=URDF)


def summary(plan):
    seconds = len(plan['targets'])/COMMAND_HZ
    worst = {key: min((v[key] for v in plan['per'].values() if key in v), key=lambda x: x[0])
             for key in ('table', 'plate', 'camera', 'self')}
    lines = [f"{len(plan['segments'])} segments, {seconds/60:.1f} min at {COMMAND_HZ:.0f} Hz; poses "
             + ', '.join(f"{n} {np.degrees(q).round(1).tolist()}" for n, q in plan['poses'])]
    lines += [f"  smallest {key}: {d*1000:.0f} mm ({link})" for key, (d, link) in worst.items()]
    lines += [f"  limits: table {plan['limits']['table']*1000:.0f} mm, plate {plan['limits']['plate']*1000:.0f} mm, "
              f"camera keep-out {CAMERA_RADIUS*1000:.0f} mm, self {plan['limits']['self']*1000:.0f} mm"]
    lines += [f"  left out ({len(plan['skipped'])}):"] + [
        f"    {s['kind']} {s.get('pose')}" + (f" J{s['joint']}" if 'joint' in s else '')
        + (f" {np.degrees(s['goal'] - s['start']):+.0f} deg" if s['kind'] == 'step' else '')
        + (f" {s['speed']} rad/s" if s['kind'] == 'ramp' else '') + f": {'; '.join(s['why'])}"
        for s in plan['skipped']]
    return '\n'.join(lines)


class PlanRunner:
    """The plan on one ArmDriver (and the hand, with hand_factory), step by step from the operator's buttons:
    connect() (arm and hand hold their measured poses), go_home() (hand, then arm along a checked straight move, to
    the plan's start), start() (from the start only: the plan, recorded), stop() (holds), disconnect() (arm motors
    off, hand torque off). Moves run in a thread, one at a time."""

    def __init__(self, plan, *, mock, max_tracking_error=.15, out_root=ROOT/'tasks/dexlong_gear3/runs', log=print,
                 hand_factory=None, hand_speed=1.):
        from gx4m_real.robot.arm import ArmDriver

        self.plan, self.mock, self.max_tracking_error, self.out_root, self.log = plan, mock, max_tracking_error, out_root, log
        self.config = plan['config']
        self.comp = self.config.compensation
        self.arm = ArmDriver(self.config, mock=mock)
        if mock:        # the simulated arm starts away from the plan's start (its default pose would cross the camera)
            self.arm.q = self.comp.to_motor(plan['targets'][0] + np.radians([-6, 4, 0, 5, 0, 0, 0]))
        self.hand_factory, self.hand, self.hand_speed = hand_factory, None, hand_speed
        self.stop_event, self.thread = threading.Event(), None
        self.status, self.progress, self.out, self.busy = '未连接', 0., None, None

    @property
    def connected(self):
        return self.arm.robot is not None

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def measured(self):
        """Measured arm joints (7,), model frame, or None."""
        return self.comp.to_model(self.arm.q) if self.connected else None

    def measured_hand(self):
        return None if self.hand is None or self.hand.q is None else np.asarray(self.hand.q)

    def at_start(self, tolerance=.02):
        q = self.measured()
        return q is not None and np.abs(q - self.plan['targets'][0]).max() <= tolerance

    def connect(self):
        if not self.connected:
            self.arm.connect()
        self.status = '机械臂已连接，保持当前位置'
        if self.hand_factory is not None and self.hand is None:
            hand = self.hand_factory()
            try:
                hand.connect()
            except Exception as exc:
                self.status = f'机械臂已连接；灵巧手连接失败：{exc}'
                raise RuntimeError(self.status) from exc
            self.hand = hand
            self.status = '机械臂和灵巧手已连接，保持当前位置'

    def disconnect(self):
        if self.running:
            raise RuntimeError('先停止')
        try:
            self.arm.disconnect()
        finally:
            hand, self.hand = self.hand, None
            if hand is not None:
                hand.close()
        self.status = '已断开：机械臂电机失能，灵巧手扭矩关闭'

    def stop(self):
        self.stop_event.set()

    def _ready(self):
        if not self.connected or self.arm.mode == 'fault' or self.running:
            raise RuntimeError('机械臂没有连接、已故障或正在运行')

    def _launch(self, target, name, status):
        self.stop_event.clear()
        self.status, self.busy, self.progress = status, name, 0.
        self.thread = threading.Thread(target=target, daemon=True, name=f'arm-ident-{name}')
        self.thread.start()

    def go_home(self):
        """Hand to its start pose, then the arm straight to the plan's start at TRANSFER_SPEED, after checking that
        move's clearances with the hand at its start pose."""
        self._ready()
        self.arm.update()
        lead = transfer(self.measured(), self.plan['targets'][0])
        limits = self.plan['limits']
        for q in lead[::3]:
            c = self.plan['geometry'].clearance(np.r_[q, self.plan['hand_q']])
            bad = [key for key in ('table', 'plate', 'camera', 'self') if c[key][0] < limits[key]]
            if bad:
                raise RuntimeError(f'从当前位置到初始位置的直线移动不安全：{bad[0]} {c[bad[0]][0]*1000:.0f} mm'
                                   f'（{c[bad[0]][1]}），请先把机械臂移到初始位置附近')
        self._launch(lambda: self._home(lead), 'home', '回初始位置中')

    def _home(self, lead):
        stopped = None
        try:
            if self.hand is not None:
                stopped = self._hand_home()
            if stopped is None:
                stopped = self._drive(lead, None, None)
        except Exception as exc:
            stopped = f'出错：{exc}'
        finally:
            stopped = self._hold(stopped)
            self.status = (f'回初始位置已停止：{stopped}' if stopped else
                           '已在初始位置' + ('' if self.hand is not None else '（灵巧手未连接）')) + '，保持当前位置'
            self.log(self.status)

    def _hand_home(self):
        goal = np.asarray(self.plan['hand_q'])
        q = self.hand.read()
        T = max(np.abs(goal - q).max()/self.hand_speed, 1/COMMAND_HZ)
        t0 = time.monotonic()
        while (s := (time.monotonic() - t0)/T) < 1.:
            if self.stop_event.is_set():
                return '操作员停止'
            self.hand.command(q + s*(goal - q))
            self.hand.read()
            time.sleep(1/COMMAND_HZ)
        self.hand.command(goal)
        time.sleep(.5)
        self.hand.read()
        return None

    def _hold(self, stopped):
        if self.arm.robot is not None and self.arm.mode != 'fault':
            try:
                self.arm.hold()
            except Exception as exc:
                stopped = (stopped or '') + f'；机械臂保持失败：{exc}'
        if self.hand is not None and stopped:
            try:
                self.hand.command(self.hand.read())
            except Exception as exc:
                stopped += f'；灵巧手保持失败：{exc}'
        return stopped

    def start(self):
        """The plan from its start (go_home first), recorded in a new directory."""
        self._ready()
        self.arm.update()
        if not self.at_start():
            raise RuntimeError('机械臂不在初始位置，先点“机械臂和灵巧手回初始位置”')
        targets, index = self.plan['targets'], self.plan['index']
        self.out = self.out_root/datetime.now().strftime('arm_ident_%Y%m%d_%H%M%S')
        self.out.mkdir(parents=True, exist_ok=False)
        (self.out/'plan.json').write_text(json.dumps(dict(
            argv=sys.argv, mock=self.mock, command_hz=COMMAND_HZ, feedback_hz=FEEDBACK_HZ,
            velocity_limit=self.config.velocity_limit, torque_limit=self.config.torque_limit,
            max_joint_step_rad=self.config.max_joint_step_rad, compensation_offsets_deg=list(self.comp.offsets_deg),
            max_tracking_error=self.max_tracking_error, limits=self.plan['limits'], segments=self.plan['segments'],
            skipped=self.plan['skipped'], hand_connected=self.hand is not None,
            hand_q=None if self.measured_hand() is None else self.measured_hand().tolist(),
            checks={str(s): {k: v[:2] for k, v in c.items()} for s, c in self.plan['per'].items()}),
            indent=1, default=float) + '\n')
        self._launch(lambda: self._collect(targets, index), 'collect', '采集中')

    def _collect(self, targets, index):
        rows, stopped = [], None
        try:
            stopped = self._drive(targets, index, rows)
        except Exception as exc:
            stopped = f'出错：{exc}'
        finally:
            stopped = self._hold(stopped)
            if rows:
                stamp, wall, ks, written, q, motor, torque = (np.array(v) for v in zip(*rows))
                np.savez_compressed(self.out/'run.npz', t=stamp, wall=wall, sample=ks, target=targets[ks],
                                    segment=index[ks], written=written, q=q, motor=motor, torque=torque)
            (self.out/'result.json').write_text(json.dumps(dict(stopped=stopped, samples=len(rows)), indent=1) + '\n')
            self.status = (f'采集已停止：{stopped}' if stopped else '采集完成') + '，保持当前位置'
            self.log(f'{self.status}; saved {self.out}')

    def _drive(self, targets, index, rows):
        """Command targets at COMMAND_HZ, read feedback at FEEDBACK_HZ (appended to rows when given); None when done,
        or why it stopped. The caller holds."""
        arm, comp, plan = self.arm, self.comp, self.plan
        k, n = -1, 0
        t0 = time.monotonic()
        while not self.stop_event.is_set():
            now = time.monotonic() - t0
            due = int(now*COMMAND_HZ)
            if due >= len(targets):
                return None
            written = due != k
            if written:
                k = due
                self.progress = k/len(targets)
                arm.command(np.clip(comp.to_motor(targets[k]), self.config.lower, self.config.upper))
            arm.update()
            if arm.mode == 'fault':
                return f'机械臂故障：{arm.error}'
            sample = arm.feedback_sample()
            motor = np.asarray(sample['positions_rad'])
            q = comp.to_model(motor)
            if rows is not None:
                torque = np.asarray(arm.robot.getJT(), float) if not self.mock else np.zeros(7)
                rows.append((sample['timestamp_monotonic'] - t0, now, k, written, q, motor, torque))
            error = np.abs(q - targets[k])
            if error.max() > self.max_tracking_error:
                return f'J{int(error.argmax()) + 1} 偏离目标 {np.degrees(error.max()):.1f}°'
            n += 1
            if n % 3 == 0:
                hand = self.measured_hand()
                c = plan['geometry'].clearance(np.r_[q, plan['hand_q'] if hand is None else hand], self_check=False)
                if c['table'][0] < plan['limits']['table']/2 or c['camera'][0] < 0:
                    return f"实测姿态太近：桌面 {c['table'][0]*1000:.0f} mm，相机 {c['camera'][0]*1000:.0f} mm"
            time.sleep(max(0., 1/FEEDBACK_HZ - (time.monotonic() - t0 - now)))
        return '操作员停止'


def control_panel(server, runner, show_measured):
    """Operator buttons for the real (or --mock) arm and hand; every motion starts from a click here."""
    gui = server.gui
    with gui.add_folder('真机控制（只能由操作员点击）'):
        status = gui.add_markdown('')
        connect = gui.add_button('1. 连接机械臂和灵巧手（保持当前位置）' if runner.hand_factory else
                                 '1. 连接机械臂（保持当前位置）')
        home = gui.add_button('2. 机械臂和灵巧手回初始位置', color='blue')
        start = gui.add_button('3. 开始采集数据', color='green')
        stop = gui.add_button('停止（保持当前位置）', color='red')
        disconnect = gui.add_button('4. 断开机械臂（电机失能，先扶住）')

    def act(f):
        def handler(_):
            try:
                f()
            except Exception as exc:
                runner.status = f'失败：{exc}'
        return handler
    connect.on_click(act(runner.connect))
    home.on_click(act(runner.go_home))
    start.on_click(act(runner.start))
    stop.on_click(act(runner.stop))
    disconnect.on_click(act(runner.disconnect))

    def refresh():
        running, connected = runner.running, runner.connected
        states = [(connect, connected and (runner.hand_factory is None or runner.hand is not None) or running),
                  (home, not connected or running), (start, not connected or running or not runner.at_start()),
                  (stop, not running), (disconnect, not connected or running)]
        for button, disabled in states:
            if button.disabled != disabled:
                button.disabled = disabled
        parts = [f"**状态**：{runner.status}" + ('（模拟）' if runner.mock else '')]
        if running:
            parts.append(f"{'回初始位置' if runner.busy == 'home' else '采集'}进度 {runner.progress*100:.0f}%")
        elif connected:
            parts.append('在初始位置，可以开始采集' if runner.at_start() else '不在初始位置：先点 2')
        if runner.out:
            parts.append(f"记录：`{runner.out}`")
        text = '\n\n'.join(parts)
        if status.content != text:
            status.content = text
        q = runner.measured()
        if q is not None:
            show_measured(q, runner.measured_hand())
    return refresh


def viewer(plan, args, runner=None):
    """Viser animation of the planned targets with the obstacles and per-sample clearances, and with a runner the
    operator's buttons; while connected the robot shows the measured pose."""
    import viser
    from viser.extras import ViserUrdf
    import yourdfpy
    from tasks.dexlong_gear3.run_step1 import pose_matrix

    iface, geometry, targets, index, segments = (plan[k] for k in ('iface', 'geometry', 'targets', 'index', 'segments'))
    model = yourdfpy.URDF.load(str(plan['urdf']), load_meshes=True)
    names = model.actuated_joint_names
    server = viser.ViserServer(host=args.host, port=args.port, label='GX7 · 机械臂关节辨识')
    server.scene.set_up_direction('+z')
    T_base = pose_matrix(iface.ref['base_pos'], iface.ref['base_quat'])
    server.scene.add_frame('/robot', show_axes=False, position=T_base[:3, 3], wxyz=np.asarray(iface.ref['base_quat']))
    robot = ViserUrdf(server, model, root_node_name='/robot')
    center = (T_base[0, 3], T_base[1, 3] + .3)
    server.scene.add_box('/table', dimensions=(1.2, 1.2, .002), color=(200, 200, 205), opacity=.5,
                         position=(*center, TABLE_Z - .001))
    server.scene.add_box('/table_margin', dimensions=(1.2, 1.2, .001), color=(240, 200, 40), opacity=.15,
                         position=(*center, TABLE_Z + plan['limits']['table']))
    server.scene.add_point_cloud('/plate', geometry.plate_points.astype(np.float32), colors=(60, 60, 70),
                                 point_size=.003)
    server.scene.add_icosphere('/camera_keepout', radius=CAMERA_RADIUS, color=(230, 60, 60), opacity=.25,
                               position=geometry.camera)
    server.scene.add_label('/camera_label', '相机禁区', position=geometry.camera + [0, 0, CAMERA_RADIUS + .03])
    gui = server.gui
    gui.add_markdown('## 机械臂关节辨识\n灰色半透明：桌面；黄色：桌面安全线；红球：相机禁区；深色点：底板训练位置。'
                     + ('\n\n预览只播放计划的目标姿态，不连接机械臂。' if runner is None else
                        '\n\n机械臂只在你点击下面的按钮后才动。连接后模型显示实测姿态。'))
    gui.add_markdown('```\n' + summary(plan) + '\n```')
    refresh = control_panel(server, runner, lambda q, hand: set_cfg(q, hand)) if runner is not None else (lambda: None)
    labels = [f"{s}: {d['kind']} {d.get('pose', '')}" + (f" J{d['joint']}" if 'joint' in d else '')
              + (f" {np.degrees(d['goal'] - d['start']):+.0f}°" if d['kind'] == 'step' else '')
              + (f" {d['speed']} rad/s" if d['kind'] == 'ramp' else '') for s, d in enumerate(segments)]
    with gui.add_folder('计划预览'):
        seg = gui.add_dropdown('段', options=labels, initial_value=labels[0])
        slider = gui.add_slider('样本', min=0, max=len(targets) - 1, step=1, initial_value=0)
        play = gui.add_checkbox('播放', initial_value=False)
        speed = gui.add_dropdown('速度', options=['1x', '2x', '5x', '10x'], initial_value='5x')
        info = gui.add_markdown('')
    starts = {s: int(np.argmax(index == s)) for s in range(len(segments))}
    state = dict(syncing=False)

    def set_cfg(arm_q, hand_q=None):
        q = dict(zip(iface.joint_order, np.r_[arm_q, plan['hand_q'] if hand_q is None else hand_q]))
        robot.update_cfg(np.array([q.get(n, 0.) for n in names]))

    def show(k):
        if runner is None or not runner.connected:
            set_cfg(targets[k])
        c = geometry.clearance(np.r_[targets[k], plan['hand_q']])
        s = int(index[k])
        mark = lambda key: '❌' if c[key][0] < plan['limits'][key] else '✅'
        info.content = (f"**{labels[s]}**，t = {k/COMMAND_HZ:.1f} s\n\n"
                        f"目标关节（度）：{np.degrees(targets[k]).round(1).tolist()}\n\n"
                        + '\n\n'.join(f"{mark(key)} {key}: {d*1000:.0f} mm（{link}）" for key, (d, link) in c.items()))
        if not state['syncing']:
            state['syncing'] = True
            seg.value = labels[s]
            state['syncing'] = False

    @slider.on_update
    def _(_):
        show(int(slider.value))

    @seg.on_update
    def _(_):
        if not state['syncing']:
            slider.value = starts[labels.index(seg.value)]

    show(0)
    print(f'Viser: http://{args.host}:{server.get_port()}', flush=True)
    try:
        while True:
            if play.value:
                slider.value = (int(slider.value) + int(speed.value[:-1])) % len(targets)
            refresh()
            time.sleep(1/COMMAND_HZ)
    except KeyboardInterrupt:
        pass
    finally:
        if runner is not None and runner.connected:
            runner.stop()
            if runner.thread is not None:
                runner.thread.join(timeout=5.)
            runner.arm.hold()
            try:
                input('Arm is holding. Support it or move it to a safe pose, then Enter to disable the motors: ')
            except (KeyboardInterrupt, EOFError):
                pass
            runner.disconnect()
        server.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true', help='Print the plan and its clearance checks')
    mode.add_argument('--preview', action='store_true', help='Viser animation of the plan, no hardware')
    mode.add_argument('--mock-run', action='store_true', help='The plan through a simulated arm, no Viser')
    mode.add_argument('--hardware', action='store_true', help='Viser with the operator buttons for the real arm')
    parser.add_argument('--mock', action='store_true', help='With --hardware: the buttons drive a simulated arm and hand')
    parser.add_argument('--hand-port', help='With --hardware: the hand (/dev/ttyUSB0), held at its start pose')
    parser.add_argument('--handeye', type=Path, default=HANDEYE)
    parser.add_argument('--start-only', action='store_true', help='Only the start pose (no descent pose)')
    parser.add_argument('--replay', action='store_true',
                        help='Add the recorded episodes (their grasp reaches the table, so they are usually left out)')
    parser.add_argument('--table-clearance', type=float, default=.05)
    parser.add_argument('--plate-clearance', type=float, default=.03)
    parser.add_argument('--self-clearance', type=float, default=.01)
    parser.add_argument('--max-tracking-error', type=float, default=.15)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8090)
    parser.add_argument('--out', type=Path, help='Directory for run records (default: tasks/dexlong_gear3/runs)')
    args = parser.parse_args(argv)
    plan = plan_from_args(args)
    print(summary(plan), flush=True)
    out_root = args.out or ROOT/'tasks/dexlong_gear3/runs'
    mock = args.mock or args.mock_run
    hand_factory = None
    if mock or args.hand_port:
        from gx4m_real.robot.hand import HandDriver
        hand_factory = lambda: HandDriver(args.hand_port, mock=mock)
    runner = None if args.preview or args.check else PlanRunner(
        plan, mock=mock, max_tracking_error=args.max_tracking_error, out_root=out_root, hand_factory=hand_factory)
    if args.preview or args.hardware:
        viewer(plan, args, runner)
    elif args.mock_run:
        runner.connect()
        for step in (runner.go_home, runner.start):
            step()
            runner.thread.join()
            print(runner.status, flush=True)
        runner.disconnect()


if __name__ == '__main__':
    main()
