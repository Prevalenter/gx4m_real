"""Joint response of the GX4M hand for system identification: one joint at a time, the others held at the step1
start pose, targets written at the policy's 30 Hz, present position, velocity and current read as fast as the bus
allows (about 70 Hz).

Per joint: steps (+-5 and +-15 deg), ramps (20 deg at 0.5, 1 and 2 rad/s) and a chirp (0.2 -> 4 Hz, amplitude
limited to 2 rad/s), all inside the URDF limits less a margin; --scale shrinks every amplitude (first runs: 0.25).

Collisions are checked before anything moves, on every target sample: the arm joints are read once (status
queries only, the arm stays as it is), and with them every hand link origin and fingertip pad must stay
--table-clearance above the table, and fingertips and links of different fingers --finger-clearance apart.
While running, a joint whose current stays near the goal current (stalled, e.g. blocked) or that stays far from
its target stops the run: the hand holds its measured pose, then torque goes off at the end.

  python tasks/dexlong_gear3/tools/identify_hand.py --port /dev/ttyUSB0 --scale .25          # hardware
  python tasks/dexlong_gear3/tools/identify_hand.py --mock --joints 1 2 --no-arm             # simulated bus
  python tasks/dexlong_gear3/tools/identify_hand.py --analyze tasks/dexlong_gear3/runs/hand_ident_<time>

Output (tasks/dexlong_gear3/runs/hand_ident_<time>/): plan.json, joint_XX.npz (t, target, q, qd, current, written:
whether a target was written this sample, segment index), report.json from --analyze.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from gx4m_real.robot.hand import HAND_JOINTS, LOWER, UPPER, HandDriver   # noqa: E402

COMMAND_HZ = 30.
MARGIN = .05                    # rad inside the URDF limits
TABLE_Z = .21                   # env frame, the plate/table surface of the reference (fixture_pos z)
STEPS_DEG = (5., 15.)
RAMP_SPEEDS = (.5, 1., 2.)      # rad/s
RAMP_DEG = 20.
CHIRP = dict(f0=.2, f1=4., seconds=10., amplitude_deg=5., max_speed=2.)
HOLD, SETTLE = 1.5, 1.
STALL_FRACTION, STALL_SEC = .9, .5
FINGERS = ('thumb', 'index', 'middle', 'ring')
FAR_RAD, FAR_SEC = .6, 1.


def segments_for(j, base, scale=1.):
    """(kind, params, target function of t, duration) of joint j around base (16,), inside the margin."""
    lo, hi = LOWER[j] + MARGIN, UPPER[j] - MARGIN
    b = float(np.clip(base[j], lo, hi))
    out = []
    for deg in STEPS_DEG:
        for sign in (1, -1):
            goal = b + sign*np.radians(deg)*scale
            if lo <= goal <= hi:
                out.append(('step', dict(start=b, goal=goal), (lambda t, g=goal: g if t < HOLD else b), HOLD + SETTLE))
    span = np.radians(RAMP_DEG)*scale
    top = b + span if b + span <= hi else b - span
    if lo <= top <= hi:
        for v in RAMP_SPEEDS:
            T = abs(top - b)/v
            f = (lambda t, T=T, top=top: b + (top - b)*min(t/T, 1.) if t < T + .5
                 else top + (b - top)*min((t - T - .5)/T, 1.))
            out.append(('ramp', dict(start=b, top=top, speed=v), f, 2*T + .5 + SETTLE))
    c = CHIRP
    amp = np.radians(c['amplitude_deg'])*scale
    center = float(np.clip(b, lo + amp, hi - amp)) if hi - lo > 2*amp else (lo + hi)/2
    k = (c['f1'] - c['f0'])/c['seconds']

    def chirp(t, amp=amp, center=center):
        if t >= c['seconds']:
            return b if t >= c['seconds'] + .5 else center
        f = c['f0'] + k*t
        a = min(amp, c['max_speed']/(2*np.pi*f))
        return center + a*np.sin(2*np.pi*(c['f0']*t + k*t*t/2))
    out.append(('chirp', dict(center=center, amplitude=amp, **c), chirp, c['seconds'] + .5 + SETTLE))
    return out


def joint_plan(j, base, scale=1.):
    """Target samples (n, 16) at COMMAND_HZ and segment index per sample for joint j, starting and ending at base."""
    rows, index = [], []
    segments = segments_for(j, base, scale)
    for s, (_, _, f, duration) in enumerate(segments):
        for t in np.arange(0., duration, 1/COMMAND_HZ):
            q = np.array(base, float)
            q[j] = f(t)
            rows.append(q)
            index.append(s)
    return np.array(rows), np.array(index), [(kind, params) for kind, params, _, _ in segments]


class Clearance:
    """Hand geometry checks on target samples: link origins and fingertip pads above the table, fingers apart."""

    def __init__(self, iface, arm_q, table_z=TABLE_Z, table_clearance=.05, finger_clearance=.012):
        self.iface, self.arm_q = iface, np.asarray(arm_q, float)
        self.table_z, self.table_clearance, self.finger_clearance = table_z, table_clearance, finger_clearance
        names = iface.fk.link_poses(dict(zip(iface.joint_order, np.r_[self.arm_q, np.zeros(16)]))).keys()
        self.hand_links = sorted(n for n in names if n.startswith('gx4m_link'))
        self.finger_of = finger_links(iface)

    def check(self, hand_q):
        """None, or why this hand pose (16,) is not allowed."""
        q = np.r_[self.arm_q, hand_q]
        poses = self.iface.fk.link_poses(dict(zip(self.iface.joint_order, q)))
        _, _, tips = self.iface.palm_and_tips(q)
        points = {n: poses[n][:3, 3] for n in self.hand_links}
        points.update({f'tip{k}': p for k, p in enumerate(tips)})
        name, low = min(((n, p[2]) for n, p in points.items()), key=lambda item: item[1])
        if low < self.table_z + self.table_clearance:
            return f'{name} {(low - self.table_z)*1000:.0f} mm above the table'
        groups = {}
        for n, p in points.items():
            finger = int(n[3:]) if n.startswith('tip') else self.finger_of.get(n)
            if finger is not None:
                groups.setdefault(finger, []).append(p)
        fingers = sorted(groups)
        for a in range(len(fingers)):
            for b in range(a + 1, len(fingers)):
                d = np.linalg.norm(np.asarray(groups[fingers[a]])[:, None] - np.asarray(groups[fingers[b]])[None], axis=2)
                if d.min() < self.finger_clearance:
                    return f'{FINGERS[fingers[a]]} and {FINGERS[fingers[b]]} fingers {d.min()*1000:.0f} mm apart'
        return None


def finger_links(iface):
    """{link name: finger index 0..3 (thumb, index, middle, ring)} for the hand links below each finger's first
    joint; the tips are tip0..tip3 in the same order (reference fingertip_bodies)."""
    parent_of = {child: parent for _, _, parent, child, _, _ in iface.fk.joints}
    owner = {}
    for k, tip in enumerate(iface.tip_bodies):
        link = tip
        while link in parent_of and parent_of[link] != iface.palm_body:
            owner[link] = k
            link = parent_of[link]
        owner[link] = k
    return owner


def run_joint(hand, j, targets, index, *, goal_current, log=print):
    """Execute one joint's target samples; returns the record and None, or the record and why it stopped."""
    t0 = time.monotonic()
    period = 1/COMMAND_HZ
    rows, k, stall_since, far_since, reason = [], -1, None, None, None
    while True:
        now = time.monotonic() - t0
        due = int(now/period)
        if due >= len(targets):
            break
        written = due != k
        if written:
            k = due
            hand.command(targets[k])
        q, qd, current = hand.read_state()
        stamp = time.monotonic() - t0
        rows.append((stamp, k, written, q, qd, current))
        stalled = abs(current[j]) >= STALL_FRACTION*goal_current
        far = abs(q[j] - targets[k][j]) > FAR_RAD
        stall_since = (stall_since or stamp) if stalled else None
        far_since = (far_since or stamp) if far else None
        if stall_since is not None and stamp - stall_since > STALL_SEC:
            reason = f'{HAND_JOINTS[j]} stalled at {current[j]:.0f} mA'
        elif far_since is not None and stamp - far_since > FAR_SEC:
            reason = f'{HAND_JOINTS[j]} {np.degrees(q[j] - targets[k][j]):+.0f} deg from its target'
        if reason:
            hand.command(q)
            break
    t, ks, written, q, qd, current = (np.array(v) for v in zip(*rows))
    return dict(t=t, target=targets[ks], q=q, qd=qd, current=current, written=written,
                segment=index[ks], sample=ks), reason


def move_to(hand, goal, speed=1.):
    q = hand.read()
    T = max(np.abs(goal - q).max()/speed, 1/COMMAND_HZ)
    t0 = time.monotonic()
    while (s := (time.monotonic() - t0)/T) < 1.:
        hand.command(q + s*(goal - q))
        time.sleep(1/COMMAND_HZ)
    hand.command(goal)
    time.sleep(.5)


def read_arm(args):
    """Arm model joints (7,) from status queries only, with the hand-eye result's compensation."""
    from gx4m_real.robot.compensation import JointCompensation
    from gx4m_real.robot.config import load_config
    from gx4m_real.robot.feedback import read_joint_positions

    c = json.loads(Path(args.handeye).read_text())['joint_compensation']
    comp = JointCompensation(tuple(c['directions']), tuple(c['offsets_deg']))
    return comp.to_model(read_joint_positions(load_config(args.config)))


def collect(args):
    from tasks.dexlong_gear3.step1 import Step1Interface

    iface = Step1Interface()
    base = np.asarray(iface.ref['initial_joint_pos'][7:], float)
    joints = [j - 1 for j in args.joints]
    arm_q = np.asarray(iface.ref['initial_joint_pos'][:7], float) if args.no_arm else read_arm(args)
    clearance = Clearance(iface, arm_q, table_clearance=args.table_clearance, finger_clearance=args.finger_clearance)
    plans = {}
    for j in joints:
        targets, index, segments = joint_plan(j, base, args.scale)
        for q in targets[::3]:
            problem = clearance.check(q)
            if problem:
                raise SystemExit(f'{HAND_JOINTS[j]}: {problem}; nothing moved. Raise the arm or lower --scale.')
        plans[j] = (targets, index, segments)
    seconds = sum(len(p[0]) for p in plans.values())/COMMAND_HZ
    print(f'Checked {len(joints)} joints ({seconds:.0f} s of motion) against the table and between fingers; '
          f'arm (model, deg) {np.degrees(arm_q).round(1).tolist()}')
    out = args.out or ROOT/'tasks/dexlong_gear3/runs'/datetime.now().strftime('hand_ident_%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    hand = HandDriver(args.port, serial_number=args.serial_number, mock=args.mock,
                      goal_current=args.goal_current, goal_pwm=args.goal_pwm)
    (out/'plan.json').write_text(json.dumps(dict(
        argv=sys.argv, scale=args.scale, command_hz=COMMAND_HZ, margin_rad=MARGIN, base=base.tolist(),
        arm_model_rad=arm_q.tolist(), curr_limit=hand.limits[0], goal_current=hand.limits[1], goal_pwm=hand.limits[2],
        mock=args.mock, joints={HAND_JOINTS[j]: [dict(kind=k, **p) for k, p in plans[j][2]] for j in joints},
        sim=dict((k, iface.ref[k][7:]) for k in ('stiffness', 'damping', 'armature', 'joint_friction',
                                                    'effort_limit', 'velocity_limit'))), indent=1) + '\n')
    if not args.yes:
        input(f'Enter: hand to the start pose, then {len(joints)} joints one at a time (Ctrl-C holds and stops) ')
    stopped = None
    try:
        hand.connect()
        move_to(hand, base)
        for j in joints:
            targets, index, segments = plans[j]
            print(f'{HAND_JOINTS[j]}: {", ".join(k for k, _ in segments)} ({len(targets)/COMMAND_HZ:.0f} s)', flush=True)
            record, stopped = run_joint(hand, j, targets, index, goal_current=hand.limits[1])
            np.savez_compressed(out/f'joint_{j + 1:02d}.npz', joint=j, **record)
            if stopped:
                print(f'Stopped: {stopped}; holding the measured pose', flush=True)
                break
            move_to(hand, base)
    except KeyboardInterrupt:
        stopped = 'interrupted'
        hand.command(hand.read())
    finally:
        hand.close()
    (out/'result.json').write_text(json.dumps(dict(stopped=stopped), indent=1) + '\n')
    print(f'Saved {out}')
    return out


def first_order_sim(target, t, k, d, inertia, friction, effort, dt=1/120):
    """The simulated joint (PD on position with torque limit, armature, Coulomb friction) driven by the same
    sample-and-hold targets, at the physics step of training (dt)."""
    times = np.arange(t[0], t[-1], dt)
    goal = target[np.searchsorted(t, times, side='right') - 1]
    q, v, out = goal[0], 0., []
    for g in goal:
        tau = np.clip(k*(g - q) - d*v, -effort, effort)
        tau -= np.sign(v)*min(friction, abs(tau)) if v else np.sign(tau)*min(friction, abs(tau))
        v += tau/inertia*dt
        q += v*dt
        out.append(q)
    return np.interp(t, times, out)


def analyze(directory):
    directory = Path(directory)
    plan = json.loads((directory/'plan.json').read_text())
    report = {}
    for path in sorted(directory.glob('joint_*.npz')):
        d = np.load(path)
        j = int(d['joint'])
        name = HAND_JOINTS[j]
        t, target, q, cur = d['t'], d['target'][:, j], d['q'][:, j], d['current'][:, j]
        s = plan['sim']
        sim = first_order_sim(target, t, s['stiffness'][j], s['damping'][j], s['armature'][j],
                              s['joint_friction'][j], s['effort_limit'][j])
        segments = plan['joints'][name]
        rows = dict(steps=[], ramps=[], chirp=None)
        for k, seg in enumerate(segments):
            m = d['segment'] == k
            if not m.any():
                continue
            ts, tg, qs, ss = t[m] - t[m][0], target[m], q[m], sim[m]
            if seg['kind'] == 'step':
                rows['steps'].append(dict(step_deg=np.degrees(seg['goal'] - seg['start']),
                                          real=step_metrics(ts, seg['start'], seg['goal'], qs),
                                          sim=step_metrics(ts, seg['start'], seg['goal'], ss),
                                          max_current_ma=float(np.abs(cur[m]).max())))
            elif seg['kind'] == 'ramp':
                moving = np.abs(np.gradient(tg, ts)) > .5*seg['speed']
                lag = lambda y: float(np.degrees(np.median(np.abs(tg[moving] - y[moving])))) if moving.any() else None
                rows['ramps'].append(dict(speed=seg['speed'], lag_deg=lag(qs), sim_lag_deg=lag(ss)))
            else:
                rows['chirp'] = dict(delay_ms=xcorr_delay(ts, tg, qs), sim_delay_ms=xcorr_delay(ts, tg, ss),
                                     amplitude_ratio_by_band=band_gain(ts, tg, qs, seg),
                                     sim_amplitude_ratio_by_band=band_gain(ts, tg, ss, seg))
        dt = np.diff(t)
        rows['read_period_ms'] = float(np.median(dt)*1000)
        report[name] = rows
    (directory/'report.json').write_text(json.dumps(report, indent=1, default=float) + '\n')
    print(f"{'joint':<12} {'step10 ms':>9} {'rise ms':>8} {'ss err':>7} {'ramp lag deg @0.5/1/2':>24} {'delay ms':>9}"
          f"   sim: {'rise ms':>7} {'ramp lag':>18} {'delay':>6}")
    for name, r in report.items():
        big = [s for s in r['steps'] if abs(s['step_deg']) > 10] or r['steps']
        mean = lambda key, side='real': np.nanmean([s[side][key] for s in big if s[side][key] is not None] or [np.nan])
        lags = '/'.join(f"{x['lag_deg']:.1f}" for x in r['ramps'])
        sim_lags = '/'.join(f"{x['sim_lag_deg']:.1f}" for x in r['ramps'])
        c = r['chirp'] or {}
        print(f"{name:<12} {mean('t10_ms'):9.0f} {mean('rise_ms'):8.0f} {mean('steady_error_deg'):7.1f} {lags:>24} "
              f"{c.get('delay_ms', np.nan):9.0f}   {mean('rise_ms', 'sim'):7.0f} {sim_lags:>18} "
              f"{c.get('sim_delay_ms', np.nan):6.0f}")
    print(f'Saved {directory/"report.json"}')
    return report


def step_metrics(t, start, goal, q):
    """10% time, 10-90% rise, overshoot and steady error (last 0.3 s of the hold) of one step. Progress is
    measured from where the joint was when the step was sent (q[0]), not the commanded start: with a deadband
    the joint rests short of it."""
    hold = t < HOLD
    start = q[0]
    span = goal - start
    if abs(span) < 1e-4:
        return dict(t10_ms=None, rise_ms=None, overshoot_pct=0., steady_error_deg=0.)
    p = (q[hold] - start)/span
    th = t[hold]
    cross = lambda level: float(th[np.argmax(p >= level)]*1000) if (p >= level).any() else None
    t10, t90 = cross(.1), cross(.9)
    late = hold & (t > HOLD - .3)
    return dict(t10_ms=t10, rise_ms=None if t10 is None or t90 is None else t90 - t10,
                overshoot_pct=float(max(p.max() - 1, 0)*100),
                steady_error_deg=float(np.degrees(np.mean(goal - q[late]))) if late.any() else None)


def xcorr_delay(t, target, q, max_delay=.5):
    """Delay (ms) of q behind target maximising their correlation, on a uniform 1 ms grid."""
    grid = np.arange(t[0], t[-1], .001)
    a = np.interp(grid, t, target) - np.mean(target)
    b = np.interp(grid, t, q) - np.mean(q)
    lags = np.arange(0, int(max_delay*1000))
    scores = [np.dot(a[:len(a) - L], b[L:]) for L in lags]
    return float(lags[int(np.argmax(scores))])


def band_gain(t, target, q, seg, bands=((.2, 1.), (1., 2.), (2., 3.), (3., 4.))):
    """Peak-to-peak ratio q / target within chirp frequency bands."""
    k = (seg['f1'] - seg['f0'])/seg['seconds']
    f = seg['f0'] + k*t
    out = {}
    for lo, hi in bands:
        m = (f >= lo) & (f < hi) & (t < seg['seconds'])
        if m.sum() > 5:
            out[f'{lo:g}-{hi:g} Hz'] = float(np.ptp(q[m])/max(np.ptp(target[m]), 1e-9))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--analyze', type=Path, help='Analyse a saved run instead of collecting')
    parser.add_argument('--port')
    parser.add_argument('--serial-number')
    parser.add_argument('--mock', action='store_true', help='Simulated hand bus')
    parser.add_argument('--joints', type=int, nargs='+', default=list(range(1, 17)), help='1..16 (gx4m_jointN)')
    parser.add_argument('--scale', type=float, default=1., help='Amplitude scale (first runs: 0.25)')
    parser.add_argument('--goal-current', type=int, default=600)
    parser.add_argument('--goal-pwm', type=int, default=200)
    parser.add_argument('--handeye', type=Path,
                        default=ROOT/'apps/charuco_handeye/runtime/he_20261102_185319_j1p87/calibration/handeye_result.json')
    parser.add_argument('--config', type=Path, default=ROOT/'apps/control/config.yaml')
    parser.add_argument('--no-arm', action='store_true', help='Check clearance at the reference arm pose (no CAN)')
    parser.add_argument('--table-clearance', type=float, default=.05)
    parser.add_argument('--finger-clearance', type=float, default=.012)
    parser.add_argument('--yes', action='store_true')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.analyze:
        return analyze(args.analyze)
    if not args.mock and not (args.port or args.serial_number):
        parser.error('--port or --serial-number (or --mock)')
    if not 0 < args.scale <= 1 or not all(1 <= j <= 16 for j in args.joints):
        parser.error('--scale in (0, 1], --joints in 1..16')
    out = collect(args)
    analyze(out)


if __name__ == '__main__':
    main()
