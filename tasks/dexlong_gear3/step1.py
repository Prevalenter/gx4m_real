"""Step1 (24-tooth gear onto the left pin) and step2 (48-tooth gear onto the right pin, the 24-tooth gear already
mounted) policy interface, outside Isaac Lab. Both steps share the observation, control and stage logic below; their
references differ only in poses and constants (step2: plate, start and goals of the 48-tooth gear, finger
smoothing 0.1).

Rebuilds the 78-wide policy observation of DexLongGear3SplitStep1Env from measured robot and gear
state, turns the 23 policy actions into joint targets as gx4m_learning's dexlong_play_control, and runs
the task's stage logic, which decides the goal inside the observation:
  stage 0: pre-insert waypoint above the pin -> stage 1: final pose on the pin -> stage 2: withdraw the hand.
A stage is reached after success_steps (not necessarily consecutive) control steps with the gear keypoints
within insertion_success_tolerance * keypoint_scale (3 mm) of the goal. With yaw_free_goal the goal turns
with the gear about its axis, so only position and tilt count (the policy input has no gear yaw either).
The withdrawal succeeds once the gear stays within retract_success_tolerance * keypoint_scale (7.5 mm) of
the final goal and every fingertip is retract_distance_threshold (20 cm) from the hub grasp point.

config/step<n>_reference.json is exported by gx4m_learning's scripts/export_gear3_split_sim2sim_reference.py
(--step n): joint order and limits, palm and fingertip bodies and offsets, robot base pose, goals, stage
settings (p2p), control constants. All positions are in the Isaac env frame: the robot root link sits at
base_pos / base_quat. Quaternions are wxyz except palm_rot inside the observation, which is xyzw like Isaac's.

Occlusion-aware references (a pose_estimator block, step2 since 2026-11-05): the camera loses the gear behind the
hand, so the policy acts on a learned estimate (pose_estimator.py) instead of the tracked pose. observation() then
also takes whether the tracked pose is current (valid), how old it is (age, s) and the tared pad forces (4, 3) N as
(normal into the pad, distal, lateral); object_pos/object_axis are the last tracked pose, held while lost.
estimated_pose() gives the estimate in the env frame for the stage logic while the camera has lost the gear.

Beam lift (key 'beam1', gx4m_learning dexlong_beam_lift step 1: grasp Beam part 2 and lift it 20 cm): the beam has no
symmetry axis, so the policy and estimator carry its full orientation as two body axes (reference['object_axes'], x
then z; the object_frame field) and the goal keeps the exact start orientation (goal_yaw_symmetry 0, not yaw-free).
A reference with a lift block (export ... --task beam_lift) has a single goal, the start pose raised by
lift['height']: reset(object_pose=...) puts it above the measured start, its first hit (success_steps near it) is
the success, and a drop (lifted drop_min_rise, then drop_fall lower relative to the palm than at its highest) ends
the episode as 'drop'. No insert or withdrawal stage.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from gx4m_real.robot.kinematics import UrdfFK, matrix_quat, quat_matrix

HERE = Path(__file__).resolve().parent
# Deployed policy per split step and the reference exported with it.
REFERENCES = {1: HERE/'config'/'step1_reference.json', 2: HERE/'config'/'step2_reference.json',
              'beam1': HERE/'config'/'beam1_reference.json'}
MODELS = {1: HERE/'models'/'step1_lstm_20261103_iter4750.pt', 2: HERE/'models'/'step2_lstm_20261105_iter4500.pt',
          'beam1': HERE/'models'/'beam1_lstm_20261107_iter8000.pt'}
# Gear pose estimator saved with each occlusion-aware policy (reference['pose_estimator']).
ESTIMATORS = {2: HERE/'models'/'step2_estimator_20261105_iter4500.pt',
              'beam1': HERE/'models'/'beam1_estimator_20261107_iter8000.pt'}
# Part each step's perception tracks (perception/gears.py).
PARTS = {1: 'gear24', 2: 'gear48', 'beam1': 'beam2'}


def step_key(value):
    """--step argument: a split step number or a named task ('beam1')."""
    return int(value) if str(value).isdigit() else str(value)
REFERENCE = REFERENCES[1]
URDF = HERE.parents[1]/'asset/gx7_right/urdf/gx7_right_gx4m_cam_xl330_pad.urdf'


def reference_urdf(reference):
    """The URDF the reference was trained with (its robot_urdf, a gx4m_learning path), from this repo's asset/.
    Same rule in perception/viser_scene.py (other environment)."""
    name = str(reference.get('robot_urdf', '')).replace('\\', '/').rsplit('/', 1)[-1]
    path = URDF.parent/name
    return path if name and path.exists() else URDF


def start_check(reference, pos, quat, z_tolerance=.005):
    """(ok, message in Chinese) for an object pose against the trained start. The gears started anywhere within
    +-2 cm in x and y at any yaw (reset_position_noise, randomized yaw); lift references started at one fixed pose,
    so the beam must lie within +-1 cm and 10 deg of it (a guess: the policy never saw another start)."""
    fixed = reference.get('lift') is not None
    xy, angle_limit = (.01, 10.) if fixed else (.02, None)
    offset = np.asarray(pos, dtype=np.float64) - reference['object_pos']
    text = (f"x {offset[0]*1000:+.0f}、y {offset[1]*1000:+.0f}、z {offset[2]*1000:+.0f} mm"
            f"（x、y 各 ±{xy*1000:.0f} mm，z ±{z_tolerance*1000:.0f} mm")
    ok = np.abs(offset[:2]).max() <= xy and abs(offset[2]) <= z_tolerance
    if angle_limit is not None:
        R = quat_matrix(np.asarray(reference['object_quat'], dtype=np.float64)).T @ quat_matrix(np.asarray(quat, dtype=np.float64))
        angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1)/2, -1., 1.))))
        text += f"；朝向差 {angle:.1f}°，要求 ≤ {angle_limit:.0f}°"
        ok = ok and angle <= angle_limit
    return bool(ok), f'离训练起点 {text}）'


def load_reference(path=REFERENCE):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])


def goal_quat(goal, obj, teeth=None):
    """Goal turned about its own z to the object's yaw: exactly (teeth=None, yaw-free goal) or by the
    whole number of teeth closest to it (gx4m_learning snap_goal_yaw)."""
    w, x, y, z = quat_mul(np.asarray(goal)*(1., -1., -1., -1.), np.asarray(obj))
    yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    half = yaw/2 if teeth is None else round(yaw/(2*math.pi/teeth))*(2*math.pi/teeth)/2
    return quat_mul(goal, np.array([math.cos(half), 0., 0., math.sin(half)]))


class Step1Interface:
    """Observation builder, action-to-target mapping and stage tracker; joint vectors are in reference['joint_order'].

    Per control step (1/30 s): obs = observation(...); targets = apply_action(actor(obs)); send the targets;
    after the step, measure again and call update(...), which advances the stage and says when to stop."""

    def __init__(self, reference=None, urdf=None, estimator=None):
        """urdf: default the reference's training URDF. estimator: the pose estimator weights (estimator_<it>.pt) a
        reference with a pose_estimator block needs."""
        r = self.ref = reference or load_reference()
        self.urdf = urdf = Path(urdf) if urdf is not None else reference_urdf(r)
        self.joint_order = list(r['joint_order'])
        self.lower, self.upper = np.array(r['lower']), np.array(r['upper'])
        self.fields = list(r['policy_fields'].items())
        self.obs_dim = sum(w for _, w in self.fields)
        self.fk = UrdfFK(urdf, r['base_pos'], r['base_quat'])
        self.palm_body, self.palm_offset = r['palm_body'], np.array(r['palm_offset'])
        self.tip_bodies, self.tip_offsets = r['fingertip_bodies'], np.array(r['fingertip_offsets'])
        c = r['control']
        if c['kind'] != 'arm_increment_hand_absolute' or c['actions'] != len(self.joint_order):
            raise ValueError(f"Unsupported control {c['kind']!r} with {c['actions']} actions")
        # control['smoothing'] is the 60 Hz constant (0.2); the 30 Hz env uses act_moving_average (step1 0.36,
        # step2 0.1) for the fingers and arm_moving_average for the arm (older references have none: the arm uses the finger value).
        self.smoothing, self.arm_speed_scale, self.dt = r['act_moving_average'], c['arm_speed_scale'], r['step_dt']
        self.arm_smoothing = r.get('arm_moving_average', self.smoothing)
        self.clip = r['clamp_abs_observations']
        self.p2p = r['p2p']
        # Goal yaw: turned with the object (yaw-free), snapped to whole teeth, or exact (no symmetry, the beam).
        self.yaw_free, self.teeth = bool(r.get('yaw_free_goal')), int(r['goal_yaw_symmetry'] or 0)
        # Body axes of the policy's object pose field: the gear's z (object_axis) or the beam's x and z (object_frame).
        self.object_axes = np.array(r.get('object_axes', [[0., 0., 1.]]), dtype=np.float64)
        self.lift = r.get('lift')
        # Isaac's palm_rot is PhysX's link quaternion, continuous in q and with w ~ 0.2 at the start pose (it
        # crosses w = 0 during the task, so a w >= 0 quaternion would flip the observation's sign there).
        start = dict(zip(self.joint_order, r['initial_joint_pos']))
        self.palm_sign = 1. if self.fk.link_quats(start)[self.palm_body][0] > 0 else -1.
        self.reward_offsets = np.array(r['reward_offsets'])
        self.grasp_position = np.array(r['grasp_position'])
        self.goals = [(np.array(r['preinsert_pos']), np.array(r['preinsert_quat'])),
                      (np.array(r['final_pos']), np.array(r['final_quat']))]
        self.estimator, self.estimate = None, None
        if r.get('pose_estimator'):
            try:
                from .pose_estimator import PoseEstimator
            except ImportError:                    # imported as a top-level module (tests, tools)
                from pose_estimator import PoseEstimator
            if estimator is None:
                raise ValueError('This reference needs the pose estimator weights (estimator=...)')
            pe = r['pose_estimator']
            if pe.get('axes', 1) != len(self.object_axes):
                raise ValueError(f"Estimator of {pe.get('axes', 1)} axes for a policy of {len(self.object_axes)}")
            self.estimator = PoseEstimator(estimator, pe['inputs'], pe['hidden'], pe.get('axes', 1))
        self.reset()

    def reset(self, targets=None, object_pose=None):
        """Start an episode in stage 0. targets: the joint targets the robot holds now (default: the
        reference initial pose, where every episode must start). object_pose: (position, wxyz) of the object at
        rest at the start; with a lift reference the goal is that pose raised by lift['height'], at the trained
        start height plus lift['height'] (default: the reference start)."""
        self.targets = np.array(self.ref['initial_joint_pos'] if targets is None else targets, dtype=np.float64)
        self.stage, self.near_steps, self.stage_steps, self.retracted = 0, 0, 0, False
        if self.lift is not None:
            r = self.ref
            pos, quat = (r['object_pos'], r['object_quat']) if object_pose is None else object_pose
            goal = np.array(pos, dtype=np.float64)
            goal[2] = r['object_pos'][2] + self.lift['height']   # the trained goal height (the start check holds z)
            self.goals = [(goal, np.array(quat, dtype=np.float64))]*2
            self.start_z = float(r['object_pos'][2])
            self.peak = None                       # (highest rise, object height above the palm then)
        if self.estimator is not None:
            self.estimator.reset()                 # fresh hidden state, as at an Isaac episode start
            self.estimate = None

    @property
    def goal(self):
        """Active goal (position, wxyz): the pre-insert waypoint in stage 0, the final pose afterwards."""
        return self.goals[min(self.stage, 1)]

    def palm_and_tips(self, q):
        """Palm point (3,), palm link quaternion wxyz (4,) with Isaac's sign and fingertip pad points (4, 3) in
        the env frame."""
        joints = dict(zip(self.joint_order, np.asarray(q, dtype=np.float64)))
        poses = self.fk.link_poses(joints)
        P = poses[self.palm_body]
        palm = P[:3, 3] + P[:3, :3] @ self.palm_offset
        tips = np.stack([poses[b][:3, 3] + poses[b][:3, :3] @ o for b, o in zip(self.tip_bodies, self.tip_offsets)])
        return palm, self.palm_sign*self.fk.link_quats(joints)[self.palm_body], tips

    def observation(self, q, object_pos, object_axis, contact, valid=True, age=0., pad_force=None):
        """q: measured joint positions (23,) rad; object_pos: gear origin (3,); object_axis: gear +z axis (3,) or the
        object rotation matrix (3, 3) (needed for the beam's object_frame), both in the env frame; contact: 0/1 object
        contact per finger (4,). With the pose estimator: object_pos and object_axis are the last tracked pose, valid
        whether it is from this step, age the seconds since it was current, pad_force the tared pad forces (4, 3) N
        (normal, distal, lateral)."""
        q = np.asarray(q, dtype=np.float64)
        palm, palm_quat, tips = self.palm_and_tips(q)
        object_pos = np.asarray(object_pos, dtype=np.float64)
        axes = self.world_axes(object_axis)
        values = dict(
            joint_pos=2*(q - self.lower)/(self.upper - self.lower) - 1,
            prev_action_targets=self.targets,
            palm_pos=palm,
            palm_rot=np.roll(palm_quat, -1),
            fingertip_pos_rel_palm=(tips - palm).ravel(),
            object_pos_rel_palm=object_pos - palm,
            object_pos_rel_goal=object_pos - self.goal[0],
            object_axis=axes[-1],
            object_frame=axes.ravel(),
            fingertip_object_contact=np.asarray(contact, dtype=np.float64),
        )
        if self.estimator is not None:
            values.update(self._estimate(values, palm, palm_quat, tips, object_pos, axes, valid, age, pad_force))
        missing = [n for n, _ in self.fields if n not in values]
        if missing:
            raise ValueError(f'Reference asks for observation fields this interface does not build: {missing}')
        obs = np.concatenate([np.ravel(values[n]) for n, _ in self.fields])
        return np.clip(obs, -self.clip, self.clip)

    def world_axes(self, object_axis):
        """Unit object axes (n, 3) in the env frame, as reference['object_axes'], from a rotation matrix or a +z axis."""
        a = np.asarray(object_axis, dtype=np.float64)
        if a.ndim == 2:
            return self.object_axes @ a.T
        if not np.array_equal(self.object_axes, [[0., 0., 1.]]):
            raise ValueError('This reference needs the object rotation matrix, not its axis')
        return (a/np.linalg.norm(a))[None]

    def _estimate(self, values, palm, palm_quat, tips, object_pos, axes, valid, age, pad_force):
        """Gear pose fields of the observation from the estimator (gx4m_learning split_env._estimate)."""
        pe = self.ref['pose_estimator']
        R = quat_matrix(palm_quat/np.linalg.norm(palm_quat))
        scale = pe['position_scale']
        force = np.zeros((4, 3)) if pad_force is None else np.asarray(pad_force, dtype=np.float64)
        x = np.concatenate((values['joint_pos'], self.targets, values['palm_rot'], ((tips - palm) @ R).ravel()*scale,
                            R.T @ (object_pos - palm)*scale, (axes @ R).ravel(), [float(valid)],
                            [min(age/pe['max_age_s'], 1.)], force.ravel()/pe['tactile_force_scale']))
        pos, est_axis, log_std = self.estimator.step(x)
        rel = R @ pos/scale
        world_axis = (est_axis.reshape(-1, 3) @ R.T).ravel()
        self.estimate = (palm + rel, world_axis, log_std)
        return dict(object_pos_rel_palm=rel, object_pos_rel_goal=palm + rel - self.goal[0], object_axis=world_axis[-3:],
                    object_frame=world_axis, object_pose_std=log_std)

    def estimated_pose(self):
        """The last estimate in the env frame: gear origin (3,) and a wxyz quaternion with the estimated axis as +z
        (yaw arbitrary, which the yaw-free goal ignores), or with two estimated axes (x, z) the full orientation;
        None before the first observation."""
        if self.estimate is None:
            return None
        pos, axis = self.estimate[:2]
        if len(axis) == 6:
            x, z = axis[:3], axis[3:]/np.linalg.norm(axis[3:])
            x = x - (x @ z)*z
            x = x/np.linalg.norm(x)
            return pos, matrix_quat(np.stack((x, np.cross(z, x), z), axis=1))
        z = np.array([0., 0., 1.])
        c, v = float(z @ axis), np.cross(z, axis)
        quat = np.array([1. + c, *v])
        if quat[0] < 1e-9:                       # axis pointing straight down
            quat = np.array([0., 1., 0., 0.])
        return pos, quat/np.linalg.norm(quat)

    def apply_action(self, action, arm_speed_scale=None):
        """Policy action (23,) in [-1, 1] -> new joint targets (23,) rad, kept for prev_action_targets.
        The arm's top target speed is arm_smoothing*arm_speed_scale rad/s (gx4m_learning joint_targets).
        arm_speed_scale overrides the trained rad/s per unit action (reference control.arm_speed_scale), e.g.
        slower for first hardware runs."""
        a = np.clip(np.asarray(action, dtype=np.float64), -1., 1.)
        speed = self.arm_speed_scale if arm_speed_scale is None else arm_speed_scale
        arm = np.clip(self.targets[:7] + speed*self.dt*a[:7], self.lower[:7], self.upper[:7])
        fingers = self.lower[7:] + .5*(a[7:] + 1)*(self.upper[7:] - self.lower[7:])
        arm = self.arm_smoothing*arm + (1 - self.arm_smoothing)*self.targets[:7]
        fingers = self.smoothing*fingers + (1 - self.smoothing)*self.targets[7:]
        self.targets = np.clip(np.concatenate((arm, fingers)), self.lower, self.upper)
        return self.targets.copy()

    def keypoint_distance(self, object_pos, object_quat):
        """Largest gear-to-goal keypoint distance (m) for the current stage. In stage 0 a gear below the
        waypoint is not penalised for the height gap (preinsert_accepts_lower)."""
        object_pos, object_quat = np.asarray(object_pos, dtype=np.float64), np.asarray(object_quat, dtype=np.float64)
        pos, quat = self.goal
        pos = pos.copy()
        if self.ref['preinsert_accepts_lower'] and self.stage == 0:
            pos[2] = min(pos[2], object_pos[2])
        if self.yaw_free or self.teeth:
            quat = goal_quat(quat, object_quat, None if self.yaw_free else self.teeth)
        a = object_pos + self.reward_offsets @ quat_matrix(object_quat).T
        b = pos + self.reward_offsets @ quat_matrix(quat).T
        return float(np.linalg.norm(a - b, axis=-1).max())

    def update(self, q, object_pos, object_quat):
        """Stage logic after a control step, from the measured state (as Isaac's p2p step). Returns a dict with
        outcome None (continue), 'success', 'fall', 'hand_far', 'drop' (lift references) or 'timeout' (per-stage
        step budget), and stage, distance (m), away (closest fingertip to the grasp point, m)."""
        p = self.p2p
        was_retract = self.stage == 2
        distance = self.keypoint_distance(object_pos, object_quat)
        self.stage_steps += 1
        if not was_retract and distance <= p['insertion_success_tolerance']*p['keypoint_scale']:
            self.near_steps += 1
        goal_hit = not was_retract and self.near_steps >= p['success_steps']
        _, _, tips = self.palm_and_tips(q)
        grasp = np.asarray(object_pos) + quat_matrix(np.asarray(object_quat)) @ self.grasp_position
        reach = np.linalg.norm(tips - grasp, axis=-1)
        away = reach.min() if p['retract_all_fingers'] else reach.mean()
        at_goal = distance <= p['retract_success_tolerance']*p['keypoint_scale']
        if was_retract and at_goal and away > p['retract_distance_threshold']:
            self.retracted = True
        if goal_hit:
            self.stage += 1
            self.near_steps, self.stage_steps = 0, 0
        dropped = False
        if self.lift is not None:
            self.retracted |= goal_hit            # the lift goal held: success (gx4m_learning Step0Env)
            palm, _, _ = self.palm_and_tips(q)
            rise, in_hand = float(object_pos[2]) - self.start_z, float(object_pos[2] - palm[2])
            if self.peak is None or rise > self.peak[0]:
                self.peak = (rise, in_hand)
            if self.lift.get('drop_min_rise') is not None:
                dropped = self.peak[0] >= self.lift['drop_min_rise'] and in_hand < self.peak[1] - self.lift['drop_fall']
        fall = object_pos[2] < p['fall_height']
        hand_far = reach.max() > p['hand_far_distance'] and self.stage != 2
        timeout = self.stage_steps >= p['episode_steps']
        outcome = ('success' if self.retracted else 'fall' if fall else 'drop' if dropped else 'hand_far' if hand_far
                   else 'timeout' if timeout else None)
        return dict(outcome=outcome, stage=self.stage, distance=distance, away=float(away))
