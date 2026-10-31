"""Step1 (24-tooth gear onto the left pin) policy interface, outside Isaac Lab.

Rebuilds the 78-wide policy observation of DexLongGear3SplitStep1Env from measured robot and gear
state, turns the 23 policy actions into joint targets as gx4m_learning's dexlong_play_control, and runs
the task's stage logic, which decides the goal inside the observation:
  stage 0: pre-insert waypoint above the pin -> stage 1: final pose on the pin -> stage 2: withdraw the hand.
A stage is reached after success_steps (not necessarily consecutive) control steps with the gear keypoints
within insertion_success_tolerance * keypoint_scale (3 mm) of the goal. With yaw_free_goal the goal turns
with the gear about its axis, so only position and tilt count (the policy input has no gear yaw either).
The withdrawal succeeds once the gear stays within retract_success_tolerance * keypoint_scale (7.5 mm) of
the final goal and every fingertip is retract_distance_threshold (20 cm) from the hub grasp point.

config/step1_reference.json is exported by gx4m_learning's scripts/export_gear3_split_sim2sim_reference.py
(--step 1): joint order and limits, palm and fingertip bodies and offsets, robot base pose, goals, stage
settings (p2p), control constants. All positions are in the Isaac env frame: the robot root link sits at
base_pos / base_quat. Quaternions are wxyz except palm_rot inside the observation, which is xyzw like Isaac's.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from gx4m_real.robot.kinematics import UrdfFK, matrix_quat, quat_matrix

HERE = Path(__file__).resolve().parent
REFERENCE = HERE/'config'/'step1_reference.json'
URDF = HERE.parents[1]/'asset/gx7_right/urdf/gx7_right_gx4m_cam_xl330.urdf'


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

    def __init__(self, reference=None, urdf=URDF):
        r = self.ref = reference or load_reference()
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
        # control['smoothing'] is the 60 Hz constant (0.2); the 30 Hz env uses act_moving_average (0.36).
        self.smoothing, self.arm_speed_scale, self.dt = r['act_moving_average'], c['arm_speed_scale'], r['step_dt']
        self.clip = r['clamp_abs_observations']
        self.p2p = r['p2p']
        self.teeth = None if r.get('yaw_free_goal') else r['goal_yaw_symmetry']
        self.reward_offsets = np.array(r['reward_offsets'])
        self.grasp_position = np.array(r['grasp_position'])
        self.goals = [(np.array(r['preinsert_pos']), np.array(r['preinsert_quat'])),
                      (np.array(r['final_pos']), np.array(r['final_quat']))]
        self.reset()

    def reset(self, targets=None):
        """Start an episode in stage 0. targets: the joint targets the robot holds now (default: the
        reference initial pose, where every episode must start)."""
        self.targets = np.array(self.ref['initial_joint_pos'] if targets is None else targets, dtype=np.float64)
        self.stage, self.near_steps, self.stage_steps, self.retracted = 0, 0, 0, False

    @property
    def goal(self):
        """Active goal (position, wxyz): the pre-insert waypoint in stage 0, the final pose afterwards."""
        return self.goals[min(self.stage, 1)]

    def palm_and_tips(self, q):
        """Palm point (3,), palm link quaternion wxyz (4,) and fingertip pad points (4, 3) in the env frame."""
        poses = self.fk.link_poses(dict(zip(self.joint_order, np.asarray(q, dtype=np.float64))))
        P = poses[self.palm_body]
        palm = P[:3, 3] + P[:3, :3] @ self.palm_offset
        tips = np.stack([poses[b][:3, 3] + poses[b][:3, :3] @ o for b, o in zip(self.tip_bodies, self.tip_offsets)])
        return palm, matrix_quat(P[:3, :3]), tips

    def observation(self, q, object_pos, object_axis, contact):
        """q: measured joint positions (23,) rad; object_pos: gear origin (3,); object_axis: gear +z axis (3,),
        both in the env frame; contact: 0/1 object contact per finger (4,)."""
        q = np.asarray(q, dtype=np.float64)
        palm, palm_quat, tips = self.palm_and_tips(q)
        object_pos = np.asarray(object_pos, dtype=np.float64)
        axis = np.asarray(object_axis, dtype=np.float64)
        values = dict(
            joint_pos=2*(q - self.lower)/(self.upper - self.lower) - 1,
            prev_action_targets=self.targets,
            palm_pos=palm,
            palm_rot=np.roll(palm_quat, -1),
            fingertip_pos_rel_palm=(tips - palm).ravel(),
            object_pos_rel_palm=object_pos - palm,
            object_pos_rel_goal=object_pos - self.goal[0],
            object_axis=axis/np.linalg.norm(axis),
            fingertip_object_contact=np.asarray(contact, dtype=np.float64),
        )
        missing = [n for n, _ in self.fields if n not in values]
        if missing:
            raise ValueError(f'Reference asks for observation fields this interface does not build: {missing}')
        obs = np.concatenate([np.ravel(values[n]) for n, _ in self.fields])
        return np.clip(obs, -self.clip, self.clip)

    def apply_action(self, action, arm_speed_scale=None):
        """Policy action (23,) in [-1, 1] -> new joint targets (23,) rad, kept for prev_action_targets.
        arm_speed_scale overrides the trained rad/s per unit action (reference control.arm_speed_scale), e.g.
        slower for first hardware runs."""
        a = np.clip(np.asarray(action, dtype=np.float64), -1., 1.)
        speed = self.arm_speed_scale if arm_speed_scale is None else arm_speed_scale
        arm = np.clip(self.targets[:7] + speed*self.dt*a[:7], self.lower[:7], self.upper[:7])
        fingers = self.lower[7:] + .5*(a[7:] + 1)*(self.upper[7:] - self.lower[7:])
        raw = np.concatenate((arm, fingers))
        self.targets = np.clip(self.smoothing*raw + (1 - self.smoothing)*self.targets, self.lower, self.upper)
        return self.targets.copy()

    def keypoint_distance(self, object_pos, object_quat):
        """Largest gear-to-goal keypoint distance (m) for the current stage. In stage 0 a gear below the
        waypoint is not penalised for the height gap (preinsert_accepts_lower)."""
        object_pos, object_quat = np.asarray(object_pos, dtype=np.float64), np.asarray(object_quat, dtype=np.float64)
        pos, quat = self.goal
        pos = pos.copy()
        if self.ref['preinsert_accepts_lower'] and self.stage == 0:
            pos[2] = min(pos[2], object_pos[2])
        quat = goal_quat(quat, object_quat, self.teeth)
        a = object_pos + self.reward_offsets @ quat_matrix(object_quat).T
        b = pos + self.reward_offsets @ quat_matrix(quat).T
        return float(np.linalg.norm(a - b, axis=-1).max())

    def update(self, q, object_pos, object_quat):
        """Stage logic after a control step, from the measured state (as Isaac's p2p step). Returns a dict with
        outcome None (continue), 'success', 'fall', 'hand_far' or 'timeout' (per-stage step budget), and stage,
        distance (m), away (closest fingertip to the grasp point, m)."""
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
        fall = object_pos[2] < p['fall_height']
        hand_far = reach.max() > p['hand_far_distance'] and self.stage != 2
        timeout = self.stage_steps >= p['episode_steps']
        outcome = ('success' if self.retracted else 'fall' if fall else 'hand_far' if hand_far
                   else 'timeout' if timeout else None)
        return dict(outcome=outcome, stage=self.stage, distance=distance, away=float(away))
