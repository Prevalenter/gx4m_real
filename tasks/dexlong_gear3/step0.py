"""Step0 (grasp the 24-tooth gear and hold it 10 cm up) policy interface, outside Isaac Lab.

Rebuilds the 78-wide policy observation of DexLongGear3SplitStep0Env from measured robot and gear
state, and turns the 23 policy actions into joint targets, as gx4m_learning's dexlong_play_control.

config/step0_reference.json is exported by gx4m_learning's scripts/export_gear3_split_sim2sim_reference.py
(--step 0): joint order and limits, palm and fingertip bodies and offsets, robot base pose, control
constants. All positions are in the Isaac env frame: the robot root link sits at base_pos / base_quat.
Quaternions are wxyz except palm_rot inside the observation, which is xyzw like Isaac's obs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gx4m_real.robot.kinematics import UrdfFK, matrix_quat

HERE = Path(__file__).resolve().parent
REFERENCE = HERE/'config'/'step0_reference.json'
URDF = HERE.parents[1]/'asset/gx7_right/urdf/gx7_right_gx4m_cam_xl330.urdf'


def load_reference(path=REFERENCE):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class Step0Interface:
    """Observation builder and action-to-target mapping; joint vectors are in reference['joint_order']."""

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
        self.smoothing, self.arm_speed_scale, self.dt = c['smoothing'], c['arm_speed_scale'], r['step_dt']
        self.clip = r['clamp_abs_observations']
        self.lift_height = r['preinsert_pos'][2] - r['object_pos'][2]
        self.targets = np.array(r['initial_joint_pos'])
        self.goal = np.array(r['preinsert_pos'])

    def reset(self, object_pos, targets=None):
        """Start an episode: the lift goal is lift_height straight above the gear's measured start position.
        targets: the joint targets the robot holds now (default: the reference initial pose)."""
        self.targets = np.array(self.ref['initial_joint_pos'] if targets is None else targets, dtype=np.float64)
        self.goal = np.asarray(object_pos, dtype=np.float64) + (0., 0., self.lift_height)

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
            object_pos_rel_goal=object_pos - self.goal,
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
        arm_speed_scale overrides the trained 3 rad/s per unit action (e.g. slower for first hardware runs)."""
        a = np.clip(np.asarray(action, dtype=np.float64), -1., 1.)
        speed = self.arm_speed_scale if arm_speed_scale is None else arm_speed_scale
        arm = np.clip(self.targets[:7] + speed*self.dt*a[:7], self.lower[:7], self.upper[:7])
        fingers = self.lower[7:] + .5*(a[7:] + 1)*(self.upper[7:] - self.lower[7:])
        raw = np.concatenate((arm, fingers))
        self.targets = np.clip(self.smoothing*raw + (1 - self.smoothing)*self.targets, self.lower, self.upper)
        return self.targets.copy()
