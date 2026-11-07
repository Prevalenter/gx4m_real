"""Gear pose estimator of the occlusion-aware split-step policies (gx4m_learning dexlong_gear3_split/pose_estimator.py).

A GRU from what the robot measures (joints, last targets, palm orientation, fingertips, the tracked gear pose held
while the camera has lost it, its valid flag and age, the 3-axis pad forces) to the gear origin and axis in the palm
frame; the policy acts on this estimate instead of the raw tracked pose. With axes=2 (the beam, no symmetry axis) the
tracked input and the output carry two body axes (x then z), i.e. the full orientation. Same network and weights as in training
(estimator_<it>.pt saved next to model_<it>.pt); Step1Interface builds the input (reference['pose_estimator']).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class GearPoseEstimator(nn.Module):
    def __init__(self, inputs, hidden=256, axes=1):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(inputs, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU())
        self.gru = nn.GRUCell(hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 3 + 3*axes + 2))
        self.hidden, self.axes = hidden, axes

    def forward(self, x, h):
        h = self.gru(self.encoder(x), h)
        out = self.head(h)
        n = 3*self.axes
        axis = nn.functional.normalize(out[:, 3:3 + n].unflatten(-1, (self.axes, 3)), dim=-1).flatten(-2)
        return out[:, :3], axis, out[:, 3 + n:5 + n].clamp(-5., 3.), h


class PoseEstimator:
    """One robot: step(x) per control step with its own hidden state, reset() at an episode start."""

    def __init__(self, path, inputs, hidden=256, axes=1):
        state = torch.load(Path(path), map_location='cpu', weights_only=False)
        if dict(state['inputs']) != dict(inputs):
            raise ValueError(f'{path}: estimator inputs {state["inputs"]} differ from the reference {inputs}')
        self.model = GearPoseEstimator(sum(inputs.values()), hidden, axes)
        self.model.load_state_dict(state['model'])
        self.model.eval()
        self.size = sum(inputs.values())
        self.reset()

    def reset(self):
        self.h = torch.zeros(1, self.model.hidden)

    @torch.no_grad()
    def step(self, x):
        """x (inputs,) -> gear origin in the palm frame (scaled), unit axes in the palm frame (3*axes,), log std (2,)."""
        x = torch.as_tensor(np.asarray(x, dtype=np.float32)).reshape(1, self.size)
        pos, axis, log_std, self.h = self.model(x, self.h)
        return pos[0].double().numpy(), axis[0].double().numpy(), log_std[0].double().numpy()
