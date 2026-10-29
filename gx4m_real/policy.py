"""Deterministic actor of an rsl_rl checkpoint (model_*.pt), without rsl_rl or Isaac Lab.

Both actor kinds rsl_rl 5 saves are read from the state dict alone:
- MLP: obs normaliser -> mlp (Linear + ELU, last layer linear)
- LSTM: obs normaliser -> rnn.rnn (one nn.LSTM step per call) -> mlp
The action is the distribution mean, i.e. the mlp output; no sampling.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class RslRlActor:
    def __init__(self, checkpoint: str | Path, device: str = 'cpu', eps: float = 1e-2):
        state = torch.load(checkpoint, map_location=device, weights_only=False)['actor_state_dict']
        self.device, self.eps = device, eps
        self.mean, self.std = state['obs_normalizer._mean'][0], state['obs_normalizer._std'][0]
        self.layers = []
        i = 0
        while f'mlp.{i}.weight' in state:
            self.layers.append((state[f'mlp.{i}.weight'], state[f'mlp.{i}.bias']))
            i += 2                                          # mlp.{odd} are the ELUs
        self.rnn = None
        if 'rnn.rnn.weight_ih_l0' in state:
            layers = sum(k.startswith('rnn.rnn.weight_ih_l') for k in state)
            hidden = state['rnn.rnn.weight_hh_l0'].shape[1]
            self.rnn = nn.LSTM(state['rnn.rnn.weight_ih_l0'].shape[1], hidden, num_layers=layers).to(device)
            self.rnn.load_state_dict({k[len('rnn.rnn.'):]: v for k, v in state.items() if k.startswith('rnn.rnn.')})
            self.rnn.eval()
        self.obs_dim = self.mean.numel()
        self.action_dim = self.layers[-1][0].shape[0]
        self.kind = 'lstm' if self.rnn is not None else 'mlp'
        self.hidden = None

    def reset(self):
        """Clear the recurrent state; call at the start of every episode."""
        self.hidden = None

    @torch.no_grad()
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """One step: obs (obs_dim,) -> action (action_dim,), float64."""
        x = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        if x.shape != (self.obs_dim,):
            raise ValueError(f'Expected observation shape ({self.obs_dim},), got {tuple(x.shape)}')
        x = (x - self.mean) / (self.std + self.eps)
        if self.rnn is not None:
            x, self.hidden = self.rnn(x[None, None], self.hidden)
            x = x[0, 0]
        for k, (w, b) in enumerate(self.layers):
            x = x @ w.T + b
            if k < len(self.layers) - 1:
                x = torch.nn.functional.elu(x)
        return x.cpu().numpy().astype(np.float64)
