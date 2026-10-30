"""Pivot + known-distance FK calibration: Gauss–Newton with truncated SVD.

Unknowns: selected joint error parameters, the probe ball centre in the last arm link, and one
centre per seat. Residuals: (tip − seat centre) / σ_pivot for every sample, and
(|c_a − c_b| − d_ab) / σ_distance for every measured seat pair.

Never estimated, because relative measurements cannot see them: the dθ of J1 and the whole origin
of J1 (a rigid motion of all seat centres), and the dθ of J7 (absorbed by the tip offset). Other
redundant combinations are dropped by the SVD truncation, so they stay at their nominal value.
The origin rotation about a joint's own axis duplicates its dθ exactly and is left out of `full`.
"""
from dataclasses import dataclass, field

import numpy as np

from chain import PARAM_NAMES
from pivot import pivot_fit

L_REF = .5          # m; angles are scaled by this so 1 mrad weighs like 0.5 mm in the SVD
MODELS = ('nominal', 'offsets', 'full')


def free_parameters(chain, model):
    """(joint index, PARAM_NAMES index) pairs estimated by a model."""
    if model == 'nominal':
        return []
    offsets = [(i, 0) for i in range(1, 6)]
    if model == 'offsets':
        return offsets
    full = []
    for i in range(1, 7):
        along_axis = np.flatnonzero(np.isclose(np.abs(chain.axes[i]), 1))
        full += [(i, k) for k in range(1, 7) if not (k >= 4 and k - 4 in along_axis)]
    return offsets + full


@dataclass
class Data:
    q: np.ndarray                                   # (N, 7) model joint angles, rad
    seats: list                                     # N seat keys
    distances: list = field(default_factory=list)   # (seat_a, seat_b, metres)
    sigma_pivot: float = 1e-4
    sigma_distance: float = 5e-5

    def subset(self, mask):
        mask = np.asarray(mask, dtype=bool)
        return Data(self.q[mask], [s for s, m in zip(self.seats, mask) if m], self.distances,
                    self.sigma_pivot, self.sigma_distance)


@dataclass
class Result:
    model: str
    free: list
    errors: np.ndarray                              # (7, 7) see chain.PARAM_NAMES
    tip: np.ndarray
    centres: dict
    std: np.ndarray                                 # 1σ of each free joint parameter
    identified: np.ndarray                          # 0..1, share of each parameter inside the kept subspace
    singular_values: np.ndarray
    rank: int
    n_unknowns: int
    iterations: int
    used_distances: list

    def param_label(self, j):
        i, k = self.free[j]
        return f'J{i + 1}.{PARAM_NAMES[k]}'


def numeric_jacobian(fun, x, h=1e-7):
    J = np.empty((len(fun(x)), len(x)))
    for j in range(len(x)):
        step = np.zeros_like(x)
        step[j] = h
        J[:, j] = (fun(x + step) - fun(x - step)) / (2 * h)
    return J


def solve(chain, data, model='full', rcond=1e-3, iterations=30):
    free = free_parameters(chain, model)
    keys = list(dict.fromkeys(data.seats))
    seat_index = np.array([keys.index(s) for s in data.seats])
    pairs = [(keys.index(a), keys.index(b), d) for a, b, d in data.distances if a in keys and b in keys]
    init = pivot_fit(chain.fk(data.q), data.seats)
    if init is None:
        raise ValueError('姿态不足或方向单一，无法做初始枢轴拟合')
    n = len(free)
    x = np.concatenate([np.zeros(n), init[0], np.concatenate([init[1][k] for k in keys])])
    scale = np.ones(len(x))
    for j, (_, k) in enumerate(free):
        if k in (0, 4, 5, 6):
            scale[j] = L_REF

    def unpack(x):
        E = np.zeros((7, 7))
        for (i, k), v in zip(free, x[:n]):
            E[i, k] = v
        return E, x[n:n + 3], x[n + 3:].reshape(-1, 3)

    def residuals(x):
        E, t, C = unpack(x)
        r = ((chain.tip(data.q, t, E) - C[seat_index]) / data.sigma_pivot).ravel()
        d = [(np.linalg.norm(C[a] - C[b]) - dist) / data.sigma_distance for a, b, dist in pairs]
        return np.concatenate([r, d])

    def truncated_svd(x):
        r = residuals(x)
        U, S, Vt = np.linalg.svd(numeric_jacobian(residuals, x) / scale, full_matrices=False)
        return r, U, S, Vt, S > rcond * S[0]

    it = 0
    for it in range(1, iterations + 1):
        r, U, S, Vt, keep = truncated_svd(x)
        step = -(Vt[keep].T @ ((U[:, keep].T @ r) / S[keep])) / scale
        x = x + step
        if np.max(np.abs(step)) < 1e-10:
            break
    r, U, S, Vt, keep = truncated_svd(x)
    rank = int(keep.sum())
    dof = len(r) - rank
    s2 = float(r @ r) / dof if dof > 0 else np.nan     # a-posteriori variance factor
    V = Vt[keep].T / S[keep]
    std = np.sqrt(s2 * np.sum(V ** 2, axis=1)) / scale
    identified = np.sum(Vt[keep] ** 2, axis=0)
    E, t, C = unpack(x)
    return Result(model, free, E, t, {k: C[i] for i, k in enumerate(keys)}, std[:n], identified[:n],
                  S, rank, len(x), it, [(keys[a], keys[b], d) for a, b, d in pairs])


def evaluate(chain, result, data):
    """Pivot error of each sample and distance error of each pair, in mm.

    A seat without a fitted centre (fully held out) is measured around its own mean, which reads
    slightly optimistic; its distance errors to the other seats are the honest check.
    """
    p = chain.tip(data.q, result.tip, result.errors)
    seats = np.array(data.seats)
    err = np.zeros(len(p))
    centres, per_seat = {}, {}
    for seat in dict.fromkeys(data.seats):
        mask = seats == seat
        fitted = seat in result.centres
        centres[seat] = result.centres[seat] if fitted else p[mask].mean(axis=0)
        err[mask] = np.linalg.norm(p[mask] - centres[seat], axis=1) * 1000
        per_seat[seat] = {'n': int(mask.sum()), 'rms_mm': rms(err[mask]), 'max_mm': float(err[mask].max()),
                          'centre': 'fitted' if fitted else 'own_mean'}
    distance = [{'pair': f'{a}–{b}', 'measured_mm': d * 1000,
                 'error_mm': (np.linalg.norm(centres[a] - centres[b]) - d) * 1000}
                for a, b, d in data.distances if a in centres and b in centres]
    return {'n': len(p), 'pivot_rms_mm': rms(err), 'pivot_max_mm': float(err.max()) if len(err) else 0.,
            'per_seat': per_seat, 'distances': distance,
            'distance_rms_mm': rms([d['error_mm'] for d in distance]), 'sample_errors_mm': err}


def rms(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values ** 2))) if values.size else float('nan')
