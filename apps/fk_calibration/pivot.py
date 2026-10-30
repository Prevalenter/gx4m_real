"""Linear pivot fit: one tip offset shared by all seats, one fixed centre per seat."""
import numpy as np


def pivot_fit(poses, seats):
    """Solve R_k t + p_k = c_{seat(k)} in least squares.

    poses: (N, 4, 4) flange poses; seats: N hashable seat keys.
    Returns (tip_offset (3,), {seat: centre (3,)}, residuals (N, 3)), or None if underdetermined.
    """
    poses = np.asarray(poses, dtype=float)
    keys = list(dict.fromkeys(seats))
    if len(poses) < 2 or len(poses) * 3 < 3 + 3 * len(keys):
        return None
    A = np.zeros((3 * len(poses), 3 + 3 * len(keys)))
    b = np.zeros(3 * len(poses))
    for k, (T, seat) in enumerate(zip(poses, seats)):
        rows = slice(3 * k, 3 * k + 3)
        j = 3 + 3 * keys.index(seat)
        A[rows, :3], A[rows, j:j + 3], b[rows] = T[:3, :3], -np.eye(3), -T[:3, 3]
    x, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < A.shape[1]:
        return None                                 # orientations not diverse enough yet
    centres = {seat: x[3 + 3 * i:6 + 3 * i] for i, seat in enumerate(keys)}
    residuals = (A @ x - b).reshape(-1, 3)
    return x[:3], centres, residuals
