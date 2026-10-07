"""Franka Panda forward kinematics (modified DH, Craig). Verified: equals DROID `cartesian_position` (flange)."""

from __future__ import annotations

import numpy as np

_A = [0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088]
_D = [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.0]
_ALPHA = [0.0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2]
FLANGE_D = 0.107


def _mdh(a: float, d: float, alpha: float, theta: np.ndarray) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    T = np.zeros(theta.shape + (4, 4), dtype=np.float64)
    T[..., 0, 0], T[..., 0, 1], T[..., 0, 3] = ct, -st, a
    T[..., 1, 0], T[..., 1, 1], T[..., 1, 2], T[..., 1, 3] = st * ca, ct * ca, -sa, -d * sa
    T[..., 2, 0], T[..., 2, 1], T[..., 2, 2], T[..., 2, 3] = st * sa, ct * sa, ca, d * ca
    T[..., 3, 3] = 1.0
    return T


def franka_fk(q: np.ndarray) -> np.ndarray:
    """(..., 7) joint angles -> (..., 4, 4) flange pose in the robot base frame."""
    q = np.asarray(q, dtype=np.float64)
    M = np.broadcast_to(np.eye(4), q.shape[:-1] + (4, 4)).copy()
    for i in range(7):
        M = M @ _mdh(_A[i], _D[i], _ALPHA[i], q[..., i])
    return M @ _mdh(0.0, FLANGE_D, 0.0, np.zeros(q.shape[:-1]))
