"""Rotation helpers. DROID euler angles are extrinsic XYZ: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""

from __future__ import annotations

import numpy as np
import torch


def euler_xyz_to_matrix(euler: np.ndarray) -> np.ndarray:
    """(..., 3) roll/pitch/yaw -> (..., 3, 3); identical to scipy Rotation.from_euler("xyz")."""
    r, p, y = euler[..., 0], euler[..., 1], euler[..., 2]
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    R = np.empty(euler.shape[:-1] + (3, 3), dtype=euler.dtype)
    R[..., 0, 0] = cy * cp
    R[..., 0, 1] = cy * sp * sr - sy * cr
    R[..., 0, 2] = cy * sp * cr + sy * sr
    R[..., 1, 0] = sy * cp
    R[..., 1, 1] = sy * sp * sr + cy * cr
    R[..., 1, 2] = sy * sp * cr - cy * sr
    R[..., 2, 0] = -sp
    R[..., 2, 1] = cp * sr
    R[..., 2, 2] = cp * cr
    return R


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """First two columns of R, concatenated: (..., 3, 3) -> (..., 6)."""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt (Zhou et al. 2019): (..., 6) -> (..., 3, 3) with columns b1, b2, b3."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def geodesic_angle(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """Angle (rad) of R1^T R2."""
    tr = torch.einsum("...ij,...ij->...", R1, R2)
    return torch.acos(((tr - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6))


def matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """Inverse of euler_xyz_to_matrix (extrinsic XYZ): (..., 3, 3) -> (..., 3) roll/pitch/yaw."""
    roll = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    pitch = -np.arcsin(np.clip(R[..., 2, 0], -1.0, 1.0))
    yaw = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    return np.stack([roll, pitch, yaw], axis=-1)
