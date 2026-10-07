import numpy as np
import torch

from sopt.data.augment import yaw_rotate
from sopt.data.features import POS, ROT, Q, raw_to_features
from sopt.utils.franka_fk import franka_fk
from sopt.utils.rotation import euler_xyz_to_matrix, matrix_to_rot6d, rot6d_to_matrix


def _matrix_to_euler_xyz(R):
    return np.stack([np.arctan2(R[..., 2, 1], R[..., 2, 2]), -np.arcsin(R[..., 2, 0]), np.arctan2(R[..., 1, 0], R[..., 0, 0])], -1)


def _raw_from_q(q):
    M = franka_fk(q)
    cart = np.concatenate([M[..., :3, 3], _matrix_to_euler_xyz(M[..., :3, :3])], -1)
    return np.concatenate([q, np.zeros(q.shape[:-1] + (1,)), cart], -1)


def test_rot6d_roundtrip():
    e = np.random.default_rng(0).uniform(-3, 3, (100, 3))
    R = euler_xyz_to_matrix(e)
    R2 = rot6d_to_matrix(torch.from_numpy(matrix_to_rot6d(R))).numpy()
    assert np.allclose(R, R2, atol=1e-6)


def test_yaw_augmentation_is_kinematically_exact():
    """Rotating q1 by theta must equal rotating the FK pose about base z."""
    rng = np.random.default_rng(0)
    q = rng.uniform(-1.5, 1.5, (32, 7))
    theta = rng.uniform(-0.5, 0.5, 32)
    feats = torch.from_numpy(raw_to_features(_raw_from_q(q)))[:, None]  # (B, T=1, D)
    rotated = yaw_rotate(feats, torch.from_numpy(theta).float())[:, 0].numpy()
    q2 = q.copy()
    q2[:, 0] += theta
    expected = raw_to_features(_raw_from_q(q2))
    for s in (Q, POS, ROT):
        assert np.allclose(rotated[:, s], expected[:, s], atol=1e-4)
