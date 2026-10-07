"""Per-frame state features: [q(7), gripper(1), ee_pos(3), ee_rot6d(6)] = 17 dims."""

from __future__ import annotations

import numpy as np

from sopt.utils.rotation import euler_xyz_to_matrix, matrix_to_rot6d

# Raw layout written by `prepare_droid.py`: [joint_position(7), gripper(1), cartesian xyz+euler(6)].
RAW_DIM = 14

FEATURE_NAMES = (
    [f"q{i}" for i in range(7)] + ["gripper", "ee_x", "ee_y", "ee_z"] + [f"rot6d_{i}" for i in range(6)]
)
FEATURE_DIM = len(FEATURE_NAMES)
Q = slice(0, 7)
GRIPPER = slice(7, 8)
POS = slice(8, 11)
ROT = slice(11, 17)


def raw_to_features(raw: np.ndarray) -> np.ndarray:
    """(N, 14) raw DROID state -> (N, 17) features. Euler -> rot6d removes the +-pi roll wrap-around."""
    raw = raw.astype(np.float64)
    rot6d = matrix_to_rot6d(euler_xyz_to_matrix(raw[:, 11:14]))
    return np.concatenate([raw[:, 0:7], raw[:, 7:8], raw[:, 8:11], rot6d], axis=1).astype(np.float32)
