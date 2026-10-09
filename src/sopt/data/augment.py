"""Batch augmentations on raw (unnormalized) features, run on GPU."""

from __future__ import annotations

import math

import torch

from sopt.data.features import POS, Q, ROT


def yaw_rotate(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rotate whole trajectories about the base z axis by theta (B,).

    Exact for Franka: joint 1 rotates about base z, so q1 += theta and EE pose <- Rz(theta) @ pose
    describe the same physical motion (ignoring the +-2.897 rad joint-1 limit).
    """
    c, s = torch.cos(theta)[:, None], torch.sin(theta)[:, None]
    x = x.clone()
    x[..., Q.start] = x[..., Q.start] + theta[:, None]
    pos_rot = [(POS.start, POS.start + 1), (ROT.start, ROT.start + 1), (ROT.start + 3, ROT.start + 4)]
    for i, j in pos_rot:
        a, b = x[..., i].clone(), x[..., j].clone()
        x[..., i] = c * a - s * b
        x[..., j] = s * a + c * b
    return x


def augment_batch(x: torch.Tensor, yaw_deg: float, *extra: torch.Tensor):
    """Rotates x and any extra (B, T, D) feature tensors (e.g. goals) by the same per-sample yaw.
    Returns x alone when no extras are given, else a tuple."""
    if yaw_deg > 0:
        theta = (torch.rand(x.shape[0], device=x.device) * 2 - 1) * math.radians(yaw_deg)
        x = yaw_rotate(x, theta)
        extra = tuple(yaw_rotate(e, theta) for e in extra)
    return (x, *extra) if extra else x
