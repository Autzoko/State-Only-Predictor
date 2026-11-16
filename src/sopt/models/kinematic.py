"""R2 baseline (reviewer control): explicit finite-difference kinematics of the proprioceptive history in place of the
SOPT encoder. Same interface as StatePrior for the host arms (`cfg.d_model`, `cfg.ctx_len`, `D`, `normalizer`,
`encode`), so an `expert_kin` arm goes through exactly the injection of `expert_droid` / `expert_scratch` (LayerNorm +
zero-initialized MLP on the last token); only the features differ.

Features of the history x (B, T, 17) on the 15 Hz grid, in the prior's normalized units (the same normalization every
arm uses), at the current frame t:
  velocity over 1 / 4 / 16 frames   (x_t - x_{t-l}) / l            (0.07 / 0.27 / 1.07 s; DV2: ~1 s of velocity memory)
  acceleration over 1 / 4 frames    (x_t - 2 x_{t-l} + x_{t-2l}) / l^2
-> 5 x 17 = 85 values, each standardized with the mean / std over the host's training samples (`fit`; training demos
only, stored as buffers so evaluation uses the same constants). The current state itself is not repeated: every host
already receives it. Early in an episode the history is padded with the first frame, so the differences are zero there,
as the SOPT encoder sees a resting history.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import torch
from torch import nn

from sopt.data.features import Q
from sopt.models.state_prior import StatePrior

VEL_LAGS = (1, 4, 16)
ACC_LAGS = (1, 4)


class KinematicPrior(nn.Module):
    def __init__(self, ref: StatePrior):
        super().__init__()
        self.D = ref.D
        n = (len(VEL_LAGS) + len(ACC_LAGS)) * self.D
        assert ref.cfg.ctx_len > max(max(VEL_LAGS), 2 * max(ACC_LAGS)), "history too short for the lags"
        self.cfg = SimpleNamespace(d_model=n, ctx_len=int(ref.cfg.ctx_len), ee_only=bool(ref.cfg.get("ee_only", False)))
        self.normalizer = copy.deepcopy(ref.normalizer)
        self.register_buffer("kin_mean", torch.zeros(n))
        self.register_buffer("kin_std", torch.ones(n))
        self.register_buffer("kin_ready", torch.zeros(()))

    def raw(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, T, D) raw features -> (B, 85) unstandardized finite differences at the last frame."""
        xn = self.normalizer.norm(x.float())
        if self.cfg.ee_only:  # joints hidden exactly as in StatePrior._hide_joints
            xn = xn.clone()
            xn[..., Q] = 0.0
        cur = xn[:, -1]
        out = [(cur - xn[:, -1 - lag]) / lag for lag in VEL_LAGS]
        out += [(cur - 2 * xn[:, -1 - lag] + xn[:, -1 - 2 * lag]) / lag**2 for lag in ACC_LAGS]
        return torch.cat(out, dim=-1)

    @torch.no_grad()
    def fit(self, ctx: torch.Tensor) -> None:
        """Standardization constants from the training samples' histories ctx (N, T, D)."""
        f = torch.cat([self.raw(c.to(self.kin_mean.device)) for c in ctx.split(4096)])
        self.kin_mean.copy_(f.mean(0))
        self.kin_std.copy_(f.std(0).clamp_min(1e-6))
        self.kin_ready.fill_(1.0)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """(B, T, D) -> (B, 1, 85): one 'token' whose features are the standardized kinematics."""
        return ((self.raw(x) - self.kin_mean) / self.kin_std)[:, None]
