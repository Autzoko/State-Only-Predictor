"""Open-loop forecasting metrics in physical units, with kinematic baselines.

Every learned model must be compared against `zero_velocity` and `constant_velocity`; at short horizons these
are strong, so gains there do not by themselves show a learned prior.
"""

from __future__ import annotations

from collections import defaultdict

import torch

from sopt.data.features import GRIPPER, POS, ROT, Q
from sopt.utils.rotation import geodesic_angle, rot6d_to_matrix


def zero_velocity(ctx: torch.Tensor, horizon: int) -> torch.Tensor:
    return ctx[:, -1:].expand(-1, horizon, -1)[:, None]


def constant_velocity(ctx: torch.Tensor, horizon: int, k: int = 3) -> torch.Tensor:
    v = (ctx[:, -1] - ctx[:, -1 - k]) / k
    steps = torch.arange(1, horizon + 1, device=ctx.device, dtype=ctx.dtype)
    pred = ctx[:, -1:] + steps[None, :, None] * v[:, None]
    pred[..., GRIPPER] = ctx[:, -1:, GRIPPER]  # gripper is closer to piecewise-constant
    return pred[:, None]


def forecast_errors(pred: torch.Tensor, fut: torch.Tensor) -> dict[str, torch.Tensor]:
    """pred: (B, K, H, D), fut: (B, H, D) -> per-(B, K) errors averaged over the horizon, plus final-step."""
    fut = fut[:, None].expand_as(pred)
    pos = (pred[..., POS] - fut[..., POS]).norm(dim=-1)  # m
    rot = geodesic_angle(rot6d_to_matrix(pred[..., ROT]), rot6d_to_matrix(fut[..., ROT]))
    return {
        "pos_ade_cm": pos.mean(-1) * 100,
        "pos_fde_cm": pos[..., -1] * 100,
        "rot_ade_deg": torch.rad2deg(rot).mean(-1),
        "joint_ade_deg": torch.rad2deg((pred[..., Q] - fut[..., Q]).abs().mean(-1)).mean(-1),
        "gripper_mae": (pred[..., GRIPPER] - fut[..., GRIPPER]).abs().mean((-1, -2)),
    }


class ForecastMeter:
    """Accumulates mean-over-samples and min-over-samples (minADE@K) errors."""

    def __init__(self):
        self.sums = defaultdict(float)
        self.n = 0

    def update(self, name: str, pred: torch.Tensor, fut: torch.Tensor) -> None:
        errs = forecast_errors(pred.float(), fut.float())
        for k, v in errs.items():
            self.sums[f"{name}/{k}"] += v.mean(1).sum().item()
            if pred.shape[1] > 1:
                self.sums[f"{name}/min{pred.shape[1]}_{k}"] += v.min(1).values.sum().item()

    def step(self, batch_size: int) -> None:
        self.n += batch_size

    def result(self) -> dict[str, float]:
        return {k: v / max(self.n, 1) for k, v in self.sums.items()}


@torch.no_grad()
def evaluate_forecast(model, loader, device, num_samples: int, max_batches: int) -> dict[str, float]:
    """Windows must be fully valid (no padding) to be scored."""
    model.eval()
    meter = ForecastMeter()
    C, H = model.cfg.ctx_len, model.H
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x, mask = batch["x"].to(device), batch["mask"].to(device)
        x = x[mask.all(1)]
        if len(x) == 0:
            continue
        ctx, fut = x[:, :C], x[:, C : C + H]
        meter.update("model", model.forecast(ctx, num_samples), fut)
        meter.update("zero_vel", zero_velocity(ctx, H), fut)
        meter.update("const_vel", constant_velocity(ctx, H), fut)
        meter.step(len(x))
    model.train()
    return meter.result()
