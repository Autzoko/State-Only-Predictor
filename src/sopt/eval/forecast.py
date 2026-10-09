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


def _traj_rms_cm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """RMS over the horizon of EE position distance, in cm. (..., H, D) x (..., H, D) -> (...)."""
    return ((a[..., POS] - b[..., POS]).pow(2).sum(-1).mean(-1)).sqrt() * 100


def energy_score_pos(pred: torch.Tensor, fut: torch.Tensor) -> torch.Tensor:
    """Energy score of the EE position trajectory (proper scoring rule; lower is better), per batch item.

    ES = E d(X, y) - 0.5 E d(X, X'), d = RMS-over-time distance in cm, unbiased over K samples.
    For a deterministic forecast it reduces to the trajectory RMSE, so deterministic and generative models
    are directly comparable (unlike min-of-K, which always favours diverse samplers).
    """
    K = pred.shape[1]
    term1 = _traj_rms_cm(pred, fut[:, None].expand_as(pred)).mean(1)
    if K == 1:
        return term1
    pair = _traj_rms_cm(pred[:, :, None], pred[:, None, :])  # (B, K, K), zero diagonal
    return term1 - 0.5 * pair.sum((1, 2)) / (K * (K - 1))


class ForecastMeter:
    """Per model: mean error over samples, min over samples (minADE@K), error of the sample mean,
    energy score, and per-step position error curves (lists, `*_by_step`)."""

    def __init__(self):
        self.sums = defaultdict(float)
        self.curves: dict[str, torch.Tensor] = {}
        self.n = 0

    def _add_curve(self, key: str, per_step: torch.Tensor) -> None:
        v = per_step.sum(0).double().cpu()
        self.curves[key] = self.curves[key] + v if key in self.curves else v

    def update(self, name: str, pred: torch.Tensor, fut: torch.Tensor) -> None:
        pred, fut = pred.float(), fut.float()
        stochastic = pred.shape[1] > 1 and not torch.equal(pred[:, 0], pred[:, 1])
        step_err = (pred[..., POS] - fut[:, None, :, POS]).norm(dim=-1) * 100  # (B, K, H) cm
        self._add_curve(f"{name}/pos_err_cm_by_step", step_err.mean(1))
        self.sums[f"{name}/energy_pos_cm"] += energy_score_pos(pred if stochastic else pred[:, :1], fut).sum().item()
        if stochastic:
            self._add_curve(f"{name}/min{pred.shape[1]}_pos_err_cm_by_step", step_err.min(1).values)
            mean_step = (pred.mean(1)[..., POS] - fut[..., POS]).norm(dim=-1) * 100
            self._add_curve(f"{name}/samplemean_pos_err_cm_by_step", mean_step)
        errs = forecast_errors(pred, fut)
        for k, v in errs.items():
            self.sums[f"{name}/{k}"] += v.mean(1).sum().item()
            if pred.shape[1] > 1:
                self.sums[f"{name}/min{pred.shape[1]}_{k}"] += v.min(1).values.sum().item()
        if stochastic:
            # Point estimate from samples (averaging rot6d then Gram-Schmidt is fine for nearby rotations);
            # this is the fair comparison against deterministic models.
            mean_errs = forecast_errors(pred.mean(1, keepdim=True), fut)
            for k, v in mean_errs.items():
                self.sums[f"{name}/samplemean_{k}"] += v[:, 0].sum().item()

    def step(self, batch_size: int) -> None:
        self.n += batch_size

    def result(self) -> dict:
        n = max(self.n, 1)
        out: dict = {k: v / n for k, v in self.sums.items()}
        out.update({k: (v / n).round(decimals=4).tolist() for k, v in self.curves.items()})
        return out


def interpolate_to(ctx: torch.Tensor, goal: torch.Tensor, horizon: int) -> torch.Tensor:
    """Kinematic baseline when the endpoint is known: straight line from the last frame to the goal."""
    a = torch.arange(1, horizon + 1, device=ctx.device, dtype=ctx.dtype)[None, :, None] / horizon
    return (ctx[:, -1:] + a * (goal[:, None] - ctx[:, -1:]))[:, None]


@torch.no_grad()
def evaluate_forecast(
    model, loader, device, num_samples: int, max_batches: int, goal_modes: tuple[str, ...] = ("none",)
) -> dict:
    """Windows must be fully valid (no padding) to be scored.

    goal_modes: 'none' always scores the unconditional forecast as `model`; 'endpoint' / 'keyframe' score a
    goal-conditioned model as `model@<mode>` (goal = state at C-1+H, or at the next gripper event after C-1;
    the latter needs batches with `kf`). 'endpoint' also scores the `interp@endpoint` baseline.
    """
    model.eval()
    meter = ForecastMeter()
    C, H = model.cfg.ctx_len, model.H
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x, mask = batch["x"].to(device), batch["mask"].to(device)
        full = mask.all(1)
        x = x[full]
        if len(x) == 0:
            continue
        ctx, fut = x[:, :C], x[:, C : C + H]
        meter.update("model", model.forecast(ctx, num_samples), fut)
        goals = {"endpoint": x[:, C - 1 + H]}
        if "kf" in batch:
            goals["keyframe"] = batch["kf"].to(device)[full][:, C - 1]
        for mode in goal_modes:
            if mode == "none":
                continue
            if getattr(model, "goal_cond", False):
                pred = model.forecast(ctx, num_samples, goal=goals[mode], goal_type=mode)
                meter.update(f"model@{mode}", pred, fut)
            if mode == "endpoint":
                meter.update("interp@endpoint", interpolate_to(ctx, goals[mode], H), fut)
        meter.update("zero_vel", zero_velocity(ctx, H), fut)
        meter.update("const_vel", constant_velocity(ctx, H), fut)
        meter.step(len(x))
    model.train()
    return meter.result()
