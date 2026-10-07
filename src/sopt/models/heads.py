"""Prediction heads mapping a token feature to a flattened future chunk (H * D)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class RegressionHead(nn.Module):
    def __init__(self, d_model: int, out_dim: int, hidden: int, depth: int):
        super().__init__()
        layers, d = [], d_model
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        self.net = nn.Sequential(*layers, nn.Linear(d, out_dim))

    def loss(self, h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-sample L1 loss, shape h.shape[:-1]."""
        return F.l1_loss(self.net(h), target, reduction="none").mean(-1)

    def sample(self, h: torch.Tensor, num_samples: int = 1, **_) -> torch.Tensor:
        return self.net(h).unsqueeze(-2).expand(*h.shape[:-1], num_samples, -1)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
    ang = t.float()[..., None] * 1000.0 * freqs
    return torch.cat([ang.cos(), ang.sin()], dim=-1)


class AdaLNBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.mod = nn.Linear(dim, 3 * dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, x, c):
        shift, scale, gate = self.mod(F.silu(c)).chunk(3, dim=-1)
        return x + gate * self.mlp(self.norm(x) * (1 + scale) + shift)


class FlowHead(nn.Module):
    """Rectified-flow head (MAR-style AdaLN MLP): x_tau = (1 - tau) * eps + tau * x1, predicts x1 - eps."""

    def __init__(self, d_model: int, out_dim: int, hidden: int, depth: int, steps: int = 10):
        super().__init__()
        self.hidden, self.steps = hidden, steps
        self.in_proj = nn.Linear(out_dim, hidden)
        self.cond = nn.Linear(d_model, hidden)
        self.time = nn.Sequential(nn.Linear(256, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList([AdaLNBlock(hidden) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_dim))

    def velocity(self, x, tau, h):
        c = self.cond(h) + self.time(timestep_embedding(tau, 256))
        z = self.in_proj(x)
        for blk in self.blocks:
            z = blk(z, c)
        return self.out(z)

    def loss(self, h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(target)
        tau = torch.sigmoid(torch.randn(target.shape[:-1], device=target.device))  # logit-normal
        x_tau = (1 - tau[..., None]) * eps + tau[..., None] * target
        return F.mse_loss(self.velocity(x_tau, tau, h), target - eps, reduction="none").mean(-1)

    @torch.no_grad()
    def sample(self, h: torch.Tensor, num_samples: int = 1, steps: int | None = None) -> torch.Tensor:
        """h: (..., d_model) -> (..., num_samples, out_dim) via Euler integration."""
        steps = steps or self.steps
        h = h.unsqueeze(-2).expand(*h.shape[:-1], num_samples, h.shape[-1])
        x = torch.randn(*h.shape[:-1], self.out[-1].out_features, device=h.device)
        for i in range(steps):
            tau = torch.full(x.shape[:-1], i / steps, device=h.device)
            x = x + self.velocity(x, tau, h) / steps
        return x
