"""StatePrior: patch-token Transformer over proprioceptive trajectories.

Objectives (cfg.objective):
  ar_regression  causal; every token predicts the next H frames (deltas w.r.t. its anchor frame), L1.
  ar_flow        same targets, rectified-flow head -> can sample multimodal futures.
  masked         bidirectional, MTM-style span / future masking; reconstructs masked patches as deltas
                 from the last visible frame before them (same target scale as the AR objectives).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sopt.models.heads import FlowHead, RegressionHead
from sopt.models.transformer import Transformer


class Normalizer(nn.Module):
    def __init__(self, stats: dict):
        super().__init__()
        for k in ("mean", "std", "delta_std"):
            self.register_buffer(k, torch.as_tensor(stats[k], dtype=torch.float32))

    def norm(self, x):
        return (x - self.mean) / self.std

    def denorm(self, x):
        return x * self.std + self.mean


class StatePrior(nn.Module):
    def __init__(self, cfg, stats: dict):
        super().__init__()
        self.cfg = cfg
        D, P, H = cfg.feature_dim, cfg.patch, cfg.horizon
        assert cfg.ctx_len % P == 0 and H % P == 0, "ctx_len and horizon must be multiples of patch"
        self.D, self.P, self.H = D, P, H
        self.objective = cfg.objective
        self.normalizer = Normalizer(stats)
        self.embed = nn.Linear(P * D, cfg.d_model)
        self.backbone = Transformer(cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
        if self.objective == "masked":
            self.mask_token = nn.Parameter(torch.zeros(cfg.d_model))
            self.head = RegressionHead(cfg.d_model, P * D, cfg.head_hidden, cfg.head_depth)
        elif self.objective == "ar_regression":
            self.head = RegressionHead(cfg.d_model, H * D, cfg.head_hidden, cfg.head_depth)
        elif self.objective == "ar_flow":
            self.head = FlowHead(cfg.d_model, H * D, cfg.head_hidden, cfg.head_depth, cfg.flow_steps)
        else:
            raise ValueError(f"unknown objective {self.objective}")

    @property
    def window(self) -> int:
        """Frames per training sample."""
        return self.cfg.ctx_len + self.H

    # ---- shared -------------------------------------------------------------------------------
    def _patchify(self, xn: torch.Tensor, mask: torch.Tensor):
        B, T, D = xn.shape
        tokens = self.embed(xn.reshape(B, T // self.P, self.P * D))
        valid = mask.reshape(B, T // self.P, self.P).all(-1)
        valid[:, 0] = True
        return tokens, valid

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Raw features (B, T, D) -> token features (B, T/P, d_model). Used for probing / transfer."""
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device) if mask is None else mask
        tokens, valid = self._patchify(self.normalizer.norm(x), mask)
        return self.backbone(tokens, valid, causal=self.objective != "masked")

    # ---- training -----------------------------------------------------------------------------
    def loss(self, x: torch.Tensor, mask: torch.Tensor, input_noise: float = 0.0) -> dict[str, torch.Tensor]:
        """x: (B, ctx_len + H, D) raw features (already augmented); mask: (B, ctx_len + H) bool."""
        if self.objective == "masked":
            return self._masked_loss(x, mask, input_noise)
        return self._ar_loss(x, mask, input_noise)

    def _ar_loss(self, x, mask, input_noise):
        C, P, H = self.cfg.ctx_len, self.P, self.H
        xn = self.normalizer.norm(x)
        inp = xn[:, :C] + input_noise * torch.randn_like(xn[:, :C])
        tokens, valid = self._patchify(inp, mask[:, :C])
        h = self.backbone(tokens, valid, causal=True)  # (B, N, d)
        anchors = torch.arange(P - 1, C, P, device=x.device)  # last frame of each patch
        fut = anchors[:, None] + 1 + torch.arange(H, device=x.device)  # (N, H)
        target = (x[:, fut] - x[:, anchors][:, :, None]) / self.normalizer.delta_std  # (B, N, H, D)
        per_token = self.head.loss(h, target.flatten(-2))
        w = mask[:, anchors].float()  # padded futures repeat the last frame = "robot at rest"
        return {"loss": (per_token * w).sum() / w.sum().clamp(min=1)}

    def _sample_patch_mask(self, B: int, N: int, device) -> torch.Tensor:
        """True = masked. Either the last H/P patches (forecasting) or ~`mask_ratio` of patches in short spans:
        max-pooling the noise over a 3-wide window makes neighbours win together, top-k fixes the count."""
        n_fut = self.H // self.P
        future = torch.rand(B, device=device) < self.cfg.future_mask_prob
        noise = F.max_pool1d(torch.rand(B, 1, N, device=device), 3, stride=1, padding=1)[:, 0]
        noise[:, 0] = -1.0  # keep one visible anchor
        n_target = min(max(1, round(self.cfg.mask_ratio * N)), N - 1)
        m = torch.zeros(B, N, dtype=torch.bool, device=device)
        m.scatter_(1, noise.topk(n_target, dim=1).indices, True)
        m[future] = False
        m[future, N - n_fut :] = True
        return m

    def _masked_targets(self, x: torch.Tensor, pm: torch.Tensor) -> torch.Tensor:
        """Each patch's frames as deltas from the last frame of the nearest visible patch at or before it."""
        B, T, D = x.shape
        N = T // self.P
        idx = torch.arange(N, device=x.device).expand(B, N)
        ref_patch = torch.where(pm, torch.zeros_like(idx), idx).cummax(dim=1).values  # patch 0 is visible
        ref = x.gather(1, ((ref_patch + 1) * self.P - 1)[..., None].expand(B, N, D))  # (B, N, D)
        delta = x.reshape(B, N, self.P, D) - ref[:, :, None]
        return (delta / self.normalizer.delta_std).flatten(-2)

    def _masked_loss(self, x, mask, input_noise):
        B = x.shape[0]
        xn = self.normalizer.norm(x)
        tokens, valid = self._patchify(xn + input_noise * torch.randn_like(xn), mask)
        pm = self._sample_patch_mask(B, tokens.shape[1], x.device)
        tokens = torch.where(pm[..., None], self.mask_token.to(tokens.dtype), tokens)
        h = self.backbone(tokens, valid, causal=False)
        per_token = self.head.loss(h, self._masked_targets(x, pm))
        w = (pm & valid).float()
        return {"loss": (per_token * w).sum() / w.sum().clamp(min=1)}

    # ---- inference ----------------------------------------------------------------------------
    @torch.no_grad()
    def forecast(self, ctx: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """ctx: (B, ctx_len, D) raw features -> (B, K, H, D) raw future frames."""
        B, C, D = ctx.shape
        assert C == self.cfg.ctx_len
        if self.objective == "masked":
            x = torch.cat([ctx, ctx[:, -1:].expand(B, self.H, D)], dim=1)
            full = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
            tokens, valid = self._patchify(self.normalizer.norm(x), full)
            n_fut = self.H // self.P
            tokens[:, -n_fut:] = self.mask_token.to(tokens.dtype)
            h = self.backbone(tokens, valid, causal=False)[:, -n_fut:]
            delta = self.head.sample(h, 1)[..., 0, :].reshape(B, self.H, D) * self.normalizer.delta_std
            return (ctx[:, -1:] + delta)[:, None].expand(B, num_samples, self.H, D)
        h = self.encode(ctx)[:, -1]
        delta = self.head.sample(h, num_samples).reshape(B, num_samples, self.H, D)
        return ctx[:, -1][:, None, None] + delta * self.normalizer.delta_std
