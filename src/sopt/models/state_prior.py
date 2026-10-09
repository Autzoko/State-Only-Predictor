"""StatePrior: patch-token Transformer over proprioceptive trajectories.

Objectives (cfg.objective):
  ar_regression  causal; every token predicts the next H frames (deltas w.r.t. its anchor frame), L1.
  ar_flow        same targets, rectified-flow head -> can sample multimodal futures.
  masked         bidirectional, MTM-style span / future masking; reconstructs masked patches as deltas
                 from the last visible frame before them (same target scale as the AR objectives).

Goal conditioning (cfg.goal_cond, post-training): a goal state (EE pose + gripper) is encoded relative to the
anchor frame and added to the token features entering the head. The backbone stays goal-agnostic, so each
token can carry its own goal without leaking goals across tokens, and the zero-initialized last layer makes
post-training start exactly at the pretrained model. Goal types: 0 none, 1 endpoint (state at anchor + H),
2 keyframe (state at the next gripper event, timing unknown).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sopt.data.features import GRIPPER, POS, ROT
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


GOAL_DIM = 12  # rel pos (3) + rot6d (6) + gripper (1) + one-hot type {endpoint, keyframe} (2)
GOAL_TYPES = {"none": 0, "endpoint": 1, "keyframe": 2}


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
        self.backbone = Transformer(
            cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.mlp_ratio, cfg.dropout, cfg.get("qk_norm", False)
        )  # checkpoints from before qk_norm existed were trained without it
        if self.objective == "masked":
            self.mask_token = nn.Parameter(torch.zeros(cfg.d_model))
            self.head = RegressionHead(cfg.d_model, P * D, cfg.head_hidden, cfg.head_depth)
        elif self.objective == "ar_regression":
            self.head = RegressionHead(cfg.d_model, H * D, cfg.head_hidden, cfg.head_depth)
        elif self.objective == "ar_flow":
            self.head = FlowHead(cfg.d_model, H * D, cfg.head_hidden, cfg.head_depth, cfg.flow_steps)
        else:
            raise ValueError(f"unknown objective {self.objective}")
        self.goal_cond = bool(cfg.get("goal_cond", False))
        if self.goal_cond:
            self.goal_proj = nn.Sequential(
                nn.Linear(GOAL_DIM, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.d_model)
            )
            nn.init.zeros_(self.goal_proj[-1].weight)
            nn.init.zeros_(self.goal_proj[-1].bias)
            self.register_buffer("goal_probs", torch.tensor(list(cfg.goal_probs), dtype=torch.float32))

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

    def goal_embedding(self, goal: torch.Tensor, anchor: torch.Tensor, gtype: torch.Tensor) -> torch.Tensor:
        """goal, anchor: (..., D) raw features; gtype: (...) long in {0, 1, 2} -> (..., d_model)."""
        nz = self.normalizer
        rel_pos = (goal[..., POS] - anchor[..., POS]) / nz.std[POS]
        g = nz.norm(goal)
        onehot = F.one_hot(gtype, 3)[..., 1:].to(goal.dtype)
        v = torch.cat([rel_pos, g[..., ROT], g[..., GRIPPER], onehot], dim=-1)
        v = v * (gtype > 0)[..., None].to(v.dtype)
        return self.goal_proj(v)

    def _sample_goal_types(self, B: int, device) -> torch.Tensor:
        return torch.multinomial(self.goal_probs, B, replacement=True).to(device)

    # ---- training -----------------------------------------------------------------------------
    def loss(
        self, x: torch.Tensor, mask: torch.Tensor, input_noise: float = 0.0, kf: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """x: (B, ctx_len + H, D) raw features (already augmented); mask: (B, ctx_len + H) bool;
        kf: (B, ctx_len + H, D) keyframe goal of every frame (needed when goal_cond)."""
        if self.goal_cond and kf is None:
            raise ValueError("goal_cond model needs keyframe goals (dataset next_event)")
        if self.objective == "masked":
            return self._masked_loss(x, mask, input_noise, kf)
        return self._ar_loss(x, mask, input_noise, kf)

    def _ar_loss(self, x, mask, input_noise, kf=None):
        C, P, H = self.cfg.ctx_len, self.P, self.H
        xn = self.normalizer.norm(x)
        inp = xn[:, :C] + input_noise * torch.randn_like(xn[:, :C])
        tokens, valid = self._patchify(inp, mask[:, :C])
        h = self.backbone(tokens, valid, causal=True)  # (B, N, d)
        anchors = torch.arange(P - 1, C, P, device=x.device)  # last frame of each patch
        fut = anchors[:, None] + 1 + torch.arange(H, device=x.device)  # (N, H)
        target = (x[:, fut] - x[:, anchors][:, :, None]) / self.normalizer.delta_std  # (B, N, H, D)
        if self.goal_cond:
            B, N = h.shape[:2]
            gtype = self._sample_goal_types(B, x.device)[:, None].expand(B, N)
            goal = torch.where((gtype == 2)[..., None], kf[:, anchors], x[:, anchors + H])
            h = h + self.goal_embedding(goal, x[:, anchors], gtype)
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
        return m, future

    def _masked_targets(self, x: torch.Tensor, pm: torch.Tensor) -> torch.Tensor:
        """Each patch's frames as deltas from the last frame of the nearest visible patch at or before it."""
        B, T, D = x.shape
        N = T // self.P
        idx = torch.arange(N, device=x.device).expand(B, N)
        ref_patch = torch.where(pm, torch.zeros_like(idx), idx).cummax(dim=1).values  # patch 0 is visible
        ref = x.gather(1, ((ref_patch + 1) * self.P - 1)[..., None].expand(B, N, D))  # (B, N, D)
        delta = x.reshape(B, N, self.P, D) - ref[:, :, None]
        return (delta / self.normalizer.delta_std).flatten(-2)

    def _masked_loss(self, x, mask, input_noise, kf=None):
        B = x.shape[0]
        xn = self.normalizer.norm(x)
        tokens, valid = self._patchify(xn + input_noise * torch.randn_like(xn), mask)
        pm, future = self._sample_patch_mask(B, tokens.shape[1], x.device)
        tokens = torch.where(pm[..., None], self.mask_token.to(tokens.dtype), tokens)
        h = self.backbone(tokens, valid, causal=False)
        if self.goal_cond:  # goals only make sense for forecasting masks (anchor = last context frame)
            C = self.cfg.ctx_len
            gtype = self._sample_goal_types(B, x.device) * future.long()
            goal = torch.where((gtype == 2)[:, None], kf[:, C - 1], x[:, C - 1 + self.H])
            h = h + self.goal_embedding(goal, x[:, C - 1], gtype)[:, None]
        per_token = self.head.loss(h, self._masked_targets(x, pm))
        w = (pm & valid).float()
        return {"loss": (per_token * w).sum() / w.sum().clamp(min=1)}

    # ---- inference ----------------------------------------------------------------------------
    @torch.no_grad()
    def forecast(
        self, ctx: torch.Tensor, num_samples: int = 1, goal: torch.Tensor | None = None, goal_type: str = "none"
    ) -> torch.Tensor:
        """ctx: (B, ctx_len, D) raw features -> (B, K, H, D) raw future frames.
        goal: (B, D) raw goal state for goal_type 'endpoint' / 'keyframe' (goal_cond models only)."""
        B, C, D = ctx.shape
        assert C == self.cfg.ctx_len
        g_emb = None
        if self.goal_cond:
            gtype = torch.full((B,), GOAL_TYPES[goal_type], dtype=torch.long, device=ctx.device)
            g_emb = self.goal_embedding(ctx[:, -1] if goal is None else goal, ctx[:, -1], gtype)
        elif goal_type != "none":
            raise ValueError("model was not trained with goal conditioning")
        if self.objective == "masked":
            x = torch.cat([ctx, ctx[:, -1:].expand(B, self.H, D)], dim=1)
            full = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
            tokens, valid = self._patchify(self.normalizer.norm(x), full)
            n_fut = self.H // self.P
            tokens[:, -n_fut:] = self.mask_token.to(tokens.dtype)
            h = self.backbone(tokens, valid, causal=False)[:, -n_fut:]
            if g_emb is not None:
                h = h + g_emb[:, None]
            delta = self.head.sample(h, 1)[..., 0, :].reshape(B, self.H, D) * self.normalizer.delta_std
            return (ctx[:, -1:] + delta)[:, None].expand(B, num_samples, self.H, D)
        h = self.encode(ctx)[:, -1]
        if g_emb is not None:
            h = h + g_emb
        delta = self.head.sample(h, num_samples).reshape(B, num_samples, self.H, D)
        return ctx[:, -1][:, None, None] + delta * self.normalizer.delta_std
