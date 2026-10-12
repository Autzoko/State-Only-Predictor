"""E4b policies. All share the same vision/task encoder so arms differ only in how actions are produced.

PriorPolicy   images + task condition a StatePrior (ar_flow) that predicts the next H states (15 Hz); an
              inverse-dynamics model turns predicted states into simulator actions. The trunk can start from
              scratch, from DROID pretraining, or from DROID -> LIBERO post-training.
DirectBC      the conventional baseline: images + task + current state -> chunk of actions (no prior, no IDM).
TrunkBC       the prior's Transformer as the policy trunk over the 15 Hz state history, conditioned on images +
              task, with an action-chunk head (same outputs/losses as DirectBC). Trunk from scratch / DROID /
              DROID -> LIBERO. Added after E4b v1: prior + IDM failed because the prior's near-term plan
              (0.05-0.2 s) is less precise than constant-velocity extrapolation (single scale for all horizons).
MotionDecoder the pretrained prior *is* the action decoder (E6): spatial vision + task tokens enter every prior
              layer through zero-initialized cross-attention (Flamingo-style; at init the decoder is exactly
              the pretrained prior), and the last token drives two heads trained jointly: a new action-chunk
              head and the pretrained flow head forecasting the next H states ("proprioceptive foresight").
              No state -> action pipeline: both heads read the shared features, actions are supervised directly.
KFMotor       "where" / "how" split (E10). The policy hidden state (pretrained history trunk + vision/task, as in
              TrunkBC) predicts the next keyframe (state at the next gripper event: EE pose + gripper, timing
              unknown). The keyframe enters through the prior's *pretrained* goal pathway (DROID post-training
              with hindsight keyframes, E3a) and conditions both the action head and the flow head's
              foresight. Keyframes come from the demos themselves (no extra labels); training mixes ground-truth
              and predicted keyframes (teacher forcing), test time uses predictions, or any external source
              (e.g. a VLM pointing at a grasp pose).
IDM           (current state, next k states at 20 Hz) -> 7-d action (6 OSC deltas + gripper open logit).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from sopt.data.features import GRIPPER, POS, ROT
from sopt.models.state_prior import StatePrior
from sopt.models.transformer import rope_cache

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


DINOV2 = ("facebook/dinov2-small", "ed25f3a31f01632728cabb09d1542f84ab7b0056")


class FrozenDINOv2(nn.Module):
    """Frozen DINOv2-S/14 at 224 px -> [CLS, mean patch] (768). Always in eval mode, no gradients."""

    out_dim = 768

    def __init__(self):
        super().__init__()
        from transformers import AutoModel

        self.net = AutoModel.from_pretrained(DINOV2[0], revision=DINOV2[1])
        self.net.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.net.eval()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
        h = self.net(pixel_values=x).last_hidden_state
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=-1).float()


class VisionTaskEncoder(nn.Module):
    """Shared image backbone over (agentview, wrist) + learned camera and task embeddings -> (B, out_dim).
    vision: 'resnet18' (ImageNet, fine-tuned; E4b / E6) or 'dinov2' (frozen DINOv2-S; E8)."""

    def __init__(self, n_tasks: int, out_dim: int, zero_init: bool, vision: str = "resnet18"):
        super().__init__()
        if vision == "resnet18":
            net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            net.fc = nn.Identity()
            self.backbone, feat = net, 512
        elif vision == "dinov2":
            self.backbone = FrozenDINOv2()
            feat = FrozenDINOv2.out_dim
        else:
            raise ValueError(vision)
        self.feat = feat
        self.cam = nn.Parameter(torch.zeros(2, feat))
        self.task = nn.Embedding(n_tasks, 64)
        self.proj = nn.Sequential(nn.Linear(2 * feat + 64, 512), nn.GELU(), nn.Linear(512, out_dim))
        if zero_init:  # a pretrained prior starts as its unconditional self
            nn.init.zeros_(self.proj[-1].weight)
            nn.init.zeros_(self.proj[-1].bias)
        self.register_buffer("mean", IMAGENET_MEAN, persistent=False)
        self.register_buffer("std", IMAGENET_STD, persistent=False)

    def forward(self, img: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        """img: (B, 2, 3, H, W) uint8."""
        B = img.shape[0]
        x = (img.flatten(0, 1).float() / 255.0 - self.mean) / self.std
        f = self.backbone(x).view(B, 2, self.feat) + self.cam
        return self.proj(torch.cat([f.flatten(1), self.task(task)], dim=-1))


class PriorPolicy(nn.Module):
    def __init__(self, prior: StatePrior, n_tasks: int):
        super().__init__()
        assert prior.objective == "ar_flow", "PriorPolicy expects an ar_flow prior"
        self.prior = prior
        self.cond = VisionTaskEncoder(n_tasks, prior.cfg.d_model, zero_init=True)

    def _h(self, ctx, img, task):
        return self.prior.encode(ctx)[:, -1] + self.cond(img, task)

    def loss(self, batch: dict) -> torch.Tensor:
        ctx, fut = batch["ctx"], batch["fut"]
        target = (fut - ctx[:, -1:]) / self.prior.normalizer.chunk_scale(self.prior.H)
        return self.prior.head.loss(self._h(ctx, batch["img"], batch["task"]), target.flatten(1)).mean()

    @torch.no_grad()
    def plan(self, ctx, img, task, num_samples: int = 1) -> torch.Tensor:
        """-> (B, K, H, D) future states."""
        h = self._h(ctx, img, task)
        B, H, D = ctx.shape[0], self.prior.H, self.prior.D
        delta = self.prior.head.sample(h, num_samples).reshape(B, num_samples, H, D)
        return ctx[:, -1][:, None, None] + delta * self.prior.normalizer.chunk_scale(self.prior.H)


class DirectBC(nn.Module):
    """Images + task + current state (and 4-step velocity) -> action chunk (L1 on OSC deltas, BCE on gripper)."""

    def __init__(self, n_tasks: int, chunk: int, state_mean: torch.Tensor, state_std: torch.Tensor,
                 vision: str = "resnet18"):
        super().__init__()
        self.chunk = chunk
        self.cond = VisionTaskEncoder(n_tasks, 512, zero_init=False, vision=vision)
        self.register_buffer("s_mean", state_mean)
        self.register_buffer("s_std", state_std)
        self.net = nn.Sequential(nn.Linear(512 + 2 * 17, 1024), nn.GELU(), nn.Linear(1024, 1024), nn.GELU(),
                                 nn.Linear(1024, chunk * 7))

    def _out(self, ctx, img, task):
        cur = (ctx[:, -1] - self.s_mean) / self.s_std
        vel = (ctx[:, -1] - ctx[:, -4]) / self.s_std
        return self.net(torch.cat([self.cond(img, task), cur, vel], -1)).view(-1, self.chunk, 7)

    def loss(self, batch: dict) -> torch.Tensor:
        out, act = self._out(batch["ctx"], batch["img"], batch["task"]), batch["act"]
        return F.l1_loss(out[..., :6], act[..., :6]) + F.binary_cross_entropy_with_logits(out[..., 6], act[..., 6])

    @torch.no_grad()
    def act(self, ctx, img, task) -> torch.Tensor:
        out = self._out(ctx, img, task)
        return torch.cat([out[..., :6].clamp(-1, 1), (out[..., 6:] > 0).float()], -1)  # gripper 1 = open


class VisionTokenEncoder(nn.Module):
    """Shared ImageNet ResNet18 feature maps of (agentview, wrist) as tokens, plus one task token -> (B, 2*S*S + 1, d)."""

    def __init__(self, n_tasks: int, d: int, img_size: int = 128):
        super().__init__()
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(net.children())[:-2])  # stride 32
        n = 2 * (img_size // 32) ** 2
        self.proj = nn.Linear(512, d)
        self.pos = nn.Parameter(torch.randn(1, n, d) * 0.02)  # camera and position
        self.task = nn.Embedding(n_tasks, d)
        self.ln = nn.LayerNorm(d)
        self.register_buffer("mean", IMAGENET_MEAN, persistent=False)
        self.register_buffer("std", IMAGENET_STD, persistent=False)

    def forward(self, img: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        B = img.shape[0]
        x = (img.flatten(0, 1).float() / 255.0 - self.mean) / self.std
        f = self.backbone(x).flatten(2).transpose(1, 2).reshape(B, -1, 512)  # (B, 2*S*S, 512)
        return self.ln(torch.cat([self.proj(f) + self.pos, self.task(task)[:, None]], dim=1))


class XAttnBlock(nn.Module):
    """Pre-LN cross-attention + MLP whose output projections start at zero (residual = identity at init)."""

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.ln_q, self.ln_ff = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        for lin in (self.attn.out_proj, self.ff[-1]):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        q = self.ln_q(x)
        x = x + self.attn(q, ctx, ctx, need_weights=False)[0]
        return x + self.ff(self.ln_ff(x))


class MotionDecoder(nn.Module):
    def __init__(self, prior: StatePrior, n_tasks: int, chunk: int, state_loss: float = 1.0, img_size: int = 128):
        super().__init__()
        assert prior.objective == "ar_flow", "MotionDecoder reuses the ar_flow head for foresight"
        self.prior, self.chunk, self.state_loss = prior, chunk, state_loss
        d, L = prior.cfg.d_model, prior.cfg.n_layers
        self.vis = VisionTokenEncoder(n_tasks, d, img_size)
        self.xattn = nn.ModuleList([XAttnBlock(d, prior.cfg.n_heads) for _ in range(L)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1024), nn.GELU(), nn.Linear(1024, 1024), nn.GELU(),
                                  nn.Linear(1024, chunk * 7))

    def prior_parameters(self):
        return self.prior.parameters()

    def _h(self, ctx, img, task) -> torch.Tensor:
        """-> (B, d) feature of the last history token, conditioned on vision + task in every layer."""
        p, bb = self.prior, self.prior.backbone
        valid = torch.ones(ctx.shape[:2], dtype=torch.bool, device=ctx.device)
        x, _ = p._patchify(p.normalizer.norm(ctx), valid)
        N = x.shape[1]
        cos, sin = rope_cache(N, bb.head_dim, x.device)
        mask = torch.ones(N, N, dtype=torch.bool, device=x.device).tril()
        v = self.vis(img, task)
        for xa, blk in zip(self.xattn, bb.blocks):
            x = blk(xa(x, v), cos, sin, mask)
        return bb.ln_f(x)[:, -1]

    def loss(self, batch: dict) -> torch.Tensor:
        ctx = batch["ctx"]
        h = self._h(ctx, batch["img"], batch["task"])
        out, act = self.head(h).view(-1, self.chunk, 7), batch["act"]
        loss = F.l1_loss(out[..., :6], act[..., :6]) + F.binary_cross_entropy_with_logits(out[..., 6], act[..., 6])
        if self.state_loss > 0:
            target = (batch["fut"] - ctx[:, -1:]) / self.prior.normalizer.chunk_scale(self.prior.H)
            loss = loss + self.state_loss * self.prior.head.loss(h, target.flatten(1)).mean()
        return loss

    @torch.no_grad()
    def act(self, ctx, img, task) -> torch.Tensor:
        out = self.head(self._h(ctx, img, task)).view(-1, self.chunk, 7)
        return torch.cat([out[..., :6].clamp(-1, 1), (out[..., 6:] > 0).float()], -1)

    @torch.no_grad()
    def foresight(self, ctx, img, task, num_samples: int = 1) -> torch.Tensor:
        """-> (B, K, H, D) forecast future states (for analysis)."""
        h = self._h(ctx, img, task)
        B, H, D = ctx.shape[0], self.prior.H, self.prior.D
        delta = self.prior.head.sample(h, num_samples).reshape(B, num_samples, H, D)
        return ctx[:, -1][:, None, None] + delta * self.prior.normalizer.chunk_scale(H)


class IDM(nn.Module):
    def __init__(self, k: int, state_mean: torch.Tensor, state_std: torch.Tensor, delta_std: torch.Tensor):
        super().__init__()
        self.k = k
        self.register_buffer("s_mean", state_mean)
        self.register_buffer("s_std", state_std)
        self.register_buffer("d_std", delta_std)
        self.net = nn.Sequential(nn.Linear(17 * (k + 1), 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(),
                                 nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 7))

    def _out(self, cur, nxt):
        x = torch.cat([(cur - self.s_mean) / self.s_std, ((nxt - cur[:, None]) / self.d_std).flatten(1)], -1)
        return self.net(x)

    def loss(self, batch: dict) -> torch.Tensor:
        out, act = self._out(batch["cur"], batch["nxt"]), batch["act"]
        return F.l1_loss(out[:, :6], act[:, :6]) + F.binary_cross_entropy_with_logits(out[:, 6], act[:, 6])

    @torch.no_grad()
    def act(self, cur, nxt) -> torch.Tensor:
        out = self._out(cur, nxt)
        return torch.cat([out[:, :6].clamp(-1, 1), (out[:, 6:] > 0).float()], -1)


class TrunkBC(nn.Module):
    def __init__(self, prior: StatePrior, n_tasks: int, chunk: int, vision: str = "resnet18"):
        super().__init__()
        self.prior, self.chunk = prior, chunk
        d = prior.cfg.d_model
        self.cond = VisionTaskEncoder(n_tasks, d, zero_init=False, vision=vision)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1024), nn.GELU(), nn.Linear(1024, 1024), nn.GELU(),
                                  nn.Linear(1024, chunk * 7))

    def _out(self, ctx, img, task):
        h = self.prior.encode(ctx)[:, -1] + self.cond(img, task)
        return self.head(h).view(-1, self.chunk, 7)

    def loss(self, batch: dict) -> torch.Tensor:
        out, act = self._out(batch["ctx"], batch["img"], batch["task"]), batch["act"]
        return F.l1_loss(out[..., :6], act[..., :6]) + F.binary_cross_entropy_with_logits(out[..., 6], act[..., 6])

    @torch.no_grad()
    def act(self, ctx, img, task) -> torch.Tensor:
        out = self._out(ctx, img, task)
        return torch.cat([out[..., :6].clamp(-1, 1), (out[..., 6:] > 0).float()], -1)


class KFMotor(nn.Module):
    def __init__(self, prior: StatePrior, n_tasks: int, chunk: int, vision: str = "resnet18",
                 state_loss: float = 1.0, teacher: float = 0.5):
        super().__init__()
        assert prior.objective == "ar_flow" and prior.goal_cond, "KFMotor needs a goal-conditioned ar_flow prior"
        self.prior, self.chunk, self.state_loss, self.teacher = prior, chunk, state_loss, teacher
        d = prior.cfg.d_model
        self.cond = VisionTaskEncoder(n_tasks, d, zero_init=False, vision=vision)
        self.kf_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 512), nn.GELU(), nn.Linear(512, 10))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1024), nn.GELU(), nn.Linear(1024, 1024), nn.GELU(),
                                  nn.Linear(1024, chunk * 7))

    # keyframe <-> normalized 10-d code (pos relative to the current EE, rot6d, gripper), the prior's own scales
    def kf_code(self, kf: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
        nz = self.prior.normalizer
        return torch.cat([(kf[:, POS] - cur[:, POS]) / nz.std[POS], nz.norm(kf)[:, ROT], nz.norm(kf)[:, GRIPPER]], -1)

    def kf_from_code(self, code: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
        nz = self.prior.normalizer
        kf = cur.clone()
        kf[:, POS] = cur[:, POS] + code[:, 0:3] * nz.std[POS]
        kf[:, ROT] = code[:, 3:9] * nz.std[ROT] + nz.mean[ROT]
        kf[:, GRIPPER] = code[:, 9:10] * nz.std[GRIPPER] + nz.mean[GRIPPER]
        return kf

    def _base(self, ctx, img, task):
        h = self.prior.encode(ctx)[:, -1] + self.cond(img, task)
        return h, self.kf_head(h)

    def _with_goal(self, h, kf, cur):
        gtype = torch.full(h.shape[:1], 2, dtype=torch.long, device=h.device)  # keyframe goal
        return h + self.prior.goal_embedding(kf, cur, gtype)

    def loss(self, batch: dict) -> torch.Tensor:
        ctx, kf = batch["ctx"], batch["kf"]
        cur = ctx[:, -1]
        h, code = self._base(ctx, batch["img"], batch["task"])
        kf_loss = F.l1_loss(code, self.kf_code(kf, cur))
        pred = self.kf_from_code(code.detach().float(), cur)
        use_gt = (torch.rand(h.shape[0], 1, device=h.device) < self.teacher)
        hg = self._with_goal(h, torch.where(use_gt, kf, pred), cur)
        out, act = self.head(hg).view(-1, self.chunk, 7), batch["act"]
        loss = F.l1_loss(out[..., :6], act[..., :6]) + F.binary_cross_entropy_with_logits(out[..., 6], act[..., 6])
        loss = loss + kf_loss
        if self.state_loss > 0:  # foresight given the true keyframe: the pretraining task, on LIBERO states
            target = (batch["fut"] - ctx[:, -1:]) / self.prior.normalizer.chunk_scale(self.prior.H)
            loss = loss + self.state_loss * self.prior.head.loss(self._with_goal(h, kf, cur), target.flatten(1)).mean()
        return loss

    @torch.no_grad()
    def keyframe(self, ctx, img, task) -> torch.Tensor:
        return self.kf_from_code(self._base(ctx, img, task)[1].float(), ctx[:, -1])

    @torch.no_grad()
    def act(self, ctx, img, task, kf: torch.Tensor | None = None) -> torch.Tensor:
        """kf: optional external keyframe (B, D) raw features, e.g. from a pointing model; else predicted."""
        h, code = self._base(ctx, img, task)
        cur = ctx[:, -1]
        kf = self.kf_from_code(code.float(), cur) if kf is None else kf
        out = self.head(self._with_goal(h, kf, cur)).view(-1, self.chunk, 7)
        return torch.cat([out[..., :6].clamp(-1, 1), (out[..., 6:] > 0).float()], -1)
