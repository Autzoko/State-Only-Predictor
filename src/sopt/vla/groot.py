"""GR00T N1.7 + SOPT (E22): NVIDIA's cross-embodiment VLA (Cosmos-Reason2 / Qwen3-VL backbone + flow-matching DiT
action head, 3B). Base checkpoint nvidia/GR00T-N1.7-3B; LIBERO appears only in separately released post-trained
variants. We use the fresh `new_embodiment` projections (never the `libero_sim` ones).

Recipe = LeRobot / Isaac-GR00T fine-tuning defaults, identical for every arm: backbone (LLM + vision) frozen,
projector + DiT + VL layer norm trainable; lr 1e-4, betas (0.95, 0.999), weight decay 1e-5 (not on biases / norms),
grad clip 1.0, bf16 compute; VLM image preprocessing on the GPU (torch path, center crop); chunk 16 (as in LeRobot's new-embodiment example), absolute (= LIBERO delta) actions,
min-max normalized state (current 17-d SOPT feature) and actions.
Arms: `none`; `expert_scratch` / `expert_droid` add the SOPT last-token feature of the 96-frame history to every
action token right after the action encoder through a zero-initialized MLP (at init they equal `none`).
E31 interfaces (docs/HOST_ANALYSIS.md section 10), same zero-init guarantee:
  adaln_*   the SOPT feature is added to the DiT's timestep embedding, which modulates every block through AdaLayerNorm
            (and the output norm): conditioning of every layer instead of one input bias.
  xattn_*   a motion cross-attention branch after every 4th DiT block (vision-language cross-attention blocks): action
            tokens attend to *all* SOPT history tokens (own softmax, so they are not diluted among the VL tokens),
            output projection zero-initialized, mixed in by a learned per-token gate sigma(W [h, temb]) that sees the
            token's VL-informed state and the denoising time -> the policy learns when to rely on the motion prior.
            --cond-drop drops the whole branch per training sample (modality dropout: the VL path must stay competent).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from sopt.models.state_prior import StatePrior

BASE, REVISION = "nvidia/GR00T-N1.7-3B", "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
# processor (tokenizer / image + video processors) of the backbone; gated repo, cached with the account's token
BACKBONE_PROCESSOR = ("nvidia/Cosmos-Reason2-2B", "9ce19a195e423419c349abfc86fd07178b230561")
IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
ARMS = ("none", "expert_scratch", "expert_droid", "adaln_scratch", "adaln_droid", "xattn_scratch", "xattn_droid")
XATTN_EVERY = 4
OPT = {"lr": 1e-4, "weight_decay": 1e-5, "clip": 1.0, "betas": (0.95, 0.999)}
CHUNK = 16


class SOPTCond(nn.Module):
    def __init__(self, prior: StatePrior, hidden: int):
        super().__init__()
        self.prior = prior
        d = prior.cfg.d_model
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.cond = None
        self.cond_drop = 0.0

    def set(self, ctx: torch.Tensor) -> None:
        from sopt.vla.smolvla import _drop

        self.cond = _drop(self.mlp(self.prior.encode(ctx.float())[:, -1]), self.cond_drop, self.training)


class SOPTMemory(nn.Module):
    """E31 xattn: all SOPT history tokens as keys / values for the motion cross-attention branches."""

    def __init__(self, prior: StatePrior, dim: int, max_tokens: int = 256):
        super().__init__()
        self.prior = prior
        self.proj = nn.Sequential(nn.LayerNorm(prior.cfg.d_model), nn.Linear(prior.cfg.d_model, dim))
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, dim))
        nn.init.normal_(self.pos, std=0.02)
        self.mem, self.keep, self.temb = None, None, None
        self.cond_drop = 0.0
        self.gates = []  # mean gate per branch of the last forward (diagnostics)

    def set(self, ctx: torch.Tensor) -> None:
        tok = self.prior.encode(ctx.float())
        self.mem = self.proj(tok) + self.pos[:, : tok.shape[1]]
        keep = torch.ones(tok.shape[0], device=tok.device)
        if self.training and self.cond_drop > 0:
            keep = (torch.rand_like(keep) >= self.cond_drop).float()
        self.keep = keep


class MotionXAttn(nn.Module):
    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.gate = nn.Linear(2 * dim, 1)  # sigma(0) = 0.5 at init; the zero `out` keeps the policy == none
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, h: torch.Tensor, mem: SOPTMemory) -> torch.Tensor:
        m = mem.mem.to(h.dtype)
        a, _ = self.attn(self.norm(h), m, m, need_weights=False)
        t = mem.temb.to(h.dtype)[:, None].expand(-1, h.shape[1], -1)
        g = torch.sigmoid(self.gate(torch.cat([h, t], -1)))
        mem.gates.append(g.detach().float().mean().item())
        return h + mem.keep.to(h.dtype)[:, None, None] * g * self.out(a)


def _dit(head):
    dits = [m for m in head.modules() if hasattr(m, "transformer_blocks") and hasattr(m, "timestep_encoder")]
    assert len(dits) == 1, len(dits)
    return dits[0]


def _condition_timestep(dit, sopt) -> None:
    """adaln: SOPT feature + timestep embedding (B, D) -> every AdaLayerNorm and the output modulation."""
    forward = dit.timestep_encoder.forward

    def wrapped(timesteps):
        temb = forward(timesteps)
        assert sopt.cond is not None and sopt.cond.shape == temb.shape, (None if sopt.cond is None else sopt.cond.shape, temb.shape)
        return temb + sopt.cond.to(temb.dtype)

    dit.timestep_encoder.forward = wrapped


def _attach_xattn(dit, mem: SOPTMemory, dim: int) -> nn.ModuleList:
    forward = dit.timestep_encoder.forward

    def capture(timesteps):  # the blocks' temb, for the gates
        mem.temb = forward(timesteps)
        mem.gates = []
        return mem.temb

    dit.timestep_encoder.forward = capture
    idx = [i for i in range(len(dit.transformer_blocks)) if i % XATTN_EVERY == 0]  # even = VL cross-attention blocks
    branches = nn.ModuleList([MotionXAttn(dim) for _ in idx])
    for i, br in zip(idx, branches):
        dit.transformer_blocks[i].register_forward_hook(lambda _m, _a, out, br=br: br(out, mem))
    return branches


def set_cond_drop(policy, p: float) -> None:
    if hasattr(policy, "sopt"):
        policy.sopt.cond_drop = p
    else:
        assert p == 0, "SOPT dropout needs an expert_* arm"


def _condition_expert(action_encoder, sopt: SOPTCond) -> None:
    forward = action_encoder.forward

    def wrapped(actions, timesteps, cat_ids):
        y = forward(actions, timesteps, cat_ids)
        assert sopt.cond is not None and sopt.cond.shape[0] == y.shape[0], "make_batch must set the SOPT ctx"
        return y + sopt.cond[:, None].to(y.dtype)

    action_encoder.forward = wrapped


def build_policy(prior: StatePrior, arm: str, action_stats: dict, device: str = "cuda", chunk: int = CHUNK,
                 state_stats: dict | None = None):
    from huggingface_hub import snapshot_download
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.groot.configuration_groot import GrootConfig
    from lerobot.policies.groot.modeling_groot import GrootPolicy
    from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors

    assert arm in ARMS, arm
    snapshot_download(BACKBONE_PROCESSOR[0], revision=BACKBONE_PROCESSOR[1], allow_patterns=["*.json", "*.txt"])  # processor only, cached (offline nodes)
    cfg = GrootConfig(base_model_path=snapshot_download(BASE, revision=REVISION), embodiment_tag="new_embodiment", chunk_size=chunk,
                      n_action_steps=chunk, use_relative_actions=False, use_bf16=True, device=device)
    cfg.input_features = {k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)) for k in IMAGE_KEYS}
    cfg.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(prior.D,))
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    policy = GrootPolicy(cfg)
    head = policy._groot_model.action_head
    if arm.startswith("expert_"):
        hidden = head.action_encoder.W3.W.shape[-1]
        policy.sopt = SOPTCond(prior, hidden)
        _condition_expert(head.action_encoder, policy.sopt)
    elif arm.startswith("adaln_"):
        dit = _dit(head)
        policy.sopt = SOPTCond(prior, dit.timestep_encoder.timestep_embedder.linear_2.out_features)
        _condition_timestep(dit, policy.sopt)
    elif arm.startswith("xattn_"):
        dit = _dit(head)
        dim = dit.timestep_encoder.timestep_embedder.linear_2.out_features
        policy.sopt = SOPTMemory(prior, dim)
        policy.sopt.branches = _attach_xattn(dit, policy.sopt, dim)
    for p in policy.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    stats = {"action": _mm(action_stats), "observation.state": _mm(state_stats)}
    pre, post = make_groot_pre_post_processors(policy.config, dataset_stats=stats)
    # The base checkpoint's processor_config sets use_albumentations=True, i.e. per-frame cv2/numpy preprocessing on
    # the CPU (~12 s/step at bs 32, GPU idle). Use LeRobot's torch path instead: same crop/resize geometry
    # (crop 230 -> 256, shortest edge 256), run on the GPU, deterministic center crop; identical for every arm.
    vlm = [s for s in pre.steps if type(s).__name__ == "GrootN17VLMEncodeStep"]
    assert len(vlm) == 1, pre.steps
    vlm[0].use_albumentations = False
    return policy.to(device), pre, post


def _mm(stats: dict) -> dict:
    return {"min": torch.as_tensor(stats["min"], dtype=torch.float32),
            "max": torch.as_tensor(stats["max"], dtype=torch.float32),
            "mean": torch.as_tensor(stats["mean"], dtype=torch.float32),
            "std": torch.as_tensor(stats["std"], dtype=torch.float32)}


def param_groups(policy, lr: float) -> list[dict]:
    """Isaac-GR00T: no weight decay on biases and normalization parameters."""
    groups = policy._build_weight_decay_parameter_groups(policy)
    return [{**g, "scale": 1.0} for g in groups]


def make_batch(policy, img: torch.Tensor, ctx: torch.Tensor, tasks: list[str]) -> dict:
    if hasattr(policy, "sopt"):
        policy.sopt.set(ctx)
    return {IMAGE_KEYS[0]: img[:, 0], IMAGE_KEYS[1]: img[:, 1], "observation.state": ctx[:, -1].float(),
            "task": tasks}


def save_trainable(policy, path: str | Path, extra: dict) -> None:
    sd = {n: p.detach().cpu() for n, p in policy.named_parameters() if p.requires_grad}
    if hasattr(policy, "sopt"):
        sd.update({n: b.detach().cpu() for n, b in policy.sopt.named_buffers(prefix="sopt")})
    torch.save({"trainable": sd, **extra}, path)


def load_trainable(policy, path: str | Path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    trainable = {n for n, p in policy.named_parameters() if p.requires_grad}
    assert trainable <= set(ck["trainable"]), trainable - set(ck["trainable"])
    missing, unexpected = policy.load_state_dict(ck["trainable"], strict=False)
    assert not unexpected, unexpected
    return ck


def write_json(path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2))
