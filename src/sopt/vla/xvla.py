"""X-VLA + SOPT (E20): a soft-prompted cross-embodiment VLA (Florence-2 encoder + 24-layer flow-matching policy
transformer, 0.9B) pretrained on DROID / RoboMind / AgiBot (290K episodes; LIBERO is *not* in its pretraining
mixture, arXiv 2510.10274; the hub's domain table also lists fine-tuning domains, so we use the default domain 0,
never the LIBERO soft prompt 3).

Recipe = LeRobot's recommended X-VLA fine-tuning, identical for every arm: everything trainable (VLM at 0.1x lr),
lr 1e-4, betas (0.9, 0.95), weight decay 1e-4, grad clip 10; action_mode 'auto' (7-d LIBERO delta EE + gripper,
MSE), chunk 30 (the base model's), 10 denoising steps, two cameras + one empty view (as in lerobot/xvla-libero).
State: current 17-d SOPT feature (MEAN_STD), padded to the model's 20-d proprio input.
Arms: `none`; `expert_scratch` / `expert_droid` add the SOPT last-token feature of the 96-frame history to every
action token right after X-VLA's action encoder through a zero-initialized MLP (at init they equal `none`).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from sopt.models.state_prior import StatePrior

BASE, REVISION = "lerobot/xvla-base", "cdb7964e4fe842935d671bfab5a5ebe00a96648c"
TOKENIZER = ("facebook/bart-large", "cb48c1365bd826bd521f650dc2e0940aee54720c")
IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
ARMS = ("none", "expert_scratch", "expert_droid")
OPT = {"lr": 1e-4, "weight_decay": 1e-4, "clip": 10.0, "betas": (0.9, 0.95)}
CHUNK = 30
VLM_LR_SCALE = 0.1


class SOPTCond(nn.Module):
    def __init__(self, prior: StatePrior, hidden: int):
        super().__init__()
        self.prior = prior
        d = prior.cfg.d_model
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.cond = None

    def set(self, ctx: torch.Tensor) -> None:
        self.cond = self.mlp(self.prior.encode(ctx.float())[:, -1])


def _condition_expert(transformer, sopt: SOPTCond) -> None:
    """Add the SOPT feature to every action token (the action encoder output, before the visual tokens are
    concatenated); runs in the training forward and at every denoising step."""
    enc = transformer.action_encoder
    forward = enc.forward

    def wrapped(x, domain_id):
        y = forward(x, domain_id)
        assert sopt.cond is not None and sopt.cond.shape[0] == y.shape[0], "make_batch must set the SOPT ctx"
        return y + sopt.cond[:, None].to(y.dtype)

    enc.forward = wrapped


def build_policy(prior: StatePrior, arm: str, action_stats: dict, device: str = "cuda", chunk: int = CHUNK,
                 state_stats: dict | None = None):
    from huggingface_hub import snapshot_download
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
    from lerobot.policies.xvla.processor_xvla import make_xvla_pre_post_processors

    assert arm in ARMS, arm
    base = snapshot_download(BASE, revision=REVISION)
    cfg = PreTrainedConfig.from_pretrained(base)
    cfg.input_features = {k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)) for k in IMAGE_KEYS}
    cfg.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(prior.D,))
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    cfg.normalization_mapping = {"VISUAL": NormalizationMode.IDENTITY, "STATE": NormalizationMode.MEAN_STD,
                                 "ACTION": NormalizationMode.MEAN_STD}
    cfg.action_mode = "auto"
    cfg.empty_cameras = 1
    cfg.num_image_views = 3
    cfg.chunk_size = cfg.n_action_steps = chunk
    cfg.tokenizer_name = snapshot_download(TOKENIZER[0], revision=TOKENIZER[1])
    cfg.freeze_vision_encoder = cfg.freeze_language_encoder = False
    cfg.train_policy_transformer = cfg.train_soft_prompts = True
    cfg.dtype = "float32"  # fp32 master weights; compute under bf16 autocast
    cfg.device = device
    policy = XVLAPolicy.from_pretrained(base, config=cfg)
    if arm != "none":
        policy.sopt = SOPTCond(prior, policy.model.transformer.hidden_size)
        _condition_expert(policy.model.transformer, policy.sopt)
    stats = {"action": _ms(action_stats), "observation.state": _ms(state_stats)}
    pre, post = make_xvla_pre_post_processors(policy.config, dataset_stats=stats)
    return policy.to(device), pre, post


def _ms(stats: dict) -> dict:
    return {"mean": torch.as_tensor(stats["mean"], dtype=torch.float32),
            "std": torch.as_tensor(stats["std"], dtype=torch.float32)}


def param_groups(policy, lr: float) -> list[dict]:
    """X-VLA fine-tuning: the Florence-2 VLM at 0.1x lr, everything else (policy transformer, soft prompts, SOPT
    conditioning) at the full lr."""
    vlm, rest = [], []
    for n, p in policy.named_parameters():
        if p.requires_grad:
            (vlm if n.startswith("model.vlm.") else rest).append(p)
    return [{"params": vlm, "scale": VLM_LR_SCALE}, {"params": rest, "scale": 1.0}]


def make_batch(policy, img: torch.Tensor, ctx: torch.Tensor, tasks: list[str]) -> dict:
    """img (B, 2, 3, H, W) float [0, 1]; X-VLA's processor expects [0, 255]. ctx (B, 96, 17)."""
    if hasattr(policy, "sopt"):
        policy.sopt.set(ctx)
    return {IMAGE_KEYS[0]: img[:, 0] * 255.0, IMAGE_KEYS[1]: img[:, 1] * 255.0,
            "observation.state": ctx[:, -1].float(), "task": tasks}


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
