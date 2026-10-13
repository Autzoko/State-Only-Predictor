"""pi0.5-DROID + SOPT (E14): does SOPT still help a VLA that was itself trained on DROID's *actions*?

SOPT is pretrained on DROID proprio only. `lerobot/pi05_droid` (pi0.5 base fine-tuned on DROID by Physical
Intelligence, LeRobot port) has already seen the same robot's motions together with images, language and actions,
so it is the strongest test of whether SOPT's motion prior is redundant once a VLA has absorbed it.

Recipe (identical for every arm except SOPT; pi0.5's own fine-tuning defaults otherwise):
  - VLM (SigLIP + Gemma-2B) frozen (`train_expert_only`); Gemma-300M action expert + action/time projections
    trained with pi0.5's flow-matching loss; lr 2.5e-5, weight decay 0.01, grad clip 1.0 (openpi defaults);
  - cameras: agent view -> base_0_rgb, wrist -> left_wrist_0_rgb, right_wrist_0_rgb absent (masked), as in DROID;
  - state: current 17-d SOPT feature, QUANTILES-normalized and discretized into the prompt (stock pi0.5);
  - actions: 7-d LIBERO delta EE + gripper, QUANTILES-normalized, padded to 32 (stock pi0.5).
Arms (same names as E7 / E12): `none`; `expert_scratch` / `expert_droid` add the SOPT last-token feature of the
96-frame history to every action token through a zero-initialized MLP, so at initialization they equal `none`.
Tokenizer: the PaliGemma SentencePiece model openpi uses (public gs://big_vision/paligemma_tokenizer.model),
converted to a HF tokenizer and checked token-for-token against SentencePiece (with BOS); the HF copy is gated.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch import nn

from sopt.models.state_prior import StatePrior

BASE, REVISION = "lerobot/pi05_droid", "72824c0a93f00ce5bb8bedb7feb58953ba1da364"
TOKENIZER = os.environ.get("PALIGEMMA_TOKENIZER", "/scratch/ll5582/SOPT_DATA/paligemma_tokenizer/hf")
IMAGE_KEYS = ("observation.images.base_0_rgb", "observation.images.left_wrist_0_rgb")
ABSENT_KEY = "observation.images.right_wrist_0_rgb"
ARMS = ("none", "expert_scratch", "expert_droid")
OPT = {"lr": 2.5e-5, "weight_decay": 0.01, "clip": 1.0}


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
        """ctx: (B, 96, 17) raw SOPT features on the 15 Hz grid."""
        self.cond = self.mlp(self.prior.encode(ctx.float())[:, -1])


def _condition_expert(model, sopt: SOPTCond) -> None:
    """Add the SOPT feature to every action token (training forward and every denoising step)."""
    embed_suffix = model.embed_suffix

    def wrapped(noisy_actions, timestep):
        embs, pad_masks, att_masks, adarms_cond = embed_suffix(noisy_actions, timestep)
        assert sopt.cond is not None and sopt.cond.shape[0] == embs.shape[0], "make_batch must set the SOPT ctx"
        return embs + sopt.cond[:, None].to(embs.dtype), pad_masks, att_masks, adarms_cond

    model.embed_suffix = wrapped


def build_policy(prior: StatePrior, arm: str, action_stats: dict, device: str = "cuda", chunk: int = 50,
                 state_stats: dict | None = None):
    """-> (policy, preprocessor, postprocessor). Stats: q01 / q99 of the training actions and current states."""
    from huggingface_hub import snapshot_download
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
    from lerobot.processor import TokenizerProcessorStep

    assert arm in ARMS, arm
    base = snapshot_download(BASE, revision=REVISION)
    cfg = PreTrainedConfig.from_pretrained(base)
    cfg.input_features = {k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)) for k in (*IMAGE_KEYS, ABSENT_KEY)}
    cfg.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(prior.D,))
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    cfg.normalization_mapping = {"VISUAL": NormalizationMode.IDENTITY, "STATE": NormalizationMode.QUANTILES,
                                 "ACTION": NormalizationMode.QUANTILES}
    cfg.chunk_size = cfg.n_action_steps = chunk
    cfg.train_expert_only = True
    cfg.dtype = "bfloat16"
    cfg.gradient_checkpointing = True  # frozen-VLM activations still sit in the joint-attention graph
    cfg.device = device
    policy = PI05Policy.from_pretrained(base, config=cfg)
    hidden = policy.model.action_in_proj.out_features
    if arm != "none":
        policy.sopt = SOPTCond(prior, hidden)
        _condition_expert(policy.model, policy.sopt)
    # keep trainable master weights in fp32; compute under bf16 autocast
    for p in policy.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    stats = {"action": _q(action_stats), "observation.state": _q(state_stats)}
    pre, post = make_pi05_pre_post_processors(policy.config, dataset_stats=stats)
    for i, step in enumerate(pre.steps):  # gated HF tokenizer -> verified local copy of the same SentencePiece model
        if isinstance(step, TokenizerProcessorStep):
            pre.steps[i] = TokenizerProcessorStep(tokenizer_name=TOKENIZER, max_length=cfg.tokenizer_max_length,
                                                  padding_side="right", padding="max_length")
    return policy.to(device), pre, post


def _q(stats: dict) -> dict:
    q01 = torch.as_tensor(stats["q01"], dtype=torch.float32)
    q99 = torch.maximum(torch.as_tensor(stats["q99"], dtype=torch.float32), q01 + 1e-3)  # constant dims
    return {"q01": q01, "q99": q99}


def make_batch(policy, img: torch.Tensor, ctx: torch.Tensor, tasks: list[str]) -> dict:
    """img (B, 2, 3, H, W) float [0, 1] on device, ctx (B, 96, 17) on device. Sets the SOPT feature."""
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
