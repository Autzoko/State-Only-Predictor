"""Diffusion Policy + SOPT (E21): the classic action-generation model (Chi et al. 2023; LeRobot implementation):
ImageNet ResNet-18 encoders with spatial-softmax keypoints per camera + a 1-D conditional U-Net denoising an action
chunk. No robot pretraining, so it is the "action model without robot priors" end of the host spectrum.

Recipe (LeRobot / paper defaults where they apply; identical for every arm): one observation step, horizon 16
(executed actions chosen by vla_eval --n-exec), ResNet-18 (ImageNet) per camera with random 240x240 crops of the
256 px frames, DDIM with 100 training / 10 inference steps, U-Net down dims (512, 1024, 2048), lr 1e-4,
betas (0.95, 0.999), weight decay 1e-6, grad clip 10. Actions MIN_MAX (DDPM clips samples to [-1, 1]); state
(current 17-d SOPT feature) MEAN_STD; images ImageNet MEAN_STD. Diffusion Policy has no language input, so the
task is given as a one-hot `observation.environment_state` (task order = the training task list).
Arms: `none`; `expert_scratch` / `expert_droid` add the SOPT last-token feature of the 96-frame history to the
U-Net's global conditioning vector through a zero-initialized MLP (at init they equal `none`).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from sopt.models.state_prior import StatePrior

IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
ARMS = ("none", "expert_scratch", "expert_droid")
OPT = {"lr": 1e-4, "weight_decay": 1e-6, "clip": 10.0, "betas": (0.95, 0.999)}
CHUNK = 16
NEEDS_TASKS = True
IMAGENET = {"mean": torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1),
            "std": torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)}


class SOPTCond(nn.Module):
    def __init__(self, prior: StatePrior, out_dim: int):
        super().__init__()
        self.prior = prior
        d = prior.cfg.d_model
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 512), nn.SiLU(), nn.Linear(512, out_dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.cond = None
        self.cond_drop = 0.0

    def set(self, ctx: torch.Tensor) -> None:
        from sopt.vla.smolvla import _drop

        self.cond = _drop(self.mlp(self.prior.encode(ctx.float())[:, -1]), self.cond_drop, self.training)


def set_cond_drop(policy, p: float) -> None:
    if hasattr(policy, "sopt"):
        policy.sopt.cond_drop = p
    else:
        assert p == 0, "SOPT dropout needs an expert_* arm"


def _condition_global(diffusion, sopt: SOPTCond) -> None:
    prepare = diffusion._prepare_global_conditioning

    def wrapped(batch):
        g = prepare(batch)
        assert sopt.cond is not None and sopt.cond.shape[0] == g.shape[0], "make_batch must set the SOPT ctx"
        return g + sopt.cond.to(g.dtype)

    diffusion._prepare_global_conditioning = wrapped


def build_policy(prior: StatePrior, arm: str, action_stats: dict, device: str = "cuda", chunk: int = CHUNK,
                 state_stats: dict | None = None, tasks: list[str] | None = None):
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors

    assert arm in ARMS, arm
    assert tasks, "Diffusion Policy needs the task list (one-hot task conditioning)"
    cfg = DiffusionConfig(
        n_obs_steps=1, horizon=chunk, n_action_steps=chunk, crop_shape=(240, 240), crop_is_random=True,
        noise_scheduler_type="DDIM", num_train_timesteps=100, num_inference_steps=10, device=device,
        normalization_mapping={"VISUAL": NormalizationMode.MEAN_STD, "STATE": NormalizationMode.MEAN_STD,
                               "ENV": NormalizationMode.IDENTITY, "ACTION": NormalizationMode.MIN_MAX},
    )
    cfg.input_features = {k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)) for k in IMAGE_KEYS}
    cfg.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(prior.D,))
    cfg.input_features["observation.environment_state"] = PolicyFeature(type=FeatureType.ENV, shape=(len(tasks),))
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    policy = DiffusionPolicy(cfg)
    policy.task_index = {t: i for i, t in enumerate(tasks)}
    if arm != "none":
        policy.sopt = SOPTCond(prior, _global_cond_dim(policy))
        _condition_global(policy.diffusion, policy.sopt)
    stats = {"action": {"min": torch.as_tensor(action_stats["min"], dtype=torch.float32),
                        "max": torch.as_tensor(action_stats["max"], dtype=torch.float32)},
             "observation.state": {k: torch.as_tensor(state_stats[k], dtype=torch.float32) for k in ("mean", "std")},
             **{k: dict(IMAGENET) for k in IMAGE_KEYS}}
    pre, post = make_diffusion_pre_post_processors(policy.config, dataset_stats=stats)
    return policy.to(device), pre, post


def _global_cond_dim(policy) -> int:
    """Width of the vector returned by _prepare_global_conditioning (state + image keypoints + task one-hot)."""
    cfg = policy.config
    enc = policy.diffusion.rgb_encoder
    feat = (enc[0].feature_dim if isinstance(enc, nn.ModuleList) else enc.feature_dim) * len(IMAGE_KEYS)
    return (cfg.robot_state_feature.shape[0] + feat + cfg.env_state_feature.shape[0]) * cfg.n_obs_steps


def make_batch(policy, img: torch.Tensor, ctx: torch.Tensor, tasks: list[str]) -> dict:
    """img (B, 2, 3, H, W) float [0, 1], ctx (B, 96, 17). One observation step: state (B, 1, 17), task one-hot."""
    if hasattr(policy, "sopt"):
        policy.sopt.set(ctx)
    onehot = torch.zeros(len(tasks), 1, len(policy.task_index), device=ctx.device)
    for i, t in enumerate(tasks):
        onehot[i, 0, policy.task_index[t]] = 1.0
    return {IMAGE_KEYS[0]: img[:, 0], IMAGE_KEYS[1]: img[:, 1], "observation.state": ctx[:, -1:].float(),
            "observation.environment_state": onehot, "task": tasks}


def save_trainable(policy, path: str | Path, extra: dict) -> None:
    sd = {n: t.detach().cpu() for n, t in policy.state_dict().items()}  # trained from scratch: everything
    torch.save({"trainable": sd, **extra}, path)


def load_trainable(policy, path: str | Path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = policy.load_state_dict(ck["trainable"], strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return ck


def write_json(path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2))
