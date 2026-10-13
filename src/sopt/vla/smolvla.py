"""SmolVLA + SOPT motion tokens (E5).

Integration point: SmolVLA's `state_proj`. Its prefix builder already accepts a (B, N, d) state embedding and
appends N state tokens after image + language tokens (image/language do not attend to them; the action expert
reads them through the KV cache). We replace `state_proj` by `MotionStateProj`:

  observation.state = the last C frames of SOPT features on the 15 Hz grid, flattened to (B, C * 17), raw
                      (identity norm). Flattened because SmolVLA's prepare_state keeps only [:, -1] of 3-D
                      states; max_state_dim = C * 17 makes its padding a no-op.
  -> [ N motion tokens = Linear(SOPT backbone token features) ] + [ 1 current-state token ]

Arms: `none` (current-state token only = vanilla SmolVLA with SOPT state features), `scratch` (random motion
encoder: controls for extra tokens / history), `droid` (DROID-pretrained SOPT backbone). Images, language, VLM
and action expert are identical across arms.

E7 arms `expert_scratch` / `expert_droid` (E5 found prefix motion tokens are not significant): the prefix is
exactly the `none` arm's (one current-state token); SOPT's last-token feature conditions the *action expert*
instead, added to every action token after SmolVLA's action-time MLP through a zero-initialized projection, so
at initialization the policy equals `none`. This mirrors TrunkBC / MotionDecoder (E4b / E6), where the motion
feature sits next to the action output and pretraining helps significantly.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from sopt.models.state_prior import StatePrior

PINS = {
    "lerobot/smolvla_base": "d9f33c94a60fb382c90dea2164c96845bd955e28",
    "HuggingFaceTB/SmolVLM2-500M-Video-Instruct": "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
}
IMAGE_KEYS = ("observation.images.image", "observation.images.image2")


ARMS = ("none", "scratch", "droid", "expert_scratch", "expert_droid")
OPT = {"lr": 1e-4, "weight_decay": 1e-10, "clip": 10.0}  # E5 / E7 recipe


class MotionStateProj(nn.Module):
    """use_motion: 'prefix' (motion tokens in the VLM prefix, E5), 'expert' (feature stashed for the action expert,
    E7) or None."""

    def __init__(self, prior: StatePrior, hidden: int, use_motion: str | None, expert_hidden: int = 0):
        super().__init__()
        self.prior, self.use_motion = prior, use_motion
        self.cur_proj = nn.Linear(prior.D, hidden)
        d = prior.cfg.d_model
        if use_motion == "prefix":
            self.motion_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden))
        elif use_motion == "expert":
            self.expert_cond = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, expert_hidden), nn.SiLU(),
                                             nn.Linear(expert_hidden, expert_hidden))
            nn.init.zeros_(self.expert_cond[-1].weight)
            nn.init.zeros_(self.expert_cond[-1].bias)
        self.cond = None  # (B, expert_hidden), set by forward when use_motion == 'expert'

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: (B, C * 17) flattened history -> (B, N + 1, hidden)."""
        x = state.float().view(state.shape[0], self.prior.cfg.ctx_len, self.prior.D)
        cur = self.cur_proj(self.prior.normalizer.norm(x[:, -1]))[:, None]
        if self.use_motion == "expert":
            self.cond = self.expert_cond(self.prior.encode(x)[:, -1])
            return cur
        if not self.use_motion:
            return cur
        motion = self.motion_proj(self.prior.encode(x))  # (B, C / patch, hidden)
        return torch.cat([motion, cur.to(motion.dtype)], dim=1)


def _condition_expert(model) -> None:
    """Add the stashed SOPT feature to every action token. embed_prefix (which runs state_proj) always precedes
    embed_suffix, in training (forward) and at inference (prefix KV cache, then each denoising step)."""
    embed_suffix = model.embed_suffix

    def wrapped(noisy_actions, timestep):
        embs, pad_masks, att_masks = embed_suffix(noisy_actions, timestep)
        cond = model.state_proj.cond
        assert cond is not None and cond.shape[0] == embs.shape[0], "state_proj must run before embed_suffix"
        return embs + cond[:, None].to(embs.dtype), pad_masks, att_masks

    model.embed_suffix = wrapped


def _snapshot(repo_id: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id, revision=PINS[repo_id])


def build_policy(prior: StatePrior, arm: str, action_stats: dict, device: str = "cuda", chunk: int = 50):
    """-> (policy, preprocessor, postprocessor). `prior` supplies normalization and, for arm='droid', weights."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    base = _snapshot("lerobot/smolvla_base")
    cfg = PreTrainedConfig.from_pretrained(base)
    cfg.input_features = {k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)) for k in IMAGE_KEYS}
    state_dim = prior.cfg.ctx_len * prior.D
    cfg.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    cfg.normalization_mapping = {"VISUAL": NormalizationMode.IDENTITY, "STATE": NormalizationMode.IDENTITY,
                                 "ACTION": NormalizationMode.MEAN_STD}
    cfg.empty_cameras = 0
    cfg.chunk_size = chunk
    cfg.n_action_steps = chunk
    cfg.load_vlm_weights = False  # VLM weights come from the smolvla_base checkpoint
    cfg.vlm_model_name = _snapshot("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    cfg.device = device
    policy = SmolVLAPolicy.from_pretrained(base, config=cfg)
    policy.config.max_state_dim = state_dim  # after loading: the base state_proj (32 -> d) is replaced below
    assert arm in ARMS, arm
    hidden = policy.model.vlm_with_expert.config.text_config.hidden_size
    use_motion = None if arm == "none" else "expert" if arm.startswith("expert") else "prefix"
    policy.model.state_proj = MotionStateProj(prior, hidden, use_motion, policy.model.vlm_with_expert.expert_hidden_size)
    if use_motion == "expert":
        _condition_expert(policy.model)
    for p in policy.model.state_proj.parameters():
        p.requires_grad = True
    # keep trainable master weights in fp32; compute under bf16 autocast
    for p in policy.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    stats = {"action": {k: torch.as_tensor(v, dtype=torch.float32) for k, v in action_stats.items()}}
    pre, post = make_smolvla_pre_post_processors(policy.config, dataset_stats=stats)
    return policy.to(device), pre, post


def make_batch(policy, img: torch.Tensor, ctx: torch.Tensor, tasks: list[str]) -> dict:
    """img (B, 2, 3, H, W) float [0, 1], ctx (B, 96, 17): the flattened history is SmolVLA's observation.state."""
    return {IMAGE_KEYS[0]: img[:, 0], IMAGE_KEYS[1]: img[:, 1], "observation.state": ctx.flatten(1), "task": tasks}


def save_trainable(policy, path: str | Path, extra: dict) -> None:
    sd = {n: p.detach().cpu() for n, p in policy.named_parameters() if p.requires_grad}
    sd.update({n: b.detach().cpu() for n, b in policy.model.state_proj.named_buffers(prefix="model.state_proj")})
    torch.save({"trainable": sd, **extra}, path)


def load_trainable(policy, path: str | Path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = policy.load_state_dict(ck["trainable"], strict=False)
    assert not unexpected, unexpected
    return ck


def write_json(path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2))
