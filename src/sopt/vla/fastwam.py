"""FastWAM (world action model, LeRobot) + SOPT (E12): does SOPT also help a pretrained *world action model*?

FastWAM = Wan2.2-TI2V-5B video DiT ("video expert") + a 1B action DiT ("action expert") in a Mixture-of-Transformers;
action tokens attend to the first-frame video tokens and to the text context (+ one proprio token). Inference
predicts actions directly from the current frame (no future imagination). Base checkpoint `lerobot/fastwam_base`:
Wan2.2 weights with the action expert interpolated from the Wan DiT (no robot data, no dataset statistics).

Fine-tuning recipe used for every arm (identical except for SOPT):
  - video expert, VAE and text encoder frozen; the action expert + proprio encoder are trained with FastWAM's own
    flow-matching action loss on the inference path (current frame -> action chunk), i.e. no video loss;
  - proprio token = current 17-d SOPT feature (normalized with the SOPT normalizer), as in E5/E7 `none`.
Arms (same names as E7): `none`, `expert_scratch`, `expert_droid`. The SOPT feature of the 96-frame history
(last token) is mapped by a zero-initialized MLP and added to every action-expert token after its input encoder
(the analogue of SmolVLA's action-time MLP), so at initialization each arm equals `none`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from sopt.models.state_prior import StatePrior

BASE = "lerobot/fastwam_base"
ARMS = ("none", "expert_scratch", "expert_droid")
IMG = 224  # per camera; agent view and wrist view are concatenated along width -> 224 x 448


class SOPTCond(nn.Module):
    def __init__(self, prior: StatePrior, hidden: int, use: bool):
        super().__init__()
        self.prior, self.use = prior, use
        d = prior.cfg.d_model
        if use:
            self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)
        self.cond = None

    def set(self, ctx: torch.Tensor) -> None:
        """ctx: (B, 96, 17) raw SOPT features on the 15 Hz grid."""
        self.cond = self.mlp(self.prior.encode(ctx.float())[:, -1]) if self.use else None

    def proprio(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.prior.normalizer.norm(ctx[:, -1].float())


def _condition_action_expert(model, sopt: SOPTCond) -> None:
    expert = model.action_expert
    pre_dit = expert.pre_dit

    def wrapped(*args, **kwargs):
        out = pre_dit(*args, **kwargs)
        if sopt.cond is not None:
            assert sopt.cond.shape[0] == out["tokens"].shape[0]
            out["tokens"] = out["tokens"] + sopt.cond[:, None].to(out["tokens"].dtype)
        return out

    expert.pre_dit = wrapped


def build_fastwam(prior: StatePrior, arm: str, chunk: int = 32, device: str = "cuda"):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
    from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy

    assert arm in ARMS, arm
    cfg = FastWAMConfig.from_pretrained(BASE)
    cfg.device = device
    cfg.proprio_dim = prior.D
    cfg.input_features = {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMG, IMG)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, IMG, IMG)),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(prior.D,)),
    }
    cfg.action_horizon = chunk
    cfg.freeze_video_expert = True
    cfg.loss = {"lambda_video": 0.0, "lambda_action": 1.0}
    policy = FastWAMPolicy.from_pretrained(BASE, config=cfg, strict=False)  # proprio encoder 8 -> 17: re-init
    model = policy.model
    for p in model.parameters():
        p.requires_grad_(False)
    model.action_expert.requires_grad_(True)  # encoder / text + time embeddings / head
    for layer in model.mot.layers:  # the action DiT blocks are re-parented onto the MoT layers
        layer.blocks["action"].requires_grad_(True)
    model.proprio_encoder.requires_grad_(True)
    sopt = SOPTCond(prior, model.action_expert.hidden_dim, use=arm != "none").to(device)
    sopt.requires_grad_(arm != "none")
    _condition_action_expert(model, sopt)
    return policy, sopt


def trainable_state(model, sopt: SOPTCond) -> dict:
    return {"model": {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad},
            "sopt": sopt.state_dict()}


def load_trainable(model, sopt: SOPTCond, state: dict) -> None:
    params = dict(model.named_parameters())
    trainable = {k for k, v in params.items() if v.requires_grad}
    assert set(state["model"]) == trainable, set(state["model"]) ^ trainable
    with torch.no_grad():
        for k, v in state["model"].items():
            params[k].copy_(v)
    sopt.load_state_dict(state["sopt"])


def images(img: torch.Tensor) -> torch.Tensor:
    """(B, 2, 3, H, W) uint8 -> (B, 3, 224, 448) float in [0, 1] (agent view | wrist view)."""
    x = img.float().div(255.0).flatten(0, 1)
    x = F.interpolate(x, size=(IMG, IMG), mode="bilinear", align_corners=False, antialias=True)
    x = x.view(img.shape[0], 2, 3, IMG, IMG)
    return torch.cat([x[:, 0], x[:, 1]], dim=-1)


class PromptCache:
    def __init__(self, model, template: str):
        self.model, self.template, self.cache = model, template, {}

    @torch.no_grad()
    def __call__(self, tasks: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        for t in set(tasks) - set(self.cache):
            self.cache[t] = self.model.encode_prompt(self.template.format(task=t))
        ctx = torch.cat([self.cache[t][0] for t in tasks])
        mask = torch.cat([self.cache[t][1] for t in tasks])
        return ctx, mask


def action_loss(model, sopt: SOPTCond, prompts: PromptCache, batch: dict, tasks: list[str],
                a_mean: torch.Tensor, a_std: torch.Tensor) -> torch.Tensor:
    """FastWAM's flow-matching action loss on its inference path (current frame -> chunk)."""
    dev = model.device
    img = images(batch["img"].to(dev, non_blocking=True))
    with torch.no_grad():
        first = model._encode_video_latents(img.unsqueeze(2).to(model.torch_dtype))[:, :, 0:1]
    ctx = batch["ctx"].to(dev, non_blocking=True)
    context, mask = prompts([tasks[int(i)] for i in batch["task"]])
    context, mask = model._append_proprio_to_context(context, mask, sopt.proprio(ctx).to(model.torch_dtype))
    sopt.set(ctx)
    action = ((batch["act"].to(dev) - a_mean) / a_std).to(model.torch_dtype)
    sch = model.train_action_scheduler
    noise = torch.randn_like(action)
    t = sch.sample_training_t(batch_size=action.shape[0], device=dev, dtype=action.dtype)
    noisy, target = sch.add_noise(action, noise, t), sch.training_target(action, noise, t)
    fuse = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
    # FastWAM decorates this inference helper with no_grad; call the undecorated function to train through it
    pred = type(model)._predict_action_noise.__wrapped__(model, first_frame_latents=first, latents_action=noisy, timestep_action=t,
                                       context=context, context_mask=mask, fuse_vae_embedding_in_latents=fuse)
    return model._compute_training_action_loss({"action_is_pad": batch["act_pad"].to(dev)}, pred, target, t)


@torch.no_grad()
def predict(model, sopt: SOPTCond, prompts: PromptCache, img: torch.Tensor, ctx: torch.Tensor, task: str,
            a_mean: torch.Tensor, a_std: torch.Tensor, chunk: int, steps: int = 10, seed: int = 0) -> torch.Tensor:
    """One environment: img (1, 2, 3, H, W) uint8, ctx (1, 96, 17) -> (chunk, 7) actions (dataset convention)."""
    dev = model.device
    context, mask = prompts([task])
    sopt.set(ctx.to(dev))
    out = model.infer_action(prompt=None, input_image=images(img.to(dev)), action_horizon=chunk,
                             proprio=sopt.proprio(ctx.to(dev)), context=context, context_mask=mask,
                             num_inference_steps=steps, seed=seed, rand_device="cpu")["action"]
    return out.to(dev).float() * a_std + a_mean
