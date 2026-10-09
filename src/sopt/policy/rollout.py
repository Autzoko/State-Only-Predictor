"""Closed-loop LIBERO evaluation for E4b policies (one process per task, shared seeds across policies)."""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf

from sopt.data.libero import features_20hz
from sopt.models.state_prior import StatePrior
from sopt.policy.data import context_indices, interp_frames
from sopt.policy.models import IDM, DirectBC, PriorPolicy
from sopt.sim.libero_env import dataset_to_sim_gripper

IMG = 128


def load_policy(path: str, device) -> tuple[torch.nn.Module, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    n_tasks = len(ck["tasks"])
    if ck["kind"] == "prior":
        prior = StatePrior(OmegaConf.create(ck["prior_config"]), ck["prior_stats"])
        model = PriorPolicy(prior, n_tasks)
    else:
        st = ck["stats"]
        model = DirectBC(n_tasks, ck["chunk"], torch.tensor(st["mean"]), torch.tensor(st["std"]))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


def load_idm(path: str, device) -> IDM:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    st = ck["stats"]
    m = IDM(ck["k"], *(torch.tensor(st[k], dtype=torch.float32) for k in ("s_mean", "s_std", "d_std")))
    m.load_state_dict(ck["model"])
    return m.to(device).eval()


def _img(o: dict) -> torch.Tensor:
    from PIL import Image

    ims = [np.asarray(Image.fromarray(o[c]).resize((IMG, IMG), Image.BILINEAR)) for c in ("agent", "wrist")]
    return torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2)[None]  # (1, 2, 3, H, W) uint8


def run_episode(env, init_index: int, policy, kind: str, idm, task_id: int, device, replan: int = 8,
                max_steps: int = 400, ctx_len: int = 96, seed: int = 0) -> dict:
    o = env.reset(init_index, seed=seed)
    hist = [features_20hz(o["q"][None], o["fingers"][None])[0]]
    task = torch.tensor([task_id], device=device)
    plan, plan_t, chunk = None, 0, None
    for t in range(max_steps):
        F = np.stack(hist)
        if t % replan == 0:
            ctx = torch.from_numpy(interp_frames(F, context_indices(len(F) - 1, ctx_len)).astype(np.float32))[None]
            img = _img(o).to(device)
            if kind == "prior":
                fut = policy.plan(ctx.to(device), img, task)[0, 0].float().cpu().numpy()  # (H, D) at 15 Hz
                plan = np.concatenate([F[-1:], fut])  # plan[j] = state at j/15 s after replanning
                plan_t = t
            else:
                chunk = policy.act(ctx.to(device), img, task)[0].cpu().numpy()
        if kind == "prior":
            # targets at the next k control steps, read off the plan on its 15 Hz grid
            dt = (t - plan_t + np.arange(1, idm.k + 1)) / 20.0
            nxt = interp_frames(plan, dt * 15.0)
            a = idm.act(torch.from_numpy(F[-1:]).to(device), torch.from_numpy(nxt[None].astype(np.float32)).to(device))
            a = a[0].cpu().numpy()
        else:
            a = chunk[(t % replan)]
        action = np.concatenate([a[:6], dataset_to_sim_gripper(a[6:7])])
        o, success = env.step(action)
        hist.append(features_20hz(o["q"][None], o["fingers"][None])[0])
        if success:
            return {"success": True, "steps": t + 1}
    return {"success": False, "steps": max_steps}
