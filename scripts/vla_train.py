#!/usr/bin/env python
"""E5 / E7 / E13 / E14 / E20: fine-tune a VLA (SmolVLA; pi0.5-DROID: --vla pi05; X-VLA: --vla xvla) with or without SOPT.

  python scripts/vla_train.py --arm droid   --k 5 --run e5_vla_droid_k5
  python scripts/vla_train.py --arm scratch --k 5 --run e5_vla_scratch_k5
  python scripts/vla_train.py --arm none    --k 5 --run e5_vla_none_k5
Same data / demos / schedule for all arms; SmolVLA defaults (frozen VLM, trainable action expert) otherwise.
"""

import argparse
import importlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sopt.models.state_prior import StatePrior
from sopt.policy.data import PolicyDataset, Raw20, build_frame_cache, select_episodes
from sopt.train.pretrain import load_checkpoint
from sopt.utils.config import REPO_ROOT
from sopt.utils.misc import JsonlLogger, git_rev, seed_everything

DROID_PRIOR = "outputs/e2_ar_flow_f100_s/best.pt"
CHUNK, CTX = 50, 96


def vla_module(name: str):
    return importlib.import_module(f"sopt.vla.{name}")


def to_lerobot(V, policy, b: dict, tasks: list[str], device) -> dict:
    img = b["img"].to(device, non_blocking=True).float() / 255.0  # (B, 2, 3, H, W)
    batch = V.make_batch(policy, img, b["ctx"].to(device, non_blocking=True), [tasks[int(i)] for i in b["task"]])
    return {**batch, "action": b["act"].to(device, non_blocking=True), "action_is_pad": b["act_pad"].to(device)}


def stats_of(x: np.ndarray) -> dict:
    return {"mean": x.mean(0), "std": np.maximum(x.std(0), 1e-3), "q01": np.quantile(x, 0.01, axis=0),
            "q99": np.quantile(x, 0.99, axis=0), "min": x.min(0), "max": np.maximum(x.max(0), x.min(0) + 1e-3)}


def lr_at(step, total, peak, warmup=1000, final=2.5e-6):
    if step < warmup:
        return peak * (step + 1) / warmup
    p = min((step - warmup) / max(1, total - warmup), 1.0)
    return final + (peak - final) * 0.5 * (1 + math.cos(math.pi * p))


def make_prior(arm: str, prior: str = DROID_PRIOR) -> StatePrior:
    ref, _ = load_checkpoint(REPO_ROOT / prior)  # same config + normalization for every arm
    if arm in ("droid", "expert_droid"):
        return ref
    return StatePrior(ref.cfg, {k: v.numpy() for k, v in ref.normalizer.state_dict().items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vla", default="smolvla", choices=["smolvla", "pi05", "xvla", "diffusion", "groot"])
    p.add_argument("--arm", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--k", type=int, default=5, help="demos per task (0 = all)")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=None, help="default: the VLA's recipe (sopt.vla.<vla>.OPT)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--suite", default="libero90",
                   help="libero90 (7 E4b tasks), libero90_17 (E23: + 10 val-split tasks) or a standard suite")
    p.add_argument("--prior", default=DROID_PRIOR, help="SOPT checkpoint (config / normalization / droid weights)")
    p.add_argument("--sopt-data", default=os.environ.get("SOPT_DATA"))
    p.add_argument("--cond-drop", type=float, default=0.0, help="E19: P(drop the SOPT feature) per training sample")
    args = p.parse_args()
    V = vla_module(args.vla)
    assert args.arm in V.ARMS, (args.arm, V.ARMS)
    args.lr = args.lr or V.OPT["lr"]
    seed_everything(args.seed)
    out = REPO_ROOT / "outputs" / args.run
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    sopt_data = Path(args.sopt_data)
    if args.suite in ("libero90", "libero90_17"):
        raw = Raw20(sopt_data / "processed/libero90_raw20")
        tf = "configs/e4b_tasks.json" if args.suite == "libero90" else "configs/libero90_17_tasks.json"
        tasks = json.loads((REPO_ROOT / tf).read_text())
        sel = select_episodes(raw.eps[raw.eps["split"].isin(["test", "val"])], tasks, args.k or None)
        assert sel["task"].nunique() == len(tasks), "missing tasks"
        frames = sopt_data / "processed/libero90_frames256"
        build_frame_cache(sel, frames, workers=args.workers, size=256)
    else:  # standard benchmark suite: all its tasks, EE-only state (see sopt/data/libero_std.py)
        from sopt.data.libero_std import SUITES, StdRaw, build

        root = sopt_data / "processed/libero_std" / args.suite
        build(args.suite, root)
        raw = StdRaw(root)
        tasks = raw.tasks()
        sel = select_episodes(raw.eps, tasks, args.k or None)
        frames = root / "frames256"
        build_frame_cache(sel, frames, workers=args.workers, size=256, repo=SUITES[args.suite][0],
                          revision=SUITES[args.suite][1])
    chunk = getattr(V, "CHUNK", CHUNK)  # the VLA's native chunk length (X-VLA: 30)
    ds = PolicyDataset(raw, sel, frames, CTX, 32, act_chunk=chunk, shift=0)
    A = np.concatenate([raw.episode(r)[1] for r in sel.itertuples()])
    action_stats = stats_of(A)
    if args.vla == "smolvla":  # E5 / E7 / E13 (unchanged: mean / std only)
        action_stats, state_stats = {k: action_stats[k] for k in ("mean", "std")}, None
        policy, pre, _ = V.build_policy(make_prior(args.arm, args.prior), args.arm, action_stats, chunk=chunk)
    else:  # current-state quantiles over the training frames
        state_stats = stats_of(np.concatenate([raw.episode(r)[0] for r in sel.itertuples()]))
        kw = {"tasks": tasks} if getattr(V, "NEEDS_TASKS", False) else {}
        policy, pre, _ = V.build_policy(make_prior(args.arm, args.prior), args.arm, action_stats, chunk=chunk,
                                        state_stats=state_stats, **kw)
    V.set_cond_drop(policy, args.cond_drop)
    params = [q for q in policy.parameters() if q.requires_grad]
    groups = V.param_groups(policy, args.lr) if hasattr(V, "param_groups") else [{"params": params, "scale": 1.0}]
    assert sum(len(g["params"]) for g in groups) == len(params)
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=V.OPT.get("betas", (0.9, 0.95)), eps=1e-8,
                            weight_decay=V.OPT["weight_decay"])
    dl = DataLoader(ds, args.bs, shuffle=True, drop_last=True, num_workers=args.workers, persistent_workers=True,
                    pin_memory=True)
    meta = {"git": git_rev(), **vars(args), "chunk": chunk, "episodes": sel["episode_index"].tolist(), "samples": len(ds),
            "trainable_M": sum(q.numel() for q in params) / 1e6}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "episodes"}), flush=True)
    logger = JsonlLogger(out / "metrics.jsonl")
    step, t0 = 0, time.time()
    policy.train()
    while step < args.steps:
        for b in dl:
            batch = pre(to_lerobot(V, policy, b, tasks, device))
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.steps, args.lr) * g["scale"]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = policy.forward(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, V.OPT["clip"])
            opt.step()
            step += 1
            if step % 100 == 0:
                logger.log(step, loss=loss.item())
                print(f"step {step} loss {loss.item():.4f} ({100 / (time.time() - t0):.2f} it/s)", flush=True)
                t0 = time.time()
            if step >= args.steps:
                break
    V.save_trainable(policy, out / "final.pt", {"arm": args.arm, "tasks": tasks, "action_stats": action_stats,
                                                "state_stats": state_stats, "meta": meta})
    print("saved", out / "final.pt", flush=True)


if __name__ == "__main__":
    main()
