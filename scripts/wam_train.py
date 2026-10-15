#!/usr/bin/env python
"""E12: fine-tune FastWAM (world action model) on the E4b LIBERO-90 tasks, with or without SOPT (see sopt/vla/fastwam.py).

  python scripts/wam_train.py --arm expert_droid --k 5 --seed 0 --run e12_wam_expert_droid_k5_s0
Same demos (k5 salted hash), 15 Hz SOPT context, action stats and schedule for all arms.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sopt.policy.data import PolicyDataset, Raw20, build_frame_cache, select_episodes
from sopt.utils.config import REPO_ROOT
from sopt.utils.misc import JsonlLogger, git_rev, seed_everything
from sopt.vla.fastwam import ARMS, PromptCache, action_loss, build_fastwam, trainable_state

CHUNK, CTX = 32, 96


def lr_at(step, total, peak, warmup=500, final=2.5e-6):
    if step < warmup:
        return peak * (step + 1) / warmup
    p = min((step - warmup) / max(1, total - warmup), 1.0)
    return final + (peak - final) * 0.5 * (1 + math.cos(math.pi * p))


def make_prior(arm: str):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("vla_train", Path(__file__).with_name("vla_train.py"))
    vt = importlib.util.module_from_spec(spec)
    sys.modules["vla_train"] = vt
    spec.loader.exec_module(vt)
    return vt.make_prior("expert_droid" if arm == "expert_droid" else "scratch")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--k", type=int, default=5, help="demos per task (0 = all)")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sopt-data", default=os.environ.get("SOPT_DATA"))
    p.add_argument("--suite", default="libero90", choices=["libero90", "libero90_17"], help="7 E4b tasks / E23 17 tasks")
    args = p.parse_args()
    seed_everything(args.seed)
    out = REPO_ROOT / "outputs" / args.run
    out.mkdir(parents=True, exist_ok=True)
    sopt_data = Path(args.sopt_data)
    raw = Raw20(sopt_data / "processed/libero90_raw20")
    tf = "configs/e4b_tasks.json" if args.suite == "libero90" else "configs/libero90_17_tasks.json"
    tasks = json.loads((REPO_ROOT / tf).read_text())
    sel = select_episodes(raw.eps[raw.eps["split"].isin(["test", "val"])], tasks, args.k or None)
    assert sel["task"].nunique() == len(tasks), "missing tasks"
    frames = sopt_data / "processed/libero90_frames256"
    build_frame_cache(sel, frames, workers=args.workers, size=256)
    ds = PolicyDataset(raw, sel, frames, CTX, 32, act_chunk=CHUNK, shift=0)
    A = np.concatenate([raw.episode(r)[1] for r in sel.itertuples()])
    action_stats = {"mean": A.mean(0), "std": np.maximum(A.std(0), 1e-3)}

    policy, sopt = build_fastwam(make_prior(args.arm), args.arm, chunk=CHUNK)
    model = policy.model
    dev = model.device
    a_mean, a_std = (torch.tensor(action_stats[k], dtype=torch.float32, device=dev) for k in ("mean", "std"))
    prompts = PromptCache(model, policy.config.prompt_template)
    params = [q for q in list(model.parameters()) + list(sopt.parameters()) if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-2)
    dl = DataLoader(ds, args.bs, shuffle=True, drop_last=True, num_workers=args.workers, persistent_workers=True,
                    pin_memory=True)
    meta = {"git": git_rev(), **vars(args), "episodes": sel["episode_index"].tolist(), "samples": len(ds),
            "trainable_M": sum(q.numel() for q in params) / 1e6, "chunk": CHUNK}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "episodes"}), flush=True)
    logger = JsonlLogger(out / "metrics.jsonl")
    step, t0 = 0, time.time()
    model.action_expert.train()
    sopt.train()
    while step < args.steps:
        for b in dl:
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.steps, args.lr)
            loss = action_loss(model, sopt, prompts, b, tasks, a_mean, a_std)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            if step % 50 == 0:
                logger.log(step, loss=loss.item())
                print(f"step {step} loss {loss.item():.4f} ({50 / (time.time() - t0):.2f} it/s, "
                      f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB)", flush=True)
                t0 = time.time()
            if step >= args.steps:
                break
    torch.save({"arm": args.arm, "tasks": tasks, "action_stats": action_stats, "meta": meta,
                "state": trainable_state(model, sopt)}, out / "final.pt")
    print("saved", out / "final.pt", flush=True)


if __name__ == "__main__":
    main()
