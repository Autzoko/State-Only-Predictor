#!/usr/bin/env python
"""E5: fine-tune SmolVLA on the E4b LIBERO-90 tasks, with or without SOPT motion tokens.

  python scripts/vla_train.py --arm droid   --k 5 --run e5_vla_droid_k5
  python scripts/vla_train.py --arm scratch --k 5 --run e5_vla_scratch_k5
  python scripts/vla_train.py --arm none    --k 5 --run e5_vla_none_k5
Same data / demos / schedule for all arms; SmolVLA defaults (frozen VLM, trainable action expert) otherwise.
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

from sopt.models.state_prior import StatePrior
from sopt.policy.data import PolicyDataset, Raw20, build_frame_cache, select_episodes
from sopt.train.pretrain import load_checkpoint
from sopt.utils.config import REPO_ROOT
from sopt.utils.misc import JsonlLogger, git_rev, seed_everything
from sopt.vla.smolvla import IMAGE_KEYS, build_policy, save_trainable

DROID_PRIOR = "outputs/e2_ar_flow_f100_s/best.pt"
CHUNK, CTX = 50, 96


def to_lerobot(b: dict, tasks: list[str], device) -> dict:
    img = b["img"].to(device, non_blocking=True).float() / 255.0  # (B, 2, 3, H, W)
    return {IMAGE_KEYS[0]: img[:, 0], IMAGE_KEYS[1]: img[:, 1],
            "observation.state": b["ctx"].to(device, non_blocking=True),
            "action": b["act"].to(device, non_blocking=True), "action_is_pad": b["act_pad"].to(device),
            "task": [tasks[int(i)] for i in b["task"]]}


def lr_at(step, total, peak, warmup=1000, final=2.5e-6):
    if step < warmup:
        return peak * (step + 1) / warmup
    p = min((step - warmup) / max(1, total - warmup), 1.0)
    return final + (peak - final) * 0.5 * (1 + math.cos(math.pi * p))


def make_prior(arm: str) -> StatePrior:
    ref, _ = load_checkpoint(REPO_ROOT / DROID_PRIOR)  # same config + normalization for every arm
    if arm == "droid":
        return ref
    return StatePrior(ref.cfg, {k: v.numpy() for k, v in ref.normalizer.state_dict().items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["none", "scratch", "droid"], required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--k", type=int, default=5, help="demos per task (0 = all)")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sopt-data", default=os.environ.get("SOPT_DATA"))
    args = p.parse_args()
    seed_everything(args.seed)
    out = REPO_ROOT / "outputs" / args.run
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    sopt_data = Path(args.sopt_data)
    raw = Raw20(sopt_data / "processed/libero90_raw20")
    tasks = json.loads((REPO_ROOT / "configs/e4b_tasks.json").read_text())
    sel = select_episodes(raw.eps[raw.eps["split"] == "test"], tasks, args.k or None)
    frames = sopt_data / "processed/libero90_frames256"
    build_frame_cache(sel, frames, workers=args.workers, size=256)
    ds = PolicyDataset(raw, sel, frames, CTX, 32, act_chunk=CHUNK, shift=0)
    A = np.concatenate([raw.episode(r)[1] for r in sel.itertuples()])
    action_stats = {"mean": A.mean(0), "std": np.maximum(A.std(0), 1e-3)}

    policy, pre, _ = build_policy(make_prior(args.arm), args.arm, action_stats, chunk=CHUNK)
    params = [q for q in policy.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10)
    dl = DataLoader(ds, args.bs, shuffle=True, drop_last=True, num_workers=args.workers, persistent_workers=True,
                    pin_memory=True)
    meta = {"git": git_rev(), **vars(args), "episodes": sel["episode_index"].tolist(), "samples": len(ds),
            "trainable_M": sum(q.numel() for q in params) / 1e6, "prior": DROID_PRIOR}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "episodes"}), flush=True)
    logger = JsonlLogger(out / "metrics.jsonl")
    step, t0 = 0, time.time()
    policy.train()
    while step < args.steps:
        for b in dl:
            batch = pre(to_lerobot(b, tasks, device))
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.steps, args.lr)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = policy.forward(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            step += 1
            if step % 100 == 0:
                logger.log(step, loss=loss.item())
                print(f"step {step} loss {loss.item():.4f} ({100 / (time.time() - t0):.2f} it/s)", flush=True)
                t0 = time.time()
            if step >= args.steps:
                break
    save_trainable(policy, out / "final.pt", {"arm": args.arm, "tasks": tasks, "action_stats": action_stats,
                                              "meta": meta})
    print("saved", out / "final.pt", flush=True)


if __name__ == "__main__":
    main()
