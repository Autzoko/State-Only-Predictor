#!/usr/bin/env python
"""E4b training: inverse dynamics (`idm`), prior-based policy (`prior`), or direct BC baseline (`bc`).

  python scripts/e4b_train.py --kind idm   --run e4b_idm
  python scripts/e4b_train.py --kind prior --run e4b_prior_droid_k5 --k 5 --init outputs/e2_ar_flow_f100_s/best.pt
  python scripts/e4b_train.py --kind prior --run e4b_prior_scratch_k5 --k 5 --init scratch
  python scripts/e4b_train.py --kind bc    --run e4b_bc_k5 --k 5
Policies use the 7 unique-language LIBERO-90 test-split tasks (configs/e4b_tasks.json): never seen by E4a.
The IDM uses LIBERO-90 train-split tasks only (state + action, no images).
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from sopt.data.dataset import TrajectoryStore
from sopt.data.normalize import compute_stats
from sopt.models.state_prior import StatePrior
from sopt.policy.data import IDMDataset, PolicyDataset, Raw20, build_frame_cache, select_episodes
from sopt.policy.models import IDM, DirectBC, PriorPolicy, TrunkBC
from sopt.train.pretrain import load_checkpoint
from sopt.utils.config import REPO_ROOT, load_config
from sopt.utils.misc import JsonlLogger, git_rev, seed_everything


def lr_at(step, total, base, warmup=500):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))


def libero_state_stats(sopt_data: Path):
    store = TrajectoryStore(sopt_data / "processed/libero90")
    train = store.episodes[store.episodes["split"] == "train"]
    return compute_stats(np.asarray(store.features), train, horizon=32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["idm", "prior", "bc", "trunkbc"], required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--sopt-data", default=os.environ.get("SOPT_DATA"))
    p.add_argument("--k", type=int, default=0, help="demos per task (0 = all)")
    p.add_argument("--init", default="scratch", help="prior checkpoint or 'scratch'")
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    seed_everything(args.seed)
    sopt_data = Path(args.sopt_data)
    out = REPO_ROOT / "outputs" / args.run
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    raw = Raw20(sopt_data / "processed/libero90_raw20")
    tasks = json.loads((REPO_ROOT / "configs/e4b_tasks.json").read_text())
    meta = {"git": git_rev(), **vars(args)}

    if args.kind == "idm":
        train = raw.eps[raw.eps["split"] == "train"]
        ds = IDMDataset(raw, train, k=4)
        F = raw.features[np.concatenate([np.arange(s, s + n) for s, n in zip(train["start"], train["length"])])]
        d = np.concatenate([raw.features[s + 1 : s + n] - raw.features[s : s + n - 1]
                            for s, n in zip(train["start"], train["length"])])
        stats = {"s_mean": F.mean(0), "s_std": np.maximum(F.std(0), 1e-3), "d_std": np.maximum(d.std(0), 1e-4)}
        model = IDM(4, *(torch.tensor(stats[k], dtype=torch.float32) for k in ("s_mean", "s_std", "d_std")))
        extra = {"k": 4, "stats": stats}
        bs = 512
    else:
        sel = select_episodes(raw.eps[raw.eps["split"] == "test"], tasks, args.k or None)
        assert sel["task"].nunique() == len(tasks), "missing tasks"
        frames = sopt_data / "processed/libero90_frames"
        build_frame_cache(sel, frames, workers=args.workers)
        meta["episodes"] = sel["episode_index"].tolist()
        if args.kind in ("prior", "trunkbc"):
            if args.init == "scratch":
                mcfg = load_config([str(REPO_ROOT / "configs/model/s.yaml"), str(REPO_ROOT / "configs/experiment/e2.yaml")],
                                   ["model.objective=ar_flow"]).model
                prior = StatePrior(mcfg, libero_state_stats(sopt_data))
            else:
                prior, _ = load_checkpoint(args.init)
            model = PriorPolicy(prior, len(tasks)) if args.kind == "prior" else TrunkBC(prior, len(tasks), 16)
            extra = {"prior_config": OmegaConf.to_container(prior.cfg), "chunk": 16,
                     "prior_stats": {k: v.cpu().numpy() for k, v in prior.normalizer.state_dict().items()}}
            ctx_len, horizon = prior.cfg.ctx_len, prior.H
        else:
            st = libero_state_stats(sopt_data)
            model = DirectBC(len(tasks), 16, torch.tensor(st["mean"]), torch.tensor(st["std"]))
            extra = {"chunk": 16, "stats": {"mean": st["mean"], "std": st["std"]}}
            ctx_len, horizon = 96, 32
        ds = PolicyDataset(raw, sel, frames, ctx_len, horizon)
        bs = args.bs
    model = model.to(device)
    dl = DataLoader(ds, bs, shuffle=True, drop_last=True, num_workers=args.workers, persistent_workers=True,
                    pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    logger = JsonlLogger(out / "metrics.jsonl")
    meta.update(samples=len(ds), n_params=sum(p.numel() for p in model.parameters()))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "episodes"}), flush=True)

    step, t0 = 0, time.time()
    model.train()
    while step < args.steps:
        for batch in dl:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.steps, args.lr)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model.loss(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 100 == 0:
                logger.log(step, loss=loss.item())
                print(f"step {step} loss {loss.item():.4f} ({100 / (time.time() - t0):.1f} it/s)", flush=True)
                t0 = time.time()
            if step >= args.steps:
                break
    torch.save({"kind": args.kind, "model": model.state_dict(), "tasks": tasks, "meta": meta, **extra},
               out / "final.pt")
    if args.kind == "idm":  # held-out tasks: is the IDM accurate enough that it is not the bottleneck?
        model.eval()
        vds = IDMDataset(raw, raw.eps[raw.eps["split"] == "val"], k=4)
        vdl = DataLoader(vds, 4096, shuffle=False, num_workers=args.workers)
        l1, grip, n = 0.0, 0.0, 0
        for b in vdl:
            b = {k: v.to(device) for k, v in b.items()}
            a = model.act(b["cur"], b["nxt"])
            l1 += (a[:, :6] - b["act"][:, :6]).abs().mean(1).sum().item()
            grip += (a[:, 6] == b["act"][:, 6]).float().sum().item()
            n += len(a)
        res = {"val_l1_osc": l1 / n, "val_gripper_acc": grip / n,
               "val_l1_zero_action": float(np.mean([vds.A[e][t][:6].__abs__().mean() for e, t in vds.items[::50]]))}
        (out / "val_metrics.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res), flush=True)
    print("saved", out / "final.pt", flush=True)


if __name__ == "__main__":
    main()
