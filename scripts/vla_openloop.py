#!/usr/bin/env python
"""Open-loop action error of a fine-tuned VLA (LIBERO-90 E4b tasks) on its own training demos and on held-out demos
of the same tasks (demos 6-10 by the same salted order; never trained on in k5). Diagnoses train / inference
mismatch (high error on training demos) vs overfitting (low train, high held-out error). Error: MSE of the first
`--n-exec` actions in action-std units (dims 0-5), gripper sign-agreement rate.

  python scripts/vla_openloop.py --ckpt outputs/e14_pi05_none_k5_s0/final.pt
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from sopt.policy.data import PolicyDataset, Raw20, select_episodes
from sopt.utils.config import REPO_ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n", type=int, default=256, help="frames per split")
    p.add_argument("--n-exec", type=int, default=10)
    p.add_argument("--sopt-data", default=os.environ.get("SOPT_DATA"))
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location("vla_train", Path(__file__).with_name("vla_train.py"))
    vt = importlib.util.module_from_spec(spec)
    sys.modules["vla_train"] = vt
    spec.loader.exec_module(vt)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    assert ck["meta"].get("suite", "libero90") == "libero90" and ck["meta"]["k"] == 5
    V = vt.vla_module(ck["meta"].get("vla", "smolvla"))
    prior = vt.make_prior(ck["arm"], ck["meta"].get("prior", vt.DROID_PRIOR))
    extra = {"state_stats": ck["state_stats"]} if ck.get("state_stats") is not None else {}
    if getattr(V, "NEEDS_TASKS", False):
        extra["tasks"] = ck["tasks"]
    chunk = ck["meta"].get("chunk", vt.CHUNK)
    policy, pre, post = V.build_policy(prior, ck["arm"], ck["action_stats"], chunk=chunk, **extra)
    V.load_trainable(policy, args.ckpt)
    policy.eval()
    sd = Path(args.sopt_data)
    raw = Raw20(sd / "processed/libero90_raw20")
    tasks = ck["tasks"]
    test = raw.eps[raw.eps["split"] == "test"]
    k5, k10 = select_episodes(test, tasks, 5), select_episodes(test, tasks, 10)
    assert set(k5["episode_index"]) == set(ck["meta"]["episodes"])
    held = k10[~k10["episode_index"].isin(k5["episode_index"])]
    std = np.asarray(ck["action_stats"]["std"])[:6] if "std" in ck["action_stats"] else None
    res = {}
    for name, sel in [("train", k5), ("heldout", held)]:
        ds = PolicyDataset(raw, sel.reset_index(drop=True), sd / "processed/libero90_frames256", vt.CTX, 32,
                           act_chunk=chunk, shift=0)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)
        errs, grip = [], []
        for i in range(0, len(idx), 16):
            items = [ds[int(j)] for j in idx[i : i + 16]]
            b = {k: torch.stack([it[k] for it in items]) for k in items[0]}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                img = b["img"].cuda().float() / 255.0
                batch = V.make_batch(policy, img, b["ctx"].cuda(), [tasks[int(t)] for t in b["task"]])
                act = policy.predict_action_chunk(pre(batch))
            act = post(act.float()).cpu().numpy()[:, : args.n_exec]
            gt = b["act"].numpy()[:, : args.n_exec]
            s = std if std is not None else gt[..., :6].reshape(-1, 6).std(0) + 1e-3
            errs.append((((act[..., :6] - gt[..., :6]) / s) ** 2).mean((1, 2)))
            grip.append(((act[..., 6] > 0.5) == (gt[..., 6] > 0.5)).mean(1))
        res[name] = {"mse_std_units": float(np.concatenate(errs).mean()), "gripper_agree": float(np.concatenate(grip).mean()),
                     "frames": len(idx), "episodes": len(sel)}
        print(name, res[name], flush=True)
    print(json.dumps({"ckpt": args.ckpt, **res}))


if __name__ == "__main__":
    main()
