#!/usr/bin/env python
"""E27: what does a host already know that SOPT would add? Linear probes on the hosts' own action-decoder input.

E24 probed the SOPT feature; here the same temporal targets (scripts/e24_probe.py: vel, acc, grip_vel, t_grip,
t_still, prog, fut05, fut10, t_next, task) are probed from the representation each fine-tuned host hands to its
action decoder, i.e. after vision, language, current state (and, for expert_* arms, SOPT) have been fused:
  xvla / groot   input of `action_decoder` (action-token hidden states), first denoising call
  pi05           input of `action_out_proj` (action-expert hidden states), first denoising call
  diffusion      the global conditioning vector (state + image keypoints [+ SOPT]) fed to the UNet
pooled as [first action token, mean over the chunk's action tokens]; the first denoising call starts from the same
seeded noise for every run (paired). Feature sets per host run: host:<run>, and for `none` runs host:<run>+sopt (the
host representation concatenated with the frozen DROID SOPT feature): its R^2 gain over host:<run> is the information
SOPT adds that the host does not already have, i.e. the redundancy test.

Windows: every 8th frame of the 17-task k5 training demos (probe fit, ridge alpha by grouped 5-fold CV) and of the
held-out demos 6-10 whose frames are cached (probe score; never seen by any host or probe fit). The held-out set
covers the 7 dev tasks (new-task held-out frames are not decoded on the offline nodes).

  python scripts/e27_host_probe.py --out outputs/e27_host_probe.json --ckpts outputs/e2*_k5_t17_s0/final.pt
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
from sopt.train.pretrain import load_checkpoint
from sopt.utils.config import REPO_ROOT


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def install_capture(policy, vla, chunk, store):
    """Record the decoder-input representation of the first call per forward (B, 2H) into `store`."""
    def pool(x):
        x = x[:, -chunk:].float()
        return torch.cat([x[:, 0], x.mean(1)], -1)

    if vla == "diffusion":
        d = policy.diffusion
        prepare = d._prepare_global_conditioning

        def wrapped(batch):
            g = prepare(batch)
            if not store:
                store.append(g.float().reshape(g.shape[0], -1).cpu())
            return g

        d._prepare_global_conditioning = wrapped
        return
    end = {"xvla": "action_decoder", "groot": "action_decoder", "pi05": "action_out_proj"}[vla]
    mod = [(n, m) for n, m in policy.named_modules() if n.endswith(end)]
    assert mod, (vla, end)
    print("capture at", [n for n, _ in mod][:3], flush=True)  # first match (one per host in LeRobot 0.6.2)
    mod = [mod[0][1]]

    def hook(_m, inp):
        if not store:
            store.append(pool(inp[0]).cpu())

    mod[0].register_forward_pre_hook(hook)


def windows(ds, stride):
    return [i for i, (e, a) in enumerate(ds.items) if a % stride == 0 and a < len(ds.F[e]) - 20]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--prior", default="outputs/e2_ar_flow_f100_s/best.pt")
    p.add_argument("--sopt-data", default=os.environ.get("SOPT_DATA"))
    args = p.parse_args()
    vt, diag, e24 = load("vla_train"), load("vla_diag"), load("e24_probe")
    sd = Path(args.sopt_data)
    raw = Raw20(sd / "processed/libero90_raw20")
    frames = sd / "processed/libero90_frames256"
    tasks = json.loads((REPO_ROOT / "configs/libero90_17_tasks.json").read_text())
    pool_eps = raw.eps[raw.eps["split"].isin(["test", "val"])]
    k5, k10 = select_episodes(pool_eps, tasks, 5), select_episodes(pool_eps, tasks, 10)
    held = k10[~k10["episode_index"].isin(k5["episode_index"])]
    held = held[[(frames / f"ep{int(e):06d}.npy").exists() for e in held["episode_index"]]]
    sets = {}
    for name, sel in (("train", k5), ("heldout", held)):
        ds = PolicyDataset(raw, sel.reset_index(drop=True), frames, vt.CTX, 32, act_chunk=32, shift=0)
        idx = windows(ds, args.stride)
        Y = {k: [] for k in e24.REG}
        for i in idx:
            e, a = ds.items[i]
            for k, v in e24.targets(ds.F[e], a).items():
                Y[k].append(v)
        sets[name] = {"ds": ds, "idx": idx, "Y": {k: np.stack(v) for k, v in Y.items()},
                      "ep": np.array([int(sel.iloc[ds.items[i][0]].episode_index) for i in idx]),
                      "task": np.array([ds.task[ds.items[i][0]] for i in idx])}
        print(f"{name}: {len(sel)} episodes, {len(idx)} windows", flush=True)
    train = np.r_[np.ones(len(sets["train"]["idx"]), bool), np.zeros(len(sets["heldout"]["idx"]), bool)]
    Y = {k: np.concatenate([sets[s]["Y"][k] for s in ("train", "heldout")]).astype(np.float64) for k in e24.REG}
    ep = np.concatenate([sets[s]["ep"] for s in ("train", "heldout")])
    task = np.concatenate([sets[s]["task"] for s in ("train", "heldout")])
    ctx_all = torch.stack([sets[s]["ds"][i]["ctx"] for s in ("train", "heldout") for i in sets[s]["idx"]])

    ref, _ = load_checkpoint(REPO_ROOT / args.prior)
    feats = {"cur": ref.normalizer.norm(ctx_all[:, -1]).numpy().astype(np.float64),
             "sopt": e24.encode(ref, ctx_all.numpy())}
    for ckpt in args.ckpts:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        meta = ck["meta"]
        assert meta["suite"] == "libero90_17" and set(k5["episode_index"]) == set(meta["episodes"]), ckpt
        V, policy, pre, post, _ = diag.build(vt, ck, ckpt)
        store, out = [], []
        install_capture(policy, meta["vla"], meta["chunk"], store)
        bi = 0
        for s in ("train", "heldout"):
            ds, idx = sets[s]["ds"], sets[s]["idx"]
            for j in range(0, len(idx), args.bs):
                items = [ds[i] for i in idx[j : j + args.bs]]
                img = torch.stack([it["img"] for it in items]).cuda().float() / 255.0
                ctx = torch.stack([it["ctx"] for it in items]).cuda()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    batch = V.make_batch(policy, img, ctx, [tasks[int(it["task"])] for it in items])
                    store.clear()
                    torch.manual_seed(1000 + bi)
                    policy.predict_action_chunk(pre(batch))
                out.append(store[0])
                bi += 1
        run = Path(ckpt).parent.name
        feats[f"host:{run}"] = torch.cat(out).numpy().astype(np.float64)
        if ck["arm"] == "none":
            feats[f"host:{run}+sopt"] = np.concatenate([feats[f"host:{run}"], feats["sopt"]], 1)
        print(f"{run}: feature dim {feats[f'host:{run}'].shape[1]}", flush=True)
        del policy
        torch.cuda.empty_cache()

    alphas = np.logspace(-2, 5, 8)
    res = {"windows_train": int(train.sum()), "windows_heldout": int((~train).sum()), "probes": {}}
    for name, X in feats.items():
        r = {k: e24.probe(X, Y[k], train, ep, alphas * len(X) / 1e4) for k in e24.REG}
        r["task"] = e24.probe(X, task, train, ep, alphas * len(X) / 1e4, classify=True)
        res["probes"][name] = r
        print(f"{name:44s} " + " ".join(f"{k}={r[k]['r2']:.3f}" for k in e24.REG) + f" task={r['task']['acc']:.3f}",
              flush=True)
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
