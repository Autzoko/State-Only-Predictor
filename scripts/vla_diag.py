#!/usr/bin/env python
"""E20d: why SOPT hurts X-VLA. Implementation checks + counterfactual open-loop probes of fine-tuned host checkpoints
(any sopt.vla host; run X-VLA next to DP / GR00T as contrasts), 17-task LIBERO-90 protocol.

Per checkpoint, on its own k5 training demos and on held-out demos of the same tasks (demos 6-10 by the same salted
order; never trained on), open-loop error of the first n_exec actions (MSE in action-std units, dims 0-5; gripper
agreement), under paired noise (same torch seed per batch) for:
  normal        as trained
  img_swap      images taken from another frame of the batch (vision reliance)
  sopt_zero     SOPT feature zeroed (expert_* arms; host's own current-state input untouched)
  sopt_swap     SOPT feature computed from another frame's history (history reliance)
Implementation checks (expert_* arms): the hook is live (zeroing changes the actions), |SOPT feature| / |action token|
at the injection point, and the drift of the fine-tuned SOPT encoder from its initialization (DROID weights, or the
seed-reproduced random init for scratch). With --init-check, a freshly built expert_droid policy must equal `none`
exactly (zero-init MLP).

  python scripts/vla_diag.py --out outputs/e20d_diag.json --ckpts outputs/e20_xvla_*_k5_t17_s0/final.pt
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from sopt.policy.data import PolicyDataset, Raw20, build_frame_cache, select_episodes
from sopt.utils.config import REPO_ROOT
from sopt.utils.misc import seed_everything


def load_vt():
    spec = importlib.util.spec_from_file_location("vla_train", Path(__file__).with_name("vla_train.py"))
    vt = importlib.util.module_from_spec(spec)
    sys.modules["vla_train"] = vt
    spec.loader.exec_module(vt)
    return vt


def build(vt, ck, ckpt=None):
    V = vt.vla_module(ck["meta"]["vla"])
    seed_everything(ck["meta"]["seed"])  # vla_train order: seed, then make_prior -> same scratch init
    prior = vt.make_prior(ck["arm"], ck["meta"].get("prior", vt.DROID_PRIOR))
    init = {k: v.detach().clone() for k, v in prior.state_dict().items()}
    extra = {"state_stats": ck["state_stats"]} if ck.get("state_stats") is not None else {}
    if getattr(V, "NEEDS_TASKS", False):
        extra["tasks"] = ck.get("vocab") or ck["tasks"]  # one-hot vocabulary (E40); evaluated tasks stay ck["tasks"]
    policy, pre, post = V.build_policy(prior, ck["arm"], ck["action_stats"], chunk=ck["meta"]["chunk"], **extra)
    if ckpt is not None:
        V.load_trainable(policy, ckpt)
    policy.eval()
    return V, policy, pre, post, init


def batches(ds, n, bs=16):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    for i in range(0, len(idx), bs):
        items = [ds[int(j)] for j in idx[i : i + bs]]
        yield {k: torch.stack([it[k] for it in items]) for k in items[0]}


def predict(V, policy, pre, post, b, tasks, mode, seed):
    """mode: normal / img_swap / sopt_zero / sopt_swap. Returns (B, chunk, 7) actions and the SOPT feature."""
    img = b["img"].cuda().float() / 255.0
    ctx = b["ctx"].cuda()
    if mode == "img_swap":
        img = img.roll(1, 0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        batch = V.make_batch(policy, img, ctx, [tasks[int(t)] for t in b["task"]])
        if hasattr(policy, "sopt"):
            if mode == "sopt_zero":
                policy.sopt.cond = torch.zeros_like(policy.sopt.cond)
            elif mode == "sopt_swap":
                policy.sopt.set(ctx.roll(1, 0))
        cond = getattr(getattr(policy, "sopt", None), "cond", None)  # (E31 xattn arms keep a token memory instead)
        cond = cond.float() if cond is not None else None
        torch.manual_seed(seed)
        act = policy.predict_action_chunk(pre(batch))
    return post(act.float()).cpu().numpy(), cond


def token_norm_hook(policy, vla, store):
    """|action token| at the SOPT injection point (output of the hooked module minus the SOPT feature)."""
    if vla == "xvla":
        mod = policy.model.transformer.action_encoder
    elif vla == "groot":
        mod = next(m for n, m in policy.named_modules() if n.endswith("action_encoder"))
    else:
        return None

    def hook(_m, _i, out):
        y = out[0] if isinstance(out, tuple) else out
        base = y.float() - policy.sopt.cond[:, None].float()
        store.append((policy.sopt.cond.float().norm(dim=-1).mean().item(), base.norm(dim=-1).mean().item()))

    return mod.register_forward_hook(hook)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=256, help="frames per split")
    p.add_argument("--n-exec", type=int, default=10)
    p.add_argument("--init-check", action="store_true")
    p.add_argument("--init-arms", nargs="*", default=["expert_droid"], help="arms whose fresh policy must equal none")
    p.add_argument("--sopt-data", default=os.environ.get("SOPT_DATA"))
    args = p.parse_args()
    vt = load_vt()
    sd = Path(args.sopt_data)
    raw = Raw20(sd / "processed/libero90_raw20")
    frames = sd / "processed/libero90_frames256"
    results = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    for ckpt in args.ckpts:
        if ckpt in results:
            continue
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        meta, tasks = ck["meta"], ck["tasks"]
        assert meta["suite"] == "libero90_17" and meta["k"] == 5, (meta["suite"], meta["k"])
        pool = raw.eps[raw.eps["split"].isin(["test", "val"])]
        k5, k10 = select_episodes(pool, tasks, 5), select_episodes(pool, tasks, 10)
        assert set(k5["episode_index"]) == set(meta["episodes"]), "training demos differ from the checkpoint's"
        held = k10[~k10["episode_index"].isin(k5["episode_index"])].reset_index(drop=True)
        try:
            build_frame_cache(held, frames, workers=8, size=256)
        except Exception as e:  # offline node without the videos: keep the held-out demos already decoded
            print("frame cache:", repr(e)[:200], flush=True)
        held = held[[(frames / f"ep{int(e):06d}.npy").exists() for e in held["episode_index"]]].reset_index(drop=True)
        V, policy, pre, post, init = build(vt, ck, ckpt)
        r = {"vla": meta["vla"], "arm": ck["arm"], "seed": meta["seed"], "held_episodes": len(held)}
        std = np.asarray(ck["action_stats"]["std"])[:6]
        has = hasattr(policy, "sopt")
        if has:  # SOPT encoder drift from its initialization (relative L2 over all prior weights)
            cur = policy.sopt.prior.state_dict()
            num = sum(((cur[k].float().cpu() - v.float()) ** 2).sum() for k, v in init.items() if v.is_floating_point())
            den = sum((v.float() ** 2).sum() for v in init.values() if v.is_floating_point())
            r["prior_drift_rel"] = float((num / den).sqrt())
        norms = []
        h = token_norm_hook(policy, meta["vla"], norms) if has else None
        modes = ["normal", "img_swap"] + (["sopt_zero", "sopt_swap"] if has else [])
        for name, sel in [("train", k5), ("heldout", held)]:
            if not len(sel):
                continue
            ds = PolicyDataset(raw, sel, frames, vt.CTX, 32, act_chunk=meta["chunk"], shift=0)
            acc = {m: {"mse": [], "grip": [], "dev_from_normal": []} for m in modes}
            for bi, b in enumerate(batches(ds, args.n)):
                gt = b["act"].numpy()[:, : args.n_exec]
                ref = None
                for m in modes:
                    a, _ = predict(V, policy, pre, post, b, tasks, m, seed=1000 + bi)
                    a = a[:, : args.n_exec]
                    acc[m]["mse"].append((((a[..., :6] - gt[..., :6]) / std) ** 2).mean((1, 2)))
                    acc[m]["grip"].append(((a[..., 6] > 0.5) == (gt[..., 6] > 0.5)).mean(1))
                    ref = a if m == "normal" else ref
                    acc[m]["dev_from_normal"].append((((a[..., :6] - ref[..., :6]) / std) ** 2).mean((1, 2)))
            r[name] = {m: {k: float(np.concatenate(v).mean()) for k, v in d.items()} for m, d in acc.items()}
            print(ckpt, name, json.dumps(r[name]), flush=True)
        if h is not None:
            h.remove()
            c, t = np.array(norms).mean(0)
            r["cond_norm"], r["token_norm"], r["cond_over_token"] = float(c), float(t), float(c / t)
        results[ckpt] = r
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(ckpt, json.dumps({k: v for k, v in r.items() if k not in ("train", "heldout")}), flush=True)
        del policy
        torch.cuda.empty_cache()
    if args.init_check:  # zero-init: a fresh expert_droid policy must reproduce `none` exactly
        ck = torch.load(args.ckpts[0], map_location="cpu", weights_only=False)
        out = {}
        for arm in ("none", *args.init_arms):
            ck2 = {**ck, "arm": arm}
            V, policy, pre, post, _ = build(vt, ck2, None)
            pool = raw.eps[raw.eps["split"].isin(["test", "val"])]
            ds = PolicyDataset(raw, select_episodes(pool, ck["tasks"], 5), frames, vt.CTX, 32,
                               act_chunk=ck["meta"]["chunk"], shift=0)
            b = next(batches(ds, 16))
            out[arm], _ = predict(V, policy, pre, post, b, ck["tasks"], "normal", seed=7)
            del policy
            torch.cuda.empty_cache()
        diff = {a: float(np.abs(out["none"] - out[a]).max()) for a in args.init_arms}
        results["init_check"] = {"vla": ck["meta"]["vla"], "max_abs_diff_none_vs_fresh": diff}
        Path(args.out).write_text(json.dumps(results, indent=2))
        print("init check", results["init_check"], flush=True)


if __name__ == "__main__":
    main()
