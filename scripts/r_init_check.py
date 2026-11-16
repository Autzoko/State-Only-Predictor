#!/usr/bin/env python
"""R2 implementation check: with identical host weights, a freshly added SOPT-structure arm (expert_kin, and
expert_droid as a reference) must reproduce the `none` policy exactly (zero-initialized injection), and the injection
must be live (perturbing its last layer changes the actions).

The host weights come from a trained `none` checkpoint and are loaded into every arm, so host initialization noise
(which differs between freshly built arms: they consume the RNG differently) plays no part. For expert_kin the
kinematic constants are fitted on the checkpoint's training demos, as vla_train does.

  python scripts/r_init_check.py --ckpt outputs/e21_dp_none_k5_t17_s0/final.pt --out outputs/r_init_dp.json
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from sopt.policy.data import PolicyDataset, Raw20, context_indices, interp_frames, select_episodes


def load_vt():
    spec = importlib.util.spec_from_file_location("vla_train", Path(__file__).with_name("vla_train.py"))
    vt = importlib.util.module_from_spec(spec)
    sys.modules["vla_train"] = vt
    spec.loader.exec_module(vt)
    return vt


def last_layer(policy) -> torch.nn.Linear:
    if hasattr(policy, "sopt"):  # Diffusion Policy (and the other global-conditioning hosts)
        return policy.sopt.mlp[-1]
    return policy.model.state_proj.expert_cond[-1]  # SmolVLA


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="a trained arm-`none` checkpoint (17-task LIBERO-90, k5)")
    p.add_argument("--arms", nargs="+", default=["expert_kin", "expert_droid"])
    p.add_argument("--out", required=True)
    p.add_argument("--sopt-data", default=os.environ.get("SOPT_DATA"))
    args = p.parse_args()
    vt = load_vt()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    meta, tasks = ck["meta"], ck["tasks"]
    assert ck["arm"] == "none" and meta["suite"] == "libero90_17" and meta["k"] == 5
    V = vt.vla_module(meta["vla"])
    sd = Path(args.sopt_data)
    raw = Raw20(sd / "processed/libero90_raw20")
    sel = select_episodes(raw.eps[raw.eps["split"].isin(["test", "val"])], tasks, 5)
    assert set(sel["episode_index"]) == set(meta["episodes"])
    ds = PolicyDataset(raw, sel, sd / "processed/libero90_frames256", meta.get("ctx", 96), 32, act_chunk=meta["chunk"], shift=0)
    idx = np.random.default_rng(0).choice(len(ds), 16, replace=False)
    b = {k: torch.stack([ds[int(i)][k] for i in idx]) for k in ("img", "ctx", "task")}
    img, ctx = b["img"].cuda().float() / 255.0, b["ctx"].cuda()
    names = [tasks[int(t)] for t in b["task"]]
    extra = {"state_stats": ck["state_stats"]} if ck.get("state_stats") is not None else {}
    if getattr(V, "NEEDS_TASKS", False):
        extra["tasks"] = ck.get("vocab") or tasks

    def build(arm):
        prior = vt.make_prior(arm, meta.get("prior", vt.DROID_PRIOR))
        policy, pre, post = V.build_policy(prior, arm, ck["action_stats"], chunk=meta["chunk"], **extra)
        # host weights of the trained `none` arm; its (unused, random) state_proj.prior of SmolVLA is not copied into
        # the SOPT arms, whose prior is their own (DROID weights / kinematic features)
        host = {k: v for k, v in ck["trainable"].items() if arm == "none" or ".prior." not in k}
        missing, unexpected = policy.load_state_dict(host, strict=False)
        trainable = {n for n, q in policy.named_parameters() if q.requires_grad}
        sopt_key = lambda k: "sopt." in k or "expert_cond" in k or ".prior." in k  # noqa: E731
        assert not unexpected, unexpected[:5]
        assert all(sopt_key(k) for k in missing if k in trainable), [k for k in missing if k in trainable][:5]
        if hasattr(prior, "fit"):
            prior.fit(torch.from_numpy(np.stack([interp_frames(ds.F[e], context_indices(a, prior.cfg.ctx_len))
                                                 for e, a in ds.items]).astype(np.float32)))
        policy.eval()
        return policy, pre, post, prior

    def act(policy, pre, post):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            batch = V.make_batch(policy, img, ctx, names)
            torch.manual_seed(7)
            a = policy.predict_action_chunk(pre(batch))
        return post(a.float()).cpu().numpy()

    ref = act(*build("none")[:3])
    res = {"ckpt": args.ckpt, "vla": meta["vla"]}
    for arm in args.arms:
        policy, pre, post, prior = build(arm)
        same = act(policy, pre, post)
        with torch.no_grad():
            feat = prior.encode(ctx)[:, -1].float()
        with torch.no_grad():  # liveness: a small random last layer must change the actions
            w = last_layer(policy).weight
            w.copy_(torch.randn_like(w) * 0.05)
        moved = act(policy, pre, post)
        res[arm] = {"max_abs_diff_vs_none_at_init": float(np.abs(same - ref).max()),
                    "max_abs_diff_after_perturbing_last_layer": float(np.abs(moved - ref).max()),
                    "feature_dim": int(feat.shape[-1]), "feature_std": float(feat.std())}
        print(arm, json.dumps(res[arm]), flush=True)
        del policy
        torch.cuda.empty_cache()
    res["pass"] = all(res[a]["max_abs_diff_vs_none_at_init"] == 0.0 and res[a]["max_abs_diff_after_perturbing_last_layer"] > 0
                      for a in args.arms)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("init check", "PASS" if res["pass"] else "FAIL", flush=True)


if __name__ == "__main__":
    main()
