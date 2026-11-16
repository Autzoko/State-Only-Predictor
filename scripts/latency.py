#!/usr/bin/env python
"""Inference cost of SOPT (reviewer request, 2026-11-16): wall time of one action-chunk prediction of fine-tuned host
checkpoints with and without SOPT, and of the SOPT encoder alone, batch 1 on one GPU, bf16 autocast as in vla_eval.

Inputs are fixed dummy tensors (images 0.5 gray, a resting 96-frame history taken from the first frame of a LIBERO
demo), so only compute is measured, not the simulator. Per checkpoint: `warmup` untimed calls, then `n` timed calls
with CUDA synchronization; reported median / p90 in ms, plus peak GPU memory.

  python scripts/latency.py --ckpts outputs/e21_dp_none_k5_t17_s0/final.pt ... --out outputs/latency.json
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def load_vt():
    spec = importlib.util.spec_from_file_location("vla_train", Path(__file__).with_name("vla_train.py"))
    vt = importlib.util.module_from_spec(spec)
    sys.modules["vla_train"] = vt
    spec.loader.exec_module(vt)
    return vt


def timed(fn, warmup: int, n: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(1000 * (time.perf_counter() - t0))
    return {"median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90)), "n": n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--n", type=int, default=200)
    args = p.parse_args()
    vt = load_vt()
    dev = torch.device("cuda")
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    for ckpt in args.ckpts:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        V = vt.vla_module(ck["meta"]["vla"])
        prior = vt.make_prior(ck["arm"], ck["meta"].get("prior", vt.DROID_PRIOR))
        extra = {"state_stats": ck["state_stats"]} if ck.get("state_stats") is not None else {}
        if getattr(V, "NEEDS_TASKS", False):
            extra["tasks"] = ck.get("vocab") or ck["tasks"]
        policy, pre, post = V.build_policy(prior, ck["arm"], ck["action_stats"], chunk=ck["meta"]["chunk"], **extra)
        V.load_trainable(policy, ckpt)
        policy.eval()
        C = ck["meta"].get("ctx", 96)
        x0 = torch.as_tensor(prior.normalizer.mean, dtype=torch.float32, device=dev)  # a plausible resting state
        ctx = x0.expand(1, C, -1).contiguous()
        img = torch.full((1, 2, 3, 256, 256), 0.5, device=dev)
        task = [ck["tasks"][0]]
        torch.cuda.reset_peak_memory_stats()

        def chunk():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                policy.predict_action_chunk(pre(V.make_batch(policy, img, ctx, task)))

        r = {"vla": ck["meta"]["vla"], "arm": ck["arm"], "chunk_call": timed(chunk, args.warmup, args.n)}
        if ck["arm"] != "none":  # the SOPT encoder alone (history -> last-token feature)
            def enc():
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    prior.encode(ctx)
            r["sopt_encode"] = timed(enc, args.warmup, args.n)
            r["sopt_params_M"] = sum(q.numel() for q in prior.parameters()) / 1e6
        r["peak_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9
        res[ckpt] = r
        print(ckpt, json.dumps(r), flush=True)
        del policy
        torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
