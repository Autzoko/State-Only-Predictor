#!/usr/bin/env python
"""E37 analysis: is SOPT's gain larger under visual distribution shift? (pre-registered in docs/SOPT_NEXT.md)

Per (host, seed, task, init) there are four outcomes: none / SOPT arm, clean (original rollouts_n50.json) / perturbed
(e37_main_<kind><level>.json). Interaction of a pair = (sopt_p - none_p) - (sopt_c - none_c).
Primary (fixed before any perturbed result): mean interaction pooled over the four visual kinds (camera, light,
texture, noise), the three hosts and seeds 0-1, with a 95% bootstrap CI (pairs resampled within host x seed x kind
strata); claim "SOPT gains more under visual shift" only if the CI is above 0. Secondary: per host / per kind
interactions, paired McNemar SOPT vs none under each perturbation, success retention (perturbed / clean) per arm.
"robot" (initial-joint offsets) is a proprioceptive shift, reported separately and never pooled into the primary.

  python scripts/e37_analyze.py --root outputs --out outputs/e37_analysis.json
"""

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

HOSTS = {"dp": ("e21_dp_none_k5_t17_s{}", "e21_dp_expert_droid_k5_t17_s{}"),
         "smolvla_goal": ("e13_libero_goal_none_k5_s{}", "e13_libero_goal_expert_droid_k5_s{}"),
         "groot": ("e22_groot_none_k5_t17_s{}", "e31_groot_xattn_droid_k5_t17_s{}")}
VISUAL = ("camera", "light", "texture", "noise")


def mcnemar(b: int, c: int) -> float:
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def outcomes(path: Path) -> dict:
    return {(e["task"], e["init"]): bool(e["success"]) for e in json.load(open(path))["episodes"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs")
    p.add_argument("--levels", default=None, help="default: <root>/e37_levels.json")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--boot", type=int, default=5000)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    root = Path(args.root)
    levels = json.load(open(args.levels or root / "e37_levels.json"))["levels"]
    rows = []  # (host, seed, kind, none_c, sopt_c, none_p, sopt_p)
    missing = []
    for h, (rn, rs) in HOSTS.items():
        for s in args.seeds:
            nc, sc = outcomes(root / rn.format(s) / "rollouts_n50.json"), outcomes(root / rs.format(s) / "rollouts_n50.json")
            for k, l in levels.items():
                fn, fs = root / rn.format(s) / f"e37_main_{k}{l}.json", root / rs.format(s) / f"e37_main_{k}{l}.json"
                if not (fn.exists() and fs.exists()):
                    missing.append(f"{h} s{s} {k}{l}")
                    continue
                npert, spert = outcomes(fn), outcomes(fs)
                keys = sorted(nc.keys() & sc.keys() & npert.keys() & spert.keys())
                assert len(keys) == len(nc) == len(npert), (h, s, k, len(keys), len(nc), len(npert))
                rows += [(h, s, k, nc[x], sc[x], npert[x], spert[x]) for x in keys]
    if missing:
        print("MISSING", missing)
    R = np.array([r[3:] for r in rows], dtype=float)
    H = np.array([r[0] for r in rows])
    S = np.array([r[1] for r in rows])
    K = np.array([r[2] for r in rows])
    inter = (R[:, 3] - R[:, 2]) - (R[:, 1] - R[:, 0])
    rng = np.random.default_rng(0)

    def boot_mean(mask) -> list:
        idx = np.flatnonzero(mask)
        strata = [idx[(H[idx] == h) & (S[idx] == s) & (K[idx] == k)] for h in HOSTS for s in args.seeds
                  for k in levels]
        strata = [g for g in strata if len(g)]
        bs = [np.mean(inter[np.concatenate([g[rng.integers(0, len(g), len(g))] for g in strata])])
              for _ in range(args.boot)] if strata else []
        return [float(inter[idx].mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))] if bs else None

    out = {"levels": levels, "seeds": args.seeds, "n_pairs": len(rows), "missing": missing, "cells": {}}
    vis = np.isin(K, VISUAL)
    out["primary"] = {"interaction_mean_ci": boot_mean(vis)}
    out["primary"]["pass"] = bool(out["primary"]["interaction_mean_ci"] and out["primary"]["interaction_mean_ci"][1] > 0)
    for h in HOSTS:
        out.setdefault("per_host_visual", {})[h] = boot_mean(vis & (H == h))
    for k in levels:
        out.setdefault("per_kind", {})[k] = boot_mean(K == k)
    for h in HOSTS:
        for k in levels:
            m = (H == h) & (K == k)
            if not m.any():
                continue
            r = R[m]
            b, c = int(((r[:, 3] == 1) & (r[:, 2] == 0)).sum()), int(((r[:, 3] == 0) & (r[:, 2] == 1)).sum())
            bc, cc = int(((r[:, 1] == 1) & (r[:, 0] == 0)).sum()), int(((r[:, 1] == 0) & (r[:, 0] == 1)).sum())
            out["cells"][f"{h}/{k}{levels[k]}"] = {
                "n": int(m.sum()), "none_clean": float(r[:, 0].mean()), "sopt_clean": float(r[:, 1].mean()),
                "none_pert": float(r[:, 2].mean()), "sopt_pert": float(r[:, 3].mean()),
                "pert_sopt_vs_none": [b, c, mcnemar(b, c)], "clean_sopt_vs_none": [bc, cc, mcnemar(bc, cc)],
                "retention_none": float(r[:, 2].mean() / max(r[:, 0].mean(), 1e-9)),
                "retention_sopt": float(r[:, 3].mean() / max(r[:, 1].mean(), 1e-9))}
            x = out["cells"][f"{h}/{k}{levels[k]}"]
            print(f"{h:13s} {k:8s} L{levels[k]} n {x['n']:4d} | clean none {100*x['none_clean']:5.1f} sopt "
                  f"{100*x['sopt_clean']:5.1f} | pert none {100*x['none_pert']:5.1f} sopt {100*x['sopt_pert']:5.1f} "
                  f"{b}:{c} p={x['pert_sopt_vs_none'][2]:.3g} | retention {x['retention_none']:.2f}/{x['retention_sopt']:.2f}")
    print("PRIMARY", json.dumps(out["primary"]))
    print("PER HOST (visual)", json.dumps(out["per_host_visual"]))
    print("PER KIND", json.dumps(out["per_kind"]))
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
