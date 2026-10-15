#!/usr/bin/env python
"""E18 aggregation: TrunkBC k5, pretraining data source: LIBERO-90 train split only / DROID + LIBERO / DROID only, pooled McNemar over seeds."""
import json
from math import comb

ARMS = {
    "bc": ["e4b_bc_k5", "e4b_bc_k5_s1", "e4b_bc_k5_s2"],
    "scratch": ["e4b_trunkbc_scratch_k5", "e4b_trunkbc_scratch_k5_s1", "e4b_trunkbc_scratch_k5_s2"],
    "joint_prior": ["e4b_trunkbc_droid_k5", "e4b_trunkbc_droid_k5_s1", "e4b_trunkbc_droid_k5_s2"],
    "libero_only": [f"e18_trunk_libero_only_k5_s{s}" for s in range(3)],
    "droid_libero": [f"e18_trunk_droid_libero_k5_s{s}" for s in range(3)],
}
R = {}
for a, runs in ARMS.items():
    try:
        R[a] = {}
        per = []
        for s, r in enumerate(runs):
            d = json.load(open(f"outputs/{r}/rollouts_n50.json"))
            per.append(100 * d["success_rate"])
            for e in d["episodes"]:
                R[a][(s, e["task"], e["init"])] = e["success"]
        m = sum(per) / 3
        sd = (sum((q - m) ** 2 for q in per) / 2) ** 0.5
        print(f"{a:14s}", [round(q, 1) for q in per], f"{m:.1f} ± {sd:.1f}")
    except FileNotFoundError as ex:
        R.pop(a)
        print(f"{a:14s} missing ({ex.filename})")


def mc(a, b):
    ks = R[a].keys() & R[b].keys()
    x = sum(R[a][k] and not R[b][k] for k in ks)
    y = sum(R[b][k] and not R[a][k] for k in ks)
    n, t = x + y, min(x, y)
    return x, y, (min(1.0, 2 * sum(comb(n, i) for i in range(t + 1)) / 2 ** n) if n else 1.0)


for a, b in [("joint_prior", "libero_only"), ("droid_libero", "joint_prior"), ("libero_only", "scratch"), ("droid_libero", "scratch")]:
    if a in R and b in R:
        x, y, p = mc(a, b)
        print(f"{a} vs {b}: {x} vs {y} p={p:.3g}")
