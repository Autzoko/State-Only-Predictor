#!/usr/bin/env python
"""E13 / E7b aggregation: none / expert_scratch / expert_droid, pooled paired McNemar over seeds and tasks.

  python scripts/agg_e13.py libero_spatial      # all demos, seed 0
  python scripts/agg_e13.py libero_spatial 5    # k5
  python scripts/agg_e13.py e7b                 # LIBERO-90 E4b tasks, all demos (SmolVLA, E7b), seeds 0-2
"""
import json
import os
import sys
from math import comb

suite = sys.argv[1]
k = int(sys.argv[2]) if len(sys.argv) > 2 else 0
ks = f"_k{k}" if k else ""


def run(arm, s):
    return f"e7b_vla_{arm}_all_s{s}" if suite == "e7b" else f"e13_{suite}_{arm}{ks}_s{s}"


R = {}
for arm in ("none", "expert_scratch", "expert_droid"):
    R[arm], per = {}, []
    for s in range(3):
        f = f"outputs/{run(arm, s)}/rollouts_n50.json"
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        per.append(100 * d["success_rate"])
        for e in d["episodes"]:
            R[arm][(s, e["task"], e["init"])] = e["success"]
    m = sum(per) / max(1, len(per))
    sd = (sum((q - m) ** 2 for q in per) / max(1, len(per) - 1)) ** 0.5
    print(f"{arm:15s} {[round(q, 1) for q in per]} {m:.1f} ± {sd:.1f}")


def mc(a, b):
    ks_ = R[a].keys() & R[b].keys()
    x = sum(R[a][q] and not R[b][q] for q in ks_)
    y = sum(R[b][q] and not R[a][q] for q in ks_)
    n, t = x + y, min(x, y)
    return x, y, (min(1.0, 2 * sum(comb(n, i) for i in range(t + 1)) / 2 ** n) if n else 1.0), len(ks_)


for a, b in [("expert_droid", "none"), ("expert_droid", "expert_scratch"), ("expert_scratch", "none")]:
    x, y, p, n = mc(a, b)
    print(f"{a} vs {b}: {x} vs {y} p={p:.3g} (n={n})")
