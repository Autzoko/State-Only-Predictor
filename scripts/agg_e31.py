#!/usr/bin/env python
"""E31 aggregation: GR00T SOPT interfaces vs the E22 arms of the same seed (17 tasks x 50 inits), paired McNemar,
overall / 7 dev / 10 new, plus the two-stage pudding task.
  python scripts/agg_e31.py 0 [1 2]
"""
import json, math, sys
from pathlib import Path

seeds = [int(s) for s in sys.argv[1:]] or [0]
dev = set(json.load(open("configs/e4b_tasks.json")))
ARMS = {"none": "e22_groot_none_k5_t17_s%d", "expert_droid": "e22_groot_expert_droid_k5_t17_s%d",
        "expert_scratch": "e22_groot_expert_scratch_k5_t17_s%d",
        "adaln_droid": "e31_groot_adaln_droid_k5_t17_s%d", "adaln_scratch": "e31_groot_adaln_scratch_k5_t17_s%d",
        "xattn_droid": "e31_groot_xattn_droid_k5_t17_s%d",
        "xattn_droid_reg": "e31_groot_xattn_droid_k5_cd03_plr01_t17_s%d",
        "xattn_scratch_reg": "e31_groot_xattn_scratch_k5_cd03_plr01_t17_s%d"}


def p_mc(x, y):
    n, t = x + y, min(x, y)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(t + 1)) / 2 ** n) if n else 1.0


R = {}
for a, pat in ARMS.items():
    for s in seeds:
        f = Path("outputs") / (pat % s) / "rollouts_n50.json"
        if f.exists():
            for e in json.load(open(f))["episodes"]:
                R.setdefault(a, {})[(s, e["task"], e["init"])] = e["success"]
tasks = sorted({k[1] for r in R.values() for k in r})
groups = (("all", set(tasks)), ("dev", dev & set(tasks)), ("new", set(tasks) - dev),
          ("pudding2", {t for t in tasks if "pudding in the top drawer" in t}))
print(f"seeds {seeds}")
for a in R:
    print(f"{a:18s} " + "  ".join(f"{g} {100 * sum(v for k, v in R[a].items() if k[1] in sel) / max(1, sum(k[1] in sel for k in R[a])):5.1f}"
                                   for g, sel in groups) + f"  (n={len(R[a])})")
for a in R:
    for b in ("none", "expert_droid"):
        if a == b or b not in R or a in ("none",):
            continue
        out = []
        for g, sel in groups:
            ks = [k for k in R[a].keys() & R[b].keys() if k[1] in sel]
            x = sum(R[a][k] and not R[b][k] for k in ks); y = sum(R[b][k] and not R[a][k] for k in ks)
            out.append(f"{g} {x}:{y} p={p_mc(x, y):.2g}")
        print(f"{a:18s} vs {b:13s} " + " | ".join(out))
