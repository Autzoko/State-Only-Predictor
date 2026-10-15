#!/usr/bin/env python
"""17-task protocol: success per arm on the 7 dev tasks (configs/e4b_tasks.json) vs the 10 new tasks, per seed set.
  python scripts/agg_split.py --prefix e22_groot --tag _t17 [--root outputs] [seeds ...]
Arms follow agg_host.py naming (<prefix>_{none,expert_scratch,expert_droid}_k5<tag>_s<seed>) unless --trunk, which uses
E23 TrunkBC names (e23_{bc,trunk_scratch,trunk_droid}_k5_t17_s<seed>). Missing runs are skipped and listed.
"""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--prefix", default="e23_smolvla")
p.add_argument("--tag", default="_t17")
p.add_argument("--root", default="outputs")
p.add_argument("--trunk", action="store_true")
p.add_argument("seeds", nargs="*", type=int)
a = p.parse_args()
dev = set(json.load(open("configs/e4b_tasks.json")))
tasks = json.load(open("configs/libero90_17_tasks.json"))
names = (("bc", "e23_bc"), ("scratch", "e23_trunk_scratch"), ("droid", "e23_trunk_droid")) if a.trunk else \
    (("none", f"{a.prefix}_none"), ("scratch", f"{a.prefix}_expert_scratch"), ("droid", f"{a.prefix}_expert_droid"))
res, missing = {}, []
for arm, stem in names:
    eps = []
    for s in a.seeds or [0, 1, 2]:
        f = Path(a.root) / f"{stem}_k5{a.tag}_s{s}" / "rollouts_n50.json"
        if not f.exists():
            missing.append(f.parent.name)
            continue
        eps += json.load(open(f))["episodes"]
    res[arm] = eps
print("missing:", missing or "none")
for grp, sel in (("all 17", set(tasks)), ("7 dev", dev & set(tasks)), ("10 new", set(tasks) - dev)):
    row = []
    for arm, eps in res.items():
        e = [x["success"] for x in eps if x["task"] in sel]
        row.append(f"{arm} {100 * sum(e) / len(e):5.1f} (n={len(e)})" if e else f"{arm} -")
    print(f"{grp:7s} | " + " | ".join(row))
