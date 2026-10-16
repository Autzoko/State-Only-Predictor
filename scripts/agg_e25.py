#!/usr/bin/env python
"""E25: TrunkBC with each pretraining variant vs the reference prior (e23_trunk_droid = e2_ar_flow_f100_s), paired
by (seed, task, init) on the 17-task protocol; dev 7 tasks = selection, new 10 = confirmation.
  python scripts/agg_e25.py [--root outputs] [variants ...]      (default: all of base event jepa speed ctx192)
"""
import argparse
import json
from math import comb
from pathlib import Path


def mcnemar(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2**n)


p = argparse.ArgumentParser()
p.add_argument("--root", default="outputs")
p.add_argument("variants", nargs="*", default=["base", "event", "jepa", "speed", "ctx192"])
a = p.parse_args()
dev = set(json.load(open("configs/e4b_tasks.json")))
tasks = set(json.load(open("configs/libero90_17_tasks.json")))
groups = (("all 17", tasks), ("7 dev", dev & tasks), ("10 new", tasks - dev))


def load(stem: str) -> dict:
    out = {}
    for s in (0, 1, 2):
        f = Path(a.root) / f"{stem}_k5_t17_s{s}" / "rollouts_n50.json"
        if f.exists():
            out.update({(s, e["task"], e["init"]): bool(e["success"]) for e in json.load(open(f))["episodes"]})
    return out


ref = load("e23_trunk_droid")
others = {"bc": load("e23_bc"), "scratch": load("e23_trunk_scratch")}
print("reference e23_trunk_droid:", " | ".join(
    f"{g} {100 * sum(v for k, v in ref.items() if k[1] in sel) / max(1, sum(k[1] in sel for k in ref)):.1f}"
    for g, sel in groups))
for v in a.variants:
    run = load(f"e25_trunk_{v}")
    if not run:
        print(f"{v:7s} no runs yet")
        continue
    seeds = sorted({k[0] for k in run})
    parts = []
    for g, sel in groups:
        keys = [k for k in run if k[1] in sel and k in ref]
        x, y = sum(run[k] for k in keys), sum(ref[k] for k in keys)
        b, c = sum(run[k] and not ref[k] for k in keys), sum(ref[k] and not run[k] for k in keys)
        parts.append(f"{g} {100 * x / len(keys):.1f} vs {100 * y / len(keys):.1f} ({100 * (x - y) / len(keys):+.1f}, "
                     f"{b}:{c} p={mcnemar(b, c):.3g})")
    print(f"{v:7s} seeds {seeds} | " + " | ".join(parts))
