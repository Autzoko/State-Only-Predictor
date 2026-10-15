#!/usr/bin/env python
"""Does the +SOPT gain depend on task difficulty? Cross-seed design to avoid regression to the mean: task difficulty is
the baseline arm's success on ONE seed; the gain (target - baseline, and target - scratch) is measured on the OTHER
seeds. Repeated with each seed as the difficulty seed. Comparisons come from the paper manifest (first arm = baseline,
second = scratch control, last = DROID target).
  python scripts/headroom.py --root <rollouts dir> --match "FastWAM" [--thr 70]
"""
import argparse
import json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--manifest", default="configs/paper_manifest.json")
ap.add_argument("--match", required=True, help="substring of the comparison name")
ap.add_argument("--thr", type=float, default=70.0, help="hard task = baseline success below this (difficulty seed)")
a = ap.parse_args()
for c in json.load(open(a.manifest))["comparisons"]:
    if a.match not in c["name"]:
        continue
    file = c.get("file", "rollouts_n50.json")
    arms = list(c["arms"].items())
    try:
        per = {name: [json.load(open(Path(a.root) / r / file))["episodes"] for r in runs] for name, runs in arms}
    except FileNotFoundError as e:
        print(f"### {c['name']}: missing {Path(e.filename).parent.name}")
        continue
    base, ctrl, tgt = arms[0][0], arms[1][0], arms[-1][0]
    tasks = sorted({e["task"] for e in per[base][0]})

    def rate(name, seeds):
        acc = {t: [] for t in tasks}
        for s in seeds:
            for e in per[name][s]:
                acc[e["task"]].append(e["success"])
        return np.array([100 * np.mean(acc[t]) for t in tasks])

    print(f"### {c['name']}  (hard = {base} < {a.thr:.0f}% on the difficulty seed)")
    n = len(per[base])
    for ds in range(n):
        rest = [s for s in range(n) if s != ds]
        d = rate(base, [ds])
        gb, gs = rate(tgt, rest) - rate(base, rest), rate(tgt, rest) - rate(ctrl, rest)
        h = d < a.thr
        f = lambda x, m: f"{x[m].mean():+5.1f}" if m.any() else "  -  "
        print(f"  diff seed {ds}: r={np.corrcoef(d, gb)[0, 1]:+.2f} | hard n={h.sum():2d} {tgt}-{base} {f(gb, h)}, "
              f"{tgt}-{ctrl} {f(gs, h)} | easy n={(~h).sum():2d} {tgt}-{base} {f(gb, ~h)}, {tgt}-{ctrl} {f(gs, ~h)}")
