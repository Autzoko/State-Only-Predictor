#!/usr/bin/env python
"""Pre-registered linear contrasts of paired closed-loop success rates (E39 / E40 / E41 / E38b), one protocol:

  A contrast is a weighted sum over host groups; inside a group it is a weighted sum of runs (terms), each term a list
  of runs indexed by seed. All terms of a group must cover the same (task, init) episodes per seed (paired design).
  Per group the unit of resampling is the (seed, task) cluster -- the same clusters are drawn for every term of the
  group, so within-cluster pairing is kept; groups are resampled independently (stratified by host).
  Estimate = sum_g weight_g * sum_terms w * success(run) (percentage points), 95% percentile CI, two-sided bootstrap p
  = 2 * min(P(boot <= 0), P(boot >= 0)) (floored at 1 / n_boot), Holm over the contrasts listed in `primary`.

  python scripts/contrast_boot.py --root outputs --spec configs/e39_contrasts.json [--out stats.md]
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load(root: Path, run: str, file: str):
    f = root / run / file
    if not f.exists():
        return None
    eps = json.loads(f.read_text())["episodes"]
    return {(e["task"], e["init"]): float(e["success"]) for e in eps}


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def group_matrix(root: Path, g: dict) -> tuple[np.ndarray | None, list[str]]:
    """-> (n_clusters,) per-cluster value of sum_terms w * mean success (in [0, 1] units), or None + missing runs."""
    missing, per_term = [], []
    for t in g["terms"]:
        runs = [load(root, r, t.get("file", "rollouts_n50.json")) for r in t["runs"]]
        missing += [r for r, x in zip(t["runs"], runs) if x is None]
        per_term.append(runs)
    if missing:
        return None, missing
    n_seed = len(per_term[0])
    assert all(len(x) == n_seed for x in per_term), "every term needs the same seeds"
    vals = []
    for s in range(n_seed):
        keys = set(per_term[0][s])
        for x in per_term[1:]:
            assert set(x[s]) == keys, f"seed {s}: terms cover different episodes"
        for task in sorted({k[0] for k in keys}):
            ks = [k for k in keys if k[0] == task]
            vals.append(sum(t["w"] * np.mean([x[s][k] for k in ks]) for t, x in zip(g["terms"], per_term)))
    return np.asarray(vals), []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root, spec = Path(args.root), json.loads(Path(args.spec).read_text())
    rng = np.random.default_rng(0)
    rows, pvals = [], {}
    for c in spec["contrasts"]:
        est, boot, missing = 0.0, np.zeros(args.boot), []
        for gname, g in c["groups"].items():
            v, miss = group_matrix(root, g)
            if v is None:
                missing += miss
                continue
            draws = rng.integers(0, len(v), size=(args.boot, len(v)))
            est += g.get("weight", 1.0) * v.mean()
            boot += g.get("weight", 1.0) * v[draws].mean(1)
        if missing:
            rows.append(f"| {c['name']} | missing: {', '.join(sorted(set(missing)))} | | | |")
            continue
        lo, hi = np.percentile(100 * boot, [2.5, 97.5])
        p = min(1.0, max(2 * min((boot <= 0).mean(), (boot >= 0).mean()), 1 / args.boot))
        pvals[c["name"]] = p
        rows.append(f"| {c['name']} | {100 * est:+.1f} | [{lo:+.1f}, {hi:+.1f}] | {p:.2g} | HOLM:{c['name']} |")
    prim = [n for n in spec.get("primary", []) if n in pvals]
    adj = dict(zip(prim, holm([pvals[n] for n in prim]))) if prim else {}
    out = []
    for r in rows:
        if "HOLM:" in r:
            n = r.split("HOLM:")[1].rstrip(" |")
            r = r.split("HOLM:")[0] + (f"{adj[n]:.2g} |" if n in adj else "– |")
        out.append(r)
    note = "" if len(prim) == len(spec.get("primary", [])) else "\n(primary contrasts with missing runs are not in Holm)"
    text = (f"### {spec.get('name', args.spec)}\n\n| contrast | estimate (pts) | 95% CI | bootstrap p | Holm p |\n"
            "|---|---|---|---|---|\n" + "\n".join(out) + note + "\n")
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
