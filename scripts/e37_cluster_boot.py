#!/usr/bin/env python
"""E37 robustness check (post hoc, conservative): the pre-registered e37_analyze.py bootstrap resamples (task, init)
pairs, treating episodes of one task as independent. Here whole (host, seed, task) clusters are resampled within
host x seed strata, so between-task heterogeneity enters the CI. Same interaction as e37_analyze.py.
  python scripts/e37_cluster_boot.py --root outputs
"""

import argparse
import json
from pathlib import Path

import numpy as np

from e37_analyze import HOSTS, VISUAL, outcomes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--boot", type=int, default=5000)
    args = p.parse_args()
    root = Path(args.root)
    levels = json.load(open(root / "e37_levels.json"))["levels"]
    rows = []  # (host, seed, task, kind, interaction)
    for h, (rn, rs) in HOSTS.items():
        for s in args.seeds:
            nc, sc = outcomes(root / rn.format(s) / "rollouts_n50.json"), outcomes(root / rs.format(s) / "rollouts_n50.json")
            for k, l in levels.items():
                npert = outcomes(root / rn.format(s) / f"e37_main_{k}{l}.json")
                spert = outcomes(root / rs.format(s) / f"e37_main_{k}{l}.json")
                assert nc.keys() == sc.keys() == npert.keys() == spert.keys(), (h, s, k)
                rows += [(h, s, x[0], k, (spert[x] - npert[x]) - (sc[x] - nc[x])) for x in nc]
    rng = np.random.default_rng(0)

    def clus(sel):
        cl = {}
        for h, s, t, k, v in rows:
            if sel(h, k):
                cl.setdefault((h, s, t), []).append(v)
        keys = list(cl)
        sums, ns = np.array([sum(cl[c]) for c in keys]), np.array([len(cl[c]) for c in keys])
        strata = {}
        for i, c in enumerate(keys):
            strata.setdefault(c[:2], []).append(i)
        bs = []
        for _ in range(args.boot):
            idx = np.concatenate([rng.choice(v, len(v)) for v in strata.values()])
            bs.append(sums[idx].sum() / ns[idx].sum())
        lo, hi = np.quantile(bs, [0.025, 0.975])
        return f"{100 * sums.sum() / ns.sum():+.2f} pts [{100 * lo:+.2f}, {100 * hi:+.2f}] ({len(keys)} clusters)"

    print("visual, all hosts:", clus(lambda h, k: k in VISUAL))
    for h in HOSTS:
        print(f"  host {h}:", clus(lambda hh, k, h=h: k in VISUAL and hh == h))
    for kind in levels:
        print(f"  kind {kind}:", clus(lambda h, k, kind=kind: k == kind))
    print("  visual without camera:", clus(lambda h, k: k in VISUAL and k != "camera"))


if __name__ == "__main__":
    main()
