#!/usr/bin/env python
"""DV2: model-free temporal redundancy of DROID proprioception (15 Hz), per split, with an episode bootstrap.

Statistics on the EE position p_t (m) and joint angles q_t (rad), within episodes only:
  * constant-velocity R^2 at horizon h: target p_{t+h} - p_t, prediction h * v_t with v_t = (p_t - p_{t-k}) / k
    (k = 1 and k = 3, the latter = the const-vel baseline of sopt.eval.forecast). R^2 = 1 - SSE / SST, SST about the
    pooled mean displacement. h = 1 is the next-frame redundancy.
  * velocity autocorrelation at lag L: Pearson correlation of v_t and v_{t+L} (v_t = p_t - p_{t-1}), pooled over axes.
  * gripper: P(binarized gripper state unchanged after L frames).
The same anchor set (t >= 3, t + h_max <= len - 1 for R^2; t >= 1, t + L_max <= len - 1 for correlations) is used for
every h / L so the curves are comparable. Sufficient statistics are kept per episode; CIs resample episodes.

  python scripts/dv_redundancy.py --splits train test --out outputs/dv2/redundancy.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from sopt.data.dataset import TrajectoryStore
from sopt.data.features import GRIPPER, POS, Q
from sopt.utils.config import DEFAULT_CONFIG

HS = (1, 2, 4, 8, 16, 32, 64)
LAGS = (1, 2, 4, 8, 16, 32, 48, 64, 96)


def episode_stats(f: np.ndarray) -> dict | None:
    L = len(f)
    if L < 4 + max(HS) or L < 2 + max(LAGS):
        return None
    out = {}
    for part, sl in (("pos", POS), ("q", Q)):
        x = f[:, sl].astype(np.float64)
        t = np.arange(3, L - max(HS))
        for k in (1, 3):
            v = (x[t] - x[t - k]) / k
            for h in HS:
                d = x[t + h] - x[t]
                r = d - h * v
                # per-episode sums: SSE, n, sum d, sum d^2 (per dimension) -> pooled SST later
                out[f"{part}/k{k}/h{h}"] = np.concatenate([[np.sum(r * r)], [d.shape[0]], d.sum(0), (d * d).sum(0)])
        v = np.diff(x, axis=0)  # v[i] = x[i+1] - x[i]
        t = np.arange(0, len(v) - max(LAGS))
        for lag in LAGS:
            a, b = v[t], v[t + lag]
            out[f"{part}/corr/L{lag}"] = np.concatenate([[len(t)], a.sum(0), b.sum(0), (a * b).sum(0), (a * a).sum(0),
                                                         (b * b).sum(0)])
        out[f"{part}/static_frac"] = np.array([np.mean(np.abs(v).max(1) == 0), 1.0])
    g = f[:, GRIPPER.start] > 0.5
    t = np.arange(0, L - max(LAGS))
    for lag in LAGS:
        out[f"grip/same/L{lag}"] = np.array([np.sum(g[t] == g[t + lag]), len(t)])
    return out


def reduce(sums: dict) -> dict:
    """Pooled statistics from (weighted) sums over episodes of the per-episode sufficient statistics."""
    res = {}
    for key, s in sums.items():
        if "/h" in key:
            D = (len(s) - 2) // 2
            sse, n, sd, sdd = s[0], s[1], s[2 : 2 + D], s[2 + D :]
            res[key] = 1 - sse / np.sum(sdd - sd * sd / n)
        elif "/corr/" in key:
            D = (len(s) - 1) // 5
            n, sa, sb, sab, saa, sbb = s[0], *(s[1 + i * D : 1 + (i + 1) * D] for i in range(5))
            cov = np.sum(sab - sa * sb / n)
            res[key] = cov / np.sqrt(np.sum(saa - sa * sa / n) * np.sum(sbb - sb * sb / n))
        else:
            res[key] = s[0] / s[1]
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    p.add_argument("--data-root", default=None)
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--max-episodes", type=int, default=0, help="debug: first N episodes per split")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    dcfg = OmegaConf.load(DEFAULT_CONFIG).data
    store = TrajectoryStore(args.data_root or dcfg.root)
    result = {"hz": 15, "horizons": HS, "lags": LAGS, "splits": {}}
    rng = np.random.default_rng(0)
    for split in args.splits:
        eps = store.split_episodes(dcfg, split)
        if args.max_episodes:
            eps = eps.iloc[: args.max_episodes]
        per = []
        for s0, L in zip(eps["start"].to_numpy(), eps["length"].to_numpy()):
            st = episode_stats(np.asarray(store.features[s0 : s0 + L]))
            if st is not None:
                per.append(st)
        keys = list(per[0])
        cols = np.cumsum([0] + [len(per[0][k]) for k in keys])
        M = np.stack([np.concatenate([e[k] for k in keys]) for e in per])  # (episodes, all sufficient stats)
        split_cols = lambda v: {k: v[cols[i] : cols[i + 1]] for i, k in enumerate(keys)}
        point = reduce(split_cols(M.sum(0)))
        boots = []
        for _ in range(0, args.boot, 100):  # multinomial episode counts, 100 replicates per matmul
            W = rng.multinomial(len(per), np.full(len(per), 1 / len(per)), size=min(100, args.boot - len(boots)))
            boots += [reduce(split_cols(row)) for row in W.astype(float) @ M]
        ci = {k: [float(np.quantile([b[k] for b in boots], q)) for q in (0.025, 0.975)] for k in point}
        result["splits"][split] = {"episodes_used": len(per), "episodes_total": int(len(eps)),
                                   "point": {k: float(v) for k, v in point.items()}, "ci95": ci}
        print(split, len(per), "of", len(eps), "episodes", flush=True)
        for k in point:
            print(f"  {k:22s} {point[k]:.4f} [{ci[k][0]:.4f}, {ci[k][1]:.4f}]")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
