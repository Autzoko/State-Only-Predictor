#!/usr/bin/env python
"""Analysis of the data-validation runs DV1 / DV3 (pre-registered in docs/EXPERIMENTS.md, 2026-10-27).

Inputs are the per-window npz files of scripts/dv_anchor_eval.py. CIs resample whole episodes (multinomial counts),
so windows of one episode stay together. Sites are the DROID test sites matched on `building`.
  python scripts/dv_analyze.py --dir outputs --episodes $SOPT_DATA/processed/droid/episodes.parquet --out dv_summary.json
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sopt.data.splits import site_ids

SITES = {"UT Austin": "^AHG", "Edinburgh": "^[Bb]ayes", "Penn": "^Penn"}
STEPS = (1, 4, 8, 16, 24, 32)


class Boot:
    """Episode-cluster bootstrap of window means: per-episode sums, resampled with multinomial episode counts."""

    def __init__(self, episode: np.ndarray, B: int, seed: int = 0):
        self.uniq, self.inv = np.unique(episode, return_inverse=True)
        self.n = np.bincount(self.inv).astype(float)
        rng = np.random.default_rng(seed)
        E = len(self.uniq)
        self.W = rng.multinomial(E, np.full(E, 1 / E), size=B).astype(float)  # (B, E)

    def sums(self, v: np.ndarray) -> np.ndarray:
        """v: (windows,) or (windows, k) -> per-episode sums (E,) or (E, k)."""
        out = np.zeros((len(self.uniq),) + v.shape[1:])
        np.add.at(out, self.inv, v)
        return out

    def mean(self, v):
        s = self.sums(v)
        return s.sum(0) / self.n.sum(), (self.W @ s) / (self.W @ self.n)[:, None] if s.ndim > 1 else (self.W @ s) / (self.W @ self.n)


def ci(point, boots):
    lo, hi = np.quantile(boots, [0.025, 0.975], axis=0)
    return {"est": np.round(point, 4).tolist(), "lo": np.round(lo, 4).tolist(), "hi": np.round(hi, 4).tolist()}


def dv1(z, b: Boot, ref: str, models: list[str]) -> dict:
    out = {"mean_es": {}, "diff_vs_ref": {}, "rel_vs_ref": {}, "adjacent": {}, "step_ratio_vs_ref": {}}
    pts, bts = {}, {}
    for m in models + ["const_vel", "zero_vel"]:
        pts[m], bts[m] = b.mean(z[f"{m}/es"])
        out["mean_es"][m] = ci(pts[m], bts[m])
    for m in models:
        if m == ref:
            continue
        out["diff_vs_ref"][m] = ci(pts[m] - pts[ref], bts[m] - bts[ref])
        out["rel_vs_ref"][m] = ci(pts[m] / pts[ref] - 1, bts[m] / bts[ref] - 1)
    order = [m for m in ("ctx4", "ctx16", "ctx48", "ctx96", "ctx192", "ctx384") if m in models]
    for a, c in zip(order[:-1], order[1:]):
        out["adjacent"][f"{a}-{c}"] = ci(pts[a] - pts[c], bts[a] - bts[c])
    inst = [m for m in models if m.startswith("ctx96")]
    if len(inst) > 1:
        P = np.array([pts[m] for m in inst]); Bm = np.stack([bts[m] for m in inst])
        out["ctx96_instance_range"] = ci(P.max() - P.min(), Bm.max(0) - Bm.min(0))
    rp, rb = b.mean(z[f"{ref}/es_step"])
    for m in ("ctx4", "ctx16", "ctx48"):
        if m in models:
            mp, mb = b.mean(z[f"{m}/es_step"])
            idx = [s - 1 for s in STEPS]
            out["step_ratio_vs_ref"][m] = ci((mp / rp)[idx], (mb / rb)[:, idx])
    return out


def sites_of(episode: np.ndarray, eps: pd.DataFrame) -> np.ndarray:
    """Test site of each window: the connected site component (sopt.data.splits.site_ids) holding a matching building."""
    comp = pd.Series(site_ids(eps), index=eps["episode_index"])
    lab = pd.Series("other", index=comp.index, dtype=object)
    for name, pat in SITES.items():
        hit = eps["building"].map(lambda x: bool(re.search(pat, x))).to_numpy()
        lab[comp.isin(np.unique(comp.to_numpy()[hit])).to_numpy()] = name
    return lab.loc[episode].to_numpy()


def dv3(z, b: Boot, name: str) -> dict:
    idx = [s - 1 for s in STEPS]
    out = {"steps": STEPS, "es_step": {}, "where_share": {}}
    pts, bts = {}, {}
    for k in (name, f"{name}@keyframe", f"{name}@endpoint", "interp@endpoint", "const_vel"):
        pts[k], bts[k] = b.mean(z[f"{k}/es_step"])
        out["es_step"][k] = ci(pts[k][idx], bts[k][:, idx])
        out.setdefault("es_mean", {})[k] = ci(*b.mean(z[f"{k}/es"]))
    for g in ("keyframe", "endpoint"):
        k = f"{name}@{g}"
        out["where_share"][g] = ci((1 - pts[k] / pts[name])[idx], (1 - bts[k] / bts[name])[:, idx])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="outputs")
    p.add_argument("--episodes", required=True)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    d = Path(args.dir)
    eps = pd.read_parquet(args.episodes)
    res = {}
    base = ["ctx4", "ctx16", "ctx48", "ctx96", "ctx96_b1", "ctx96_b2"]
    for tag, models in (("a96_test", base), ("a96_val", base), ("a384_test", base + ["ctx192", "ctx384"])):
        z = np.load(d / "dv1" / f"{tag}.npz")
        b = Boot(z["episode"], args.boot)
        res[tag] = {"windows": int(len(z["episode"])), "episodes": int(len(b.uniq))} | dv1(z, b, "ctx96", models)
        if tag == "a96_test":
            lab = sites_of(z["episode"], eps)
            res[tag]["by_site"] = {}
            for s in SITES:
                m = lab == s
                zs = {k: z[k][m] for k in z.files}
                bs = Boot(zs["episode"], args.boot)
                r = dv1(zs, bs, "ctx96", ["ctx4", "ctx96"])
                res[tag]["by_site"][s] = {"windows": int(m.sum()), "episodes": int(len(bs.uniq)),
                                          "diff": r["diff_vs_ref"]["ctx4"], "rel": r["rel_vs_ref"]["ctx4"]}
            assert (lab != "other").all(), "test windows outside the three test sites"
    for name in ("gc_ar_flow_ft", "gc_ar_flow_scratch"):
        z = np.load(d / "dv3" / f"{name}_test.npz")
        res[f"dv3_{name}"] = dv3(z, Boot(z["episode"], args.boot), name)
    Path(args.out).write_text(json.dumps(res, indent=1))
    t = res["a96_test"]
    print("DV1 primary (test, 96-frame anchors): ES ctx4 - ctx96 =", t["diff_vs_ref"]["ctx4"], "rel", t["rel_vs_ref"]["ctx4"])


if __name__ == "__main__":
    main()
