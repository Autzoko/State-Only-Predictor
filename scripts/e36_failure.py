#!/usr/bin/env python
"""E36a: host-agnostic runtime failure prediction from proprioception alone (docs/SOPT_NEXT.md).

Input: closed-loop rollouts logged with `vla_eval.py --log-npz` (per-step joints + fingers, executed actions, every
predicted chunk). Every score at check time t uses only what has happened up to t (causal):
  sopt_ee     energy score of the frozen DROID EE-only prior's forecast issued at t - 1 s, against the realised
              EE pose + gripper over (t - 1 s, t]  (K samples; same prior for every host and suite)
  sopt_full   the same with the full-state DROID prior (LIBERO-90 only: needs joints)
  rand_ee     the same architecture / normalisation, random init (isolates DROID pretraining)
  constvel    constant-velocity forecast from t - 1 s, same error metric (isolates any learned prior)
  still       EE path length over the last 2 s, negated (stalling)
  gripcyc     closing -> opening transitions of the gripper command so far
  blocked     fraction of 4-step windows in the last 2 s with a large commanded translation but < 30% of the
              achieved-per-command gain of the calibration successes (E26 definition, online)
  chunkcons   host action-chunk inconsistency: mean |overlap difference| of consecutive predicted chunks (dims 0-5),
              a Sentinel-style temporal-consistency score -- host-specific, needs the host's own predictions
Aggregation over checks: running max (key) or current value (key + "_last"; e.g. stalling is normal right after
the start, which a running max would keep). Baselines are credited with the better of the two.
Splits: inits < 25 = calibration (thresholds; successes only, as in FAIL-Detect), inits >= 25 = test.
Metrics: AUROC at 25 / 50 / 75 % of the step limit among test episodes still running; sequential detection with
time-varying conformal thresholds (alpha 0.1) -> TPR, FPR, median detection time; cross-run threshold transfer.
Pre-registered primary: mean AUROC at 50 % over the clean LIBERO-90 runs, sopt_ee (running max) vs the best
non-SOPT score (either aggregation, chosen by the same mean); pass if sopt_ee >= 0.75, exceeds that baseline by
>= 0.05 with the pooled paired-bootstrap 95% CI of the mean difference > 0, and under cross-run threshold transfer
the TPR drops by <= 0.05 on average while the FPR stays <= alpha + 0.05.

  python scripts/e36_failure.py --out outputs/e36_failure.json --runs groot_none=outputs/.../log_n50.npz ...
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sopt.data.features import GRIPPER, POS, ROT
from sopt.data.libero import features_20hz
from sopt.data.libero_std import hide_joints
from sopt.models.state_prior import StatePrior
from sopt.policy.data import context_indices, interp_frames
from sopt.train.pretrain import load_checkpoint

SIM_HZ, PRIOR_HZ = 20.0, 15.0
H_STEPS = 20  # forecast lead: 1 s at 20 Hz
M = int(H_STEPS * PRIOR_HZ / SIM_HZ)  # 15 prior frames span (t - 1 s, t]
CHECK = 10  # check every 0.5 s
T0 = 2 * H_STEPS  # first check at 2 s: every forecast has >= 1 s of real history (the start from rest is shared)
DIMS = np.r_[GRIPPER.start, np.arange(POS.start, POS.stop), np.arange(ROT.start, ROT.stop)]
SCORES = ("sopt_ee", "sopt_full", "rand_ee", "constvel", "still", "gripcyc", "blocked", "chunkcons")


def auroc(pos, neg) -> float | None:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    allv = np.concatenate([pos, neg])
    ranks = np.empty(len(allv))
    order = allv.argsort(kind="mergesort")
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):  # average ranks for ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _v(x) -> float:  # AUROC that could not be computed (one class empty) counts as chance
    return 0.5 if x is None else x


def _f(x) -> str:
    return "  -  " if x is None else f"{x:.3f}"


def load_run(path: str) -> dict:
    z = np.load(path)
    n = len(z["success"])
    eps = []
    for k in range(n):
        qf = z["qf"][z["qf_off"][k]: z["qf_off"][k + 1]].astype(np.float64)
        eps.append({"qf": qf, "act": z["act"][z["act_off"][k]: z["act_off"][k + 1]],
                    "chunk": z["chunk"][z["chunk_off"][k]: z["chunk_off"][k + 1]],
                    "chunk_t": z["chunk_t"][z["chunk_off"][k]: z["chunk_off"][k + 1]],
                    "success": bool(z["success"][k]), "init": int(z["init"][k]), "task": int(z["task"][k])})
    suite = str(z["suite"])
    max_steps = int(z["max_steps"])  # the eval's step limit (failures run exactly this long)
    assert all(len(e["act"]) == max_steps for e in eps if not e["success"])
    return {"eps": eps, "suite": suite, "n_exec": int(z["n_exec"]), "max_steps": max_steps}


def episode_features(e: dict, ee_only: bool) -> np.ndarray:
    F = features_20hz(e["qf"][:, :7], e["qf"][:, 7:9])
    return hide_joints(F) if ee_only else F


@torch.no_grad()
def forecast_scores(prior: StatePrior, feats: list[np.ndarray], checks: list[np.ndarray], K: int, device,
                    bs: int = 1024) -> list[np.ndarray]:
    """Energy score per check, in units of the prior's per-dim delta scale (EE pose + gripper dims only)."""
    C = prior.cfg.ctx_len
    scale = prior.normalizer.chunk_scale(prior.H)[:M].cpu().numpy()[:, DIMS]  # (M, d)
    ctxs, ys, owner = [], [], []
    for j, (F, ts) in enumerate(zip(feats, checks)):
        for t in ts:
            s = t - H_STEPS
            ctxs.append(interp_frames(F[: s + 1], context_indices(s, C)))  # strictly causal: frames <= s
            ys.append(interp_frames(F[: t + 1], s + np.arange(1, M + 1) * SIM_HZ / PRIOR_HZ)[:, DIMS])
            owner.append(j)
    out = np.zeros(len(ctxs))
    for a in range(0, len(ctxs), bs):
        ctx = torch.from_numpy(np.stack(ctxs[a: a + bs]).astype(np.float32)).to(device)
        X = prior.forecast(ctx, K)[:, :, :M].cpu().numpy()[..., DIMS] / scale  # (b, K, M, d)
        Y = np.stack(ys[a: a + bs]) / scale  # (b, M, d)
        Xf, Yf = X.reshape(len(X), K, -1), Y.reshape(len(Y), -1)
        t1 = np.linalg.norm(Xf - Yf[:, None], axis=-1).mean(1)
        t2 = np.linalg.norm(Xf[:, :, None] - Xf[:, None], axis=-1).sum((1, 2)) / (K * (K - 1))
        out[a: a + bs] = t1 - 0.5 * t2
    res, i = [], 0
    for ts in checks:
        res.append(out[i: i + len(ts)])
        i += len(ts)
    return res


def constvel_scores(F: np.ndarray, ts: np.ndarray, scale: np.ndarray) -> np.ndarray:
    out = []
    for t in ts:
        s = t - H_STEPS
        x0 = interp_frames(F[: s + 1], np.array([s - SIM_HZ / PRIOR_HZ, float(s)]))
        tau = np.arange(1, M + 1)[:, None]
        pred = (x0[1] + tau * (x0[1] - x0[0]))[:, DIMS]
        y = interp_frames(F[: t + 1], s + np.arange(1, M + 1) * SIM_HZ / PRIOR_HZ)[:, DIMS]
        out.append(np.linalg.norm(((pred - y) / scale).ravel()))
    return np.array(out)


def heuristic_scores(e: dict, F: np.ndarray, ts: np.ndarray, gain: float | None) -> dict:
    ee = F[:, POS]
    cmd = e["act"][:, 6] > 0  # sim convention: +1 = close
    still, cyc, blk = [], [], []
    for t in ts:
        w = ee[max(0, t - 40): t + 1]
        still.append(-float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum()))
        c = cmd[:t]
        cyc.append(float(np.sum(c[:-1] & ~c[1:])) if len(c) > 1 else 0.0)
        if gain is None:
            blk.append(0.0)
            continue
        fr = []
        for a in range(max(0, t - 40), t - 3, 4):
            cm = np.linalg.norm(e["act"][a: a + 4, :3].sum(0))
            if cm > 0.5:
                fr.append(np.linalg.norm(ee[a + 4] - ee[a]) < 0.3 * gain * cm)
        blk.append(float(np.mean(fr)) if fr else 0.0)
    return {"still": np.array(still), "gripcyc": np.array(cyc), "blocked": np.array(blk)}


def exec_gain(eps: list[dict], feats: list[np.ndarray]) -> float | None:
    """Achieved EE translation per unit command over 4-step windows of the calibration successes."""
    r = []
    for e, F in zip(eps, feats):
        if not (e["success"] and e["init"] < 25):
            continue
        for a in range(0, len(e["act"]) - 4, 4):
            cm = np.linalg.norm(e["act"][a: a + 4, :3].sum(0))
            if cm > 0.5:
                r.append(np.linalg.norm(F[a + 4, POS] - F[a, POS]) / cm)
    return float(np.median(r)) if r else None


def chunk_scores(e: dict, ts: np.ndarray, n_exec: int) -> np.ndarray:
    ch, ct = e["chunk"], e["chunk_t"]
    L = ch.shape[1]
    O = min(L - n_exec, 8)
    inc = [(ct[k + 1], float(np.abs(ch[k, n_exec: n_exec + O, :6] - ch[k + 1, :O, :6]).mean()))
           for k in range(len(ct) - 1)] if O > 0 else []
    out = []
    for t in ts:
        v = [s for tt, s in inc if tt <= t]
        out.append(v[-1] if v else 0.0)
    return np.array(out)


def evaluate(run: dict, scores: dict[str, list[np.ndarray]], checks: list[np.ndarray], alpha: float = 0.1,
             thresholds: dict | None = None) -> dict:
    eps, T = run["eps"], run["max_steps"]
    res = {"n": len(eps), "fail_rate": float(np.mean([not e["success"] for e in eps])), "auroc": {}, "seq": {}}
    grid = np.arange(T0, T + 1, CHECK)
    cm = {k: vals if k.endswith("_last") else [np.maximum.accumulate(v) if len(v) else v for v in vals]
          for k, vals in scores.items()}

    def at(k, j, t):  # running-max score of episode j at time t (None if no check yet)
        ts = checks[j]
        i = np.searchsorted(ts, t, side="right") - 1
        return cm[k][j][i] if i >= 0 else None

    test = [j for j, e in enumerate(eps) if e["init"] >= 25]
    cal = [j for j, e in enumerate(eps) if e["init"] < 25 and e["success"]]
    for k in scores:
        res["auroc"][k] = {}
        for frac in (0.25, 0.5, 0.75):
            t = int(round(frac * T / CHECK) * CHECK)
            run_j = [j for j in test if len(eps[j]["act"]) > t]  # still running at t
            pos = [at(k, j, t) for j in run_j if not eps[j]["success"]]
            neg = [at(k, j, t) for j in run_j if eps[j]["success"]]
            res["auroc"][k][f"{frac:.2f}"] = auroc([p for p in pos if p is not None], [q for q in neg if q is not None])
        # time-varying conformal thresholds from calibration successes still running at t (last stable one after)
        if thresholds is None:
            th, last = {}, None
            for t in grid:
                v = [at(k, j, t) for j in cal if len(eps[j]["act"]) >= t]
                v = [x for x in v if x is not None]
                if len(v) >= 20:
                    q = min(1.0, np.ceil((len(v) + 1) * (1 - alpha)) / len(v))
                    last = float(np.quantile(v, q))
                th[int(t)] = last
        else:
            th = thresholds[k]
        tp, fp, det_t = 0, 0, []
        nf = sum(not eps[j]["success"] for j in test)
        ns = len(test) - nf
        for j in test:
            L = len(eps[j]["act"])
            hit = next((t for t in grid if t <= L and th.get(int(t)) is not None and at(k, j, t) is not None
                        and at(k, j, t) > th[int(t)]), None)
            if hit is None:
                continue
            if eps[j]["success"]:
                fp += 1
            else:
                tp += 1
                det_t.append(hit / SIM_HZ)
        res["seq"][k] = {"tpr": tp / nf if nf else None, "fpr": fp / ns if ns else None,
                         "median_detect_s": float(np.median(det_t)) if det_t else None}
        res.setdefault("_thresholds", {})[k] = th
    return res


def at_frac(run: dict, scores: dict, checks: list, k: str, frac: float):
    """(score at frac of the step limit, failure label) of the test episodes still running then."""
    eps, T = run["eps"], run["max_steps"]
    t = int(round(frac * T / CHECK) * CHECK)
    js = [j for j, e in enumerate(eps) if e["init"] >= 25 and len(e["act"]) > t]

    def val(j):
        i = np.searchsorted(checks[j], t, side="right") - 1
        v = scores[k][j] if k.endswith("_last") else np.maximum.accumulate(scores[k][j])
        return v[i] if i >= 0 else -np.inf

    return np.array([val(j) for j in js]), np.array([not eps[j]["success"] for j in js])


def bootstrap_mean_diff(items: list, a: str, b: str, frac: float = 0.5, n: int = 1000, seed: int = 0) -> list:
    """Pooled paired bootstrap of mean over runs of AUROC(a) - AUROC(b): episodes resampled within each run."""
    data = [(at_frac(r, s, c, a, frac)[0], *at_frac(r, s, c, b, frac)) for r, s, c in items]
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        per = []
        for A, B, y in data:
            s = rng.integers(0, len(y), len(y))
            ra, rb = auroc(A[s][y[s]], A[s][~y[s]]), auroc(B[s][y[s]], B[s][~y[s]])
            if ra is not None and rb is not None:
                per.append(ra - rb)
        if per:
            d.append(np.mean(per))
    return [float(np.mean(d)), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="name=path.npz (from vla_eval.py --log-npz)")
    p.add_argument("--prior-ee", default="outputs/e13_ar_flow_f100_s_ee/best.pt")
    p.add_argument("--prior-full", default="outputs/e2_ar_flow_f100_s/best.pt")
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--primary", nargs="*", default=None, help="run names entering the pre-registered primary mean")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    p_ee, _ = load_checkpoint(args.prior_ee, device)
    p_full, _ = load_checkpoint(args.prior_full, device)
    assert p_ee.cfg.get("ee_only", False) and not p_full.cfg.get("ee_only", False)
    torch.manual_seed(0)
    p_rand = StatePrior(p_ee.cfg, {k: v.cpu().numpy() for k, v in p_ee.normalizer.state_dict().items()}).to(device)
    for m in (p_ee, p_full, p_rand):
        m.eval()
    scale = p_ee.normalizer.chunk_scale(p_ee.H)[:M].cpu().numpy()[:, DIMS]
    out = {"config": vars(args), "runs": {}}
    cache = {}
    for spec in args.runs:
        name, path = spec.split("=", 1)
        run = load_run(path)
        eps = run["eps"]
        ee_feats = [episode_features(e, True) for e in eps]
        full_feats = [episode_features(e, False) for e in eps]
        checks = [np.arange(T0, len(e["act"]) + 1, CHECK) for e in eps]
        sc = {"sopt_ee": forecast_scores(p_ee, ee_feats, checks, args.samples, device),
              "rand_ee": forecast_scores(p_rand, ee_feats, checks, args.samples, device),
              "constvel": [constvel_scores(F, ts, scale) for F, ts in zip(ee_feats, checks)]}
        if run["suite"] in ("libero90", "libero90_17"):
            sc["sopt_full"] = forecast_scores(p_full, full_feats, checks, args.samples, device)
        gain = exec_gain(eps, ee_feats)
        heur = [heuristic_scores(e, F, ts, gain) for e, F, ts in zip(eps, ee_feats, checks)]
        for k in ("still", "gripcyc", "blocked"):
            sc[k] = [h[k] for h in heur]
        sc["chunkcons"] = [chunk_scores(e, ts, run["n_exec"]) for e, ts in zip(eps, checks)]
        sc.update({f"{k}_last": v for k, v in list(sc.items())})
        res = evaluate(run, sc, checks)
        base = [k for k in sc if not k.startswith("sopt")]
        res["best_baseline"] = max(base, key=lambda k: _v(res["auroc"][k]["0.50"]))
        res["exec_gain"] = gain
        cache[name] = (run, sc, checks, res.pop("_thresholds"))
        out["runs"][name] = res
        print(f"{name:34s} n {res['n']} fail {res['fail_rate']:.2f} | AUROC@50% " +
              " ".join(f"{k} {_f(res['auroc'][k]['0.50'])}" for k in sc) +
              f" | best base {res['best_baseline']} | seq sopt_ee {res['seq']['sopt_ee']}", flush=True)
    # cross-run transfer: thresholds calibrated on run A applied to the test inits of run B
    names = list(cache)
    out["transfer"] = {}
    for a in names:
        for b in names:
            if a == b:
                continue
            rb, scb, chb, _ = cache[b]
            th = cache[a][3]
            ks = ("sopt_ee", "constvel", "chunkcons")
            r = evaluate(rb, {k: scb[k] for k in ks}, chb, thresholds={k: th[k] for k in ks})
            out["transfer"][f"{a}->{b}"] = r["seq"]
    prim = args.primary or [n for n in names if cache[n][0]["suite"] in ("libero90", "libero90_17")]
    if prim:
        mean = lambda k: float(np.mean([_v(out["runs"][n]["auroc"][k]["0.50"]) for n in prim]))  # noqa: E731
        own = {n: out["runs"][n]["seq"]["sopt_ee"]["tpr"] for n in prim}
        pairs = [(b, v["sopt_ee"]) for ab, v in out["transfer"].items() for a, b in [ab.split("->")]
                 if a in prim and b in prim and v["sopt_ee"]["tpr"] is not None and own[b] is not None]
        drops = [own[b] - v["tpr"] for b, v in pairs]
        fprs = [v["fpr"] for _, v in pairs if v["fpr"] is not None]
        base_keys = [k for k in out["runs"][prim[0]]["auroc"] if not k.startswith("sopt")]
        best = max(base_keys, key=mean)
        items = [cache[n][:3] for n in prim]
        pr = out["primary"] = {
            "runs": prim, "sopt_ee": mean("sopt_ee"), "best_baseline": best, "best_value": mean(best),
            "boot_sopt_ee_minus_best": bootstrap_mean_diff(items, "sopt_ee", best),
            "boot_sopt_ee_minus_rand": bootstrap_mean_diff(items, "sopt_ee", "rand_ee"),
            "mean_transfer_tpr_drop": float(np.mean(drops)) if drops else None,
            "mean_transfer_fpr": float(np.mean(fprs)) if fprs else None}
        pr["pass"] = bool(pr["sopt_ee"] >= 0.75 and pr["sopt_ee"] - pr["best_value"] >= 0.05
                          and pr["boot_sopt_ee_minus_best"][1] > 0
                          and pr["mean_transfer_tpr_drop"] is not None and pr["mean_transfer_tpr_drop"] <= 0.05
                          and pr["mean_transfer_fpr"] <= 0.1 + 0.05)
        print("PRIMARY", json.dumps(pr))
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
