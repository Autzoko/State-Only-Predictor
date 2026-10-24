#!/usr/bin/env python
"""E36 sanity: does the frozen DROID EE prior actually forecast LIBERO sim motion better than a random-init prior and a
constant-velocity extrapolation, on the exact inputs e36_failure.py builds? Mean expected error E||X - Y|| (scaled units),
then position error per horizon step (cm). CPU is enough (60 random episodes, every 4th check).
  PYTHONPATH=src:scripts python scripts/e36_sanity.py --prior outputs/e13_ar_flow_f100_s_ee/best.pt \
      --npz outputs/e21_dp_none_k5_t17_s0/e36_clean_n50.npz
"""
import argparse

import numpy as np
import torch

import e36_failure as E

ap = argparse.ArgumentParser()
ap.add_argument("--prior", required=True)
ap.add_argument("--npz", required=True)
args = ap.parse_args()
torch.manual_seed(0)
p, _ = E.load_checkpoint(args.prior, torch.device("cpu"))
torch.manual_seed(0)
r = E.StatePrior(p.cfg, {k: v.cpu().numpy() for k, v in p.normalizer.state_dict().items()})
p.eval(); r.eval()
run = E.load_run(args.npz)
rng = np.random.default_rng(0)
eps = [run["eps"][i] for i in rng.choice(len(run["eps"]), 60, replace=False)]
feats = [E.episode_features(e, True) for e in eps]
checks = [np.arange(E.T0, len(e["act"]) + 1, E.CHECK)[::4] for e in eps]
scale = p.normalizer.chunk_scale(p.H)[:E.M].cpu().numpy()[:, E.DIMS]


@torch.no_grad()
def err(prior, K=8):
    C = prior.cfg.ctx_len
    ctxs, ys = [], []
    for F, ts in zip(feats, checks):
        for t in ts:
            s = t - E.H_STEPS
            ctxs.append(E.interp_frames(F[: s + 1], E.context_indices(s, C)))
            ys.append(E.interp_frames(F[: t + 1], s + np.arange(1, E.M + 1) * E.SIM_HZ / E.PRIOR_HZ)[:, E.DIMS])
    X = prior.forecast(torch.from_numpy(np.stack(ctxs).astype(np.float32)), K)[:, :, :E.M].numpy()[..., E.DIMS] / scale
    Y = np.stack(ys) / scale
    e = np.linalg.norm((X - Y[:, None]).reshape(len(X), K, -1), axis=-1).mean(1)
    zero = np.linalg.norm((Y - np.stack(ctxs)[:, -1:, E.DIMS] / scale).reshape(len(Y), -1), axis=-1)  # "stay put"
    return e, zero


es, zero = err(p)
er, _ = err(r)
cv = np.concatenate([E.constvel_scores(F, ts, scale) for F, ts in zip(feats, checks)])
print(f"n checks {len(es)} | mean error: sopt_ee {es.mean():.2f}  rand {er.mean():.2f}  constvel {cv.mean():.2f}  "
      f"hold-last {zero.mean():.2f}")
print(f"median: sopt_ee {np.median(es):.2f} rand {np.median(er):.2f} constvel {np.median(cv):.2f} hold {np.median(zero):.2f}")
print(f"frac checks sopt < constvel {np.mean(es < cv):.2f}; sopt < hold {np.mean(es < zero):.2f}")


@torch.no_grad()
def per_step(prior, H, K=8):
    """cm position error per horizon step (1..H) for prior / constvel / hold."""
    C = prior.cfg.ctx_len
    P, CV, HO = [], [], []
    for e, F, ts in zip(eps, feats, checks):
        for t in ts:
            s = t - E.H_STEPS
            if s + H * E.SIM_HZ / E.PRIOR_HZ > len(F) - 1:
                continue
            ctx = E.interp_frames(F[: s + 1], E.context_indices(s, C))
            y = E.interp_frames(F, s + np.arange(1, H + 1) * E.SIM_HZ / E.PRIOR_HZ)[:, E.POS]
            x = prior.forecast(torch.from_numpy(ctx[None].astype(np.float32)), K)[0, :, :H].numpy()[..., E.POS]
            x0 = E.interp_frames(F[: s + 1], np.array([s - E.SIM_HZ / E.PRIOR_HZ, float(s)]))[:, E.POS]
            cv = x0[1] + np.arange(1, H + 1)[:, None] * (x0[1] - x0[0])
            P.append(100 * np.linalg.norm(x - y[None], axis=-1).mean(0))
            CV.append(100 * np.linalg.norm(cv - y, axis=-1))
            HO.append(100 * np.linalg.norm(x0[1][None] - y, axis=-1))
    P, CV, HO = map(np.array, (P, CV, HO))
    for k in (1, 3, 8, 15, 32):
        if k <= H:
            print(f"  step {k:2d} ({k / 15:.2f}s): prior {P[:, k - 1].mean():.2f} cm  constvel {CV[:, k - 1].mean():.2f}  hold {HO[:, k - 1].mean():.2f}  (n {len(P)})")


print("EE prior e13, per horizon step:")
per_step(p, 32)
