#!/usr/bin/env python
"""E24: what does the SOPT feature handed to the hosts actually encode? Linear probes on frozen features.

The hosts receive one vector per decision step: the SOPT backbone's last-token feature of the 96-frame (6.4 s,
15 Hz) proprio history. Every host already sees the *current* frame, so the only new information SOPT can carry is
temporal. We probe that feature (and baselines) for temporal quantities that a single frame cannot give:

  vel        EE velocity over the last 0.2 s (m/s, 3)            acc     EE acceleration over 0.4 s (3)
  grip_vel   gripper change over the last 0.2 s (1)              t_grip  time since the last gripper event (s, <=6.4)
  t_still    time since the EE last moved > 1 cm/s (s, <=6.4)    prog    episode progress a / (n - 1)
  fut05/fut10  EE displacement over the next 0.5 / 1.0 s (3)     t_next  time to the next gripper event (s, <=3)
  task       which of the 17 tasks (accuracy; history -> task decodability = copycat-shortcut risk)

Feature sets: cur (normalized current frame, what hosts already have), hand (cur + velocity / acceleration /
1 s of gripper / 1-2 s EE displacement), raw (the whole normalized 96x17 history, flattened), rand (same SOPT
architecture, random init, 3 seeds), droid (DROID-pretrained prior), droid_ee (EE-only prior, E13/E15), and
ft:<run> (the SOPT encoder *after* downstream fine-tuning inside a host, read from its checkpoint).

Data: LIBERO-90 demos of the 17 main-table tasks (split test + val), windows built exactly like the policy inputs;
probes are fit on 70% of the episodes and scored on the other 30% (split by episode hash, stratified by task).
Ridge regression on standardized features, ridge alpha by 5-fold grouped CV on the training episodes. Diagnostic
only (no policy is trained or selected with it).

  python scripts/e24_probe.py --out outputs/e24_probe.json [--ft e7_vla_expert_droid_k5_s0 ...]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from sopt.data.features import GRIPPER, POS
from sopt.data.splits import unit_hash
from sopt.models.state_prior import StatePrior
from sopt.policy.data import Raw20, context_indices, interp_frames, select_episodes
from sopt.train.pretrain import load_checkpoint
from sopt.utils.config import REPO_ROOT

FPS = 20.0
CLIP_PAST, CLIP_NEXT = 6.4, 3.0
REG = ["vel", "acc", "grip_vel", "t_grip", "t_still", "prog", "fut05", "fut10", "t_next"]


def targets(F: np.ndarray, a: int) -> dict:
    p, g = F[:, POS], F[:, GRIPPER.start]
    i = lambda k: min(max(k, 0), len(F) - 1)  # noqa: E731
    c = g > 0.5
    ev = np.flatnonzero(c[1:] != c[:-1]) + 1
    past, nxt = ev[ev <= a], ev[ev > a]
    speed = np.linalg.norm(p - p[np.maximum(np.arange(len(F)) - 2, 0)], axis=1) * FPS / 2
    moving = np.flatnonzero(speed[: a + 1] > 0.01)
    return {
        "vel": (p[a] - p[i(a - 4)]) * FPS / 4,
        "acc": (p[a] - 2 * p[i(a - 4)] + p[i(a - 8)]) * (FPS / 4) ** 2,
        "grip_vel": np.array([(g[a] - g[i(a - 4)]) * FPS / 4]),
        "t_grip": np.array([min((a - past[-1]) / FPS if len(past) else a / FPS, CLIP_PAST)]),
        "t_still": np.array([min((a - moving[-1]) / FPS if len(moving) else a / FPS, CLIP_PAST)]),
        "prog": np.array([a / (len(F) - 1)]),
        "fut05": p[i(a + 10)] - p[a],
        "fut10": p[i(a + 20)] - p[a],
        "t_next": np.array([min((nxt[0] - a) / FPS if len(nxt) else CLIP_NEXT, CLIP_NEXT)]),
    }


def build(raw, sel, C: int, stride: int):
    ctx, Y, ep, task, split = [], {k: [] for k in REG}, [], [], []
    for row in sel.itertuples():
        F = raw.episode(row)[0]
        test = unit_hash(str(row.episode_index), "sopt-e24") < 0.3
        for a in range(0, len(F) - 20, stride):
            ctx.append(interp_frames(F, context_indices(a, C)))
            for k, v in targets(F, a).items():
                Y[k].append(v)
            ep.append(int(row.episode_index)), task.append(int(row.task_id)), split.append(test)
    return (np.stack(ctx).astype(np.float32), {k: np.stack(v).astype(np.float64) for k, v in Y.items()},
            np.array(ep), np.array(task), np.array(split))


def hand_features(ctx_n: np.ndarray) -> np.ndarray:
    """Hand-crafted history on the normalized 15 Hz grid: cur, 0.2 s velocity, acceleration, 1 s of gripper,
    EE displacement over 1 s and 2 s."""
    x, P = ctx_n, POS
    vel = x[:, -1, P] - x[:, -4, P]
    acc = x[:, -1, P] - 2 * x[:, -4, P] + x[:, -7, P]
    grip = x[:, -15:, GRIPPER.start]
    d1, d2 = x[:, -1, P] - x[:, -16, P], x[:, -1, P] - x[:, -31, P]
    return np.concatenate([x[:, -1], vel, acc, grip, d1, d2], axis=1)


@torch.no_grad()
def encode(model: StatePrior, ctx: np.ndarray, bs: int = 1024) -> np.ndarray:
    model.eval()
    out = [model.encode(torch.from_numpy(ctx[i : i + bs]))[:, -1].numpy() for i in range(0, len(ctx), bs)]
    return np.concatenate(out).astype(np.float64)


def load_finetuned(ref: StatePrior, path: Path) -> StatePrior:
    """The SOPT encoder inside a downstream checkpoint (keys '...prior.<name>'), on a copy of the reference."""
    ck = torch.load(path, map_location="cpu", weights_only=False)

    def find(d):
        if isinstance(d, dict):
            if any(isinstance(k, str) and "prior.backbone." in k for k in d):
                return d
            for v in d.values():
                r = find(v)
                if r is not None:
                    return r
        return None

    sd = find(ck)
    assert sd is not None, f"no SOPT prior weights in {path}"
    model = StatePrior(ref.cfg, {k: v.numpy() for k, v in ref.normalizer.state_dict().items()})
    own = model.state_dict()
    new = {k[k.index("prior.") + 6 :]: v for k, v in sd.items() if "prior." in k}
    new = {k: v.float() for k, v in new.items() if k in own}
    need = [k for k in own if k.startswith(("embed.", "backbone."))]
    assert all(k in new for k in need), sorted(set(need) - set(new))[:5]
    model.load_state_dict({**own, **new})
    return model


def ridge_fit(X, Y, alpha):
    """Closed-form ridge via the eigendecomposition of X'X (X, Y standardized, no intercept)."""
    lam, Q = np.linalg.eigh(X.T @ X)
    XtY = Q.T @ (X.T @ Y)
    return [Q @ (XtY / (lam + a)[:, None]) for a in np.atleast_1d(alpha)]


def probe(X, Y, train, groups, alphas, classify=False):
    mu, sd = X[train].mean(0), X[train].std(0) + 1e-6
    X = (X - mu) / sd
    if classify:
        Y = np.eye(Y.max() + 1)[Y]
    ym, ys = Y[train].mean(0), Y[train].std(0) + 1e-9
    Yn = (Y - ym) / ys
    tr = np.flatnonzero(train)
    ug = np.unique(groups[tr])
    fold = dict(zip(np.random.default_rng(0).permutation(ug), np.arange(len(ug)) % 5))
    fid = np.array([fold[g] for g in groups[tr]])
    cv = np.zeros(len(alphas))
    for f in range(5):
        a_, b_ = tr[fid != f], tr[fid == f]
        for j, W in enumerate(ridge_fit(X[a_], Yn[a_], alphas)):
            cv[j] += ((X[b_] @ W - Yn[b_]) ** 2).sum()
    best = alphas[int(np.argmin(cv))]
    W = ridge_fit(X[tr], Yn[tr], best)[0]
    te = np.flatnonzero(~train)
    pred = X[te] @ W
    if classify:
        return {"acc": float((pred.argmax(1) == Y[te].argmax(1)).mean()), "alpha": float(best)}
    r2 = 1 - ((pred - Yn[te]) ** 2).sum(0) / ((Yn[te] - Yn[te].mean(0)) ** 2).sum(0)
    return {"r2": float(r2.mean()), "r2_dims": r2.round(4).tolist(), "alpha": float(best)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prior", default="outputs/e2_ar_flow_f100_s/best.pt")
    p.add_argument("--prior-ee", default="outputs/e13_ar_flow_f100_s_ee/best.pt")
    p.add_argument("--ft", nargs="*", default=[], help="downstream runs whose fine-tuned SOPT encoder is probed")
    p.add_argument("--sopt-data", default=os.environ.get("SOPT_DATA"))
    p.add_argument("--stride", type=int, default=8, help="anchor stride in 20 Hz frames")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    ref, _ = load_checkpoint(REPO_ROOT / args.prior)
    raw = Raw20(Path(args.sopt_data) / "processed/libero90_raw20")
    tasks = json.loads((REPO_ROOT / "configs/libero90_17_tasks.json").read_text())
    sel = select_episodes(raw.eps[raw.eps["split"].isin(["test", "val"])], tasks, None)
    ctx, Y, ep, task, test = build(raw, sel, ref.cfg.ctx_len, args.stride)
    train = ~test
    print(f"episodes {len(sel)} windows {len(ctx)} (test {test.mean():.2f})", flush=True)
    ctx_n = ref.normalizer.norm(torch.from_numpy(ctx)).numpy().astype(np.float64)

    feats = {"cur": ctx_n[:, -1], "hand": hand_features(ctx_n), "raw": ctx_n.reshape(len(ctx_n), -1)}
    for s in range(3):
        torch.manual_seed(s)
        rnd = StatePrior(ref.cfg, {k: v.numpy() for k, v in ref.normalizer.state_dict().items()})
        feats[f"rand{s}"] = encode(rnd, ctx)
    feats["droid"] = encode(ref, ctx)
    if args.prior_ee and (REPO_ROOT / args.prior_ee).exists():
        feats["droid_ee"] = encode(load_checkpoint(REPO_ROOT / args.prior_ee)[0], ctx)
    for run in args.ft:
        feats[f"ft:{run}"] = encode(load_finetuned(ref, REPO_ROOT / "outputs" / run / "final.pt"), ctx)

    alphas = np.logspace(-2, 5, 8)
    res = {"prior": args.prior, "episodes": len(sel), "windows": len(ctx), "stride": args.stride, "probes": {}}
    for name, X in feats.items():
        r = {k: probe(X, Y[k], train, ep, alphas * len(X) / 1e4) for k in REG}
        r["task"] = probe(X, task, train, ep, alphas * len(X) / 1e4, classify=True)
        res["probes"][name] = r
        print(f"{name:34s} " + " ".join(f"{k}={r[k]['r2']:.3f}" for k in REG) + f" task={r['task']['acc']:.3f}",
              flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
