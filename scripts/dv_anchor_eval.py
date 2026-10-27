#!/usr/bin/env python
"""Data-validation forecasts (DV1 history length, DV3 goal decomposition) on one fixed set of anchors.

`evaluate.py` builds its windows from each checkpoint's own ctx_len (stride = ctx_len, history >= ctx_len), so
models with different context lengths would be scored on different windows. Here every window has `--anchor-ctx`
frames of real history and `H` real future frames; a model with context C reads the last C history frames.
All models therefore predict the same futures from the same anchor frames, with the same sampling noise per batch
(common random numbers), and per-window results are saved for an episode-cluster bootstrap.

  python scripts/dv_anchor_eval.py --split test --out outputs/dv1/test.npz \
      --ckpt ctx4=outputs/dv1_ctx4/best.pt ctx96=outputs/e2_ar_flow_f100_s/best.pt [--goal-modes none endpoint]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from sopt.data.dataset import TrajectoryStore, WindowDataset
from sopt.data.features import GRIPPER, POS
from sopt.eval.forecast import (constant_velocity, energy_score_pos, energy_score_pos_by_step, interpolate_to,
                                zero_velocity)
from sopt.train.pretrain import load_checkpoint
from sopt.utils.config import DEFAULT_CONFIG


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True, help="name=path")
    p.add_argument("--split", default="test", choices=["train", "val", "id_val", "test"])
    p.add_argument("--anchor-ctx", type=int, default=96, help="real history frames required before each anchor")
    p.add_argument("--stride", type=int, default=32, help="frames between anchors within an episode")
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-windows", type=int, default=0, help="0 = all windows")
    p.add_argument("--goal-modes", nargs="+", default=["none"], help="none / endpoint / keyframe (goal_cond models)")
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        m, ck = load_checkpoint(path, device)
        m.eval()
        models[name] = (m, ck, path)
    Hs = {m.H for m, _, _ in models.values()}
    assert len(Hs) == 1, f"all models must share the horizon, got {Hs}"
    H = Hs.pop()
    for name, (m, ck, _) in models.items():
        assert m.cfg.ctx_len <= args.anchor_ctx, f"{name}: ctx_len {m.cfg.ctx_len} > anchor_ctx {args.anchor_ctx}"
        if any(g != "none" for g in args.goal_modes):
            assert m.goal_cond, f"{name} is not goal-conditioned"

    # Split definition: all checkpoints must agree on it (same sites / salt), else the held-out set is ambiguous.
    dcfgs = [OmegaConf.merge(OmegaConf.load(DEFAULT_CONFIG).data, ck["config"]["data"]) for _, ck, _ in models.values()]
    keys = ("val_sites", "test_sites", "id_val_frac", "split_salt")
    for d in dcfgs[1:]:
        assert all(OmegaConf.to_container(d)[k] == OmegaConf.to_container(dcfgs[0])[k] for k in keys), "split mismatch"
    dcfg = dcfgs[0]
    store = TrajectoryStore(args.data_root or dcfg.root)
    eps = store.split_episodes(dcfg, args.split)
    A = args.anchor_ctx
    ne = store.next_event if "keyframe" in args.goal_modes else None
    ds = WindowDataset(store.features, eps, A + H, stride=args.stride, next_event=ne, left_pad=0)
    idx = np.arange(len(ds))
    if args.max_windows and len(idx) > args.max_windows:
        idx = np.sort(np.random.default_rng(0).choice(idx, args.max_windows, replace=False))
    sub = torch.utils.data.Subset(ds, idx.tolist())
    dl = DataLoader(sub, args.batch_size, shuffle=False, num_workers=8)

    rec = {"episode": [], "t": []}
    names = []
    for name, (m, _, _) in models.items():
        names += [f"{name}@{g}" if g != "none" else name for g in args.goal_modes]
    names += ["const_vel", "zero_vel"] + (["interp@endpoint"] if "endpoint" in args.goal_modes else [])
    out = {n: {"es": [], "es_step": [], "err_step": [], "grip": []} for n in names}
    ep_index = eps["episode_index"].to_numpy()
    seen = 0
    for b, batch in enumerate(dl):
        x, mask = batch["x"].to(device), batch["mask"].to(device)
        full = mask.all(1)
        sel = idx[seen : seen + len(x)]
        seen += len(x)
        full_np = full.cpu().numpy()
        x = x[full]
        if len(x) == 0:
            continue
        rec["episode"].append(ep_index[ds.item_ep[sel[full_np]]])
        rec["t"].append(ds.item_t[sel[full_np]])
        hist, fut = x[:, :A], x[:, A : A + H]
        goals = {"endpoint": x[:, A - 1 + H]}
        if ne is not None:
            goals["keyframe"] = batch["kf"].to(device)[full][:, A - 1]

        def score(n, pred):
            pred = pred.float()
            stoch = pred.shape[1] > 1 and not torch.equal(pred[:, 0], pred[:, 1])
            pr = pred if stoch else pred[:, :1]
            out[n]["es"].append(energy_score_pos(pr, fut).cpu().numpy())
            out[n]["es_step"].append(energy_score_pos_by_step(pr, fut).cpu().numpy().astype(np.float32))
            out[n]["err_step"].append(((pr.mean(1)[..., POS] - fut[..., POS]).norm(dim=-1) * 100).cpu().numpy())
            out[n]["grip"].append((pr[..., GRIPPER] - fut[:, None, :, GRIPPER]).abs().mean((1, 2, 3)).cpu().numpy())

        with torch.no_grad():
            for name, (m, _, _) in models.items():
                ctx = hist[:, A - m.cfg.ctx_len :]
                for g in args.goal_modes:
                    torch.manual_seed(1000 + b)  # same flow noise for every model / mode on this batch
                    if g == "none":
                        pred = m.forecast(ctx, args.num_samples)
                    else:
                        pred = m.forecast(ctx, args.num_samples, goal=goals[g], goal_type=g)
                    score(f"{name}@{g}" if g != "none" else name, pred)
            score("const_vel", constant_velocity(hist, H))
            score("zero_vel", zero_velocity(hist, H))
            if "endpoint" in args.goal_modes:
                score("interp@endpoint", interpolate_to(hist, goals["endpoint"], H))
        if b % 20 == 0:
            print(f"batch {b} windows {sum(len(e) for e in rec['episode'])}", flush=True)

    arrays = {"episode": np.concatenate(rec["episode"]), "t": np.concatenate(rec["t"])}
    for n, d in out.items():
        for k, v in d.items():
            arrays[f"{n}/{k}"] = np.concatenate(v)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    summary = {
        "split": args.split, "anchor_ctx": A, "horizon": H, "stride": args.stride, "num_samples": args.num_samples,
        "goal_modes": args.goal_modes, "windows": int(len(arrays["episode"])),
        "episodes": int(len(np.unique(arrays["episode"]))), "split_episodes": int(len(eps)),
        "models": {k: {"path": p, "ctx_len": int(m.cfg.ctx_len), "step": int(ck["step"]), "seed": ck["config"].get("seed")}
                   for k, (m, ck, p) in models.items()},
        "mean_es": {n: float(arrays[f"{n}/es"].mean()) for n in names},
    }
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
