#!/usr/bin/env python
"""E16: how well does the DROID-pretrained SOPT prior (zero-shot, no LIBERO data) forecast each evaluation domain?

Diagnostic for the suite dependence of the downstream SOPT gain (E7 / E13): windows are built exactly like the
policy inputs (96-frame 15 Hz history interpolated from the 20 Hz demos, clamped at the episode start) from the
training demos of each domain, and the prior's 16-step forecast is scored against the constant-velocity baseline.
Ratio < 1 = the prior knows something about this domain's motion beyond kinematics. Demos only; no rollouts.

  python scripts/e16_prior_fit.py --prior outputs/e13_ar_flow_f100_s_ee/best.pt --out outputs/e16_prior_fit.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from sopt.eval.forecast import ForecastMeter, constant_velocity, zero_velocity
from sopt.policy.data import Raw20, context_indices, future_indices, interp_frames, select_episodes
from sopt.train.pretrain import load_checkpoint
from sopt.utils.config import REPO_ROOT

DOMAINS = ["libero90_e4b", "libero_spatial", "libero_object", "libero_goal", "libero_10"]


def load_domain(name: str, sopt_data: Path):
    if name == "libero90_e4b":
        raw = Raw20(sopt_data / "processed/libero90_raw20")
        tasks = json.loads((REPO_ROOT / "configs/e4b_tasks.json").read_text())
        return raw, select_episodes(raw.eps[raw.eps["split"] == "test"], tasks, None)
    from sopt.data.libero_std import StdRaw

    raw = StdRaw(sopt_data / "processed/libero_std" / name)
    return raw, select_episodes(raw.eps, raw.tasks(), None)


@torch.no_grad()
def score(model, raw, sel, stride: int, samples: int, bs: int, device) -> dict:
    C, H = model.cfg.ctx_len, model.H
    ctx, fut = [], []
    for row in sel.itertuples():
        F = raw.episode(row)[0]
        last = len(F) - 1 - future_indices(0, H)[-1]  # future must stay inside the episode
        for a in range(0, int(last) + 1, stride):
            ctx.append(interp_frames(F, context_indices(a, C)))
            fut.append(interp_frames(F, future_indices(a, H)))
    ctx, fut = torch.tensor(np.stack(ctx), dtype=torch.float32), torch.tensor(np.stack(fut), dtype=torch.float32)
    meter = ForecastMeter()
    for i in range(0, len(ctx), bs):
        c, f = ctx[i : i + bs].to(device), fut[i : i + bs].to(device)
        meter.update("model", model.forecast(c, samples), f)
        meter.update("const_vel", constant_velocity(c, H), f)
        meter.update("zero_vel", zero_velocity(c, H), f)
        meter.step(len(c))
    r = {k: v for k, v in meter.result().items() if not isinstance(v, list)}
    r["windows"], r["episodes"] = len(ctx), len(sel)
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prior", default="outputs/e13_ar_flow_f100_s_ee/best.pt")
    p.add_argument("--sopt-data", default=os.environ.get("SOPT_DATA"))
    p.add_argument("--domains", nargs="*", default=DOMAINS)
    p.add_argument("--stride", type=int, default=10, help="anchor stride in 20 Hz frames")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(REPO_ROOT / args.prior, device)
    model.eval()
    res = {"prior": args.prior, "stride": args.stride, "samples": args.samples, "domains": {}}
    for d in args.domains:
        raw, sel = load_domain(d, Path(args.sopt_data))
        r = res["domains"][d] = score(model, raw, sel, args.stride, args.samples, args.bs, device)
        es, cv = r["model/energy_pos_cm"], r["const_vel/energy_pos_cm"]
        print(f"{d:16s} eps {r['episodes']:4d} win {r['windows']:6d}  ES model {es:.2f} cm  const_vel {cv:.2f} cm  "
              f"ratio {es / cv:.3f}", flush=True)
    Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
