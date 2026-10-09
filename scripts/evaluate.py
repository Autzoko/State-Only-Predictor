#!/usr/bin/env python
"""Forecasting metrics of a checkpoint on a held-out split (default: val; use `test` only for final numbers).

  python scripts/evaluate.py --ckpt outputs/e1_ar_flow_s/best.pt --split val
"""

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from sopt.data.dataset import TrajectoryStore, WindowDataset
from sopt.eval.forecast import evaluate_forecast
from sopt.train.pretrain import goal_modes, load_checkpoint
from sopt.utils.config import DEFAULT_CONFIG


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="val", choices=["val", "id_val", "test"])
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--data-root", default=None, help="evaluate on another processed dataset (zero-shot)")
    p.add_argument("--data-config", default=None,
                   help="yaml whose `data` section (split / padding / stride) replaces the checkpoint's, "
                        "e.g. configs/data/libero90.yaml for zero-shot DROID -> LIBERO")
    p.add_argument("--tag", default="", help="suffix of the output json name")
    p.add_argument("--goal-modes", nargs="*", default=None,
                   help="none / endpoint / keyframe; default: all modes the model was trained for. For an "
                        "unconditional model, 'endpoint' adds the interp@endpoint baseline on the same windows.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.ckpt, device)
    cfg = model.cfg
    dcfg = OmegaConf.merge(OmegaConf.load(DEFAULT_CONFIG).data, ckpt["config"]["data"])
    if args.data_config:
        dcfg = OmegaConf.merge(dcfg, OmegaConf.load(args.data_config).data)
    store = TrajectoryStore(args.data_root or dcfg.root)
    eps = store.split_episodes(dcfg, args.split)
    torch.manual_seed(0)
    modes = tuple(args.goal_modes) if args.goal_modes else goal_modes(cfg)
    ne = store.next_event if "keyframe" in modes else None
    ds = WindowDataset(store.features, eps, model.window, stride=dcfg.get("eval_stride") or cfg.ctx_len,
                       next_event=ne, left_pad=min(dcfg.get("left_pad", 0), cfg.ctx_len - cfg.patch))
    dl = DataLoader(ds, ckpt["config"]["eval"]["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(0))
    ev = ckpt["config"]["eval"]
    res = evaluate_forecast(
        model, dl, device, args.num_samples or ev["num_samples"], args.max_batches or ev["max_batches"], modes
    )
    res.update(goal_modes=list(modes), split=args.split, step=ckpt["step"], episodes=len(eps), ckpt=str(args.ckpt))
    print(json.dumps(res, indent=2))
    out = Path(args.ckpt).parent / f"eval_{args.split}{args.tag}.json"
    out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
