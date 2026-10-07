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
from sopt.train.pretrain import load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="val", choices=["val", "id_val", "test"])
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument("--data-root", default=None, help="evaluate on another processed dataset (zero-shot)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.ckpt, device)
    cfg = model.cfg
    data_cfg = ckpt["config"]["data"]
    store = TrajectoryStore(args.data_root or data_cfg["root"])
    eps = store.split_episodes(OmegaConf.create(data_cfg), args.split)
    torch.manual_seed(0)
    ds = WindowDataset(store.features, eps, model.window, stride=cfg.ctx_len)
    dl = DataLoader(ds, ckpt["config"]["eval"]["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(0))
    ev = ckpt["config"]["eval"]
    res = evaluate_forecast(model, dl, device, args.num_samples or ev["num_samples"], args.max_batches or ev["max_batches"])
    res.update(split=args.split, step=ckpt["step"], episodes=len(eps), ckpt=str(args.ckpt))
    print(json.dumps(res, indent=2))
    out = Path(args.ckpt).parent / f"eval_{args.split}.json"
    out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
