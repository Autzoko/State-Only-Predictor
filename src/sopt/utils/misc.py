from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_rev() -> str:
    try:
        root = Path(__file__).resolve().parents[3]
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], cwd=root) != 0
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


class JsonlLogger:
    """Append-only metrics log; optionally mirrors to wandb when `use_wandb` and installed."""

    def __init__(self, path: str | os.PathLike, use_wandb: bool = False, wandb_kwargs: dict | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.wandb = None
        if use_wandb:
            import wandb

            self.wandb = wandb.init(**(wandb_kwargs or {}))

    def log(self, step: int, **metrics) -> None:
        rec = {"step": step, **{k: float(v) for k, v in metrics.items()}}
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if self.wandb is not None:
            self.wandb.log(metrics, step=step)
