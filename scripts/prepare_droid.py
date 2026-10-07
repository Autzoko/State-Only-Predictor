#!/usr/bin/env python
"""Raw DROID parquet -> flat arrays + 17-d feature cache + split summary.

  python scripts/prepare_droid.py --src data/raw/droid_1.0.1 --out data/processed/droid --workers 32
"""

import argparse
import json

from omegaconf import OmegaConf

from sopt.data.dataset import TrajectoryStore
from sopt.data.droid import convert
from sopt.data.splits import assign_splits
from sopt.utils.config import DEFAULT_CONFIG


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    episodes = convert(args.src, args.out, args.workers, args.limit)
    TrajectoryStore(args.out)  # builds and caches features.npy
    d = OmegaConf.load(DEFAULT_CONFIG).data
    split = assign_splits(episodes, d.val_frac, d.test_frac, d.split_salt)
    summary = {
        "episodes": len(episodes),
        "frames": int(episodes["length"].sum()),
        "buildings": int(episodes["building"].nunique()),
        "success_rate": float(episodes["success"].mean()),
        "split_episodes": {s: int((split == s).sum()) for s in ("train", "val", "test")},
        "split_frames": {s: int(episodes["length"][split == s].sum()) for s in ("train", "val", "test")},
        "split_buildings": {s: int(episodes["building"][split == s].nunique()) for s in ("train", "val", "test")},
        "split_salt": d.split_salt,
    }
    print(json.dumps(summary, indent=2))
    with open(f"{args.out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
