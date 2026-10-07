#!/usr/bin/env python
"""Build the DROID state dataset. Only state + episode-metadata columns are fetched (parquet column projection
over HTTP, ~1.4 GB); videos and other columns are never downloaded. Resumable.

  python scripts/prepare_droid.py --out ~/langtian/SOPT_DATA/processed/droid --workers 16
  python scripts/prepare_droid.py --out data/processed/droid_debug --max-files 2      # local pipeline check
  python scripts/prepare_droid.py --src /path/to/droid_1.0.1_v30 --out data/processed/droid   # local copy
"""

import argparse
import json

from omegaconf import OmegaConf

from sopt.data.dataset import TrajectoryStore
from sopt.data.droid import build
from sopt.data.splits import assign_splits
from sopt.utils.config import DEFAULT_CONFIG


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--src", default=None, help="local LeRobot v3.0 copy; default: read from the HF hub")
    p.add_argument("--max-files", type=int, default=None, help="debug: only the first N data files")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    episodes = build(args.out, args.src, args.workers, args.max_files)
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
