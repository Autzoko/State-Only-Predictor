#!/usr/bin/env python
"""Build the DROID state dataset: only state columns + episode metadata are stored, videos are never fetched.

  # stream from the HF hub chunk by chunk (raw parquet is deleted right after each chunk; resumable)
  python scripts/prepare_droid.py --out ~/langtian/SOPT_DATA/processed/droid --workers 32
  # local pipeline check: 3 chunks x 10 episodes
  python scripts/prepare_droid.py --out data/processed/droid_debug --chunks 0 30 60 --max-per-chunk 10
  # from an existing local copy
  python scripts/prepare_droid.py --src /path/to/droid_1.0.1 --out data/processed/droid
"""

import argparse
import json

from omegaconf import OmegaConf

from sopt.data.dataset import TrajectoryStore
from sopt.data.droid import build_from_hub, build_from_local
from sopt.data.splits import assign_splits
from sopt.utils.config import DEFAULT_CONFIG


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--src", default=None, help="local LeRobot copy; default: stream from the HF hub")
    p.add_argument("--chunks", type=int, nargs="*", default=None)
    p.add_argument("--max-per-chunk", type=int, default=None)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    if args.src:
        episodes = build_from_local(args.src, args.out, args.workers)
    else:
        episodes = build_from_hub(args.out, args.chunks, args.workers, args.max_per_chunk)
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
