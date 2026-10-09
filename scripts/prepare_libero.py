#!/usr/bin/env python
"""Build LIBERO-90 states in the DROID feature convention (reads joint/gripper columns only, no images).

  python scripts/prepare_libero.py --out $SOPT_DATA/processed/libero90 --workers 16
"""

import argparse
import json

from sopt.data.dataset import TrajectoryStore
from sopt.data.libero import build


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-files", type=int, default=None)
    args = p.parse_args()
    eps = build(args.out, args.workers, max_files=args.max_files)
    store = TrajectoryStore(args.out)
    store.next_event  # noqa: B018  (build and cache keyframe goals)
    summary = {
        "episodes": len(eps), "frames": int(eps["length"].sum()), "tasks": int(eps["task_index"].nunique()),
        "mean_length_15hz": float(eps["length"].mean()),
        "split_episodes": eps["split"].value_counts().to_dict(),
        "split_tasks": eps.groupby("split")["task_index"].nunique().to_dict(),
        "episodes_per_task_train": float(eps[eps["split"] == "train"].groupby("task_index").size().mean()),
    }
    print(json.dumps(summary, indent=2))
    with open(f"{args.out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
