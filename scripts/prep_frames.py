#!/usr/bin/env python
"""E39 / E40: decode the LIBERO-90 camera videos of every demo the new runs need into the frame cache, once, on a CPU
node with network access (training jobs run with HF_HUB_OFFLINE=1 and would otherwise fail on a missing episode):
  17 evaluated tasks, all demos (E39 k = all)  +  every train-split task demo (E40 host pretraining).
Selection is exactly vla_train.py's (select_episodes over the same pools), so the cache covers each run's episodes.

  HF_HUB_OFFLINE=0 python scripts/prep_frames.py --workers 16
"""

import argparse
import json
import os
from pathlib import Path

from sopt.policy.data import Raw20, build_frame_cache, select_episodes
from sopt.utils.config import REPO_ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--sopt-data", default=os.environ.get("SOPT_DATA"))
    args = p.parse_args()
    root = Path(args.sopt_data)
    raw = Raw20(root / "processed/libero90_raw20")
    frames = root / "processed/libero90_frames256"
    t17 = json.loads((REPO_ROOT / "configs/libero90_17_tasks.json").read_text())
    ev = select_episodes(raw.eps[raw.eps["split"].isin(["test", "val"])], t17, None)
    pool = raw.eps[raw.eps["split"] == "train"]
    tr = select_episodes(pool, sorted(pool["task"].unique()), None)
    assert not set(t17) & set(tr["task"]), "evaluated task in the train split"
    for name, sel in (("17 tasks, all demos", ev), ("train split, all demos", tr)):
        have = sum((frames / f"ep{int(e):06d}.npy").exists() for e in sel["episode_index"])
        print(f"{name}: {len(sel)} episodes, {sel['task'].nunique()} tasks, cached before {have}", flush=True)
        build_frame_cache(sel, frames, workers=args.workers, size=256)
        have = sum((frames / f"ep{int(e):06d}.npy").exists() for e in sel["episode_index"])
        print(f"{name}: cached after {have} / {len(sel)}", flush=True)
        assert have == len(sel)


if __name__ == "__main__":
    main()
