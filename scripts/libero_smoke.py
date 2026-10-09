#!/usr/bin/env python
"""Check the LIBERO sim on this machine: EGL rendering, joint readout, image orientation vs the dataset.

  python scripts/libero_smoke.py --sopt-data $SOPT_DATA --out /tmp/libero_smoke
Writes sim_agent.png / sim_wrist.png and data_agent.png / data_wrist.png (first frame of a demo of the same task).
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sopt-data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--task", default=None, help="task language; default: first libero_90 task")
    args = p.parse_args()

    from sopt.sim.libero_env import LiberoTaskEnv, setup_env_vars, suite_tasks

    setup_env_vars(args.sopt_data)
    import av
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from PIL import Image

    from sopt.data.libero import HF_REPO, HF_REVISION
    from sopt.utils.franka_fk import franka_fk

    tasks = suite_tasks("libero_90")
    lang = args.task or tasks[0]
    tid = tasks.index(lang)
    env = LiberoTaskEnv("libero_90", tid)
    o = env.reset(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(o["agent"]).save(out / "sim_agent.png")
    Image.fromarray(o["wrist"]).save(out / "sim_wrist.png")
    flange = franka_fk(o["q"])[:3, 3]
    ok = 0
    for _ in range(20):
        o, done = env.step(np.zeros(7) + np.array([0, 0, 0, 0, 0, 0, -1.0]))
        ok += 1

    eps = pd.read_parquet(Path(args.sopt_data) / "processed/libero90/episodes.parquet")
    ep = int(eps[eps["task"].str.lower() == lang.lower()]["episode_index"].iloc[0])
    chunk = ep // 1000
    for key, name in [("observation.images.image", "agent"), ("observation.images.wrist_image", "wrist")]:
        f = hf_hub_download(HF_REPO, f"videos/chunk-{chunk:03d}/{key}/episode_{ep:06d}.mp4", repo_type="dataset",
                            revision=HF_REVISION)
        with av.open(f) as c:
            frame = next(c.decode(video=0)).to_ndarray(format="rgb24")
        Image.fromarray(frame).save(out / f"data_{name}.png")
    print(json.dumps({"task": lang, "task_id": tid, "episode": ep, "q": o["q"].round(3).tolist(),
                      "fingers": o["fingers"].round(4).tolist(), "flange_xyz": flange.round(3).tolist(),
                      "steps_ok": ok, "n_libero90_tasks": len(tasks),
                      "dataset_tasks_found": int(eps["task"].str.lower().isin([t.lower() for t in tasks]).sum())}))


if __name__ == "__main__":
    main()
