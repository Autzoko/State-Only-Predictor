#!/usr/bin/env python
"""Leakage audit: do LIBERO's fixed evaluation init states coincide with the initial scenes of training demos?

For each E4b task, render every benchmark init state (after the same settle steps as evaluation) and compare
its agent-view image with the first frame of every demo of that task (frame cache, 128 px). An init state that
reproduces a demo's scene has a nearest-demo distance far below the typical demo-to-demo distance.

  python scripts/check_init_leakage.py --out outputs/init_leakage.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from sopt.policy.data import Raw20, select_episodes
from sopt.sim.libero_env import LiberoTaskEnv, setup_env_vars, suite_tasks
from sopt.utils.config import REPO_ROOT


def small(a: np.ndarray, size: int = 64) -> np.ndarray:
    return np.asarray(Image.fromarray(a).resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sopt-data", default=os.environ.get("SOPT_DATA"))
    p.add_argument("--frames", default="processed/libero90_frames")
    p.add_argument("--k-train", type=int, default=5, help="report which init states match the k selected demos")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    sopt_data = Path(args.sopt_data)
    setup_env_vars(sopt_data)
    raw = Raw20(sopt_data / "processed/libero90_raw20")
    tasks = json.loads((REPO_ROOT / "configs/e4b_tasks.json").read_text())
    test = raw.eps[raw.eps["split"] == "test"]
    all_sel, k_sel = select_episodes(test, tasks, None), select_episodes(test, tasks, args.k_train)
    report = {}
    for t_i, lang in enumerate(tasks):
        eps = all_sel[all_sel["task"] == lang]
        firsts, ids = [], []
        for e in eps["episode_index"]:
            f = Path(sopt_data / args.frames / f"ep{int(e):06d}.npy")
            if f.exists():
                firsts.append(small(np.load(f, mmap_mode="r")[0, 0])), ids.append(int(e))
        D = np.stack(firsts)  # (n_demo, s, s, 3)
        env = LiberoTaskEnv("libero_90", suite_tasks("libero_90").index(lang))
        inits = []
        for i in range(len(env.init_states)):
            inits.append(small(env.reset(i, seed=i)["agent"]))
        env.close()
        I = np.stack(inits)
        d = np.sqrt(((I[:, None] - D[None]) ** 2).mean((2, 3, 4)))  # (n_init, n_demo) RMS
        dd = np.sqrt(((D[:, None] - D[None]) ** 2).mean((2, 3, 4)))
        dd = dd[~np.eye(len(D), dtype=bool)]
        nn_d, nn_j = d.min(1), d.argmin(1)
        ratio = nn_d / np.sort(d, 1)[:, 1]  # a copied scene: nearest << second nearest (ratio near 0)
        k_ids = set(k_sel[k_sel["task"] == lang]["episode_index"].astype(int))
        report[lang] = {
            "n_demos": len(ids), "n_inits": len(I),
            "init_to_nearest_demo": {"min": float(nn_d.min()), "median": float(np.median(nn_d)), "max": float(nn_d.max())},
            "demo_to_demo": {"min": float(dd.min()), "p5": float(np.percentile(dd, 5)), "median": float(np.median(dd))},
            "nearest_demo_of_eval_inits_0_19": [ids[j] for j in nn_j[:20]],
            "eval_inits_whose_nearest_demo_is_in_k_train": [i for i in range(20) if ids[nn_j[i]] in k_ids],
            "nn_over_2nd_ratio": {"min": float(ratio.min()), "median": float(np.median(ratio))},
            "nn_dist_eval_inits_0_19": [round(float(x), 4) for x in nn_d[:20]],
        }
        print(lang, json.dumps(report[lang]["init_to_nearest_demo"]), json.dumps(report[lang]["demo_to_demo"]),
              json.dumps(report[lang]["nn_over_2nd_ratio"]), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
