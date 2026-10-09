#!/usr/bin/env python
"""Closed-loop LIBERO-90 success rate of an E4b policy (one worker process per task).

  python scripts/e4b_eval.py --policy outputs/e4b_prior_droid_k5/final.pt --idm outputs/e4b_idm/final.pt \
      --n-init 20 --out outputs/e4b_prior_droid_k5/rollouts.json
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path


def worker(args_tuple):
    policy_path, idm_path, sopt_data, task_lang, task_id, inits, replan, max_steps, gpu = args_tuple
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    import torch

    from sopt.policy.rollout import load_idm, load_policy, run_episode
    from sopt.sim.libero_env import LiberoTaskEnv, setup_env_vars, suite_tasks

    setup_env_vars(sopt_data)
    torch.set_num_threads(2)
    device = torch.device("cuda")
    policy, ck = load_policy(policy_path, device)
    idm = load_idm(idm_path, device) if ck["kind"] == "prior" else None
    ctx_len = ck["prior_config"]["ctx_len"] if ck["kind"] == "prior" else 96
    env = LiberoTaskEnv("libero_90", suite_tasks("libero_90").index(task_lang))
    res = []
    for i in inits:
        torch.manual_seed(1000 * task_id + i)
        r = run_episode(env, i, policy, ck["kind"], idm, task_id, device, replan, max_steps, ctx_len, seed=i)
        res.append({"task": task_lang, "init": i, **r})
    env.close()
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--idm", required=True)
    p.add_argument("--sopt-data", default=os.environ.get("SOPT_DATA"))
    p.add_argument("--n-init", type=int, default=20)
    p.add_argument("--replan", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--gpu", default="0")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    import torch

    tasks = torch.load(args.policy, map_location="cpu", weights_only=False)["tasks"]
    jobs = [(args.policy, args.idm, args.sopt_data, t, i, list(range(args.n_init)), args.replan, args.max_steps,
             args.gpu) for i, t in enumerate(tasks)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(len(jobs)) as pool:
        rows = [r for rs in pool.map(worker, jobs) for r in rs]
    per_task = {t: sum(r["success"] for r in rows if r["task"] == t) / args.n_init for t in tasks}
    summary = {"policy": args.policy, "success_rate": sum(r["success"] for r in rows) / len(rows),
               "per_task": per_task, "n_episodes": len(rows), "wall_s": time.time() - t0, "episodes": rows}
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
