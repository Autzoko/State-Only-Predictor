#!/usr/bin/env python
"""Closed-loop LIBERO-90 success of an E12 FastWAM policy (7 tasks in lockstep, one model copy).

  python scripts/wam_eval.py --ckpt outputs/e12_wam_none_k5_s0/final.pt --out outputs/e12_wam_none_k5_s0/rollouts_n50.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from sopt.data.libero import features_20hz
from sopt.policy.data import context_indices, interp_frames
from sopt.policy.rollout import trace_row
from sopt.sim.libero_env import dataset_to_sim_gripper
from sopt.sim.vec_env import TaskVecEnv
from sopt.vla.fastwam import PromptCache, build_fastwam, load_trainable, predict

CTX = 96


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-init", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--n-exec", type=int, default=10, help="actions executed per predicted chunk")
    p.add_argument("--denoise-steps", type=int, default=10)
    p.add_argument("--trace", type=int, default=0)
    p.add_argument("--sopt-data", default=os.environ.get("SOPT_DATA"))
    args = p.parse_args()
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("wam_train", Path(__file__).with_name("wam_train.py"))
    wt = importlib.util.module_from_spec(spec)
    sys.modules["wam_train"] = wt
    spec.loader.exec_module(wt)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    tasks, arm, chunk = ck["tasks"], ck["arm"], ck["meta"]["chunk"]
    policy, sopt = build_fastwam(wt.make_prior(arm, ck["meta"].get("prior")), arm, chunk=chunk)  # E38: run's prior
    model = policy.model
    load_trainable(model, sopt, ck["state"])
    model.eval()
    sopt.eval()
    dev = model.device
    a_mean, a_std = (torch.tensor(ck["action_stats"][k], dtype=torch.float32, device=dev) for k in ("mean", "std"))
    prompts = PromptCache(model, policy.config.prompt_template)
    venv = TaskVecEnv(args.sopt_data, tasks)
    rows, t0 = [], time.time()
    for init in range(args.n_init):
        torch.manual_seed(init)
        obs = venv.reset(init)
        hist = [[features_20hz(o["q"][None], o["fingers"][None])[0]] for o in obs]
        active = {i: True for i in range(len(tasks))}
        chunks, traces = {}, {i: [] for i in range(len(tasks))}
        for t in range(args.max_steps):
            live = [i for i in active if active[i]]
            if not live:
                break
            if t % args.n_exec == 0:
                for i in live:
                    ctx = interp_frames(np.stack(hist[i]), context_indices(len(hist[i]) - 1, CTX))
                    img = torch.from_numpy(np.stack([obs[i]["agent"], obs[i]["wrist"]])).permute(0, 3, 1, 2)[None]
                    act = predict(model, sopt, prompts, img, torch.from_numpy(ctx.astype(np.float32))[None],
                                  tasks[i], a_mean, a_std, chunk, args.denoise_steps, seed=1000 * init + i)
                    chunks[i] = act.cpu().numpy()
            actions = {}
            for i in live:
                a = chunks[i][t % args.n_exec]
                actions[i] = np.concatenate([np.clip(a[:6], -1, 1), dataset_to_sim_gripper(np.array([float(a[6] > 0.5)]))])
            if args.trace and t % args.trace == 0:
                for i in live:
                    traces[i].append(trace_row(t, obs[i], actions[i]))
            for i, (o, success) in venv.step(actions).items():
                obs[i] = o
                hist[i].append(features_20hz(o["q"][None], o["fingers"][None])[0])
                if success:
                    active[i] = False
                    rows.append({"task": tasks[i], "init": init, "success": True, "steps": t + 1,
                                 **({"trace": traces[i]} if args.trace else {})})
        rows += [{"task": tasks[i], "init": init, "success": False, "steps": args.max_steps,
                  **({"trace": traces[i]} if args.trace else {})} for i in active if active[i]]
        print(f"init {init}: running success {np.mean([r['success'] for r in rows]):.3f} ({time.time() - t0:.0f}s)",
              flush=True)
    venv.close()
    per_task = {t: float(np.mean([r["success"] for r in rows if r["task"] == t])) for t in tasks}
    summary = {"ckpt": args.ckpt, "arm": arm, "success_rate": float(np.mean([r["success"] for r in rows])),
               "per_task": per_task, "n_episodes": len(rows), "n_exec": args.n_exec, "wall_s": time.time() - t0,
               "episodes": rows}
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
