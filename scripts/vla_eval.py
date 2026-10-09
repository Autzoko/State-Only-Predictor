#!/usr/bin/env python
"""Closed-loop LIBERO-90 success of an E5 SmolVLA policy (7 tasks in lockstep, one policy copy).

  python scripts/vla_eval.py --ckpt outputs/e5_vla_droid_k5/final.pt --out outputs/e5_vla_droid_k5/rollouts.json
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
from sopt.sim.libero_env import dataset_to_sim_gripper
from sopt.sim.vec_env import TaskVecEnv
from sopt.vla.smolvla import IMAGE_KEYS, build_policy, load_trainable

CTX = 96


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-init", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--n-exec", type=int, default=10, help="actions executed per predicted chunk")
    p.add_argument("--sopt-data", default=os.environ.get("SOPT_DATA"))
    args = p.parse_args()
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("vla_train", Path(__file__).with_name("vla_train.py"))
    vt = importlib.util.module_from_spec(spec)
    sys.modules["vla_train"] = vt
    spec.loader.exec_module(vt)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    tasks, arm = ck["tasks"], ck["arm"]
    device = torch.device("cuda")
    policy, pre, post = build_policy(vt.make_prior(arm), arm, ck["action_stats"], chunk=vt.CHUNK)
    load_trainable(policy, args.ckpt)
    policy.eval()
    venv = TaskVecEnv(args.sopt_data, tasks)
    rows, t0 = [], time.time()
    for init in range(args.n_init):
        torch.manual_seed(init)
        obs = venv.reset(init)
        hist = [[features_20hz(o["q"][None], o["fingers"][None])[0]] for o in obs]
        active = {i: True for i in range(len(tasks))}
        chunks = {}
        for t in range(args.max_steps):
            live = [i for i in active if active[i]]
            if not live:
                break
            if t % args.n_exec == 0:
                ctx = np.stack([interp_frames(np.stack(hist[i]), context_indices(len(hist[i]) - 1, CTX)) for i in live])
                imgs = [np.stack([obs[i]["agent"], obs[i]["wrist"]]) for i in live]
                img = torch.from_numpy(np.stack(imgs)).permute(0, 1, 4, 2, 3).float().div(255).to(device)
                batch = {IMAGE_KEYS[0]: img[:, 0], IMAGE_KEYS[1]: img[:, 1],
                         "observation.state": torch.from_numpy(ctx.astype(np.float32)).to(device),
                         "task": [tasks[i] for i in live]}
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    act = policy.predict_action_chunk(pre(batch))
                act = post(act.float()).cpu().numpy()  # (B, chunk, 7), dataset convention
                chunks.update({i: act[j] for j, i in enumerate(live)})
            actions = {}
            for i in live:
                a = chunks[i][t % args.n_exec]
                actions[i] = np.concatenate([np.clip(a[:6], -1, 1), dataset_to_sim_gripper(np.array([float(a[6] > 0.5)]))])
            for i, (o, success) in venv.step(actions).items():
                obs[i] = o
                hist[i].append(features_20hz(o["q"][None], o["fingers"][None])[0])
                if success:
                    active[i] = False
                    rows.append({"task": tasks[i], "init": init, "success": True, "steps": t + 1})
        rows += [{"task": tasks[i], "init": init, "success": False, "steps": args.max_steps}
                 for i in active if active[i]]
        sr = np.mean([r["success"] for r in rows])
        print(f"init {init}: running success {sr:.3f} ({time.time() - t0:.0f}s)", flush=True)
    venv.close()
    per_task = {t: float(np.mean([r["success"] for r in rows if r["task"] == t])) for t in tasks}
    summary = {"ckpt": args.ckpt, "arm": arm, "success_rate": float(np.mean([r["success"] for r in rows])),
               "per_task": per_task, "n_episodes": len(rows), "n_exec": args.n_exec, "wall_s": time.time() - t0,
               "episodes": rows}
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
