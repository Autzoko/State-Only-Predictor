#!/usr/bin/env python
"""Closed-loop LIBERO success of a fine-tuned VLA (SmolVLA or pi0.5; tasks in lockstep, one policy copy).

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
from sopt.policy.rollout import trace_row
from sopt.sim.libero_env import dataset_to_sim_gripper
from sopt.sim.vec_env import TaskVecEnv

CTX = 96


def mask_sopt(policy, keep: list[bool], device) -> None:
    """E34: zero the SOPT input of the samples with keep = False for this prediction (after make_batch has set it).
    xattn arms: the branch's keep mask (as in modality-dropout training); expert / adaln arms: the feature itself."""
    s = getattr(policy, "sopt", None)
    assert s is not None, "--sopt-mask-grasp needs a SOPT arm"
    m = torch.tensor(keep, dtype=torch.float32, device=device)
    if getattr(s, "keep", None) is not None:
        s.keep = s.keep * m
    else:
        s.cond = s.cond * m[:, None].to(s.cond.dtype)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-init", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=None, help="default: 400 (libero90) / OpenVLA per-suite limits")
    p.add_argument("--n-exec", type=int, default=10, help="actions executed per predicted chunk")
    p.add_argument("--trace", type=int, default=0, help="log simulator state every N steps (failure taxonomy)")
    p.add_argument("--act-log", action="store_true", help="E26: log every executed action (chunk-boundary metrics)")
    p.add_argument("--sopt-mask-grasp", type=float, default=0.0,
                   help="E34: switch the SOPT input off for this many seconds after each gripper-close command (<0: always off)")
    p.add_argument("--perturb", default=None, help="E37: <kind>:<level>, kinds in sopt/sim/perturb.py (default: none)")
    p.add_argument("--log-npz", default=None,
                   help="E36: per-step raw proprio (q, fingers), executed actions and every predicted chunk -> .npz")
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
    suite = ck["meta"].get("suite", "libero90")
    V = vt.vla_module(ck["meta"].get("vla", "smolvla"))
    prior = vt.make_prior(arm, ck["meta"].get("prior", vt.DROID_PRIOR))
    extra = {"state_stats": ck["state_stats"]} if ck.get("state_stats") is not None else {}
    if getattr(V, "NEEDS_TASKS", False):
        extra["tasks"] = ck["tasks"]
    policy, pre, post = V.build_policy(prior, arm, ck["action_stats"], chunk=ck["meta"].get("chunk", vt.CHUNK), **extra)
    V.load_trainable(policy, args.ckpt)
    policy.eval()
    if suite in ("libero90", "libero90_17"):
        venv, feat = TaskVecEnv(args.sopt_data, tasks, perturb=args.perturb), features_20hz
        args.max_steps = args.max_steps or 400
    else:  # EE-only features: joints hidden exactly as in training (sopt/data/libero_std.py)
        from sopt.data.libero_std import MAX_STEPS, hide_joints

        venv = TaskVecEnv(args.sopt_data, tasks, suite=suite, perturb=args.perturb)
        feat = lambda q, f: hide_joints(features_20hz(q, f))  # noqa: E731
        args.max_steps = args.max_steps or MAX_STEPS[suite]
    C = ck["meta"].get("ctx", CTX)  # history length the policy was trained with (E33)
    rows, t0 = [], time.time()
    log = []  # E36: one dict of arrays per finished episode, in the order of `rows`
    for init in range(args.n_init):
        torch.manual_seed(init)
        obs = venv.reset(init)
        hist = [[feat(o["q"][None], o["fingers"][None])[0]] for o in obs]
        active = {i: True for i in range(len(tasks))}
        chunks, traces, alog = {}, {i: [] for i in range(len(tasks))}, {i: [] for i in range(len(tasks))}
        last_close, was_open = {i: -10**9 for i in range(len(tasks))}, {i: True for i in range(len(tasks))}
        lq = {i: [np.concatenate([o["q"], o["fingers"]])] for i, o in enumerate(obs)}  # state before step t at [t]
        la, lc = {i: [] for i in range(len(tasks))}, {i: [] for i in range(len(tasks))}
        pinfo = {i: o.get("perturb", {}) for i, o in enumerate(obs)}

        def finish(i, success, steps):
            rows.append({"task": tasks[i], "init": init, "success": success, "steps": steps,
                         **({"perturb": pinfo[i]} if args.perturb else {}),
                         **({"trace": traces[i]} if args.trace else {}),
                         **({"acts": alog[i]} if args.act_log else {})})
            if args.log_npz:
                log.append({"qf": np.asarray(lq[i], np.float32), "act": np.asarray(la[i], np.float32),
                            "chunk_t": np.asarray([c[0] for c in lc[i]], np.int32),
                            "chunk": np.asarray([c[1] for c in lc[i]], np.float32)})
        for t in range(args.max_steps):
            live = [i for i in active if active[i]]
            if not live:
                break
            if t % args.n_exec == 0:
                ctx = np.stack([interp_frames(np.stack(hist[i]), context_indices(len(hist[i]) - 1, C)) for i in live])
                imgs = [np.stack([obs[i]["agent"], obs[i]["wrist"]]) for i in live]
                img = torch.from_numpy(np.stack(imgs)).permute(0, 1, 4, 2, 3).float().div(255).to(device)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    batch = V.make_batch(policy, img, torch.from_numpy(ctx.astype(np.float32)).to(device),
                                         [tasks[i] for i in live])
                    if args.sopt_mask_grasp:
                        mask_sopt(policy, [args.sopt_mask_grasp >= 0 and t - last_close[i] > 20 * args.sopt_mask_grasp
                                           for i in live], device)
                    act = policy.predict_action_chunk(pre(batch))
                act = post(act.float()).cpu().numpy()  # (B, chunk, 7), dataset convention
                chunks.update({i: act[j] for j, i in enumerate(live)})
                for j, i in enumerate(live):
                    lc[i].append((t, act[j]))
            actions = {}
            for i in live:
                a = chunks[i][t % args.n_exec]
                actions[i] = np.concatenate([np.clip(a[:6], -1, 1), dataset_to_sim_gripper(np.array([float(a[6] > 0.5)]))])
                if was_open[i] and a[6] <= 0.5:  # dataset convention: 1 = open
                    last_close[i] = t
                was_open[i] = a[6] > 0.5
                if args.act_log:
                    alog[i].append([round(float(x), 3) for x in a[:7]])
            for i in live:
                la[i].append(actions[i])
            if args.trace and t % args.trace == 0:
                for i in live:
                    traces[i].append(trace_row(t, obs[i], actions[i]))
            for i, (o, success) in venv.step(actions).items():
                obs[i] = o
                hist[i].append(feat(o["q"][None], o["fingers"][None])[0])
                lq[i].append(np.concatenate([o["q"], o["fingers"]]))
                if success:
                    active[i] = False
                    finish(i, True, t + 1)
        for i in active:
            if active[i]:
                finish(i, False, args.max_steps)
        sr = np.mean([r["success"] for r in rows])
        print(f"init {init}: running success {sr:.3f} ({time.time() - t0:.0f}s)", flush=True)
    venv.close()
    per_task = {t: float(np.mean([r["success"] for r in rows if r["task"] == t])) for t in tasks}
    summary = {"ckpt": args.ckpt, "arm": arm, "vla": ck["meta"].get("vla", "smolvla"), "suite": suite, "max_steps": args.max_steps, "success_rate": float(np.mean([r["success"] for r in rows])),
               "per_task": per_task, "n_episodes": len(rows), "n_exec": args.n_exec, "trace_every": args.trace, "sopt_mask_grasp": args.sopt_mask_grasp,
               "perturb": args.perturb, "ctx": C, "wall_s": time.time() - t0,
               "episodes": rows}
    Path(args.out).write_text(json.dumps(summary, indent=2))
    if args.log_npz:  # flat arrays + per-episode offsets (row k of `episodes` <-> episode k here)
        cat = lambda k: np.concatenate([e[k] for e in log]) if log else np.zeros((0,))  # noqa: E731
        off = lambda k: np.cumsum([0] + [len(e[k]) for e in log]).astype(np.int64)  # noqa: E731
        np.savez_compressed(args.log_npz, qf=cat("qf"), qf_off=off("qf"), act=cat("act"), act_off=off("act"),
                            chunk=cat("chunk"), chunk_t=cat("chunk_t"), chunk_off=off("chunk"),
                            success=np.array([r["success"] for r in rows]), init=np.array([r["init"] for r in rows]),
                            task=np.array([tasks.index(r["task"]) for r in rows]), tasks=np.array(tasks),
                            suite=suite, n_exec=args.n_exec, max_steps=args.max_steps)
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
