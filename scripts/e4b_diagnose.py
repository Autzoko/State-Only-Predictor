#!/usr/bin/env python
"""Offline diagnosis of E4b policies on held-out demos of the E4b tasks (not used for k=5 training).

For random anchors, compares the demo action with
  oracle_idm : IDM(current state, TRUE next 4 states)            -> is the IDM itself fine?
  prior_idm  : IDM(current state, states read off the policy plan) -> what the prior arm executes
  bc         : first action of the BC chunk
and reports OSC L1, gripper accuracy (overall and near gripper switches), plan ADE (cm) at 0.2 s / 1 s.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from sopt.data.features import GRIPPER, POS
from sopt.policy.data import PolicyDataset, Raw20, select_episodes
from sopt.policy.rollout import load_idm, load_policy
from sopt.utils.config import REPO_ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prior", required=True)
    p.add_argument("--bc", required=True)
    p.add_argument("--idm", required=True)
    p.add_argument("--n", type=int, default=2000)
    args = p.parse_args()
    dev = torch.device("cuda")
    sopt_data = Path(os.environ["SOPT_DATA"])
    raw = Raw20(sopt_data / "processed/libero90_raw20")
    tasks = json.loads((REPO_ROOT / "configs/e4b_tasks.json").read_text())
    prior, pck = load_policy(args.prior, dev)
    bc, _ = load_policy(args.bc, dev)
    idm = load_idm(args.idm, dev)
    allsel = select_episodes(raw.eps[raw.eps["split"] == "test"], tasks, None)
    held = allsel[~allsel["episode_index"].isin(pck["meta"]["episodes"])]
    ds = PolicyDataset(raw, held, sopt_data / "processed/libero90_frames", prior.prior.cfg.ctx_len, prior.prior.H,
                       shift=0)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)
    stats = {k: [] for k in ["oracle_l1", "oracle_g", "prior_l1", "prior_g", "bc_l1", "bc_g", "near_switch",
                             "ade02", "ade1", "plan_grip_err"]}
    for i0 in range(0, len(idx), 128):
        items = [ds[int(i)] for i in idx[i0 : i0 + 128]]
        b = {k: torch.stack([it[k] for it in items]).to(dev) for k in items[0]}
        meta = [ds.items[int(i)] for i in idx[i0 : i0 + 128]]
        F = [ds.F[e] for e, _ in meta]
        cur = b["ctx"][:, -1]
        true_nxt = torch.stack([torch.from_numpy(F[j][np.minimum(np.arange(a + 1, a + 5), len(F[j]) - 1)])
                                for j, (_, a) in enumerate(meta)]).to(dev)
        act = b["act"][:, 0]
        a_or = idm.act(cur, true_nxt)
        plan = prior.plan(b["ctx"], b["img"], b["task"])[:, 0]  # (B, H, D) at 15 Hz
        plan = torch.cat([cur[:, None], plan], 1)
        t15 = torch.arange(1, 5, device=dev) / 20.0 * 15.0
        lo = t15.floor().long()
        w = (t15 - lo)[None, :, None]
        nxt = (1 - w) * plan[:, lo] + w * plan[:, lo + 1]
        a_pr = idm.act(cur, nxt)
        a_bc = bc.act(b["ctx"], b["img"], b["task"])[:, 0]
        for name, a in [("oracle", a_or), ("prior", a_pr), ("bc", a_bc)]:
            stats[f"{name}_l1"] += (a[:, :6] - act[:, :6]).abs().mean(1).tolist()
            stats[f"{name}_g"] += (a[:, 6] == act[:, 6]).float().tolist()
        switch = [bool(np.any(ds.A[e][max(a - 10, 0) : a + 10, 6] != ds.A[e][a, 6])) for e, a in meta]
        stats["near_switch"] += switch
        fut = b["fut"]
        stats["ade02"] += ((plan[:, 3, POS] - fut[:, 2, POS]).norm(dim=-1) * 100).tolist()  # ~0.2 s
        stats["ade1"] += ((plan[:, 15, POS] - fut[:, 14, POS]).norm(dim=-1) * 100).tolist()  # ~1 s
        stats["plan_grip_err"] += (plan[:, 1:16, GRIPPER] - fut[:, :15, GRIPPER]).abs().mean((1, 2)).tolist()
    s = {k: np.asarray(v, dtype=float) for k, v in stats.items()}
    ns = s["near_switch"] > 0
    res = {"n": len(s["prior_l1"]), "held_out_episodes": len(held)}
    for name in ["oracle", "prior", "bc"]:
        res[name] = {"osc_l1": s[f"{name}_l1"].mean(), "grip_acc": s[f"{name}_g"].mean(),
                     "grip_acc_near_switch": s[f"{name}_g"][ns].mean()}
    res["plan"] = {"pos_err_cm_0.2s": s["ade02"].mean(), "pos_err_cm_1s": s["ade1"].mean(),
                   "gripper_state_err": s["plan_grip_err"].mean()}
    res["zero_action_osc_l1"] = None
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
