#!/usr/bin/env python
"""Why are some LIBERO scenes not reproducible across resets? (E37 probe found two resets of the same (task, init,
seed) differ in kitchen / cabinet scenes, also without any perturbation.)
For each scene: reset twice (same init, same seed) in one env and once in a fresh env, then report which model /
state fields differ: fixture body poses (model.body_pos / body_quat, sampled by LIBERO at every hard reset and not
part of the init state), qpos after set_init_state, qpos after settling, images.
  python scripts/e37_diag.py
"""

import json
import os

import numpy as np

from sopt.sim.libero_env import SETTLE_STEPS, LiberoTaskEnv, setup_env_vars, suite_tasks


def grab(env: LiberoTaskEnv, init: int, seed: int) -> dict:
    e = env.env
    e.seed(seed)
    e.reset()
    m = e.sim.model
    pre = {"body_pos": np.array(m.body_pos), "body_quat": np.array(m.body_quat),
           "rng": np.random.get_state()[1][:4].copy()}
    obs = e.set_init_state(env.init_states[init])
    pre["qpos_init"] = np.array(e.sim.data.qpos)
    for _ in range(SETTLE_STEPS):
        obs, *_ = e.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
    o = env.parse(obs)
    pre.update(qpos_settled=np.array(e.sim.data.qpos), agent=o["agent"].astype(int), wrist=o["wrist"].astype(int),
               names=[m.body(i).name for i in range(m.nbody)])
    return pre


def compare(a: dict, b: dict) -> dict:
    dpos = np.abs(a["body_pos"] - b["body_pos"]).max(1)
    moved = [(a["names"][i], round(float(dpos[i]), 4)) for i in np.flatnonzero(dpos > 0)]
    return {"bodies_moved": moved[:12], "n_bodies_moved": len(moved),
            "max_body_quat": float(np.abs(a["body_quat"] - b["body_quat"]).max()),
            "qpos_init": float(np.abs(a["qpos_init"] - b["qpos_init"]).max()),
            "qpos_settled": float(np.abs(a["qpos_settled"] - b["qpos_settled"]).max()),
            "agent_px_frac": float((np.abs(a["agent"] - b["agent"]).max(-1) > 0).mean()),
            "agent_max": int(np.abs(a["agent"] - b["agent"]).max()),
            "wrist_px_frac": float((np.abs(a["wrist"] - b["wrist"]).max(-1) > 0).mean()),
            "rng_same_after_reset": bool(np.array_equal(a["rng"], b["rng"]))}


def main():
    setup_env_vars(os.environ["SOPT_DATA"])
    cases = [("libero_90", "close the bottom drawer of the cabinet"), ("libero_90", "put the frying pan on the stove"),
             ("libero_90", "put the middle black bowl on the plate"),
             ("libero_90", "pick up the cream cheese and put it in the tray"),
             ("libero_goal", "open the middle drawer of the cabinet"),
             ("libero_10", "turn on the stove and put the moka pot on it")]
    out = []
    for suite, task in cases:
        tid = suite_tasks(suite).index(task)
        env = LiberoTaskEnv(suite, tid)
        a, b = grab(env, 0, 0), grab(env, 0, 0)
        env2 = LiberoTaskEnv(suite, tid)
        c = grab(env2, 0, 0)
        d = grab(env, 1, 0)  # another init, same seed: are fixture poses tied to the seed or to the init?
        r = {"suite": suite, "task": task, "same_env": compare(a, b), "fresh_env": compare(a, c),
             "other_init": compare(a, d)}
        print(json.dumps(r), flush=True)
        out.append(r)
        env.close()
        env2.close()
    os.makedirs("outputs/e37_probe", exist_ok=True)
    json.dump(out, open("outputs/e37_probe/diag.json", "w"), indent=1)


if __name__ == "__main__":
    main()
