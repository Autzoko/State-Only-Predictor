#!/usr/bin/env python
"""Follow-up of scripts/e37_diag.py: LIBERO fixture poses (cabinet, rack, ...) are re-sampled at every hard reset and
differ between the 1st and 2nd reset of one env although the global RNG state is identical. Does the drift
(a) accumulate over the 50 sequential resets that vla_eval / TaskVecEnv performs per worker, and
(b) depend on what happened in the previous episode (i.e. on the policy -> arm-dependent)?
Env A: reset(init, seed=init) for init 0..49 with no steps in between (as an all-identical policy would);
Env B: same, but 60 random actions after every reset (a different "policy").
  python scripts/e37_diag2.py
"""

import json
import os

import numpy as np

from sopt.sim.libero_env import LiberoTaskEnv, setup_env_vars, suite_tasks


def fixtures(env: LiberoTaskEnv) -> dict:
    e = env.env.env
    m = env.env.sim.model
    return {k: np.array(m.body_pos[m.body_name2id(f.root_body)]) for k, f in e.fixtures_dict.items()}


def main():
    setup_env_vars(os.environ["SOPT_DATA"])
    out = []
    for suite, task in [("libero_90", "close the bottom drawer of the cabinet"),
                        ("libero_90", "put the right moka pot on the stove"),
                        ("libero_goal", "open the middle drawer of the cabinet")]:
        tid = suite_tasks(suite).index(task)
        A, B = LiberoTaskEnv(suite, tid), LiberoTaskEnv(suite, tid)
        rng = np.random.default_rng(0)
        fa, fb = [], []
        for init in range(50):
            A.reset(init, seed=init)
            fa.append(fixtures(A))
            B.reset(init, seed=init)
            fb.append(fixtures(B))
            for _ in range(60):
                a = rng.uniform(-1, 1, 7)
                B.step(a)
        names = list(fa[0])
        dev_first = {k: [float(np.abs(f[k] - fa[0][k]).max()) for f in fa] for k in names}
        ab = {k: [float(np.abs(x[k] - y[k]).max()) for x, y in zip(fa, fb)] for k in names}
        r = {"suite": suite, "task": task, "fixtures": names,
             "A_dev_from_reset0_max": {k: max(v) for k, v in dev_first.items()},
             "A_dev_from_reset0_last5": {k: [round(x, 4) for x in v[-5:]] for k, v in dev_first.items()},
             "A_vs_B_max": {k: max(v) for k, v in ab.items()},
             "A_vs_B_n_differ": {k: int(sum(x > 0 for x in v)) for k, v in ab.items()},
             "A_pos_first10": {k: [[round(float(c), 4) for c in f[k]] for f in fa[:10]] for k in names}}
        print(json.dumps(r), flush=True)
        out.append(r)
        A.close()
        B.close()
    json.dump(out, open("outputs/e37_probe/diag2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
