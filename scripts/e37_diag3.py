#!/usr/bin/env python
"""E37 probe follow-up: in the kitchen scenes texture:1 left the image bit-identical while texture:2/3 recoloured the
table. Trace mat_rgba of the arena materials through apply_scene -> settle steps -> observation, and the image change,
for every texture level (and light:1 as a control) in one kitchen and one living-room scene.
  DIAG=e37_diag3 sbatch scripts/slurm/e37_diag.sbatch
"""

import os

import numpy as np

from sopt.sim.libero_env import LiberoTaskEnv, setup_env_vars, suite_tasks
from sopt.sim.perturb import arena_materials


def main():
    setup_env_vars(os.environ["SOPT_DATA"])
    for task in ("put the middle black bowl on the plate", "close the bottom drawer of the cabinet",
                 "pick up the cream cheese and put it in the tray"):
        tid = suite_tasks("libero_90").index(task)
        base = LiberoTaskEnv("libero_90", tid)
        o0 = base.reset(0, seed=0)
        m0 = base.env.sim.model
        mats = arena_materials(m0)
        print(task, {m0.material(i).name: (np.round(m0.mat_rgba[i], 3).tolist(), int(m0.mat_texid[i])
                                            if np.ndim(m0.mat_texid[i]) == 0 else m0.mat_texid[i].tolist())
                     for i in mats}, flush=True)
        # which geoms use the arena materials, and do they have their own rgba?
        g = [(m0.geom(j).name, int(m0.geom_matid[j]), np.round(m0.geom_rgba[j], 2).tolist())
             for j in range(m0.ngeom) if m0.geom_matid[j] in mats]
        print("  geoms using arena materials:", g[:15], flush=True)
        base.close()
        for spec in ("texture:1", "texture:2", "texture:3", "light:1"):
            env = LiberoTaskEnv("libero_90", tid, perturb=spec)
            a = env.reset(0, seed=0)
            m = env.env.sim.model
            d = np.abs(a["agent"].astype(int) - o0["agent"].astype(int)).max(-1)
            print(f"  {spec}: rgba after reset+settle {[np.round(m.mat_rgba[i], 3).tolist() for i in mats]} "
                  f"model is env.sim.model {m is env.env.sim.model} px changed {(d > 0).mean():.3f} info {a['perturb']}",
                  flush=True)
            # render straight from the sim as well (bypasses robosuite observables)
            img = env.env.sim.render(camera_name="agentview", height=256, width=256)[::-1, ::-1]
            print(f"     direct render differs from obs image: {(np.abs(img.astype(int) - a['agent'].astype(int)).max(-1) > 0).mean():.3f}",
                  flush=True)
            env.close()


if __name__ == "__main__":
    main()
