#!/usr/bin/env python
"""E37 checks, run on a GPU node before any perturbed evaluation (scripts/slurm/e37_probe.sbatch):
  1. unperturbed env == env without the perturbation code path (images, q) -> the benchmark itself is unchanged;
  2. every (kind, level) is deterministic: two resets of the same (task, init) give identical images and q;
  3. hard reset restores the scene: camera / lights / materials after a plain reset equal the XML values;
  4. per scene: matched arena materials (texture), camera parent body, light count, realised parameters;
  5. images of init 0 for every task family and (kind, level) -> outputs/e37_probe/*.png (inspected by eye).
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from sopt.sim.libero_env import LiberoTaskEnv, setup_env_vars, suite_tasks
from sopt.sim.perturb import KINDS, LEVELS, arena_materials


def snapshot(env) -> dict:
    m = env.env.sim.model
    cid = m.camera_name2id("agentview")
    return {"cam_pos": np.array(m.cam_pos[cid]), "cam_quat": np.array(m.cam_quat[cid]),
            "light": np.array(m.light_diffuse[: m.nlight]), "mat": np.array(m.mat_rgba)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="outputs/e37_probe")
    p.add_argument("--sopt-data", default=os.environ.get("SOPT_DATA"))
    args = p.parse_args()
    setup_env_vars(args.sopt_data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t17 = json.load(open("configs/libero90_17_tasks.json"))
    t17 = t17["tasks"] if isinstance(t17, dict) else t17
    cases = [("libero_90", t) for t in t17] + [("libero_goal", t) for t in suite_tasks("libero_goal")[:2]] + \
            [("libero_10", t) for t in suite_tasks("libero_10")[:3]]
    report, ok = [], True
    for suite, task in cases:
        tid = suite_tasks(suite).index(task)
        base = LiberoTaskEnv(suite, tid)
        o0 = base.reset(0, seed=0)
        ref = snapshot(base)
        m = base.env.sim.model
        mats = arena_materials(m)
        names = [m.material(i).name for i in range(m.nmat)]
        rec = {"suite": suite, "task": task, "camera_body": int(m.cam_bodyid[m.camera_name2id("agentview")]),
               "n_light": int(m.nlight), "arena_materials": [names[i] for i in mats],
               "other_materials": [n for i, n in enumerate(names) if i not in mats], "levels": {}}
        # 1. the perturbation-capable env with perturb=None reproduces the plain env
        none = LiberoTaskEnv(suite, tid, perturb=None)
        o1 = none.reset(0, seed=0)
        same = all(np.array_equal(o0[k], o1[k]) for k in ("agent", "wrist", "q", "fingers"))
        ok &= same
        rec["none_identical"] = bool(same)
        none.close()
        fam = task[:40].replace(" ", "_")
        if suite != "libero_90" or task in t17[:4]:
            Image.fromarray(np.concatenate([o0["agent"], o0["wrist"]], 1)).save(out / f"{suite}_{fam}_clean.png")
        for kind in KINDS:
            for level in LEVELS[kind]:
                env = LiberoTaskEnv(suite, tid, perturb=f"{kind}:{level}")
                a = env.reset(0, seed=0)
                b = env.reset(0, seed=0)  # 2. deterministic; 3. a second hard reset starts from the XML scene again
                det = all(np.array_equal(a[k], b[k]) for k in ("agent", "wrist", "q", "fingers"))
                env.env.reset()  # plain reset, no perturbation applied -> must equal the unperturbed scene
                s = snapshot(env)
                restored = all(np.allclose(s[k], ref[k]) for k in ref)
                changed = not np.array_equal(a["agent"], o0["agent"]) if kind != "robot" else \
                    not np.allclose(a["q"], o0["q"])
                if kind == "robot":  # gripper must be untouched; the arm moved by at most the level
                    changed &= np.allclose(a["fingers"], o0["fingers"], atol=2e-3)
                ok &= det and restored and changed
                rec["levels"][f"{kind}:{level}"] = {"deterministic": bool(det), "restored": bool(restored),
                                                    "changed": bool(changed), "info": a.get("perturb", {}),
                                                    "max_dq": float(np.abs(a["q"] - o0["q"]).max())}
                if suite != "libero_90" or task in t17[:4]:
                    Image.fromarray(np.concatenate([a["agent"], a["wrist"]], 1)).save(
                        out / f"{suite}_{fam}_{kind}{level}.png")
                env.close()
        base.close()
        report.append(rec)
        bad = {k: v for k, v in rec["levels"].items() if not (v["deterministic"] and v["restored"] and v["changed"])}
        print(f"{suite:11s} {task[:60]:60s} cam_body {rec['camera_body']} lights {rec['n_light']} "
              f"arena {rec['arena_materials']} none_identical {same} bad {list(bad)}", flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
