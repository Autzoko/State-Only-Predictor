#!/usr/bin/env python
"""E37 checks, run on a GPU node before any perturbed evaluation (scripts/slurm/e37_probe.sbatch):
  1. unperturbed env == env without the perturbation code path (q exact, images up to render jitter) -> the benchmark
     itself is unchanged;
  2. every (kind, level) is deterministic: first resets of two fresh envs give the same q, the same realised parameters
     and the same images up to render jitter (LIBERO stove scenes flicker in a few wrist pixels even unperturbed);
  3. hard reset restores the scene: camera / lights / materials / texture data after a plain reset equal the XML values;
  4. every visual kind visibly changes the agent image at every level (>= 5% of pixels) and the robot kind moves the
     arm but not the gripper; mean |diff| per (kind, level) is logged to compare strengths across scenes;
  5. perturbing does not alter LIBERO's reset sequence: fixture poses over inits 0-4 equal those of an unperturbed env
     (fixtures are re-sampled at every hard reset by a stateful sampler, scripts/e37_diag2.py);
  6. images of init 0 for the task families and every (kind, level) -> outputs/e37_probe/*.png (inspected by eye).
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from sopt.sim.libero_env import LiberoTaskEnv, setup_env_vars, suite_tasks
from sopt.sim.perturb import KINDS, LEVELS, arena_materials


JITTER = 1e-3  # max fraction of pixels allowed to differ between identical resets (LIBERO render flicker)
MIN_CHANGE = 0.05  # min fraction of agent-view pixels a visual perturbation must change


def snapshot(env) -> dict:
    m = env.env.sim.model
    cid = m.camera_name2id("agentview")
    return {"cam_pos": np.array(m.cam_pos[cid]), "cam_quat": np.array(m.cam_quat[cid]),
            "light": np.array(m.light_diffuse[: m.nlight]), "mat": np.array(m.mat_rgba),
            "tex": np.array(m._model.tex_data, dtype=np.int64)}


def pix(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of pixels that differ in any channel."""
    return float((np.abs(a.astype(int) - b.astype(int)).max(-1) > 0).mean())


def same_obs(a: dict, b: dict) -> bool:
    return bool(np.array_equal(a["q"], b["q"]) and np.array_equal(a["fingers"], b["fingers"])
                and pix(a["agent"], b["agent"]) <= JITTER and pix(a["wrist"], b["wrist"]) <= JITTER)


def fixture_poses(env) -> np.ndarray:
    e, m = env.env.env, env.env.sim.model
    return np.array([m.body_pos[m.body_name2id(f.root_body)] for f in e.fixtures_dict.values()]).reshape(-1, 3)


def reset_sequence(suite: str, tid: int, perturb: str | None, n: int = 5) -> np.ndarray:
    env = LiberoTaskEnv(suite, tid, perturb=perturb)
    seq = []
    for init in range(n):  # as vla_eval / TaskVecEnv: reset(init, seed=init) in init order
        env.reset(init, seed=init)
        seq.append(fixture_poses(env))
    env.close()
    return np.array(seq)


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
               "fixtures": list(base.env.env.fixtures_dict), "levels": {}}
        base.close()
        # 1. the perturbation-capable env with perturb=None reproduces the plain env
        none = LiberoTaskEnv(suite, tid, perturb=None)
        o1 = none.reset(0, seed=0)
        same = same_obs(o0, o1)
        rec["none_identical"] = same
        rec["none_jitter"] = [pix(o0["agent"], o1["agent"]), pix(o0["wrist"], o1["wrist"])]
        ok &= same
        none.close()
        fam = task[:40].replace(" ", "_")
        if suite != "libero_90" or task in t17[:4]:
            Image.fromarray(np.concatenate([o0["agent"], o0["wrist"]], 1)).save(out / f"{suite}_{fam}_clean.png")
        for kind in KINDS:
            for level in LEVELS[kind]:
                spec = f"{kind}:{level}"
                env = LiberoTaskEnv(suite, tid, perturb=spec)
                a = env.reset(0, seed=0)
                env.env.reset()  # 3. plain reset (no perturbation applied) -> must equal the unperturbed scene
                s = snapshot(env)
                restored = all(np.allclose(s[k], ref[k]) for k in ref)
                env.close()
                env2 = LiberoTaskEnv(suite, tid, perturb=spec)  # 2. a fresh env gives the same perturbed episode
                b = env2.reset(0, seed=0)
                env2.close()
                det = same_obs(a, b) and a.get("perturb") == b.get("perturb")
                if kind == "robot":  # arm moved, gripper untouched
                    changed = (not np.allclose(a["q"], o0["q"])) and np.allclose(a["fingers"], o0["fingers"], atol=2e-3)
                else:  # 4. visibly changed
                    changed = pix(a["agent"], o0["agent"]) >= MIN_CHANGE
                ok &= det and restored and changed
                rec["levels"][spec] = {"deterministic": bool(det), "restored": bool(restored), "changed": bool(changed),
                                       "px_changed": pix(a["agent"], o0["agent"]),
                                       "mean_abs": float(np.abs(a["agent"].astype(int) - o0["agent"].astype(int)).mean()),
                                       "info": a.get("perturb", {}),
                                       "max_dq": float(np.abs(a["q"] - o0["q"]).max())}
                if suite != "libero_90" or task in t17[:4]:
                    Image.fromarray(np.concatenate([a["agent"], a["wrist"]], 1)).save(
                        out / f"{suite}_{fam}_{kind}{level}.png")
        # 5. the fixture-pose sequence over inits 0-4 is unchanged by any perturbation kind
        if rec["fixtures"]:
            seq0 = reset_sequence(suite, tid, None)
            rec["fixture_seq_same"] = {k: bool(np.array_equal(seq0, reset_sequence(suite, tid, f"{k}:2"))) for k in KINDS}
            rec["fixture_jitter_cm"] = float(100 * np.abs(seq0 - seq0[0]).max())
            ok &= all(rec["fixture_seq_same"].values())
        report.append(rec)
        bad = {k: v for k, v in rec["levels"].items() if not (v["deterministic"] and v["restored"] and v["changed"])}
        print(f"{suite:11s} {task[:60]:60s} lights {rec['n_light']} fixtures {rec['fixtures']} "
              f"none_identical {same} {rec['none_jitter']} fixture_seq {rec.get('fixture_seq_same')} bad {list(bad)}",
              flush=True)
        print("    mean|diff| " + " ".join(f"{k}{v['mean_abs']:.1f}" for k, v in rec["levels"].items()), flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
