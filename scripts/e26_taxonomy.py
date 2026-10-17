#!/usr/bin/env python
"""E26: closed-loop failure taxonomy of the 17-task host checkpoints (traces from vla_eval.py --trace 4 --act-log).

Generalizes failure_taxonomy.py (E9a, 6 hand-mapped tasks) to any task without hand-tuned constants. Calibration is
fixed before looking at arm differences and pooled over *all* traced runs given (every host and arm alike):
  target object of a task   the object displaced most (max over the episode) in the task's successful episodes
                            (majority vote); a task whose successes move no object > 3 cm is "articulated"
  grasp distance            median over those successes of the minimum EE-to-target distance
Per failed episode (E9a rules): place_fail (target lifted > 3 cm, task not done) / grasp_miss (EE came within
grasp distance + 6 cm, never lifted) / wrong_target (never approached, another object moved > 3 cm) / no_reach;
articulated tasks -> artic_fail.
Closed-loop behaviour, per episode (the copycat / inertia signatures of history-conditioned imitation, Wen et al. 2020):
  frozen        failed and the EE moved < 2 cm over the last 5 s (stalled until the time limit)
  grip_cycles   close -> open transitions of the gripper command (re-grasp attempts)
  jerk_ratio    mean |a_t - a_{t-1}| (dims 0-5) at chunk boundaries / within chunks (closed-loop chunk stitching)
Paired analysis (same task + init): episodes where one arm succeeds and the other fails, by the failing arm's
category -> which failure modes SOPT creates vs removes.

  python scripts/e26_taxonomy.py --out outputs/e26_taxonomy.json outputs/e2*_*_k5_t17_s0/trace_n50.json
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

NEAR, LIFT, MOVED, FREEZE_M, FREEZE_S, FPS = 0.06, 0.03, 0.03, 0.02, 5.0, 20.0
CATS = ["success", "wrong_target", "no_reach", "grasp_miss", "place_fail", "artic_fail"]


def objects(tr):
    return [k for k in tr[0]["objs"] if not k.endswith("_to_robot0_eef")]


def disp(tr, k):
    p = np.array([r["objs"][k] for r in tr])
    return np.linalg.norm(p - p[0], axis=1).max()


def calibrate(eps):
    """task -> (target object or None, grasp distance), from the pooled successful episodes."""
    votes, dist = defaultdict(Counter), defaultdict(list)
    for e in eps:
        if e["success"] and e.get("trace"):
            d = {k: disp(e["trace"], k) for k in objects(e["trace"])}
            k = max(d, key=d.get)
            votes[e["task"]][k if d[k] > MOVED else None] += 1
    cal = {}
    for task, v in votes.items():
        tgt = v.most_common(1)[0][0]
        if tgt is not None:
            for e in eps:
                if e["task"] == task and e["success"] and e.get("trace") and tgt + "_to_robot0_eef" in e["trace"][0]["objs"]:
                    dist[task].append(np.linalg.norm(np.array([r["objs"][tgt + "_to_robot0_eef"] for r in e["trace"]]), axis=1).min())
        cal[task] = {"target": tgt, "grasp_dist": float(np.median(dist[task])) if dist[task] else None,
                     "n_success": sum(v.values())}
    return cal


def classify(e, cal):
    if e["success"]:
        return "success"
    c = cal.get(e["task"])
    if c is None or c["target"] is None or c["grasp_dist"] is None:
        return "artic_fail"
    tr, tgt = e["trace"], c["target"]
    pos = np.array([r["objs"][tgt] for r in tr])
    if (pos[:, 2] - pos[0, 2]).max() > LIFT:
        return "place_fail"
    if np.linalg.norm(np.array([r["objs"][tgt + "_to_robot0_eef"] for r in tr]), axis=1).min() < c["grasp_dist"] + NEAR:
        return "grasp_miss"
    moved = any(disp(tr, k) > MOVED for k in objects(tr) if k != tgt)
    return "wrong_target" if moved else "no_reach"


def behaviour(e, every, n_exec):
    tr = e["trace"]
    eef = np.array([r["eef"] for r in tr])
    last = max(1, int(FREEZE_S * FPS / every))
    frozen = (not e["success"]) and len(eef) > last and np.linalg.norm(eef[-last:] - eef[-1], axis=1).max() < FREEZE_M
    out = {"frozen": bool(frozen)}
    a = np.array(e.get("acts") or [])
    if len(a) > n_exec:
        g = a[:, 6] > 0.5  # dataset convention: 1 = open
        out["grip_cycles"] = int(((~g[:-1]) & g[1:]).sum())
        d = np.abs(np.diff(a[:, :6], axis=0)).mean(1)  # d[t-1] = |a_t - a_{t-1}|
        t = np.arange(1, len(a))
        b, w = d[t % n_exec == 0], d[t % n_exec != 0]
        out["jerk_ratio"] = float(b.mean() / (w.mean() + 1e-6)) if len(b) and len(w) else None
    return out


def run_name(path):
    return Path(path).parent.name


def arm_of(name):
    m = re.search(r"_(none|expert_scratch|expert_droid)_", name)
    return m.group(1) if m else name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("traces", nargs="+")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    runs = {run_name(t): json.load(open(t)) for t in args.traces}
    cal = calibrate([e for d in runs.values() for e in d["episodes"]])
    res = {"calibration": cal, "runs": {}, "paired": {}}
    per_ep = {}
    print(f"{'run':40s}" + "".join(f"{c:>13s}" for c in CATS) + f"{'frozen':>8s}{'gcyc':>6s}{'jerk':>6s}")
    for name, d in runs.items():
        every, n_exec = d.get("trace_every") or 4, d.get("n_exec", 10)
        rows = {}
        for e in d["episodes"]:
            rows[(e["task"], e["init"])] = {"cat": classify(e, cal), **behaviour(e, every, n_exec)}
        per_ep[name] = rows
        c = Counter(r["cat"] for r in rows.values())
        fails = [r for r in rows.values() if r["cat"] != "success"]
        gc = [r["grip_cycles"] for r in rows.values() if "grip_cycles" in r]
        jr = [r["jerk_ratio"] for r in rows.values() if r.get("jerk_ratio") is not None]
        res["runs"][name] = {"counts": dict(c), "n": len(rows), "frozen_fail": sum(r["frozen"] for r in fails),
                             "grip_cycles_mean": float(np.mean(gc)) if gc else None,
                             "jerk_ratio_mean": float(np.mean(jr)) if jr else None,
                             "per_task": {t: dict(Counter(r["cat"] for (tt, _), r in rows.items() if tt == t))
                                          for t in sorted({k[0] for k in rows})}}
        r = res["runs"][name]
        print(f"{name:40s}" + "".join(f"{c[k]:13d}" for k in CATS) + f"{r['frozen_fail']:8d}"
              + f"{(r['grip_cycles_mean'] or 0):6.2f}{(r['jerk_ratio_mean'] or 0):6.2f}")
    # paired: within a host / seed group, each SOPT arm vs none
    groups = defaultdict(dict)
    for name in per_ep:
        groups[re.sub(r"_(none|expert_scratch|expert_droid)_", "_*_", name)][arm_of(name)] = name
    for g, arms in groups.items():
        if "none" not in arms:
            continue
        for a in ("expert_scratch", "expert_droid"):
            if a not in arms:
                continue
            A, N = per_ep[arms[a]], per_ep[arms["none"]]
            ks = A.keys() & N.keys()
            created = Counter(A[k]["cat"] + ("+frozen" if A[k]["frozen"] else "") for k in ks
                              if A[k]["cat"] != "success" and N[k]["cat"] == "success")
            removed = Counter(N[k]["cat"] + ("+frozen" if N[k]["frozen"] else "") for k in ks
                              if N[k]["cat"] != "success" and A[k]["cat"] == "success")
            res["paired"][f"{g}:{a}_vs_none"] = {"sopt_fails_none_ok": dict(created), "none_fails_sopt_ok": dict(removed)}
            print(f"{g} {a} vs none | SOPT-only failures {dict(created)} | none-only failures {dict(removed)}")
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
