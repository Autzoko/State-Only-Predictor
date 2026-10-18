#!/usr/bin/env python
"""E26 follow-up on the traces + e26_taxonomy.py output (local analysis, no simulation):
  1. net failures created by SOPT per category (paired, SOPT-only minus none-only) per host / arm
  2. grasp timing: EE-to-target distance at the first gripper-close command, and retries, in grasp_miss failures
     phantom transport: after closing on nothing, the EE rises > 5 cm within 2 s with the gripper still closed
     (the policy carries on as if the grasp succeeded)
  3. two-stage pudding task: stage reached (pudding moved > 10 cm and released = stage 1 done); failure categories;
     failure rate on the two multi-stage tasks vs the 15 single-stage ones
  4. A1 gate part 2: share of failures with blocked > success mean + 1 SD
  python scripts/analysis/e26_followup.py <e26_taxonomy.json> <trace_n50.json ...>
"""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

tax = json.load(open(sys.argv[1]))
cal = tax["calibration"]
ARMS = ("none", "expert_scratch", "expert_droid")
host = lambda n: n.split("_")[1]
arm = lambda n: re.search(r"_(none|expert_scratch|expert_droid)_", n).group(1)
rows = {n: {(r["task"], r["init"]): r for r in eps} for n, eps in tax["episodes"].items()}

print("== 1. net SOPT-created failures (SOPT-only fail minus none-only fail), paired by task+init")
for n in sorted(rows):
    if arm(n) == "none":
        continue
    N = rows[n.replace(arm(n), "none")]
    A = rows[n]
    net = Counter()
    for k in A.keys() & N.keys():
        if A[k]["cat"] != "success" and N[k]["cat"] == "success":
            net[A[k]["cat"]] += 1; net["frozen"] += A[k]["frozen"]
        if N[k]["cat"] != "success" and A[k]["cat"] == "success":
            net[N[k]["cat"]] -= 1; net["frozen"] -= N[k]["frozen"]
    print(f"  {n:42s}", {c: net[c] for c in ("grasp_miss", "place_fail", "wrong_target", "no_reach", "artic_fail", "frozen")})


def close_events(a):
    g = a[:, 6] > 0.5  # 1 = open
    return [t for t in range(1, len(a)) if g[t - 1] and not g[t]]


print("== 2. grasp timing in grasp_miss failures / all episodes of grasp tasks")
print(f"  {'run':42s} {'d@close_miss':>12s} {'d@close_succ':>12s} {'early%_miss':>11s} {'closes_miss':>11s} {'n_miss':>6s} {'phantom%':>8s} {'n_phantom':>9s}")
stage = {}
blocked_rows = {}
for path in sys.argv[2:]:
    n = Path(path).parent.name
    d = json.load(open(path)); every = d.get("trace_every") or 4
    dm, ds, nc, early, ph = [], [], [], [], []
    st = Counter()
    for e in d["episodes"]:
        c = cal[e["task"]]; r = rows[n][(e["task"], e["init"])]
        if c["target"] is None or not e.get("acts"):
            continue
        a = np.array(e["acts"]); tr = e["trace"]; ts = np.array([q["t"] for q in tr])
        rel = np.array([q["objs"][c["target"] + "_to_robot0_eef"] for q in tr])
        ce = close_events(a)
        if ce:
            j = int(np.argmin(np.abs(ts - ce[0])))
            dist = float(np.linalg.norm(rel[j]))
            if r["cat"] == "grasp_miss":
                dm.append(dist); nc.append(len(ce)); early.append(dist > c["grasp_dist"] + 0.03)
                g = a[:, 6] > 0.5; t1 = ce[0]
                while t1 < len(a) and not g[t1] and t1 < ce[0] + 40:
                    t1 += 1
                k = int(np.argmin(np.abs(ts - t1))); eef = np.array([q["eef"] for q in tr])
                ph.append(k > j and (eef[j:k + 1, 2] - eef[j, 2]).max() > 0.05)
            elif r["cat"] == "success":
                ds.append(dist)
        elif r["cat"] == "grasp_miss":
            ph.append(False)
        if "pudding in the top drawer" in e["task"]:
            pos = np.array([q["objs"][c["target"]] for q in tr])
            moved = np.linalg.norm(pos[-1] - pos[5]) > 0.10
            opened = a[-1, 6] > 0.5
            if not e["success"]:
                st["cat:" + r["cat"] + ("+frozen" if r["frozen"] else "")] += 1
            st["success" if e["success"] else ("stage1_done" if moved and opened else ("carrying" if moved else "stage1_fail"))] += 1
    stage[n] = dict(st)
    print(f"  {n:42s} {np.median(dm) if dm else 0:12.3f} {np.median(ds) if ds else 0:12.3f} {100*np.mean(early) if early else 0:11.1f} {np.mean(nc) if nc else 0:11.2f} {len(dm):6d} {100*np.mean(ph) if ph else 0:8.1f} {int(np.sum(ph)):9d}")

print("== 3. pudding two-stage task (drawer + close), 50 inits")
for n in sorted(stage):
    print(f"  {n:42s}", stage[n])

multi = lambda t: "and close" in t or "stack the left bowl" in t
for n in sorted(rows):
    m = [r["cat"] != "success" for (t, _), r in rows[n].items() if multi(t)]
    o = [r["cat"] != "success" for (t, _), r in rows[n].items() if not multi(t)]
    print(f"  {n:42s} failure rate multi-stage {100*np.mean(m):5.1f}%  single-stage {100*np.mean(o):5.1f}%")

print("== 4. A1 gate part 2: failures with blocked > success mean + 1 SD")
for n in sorted(rows):
    b_s = np.array([r["blocked"] for r in rows[n].values() if r["cat"] == "success" and r["blocked"] is not None])
    b_f = np.array([r["blocked"] for r in rows[n].values() if r["cat"] != "success" and r["blocked"] is not None])
    thr = b_s.mean() + b_s.std()
    print(f"  {n:42s} thr {thr:.3f} share {100*(b_f > thr).mean():5.1f}%  AUC {tax['runs'][n]['blocked_auc_fail']:.2f}")
