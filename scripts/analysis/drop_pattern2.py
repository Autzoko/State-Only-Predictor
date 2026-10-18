# Per-task drop-pattern analysis (2026-10-18). Input: per_task_dump.json = {run: {task: [[init, success], ...]}} dumped from outputs/*/rollouts_n50.json; run from the directory holding it.
import json, numpy as np, math
d = json.load(open("per_task_dump.json"))
tasks = json.load(open("/Users/ll5582/Desktop/NYU AIR Lab/State-Only Pretrain/configs/libero90_17_tasks.json"))
def mcn(a, b):
    x = sum(1 for k in a if a[k] and not b[k]); y = sum(1 for k in a if b[k] and not a[k]); n = x + y
    p = min(1, 2 * sum(math.comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n) if n else 1
    return x, y, p
def paired(pat, n, arm, t):
    A = {(s, i): v for s in range(n) for i, v in d[pat % (arm, s)][t]}
    B = {(s, i): v for s in range(n) for i, v in d[pat % ("none", s)][t]}
    return mcn(A, B)
cases = [("X-VLA droid", "e20_xvla_%s_k5_t17_s%d", 3, "expert_droid"), ("GR00T scratch", "e22_groot_%s_k5_t17_s%d", 3, "expert_scratch"),
         ("pi05 scratch", "e14_pi05_%s_k5_lr1e-4_t17_s%d", 2, "expert_scratch"), ("DP scratch", "e21_dp_%s_k5_t17_s%d", 3, "expert_scratch"),
         ("GR00T droid", "e22_groot_%s_k5_t17_s%d", 3, "expert_droid"), ("pi05 droid", "e14_pi05_%s_k5_lr1e-4_t17_s%d", 2, "expert_droid")]
print("per-task significant changes (p<0.05, paired over seeds x 50 inits):")
for name, pat, n, arm in cases:
    sig = [(t[:40], x, y, p) for t in tasks for x, y, p in [paired(pat, n, arm, t)] if p < 0.05]
    print(f"  {name:14s}", "; ".join(f"{t} {x}:{y} p={p:.3f}" for t, x, y, p in sig) or "none")
REF = {"pick up the book on the right and place it on the cabinet shelf", "put the middle black bowl on the plate",
       "put the right moka pot on the stove", "stack the left bowl on the right bowl and place them in the tray"}
print("\nmean delta: tasks whose target object is picked out by a spatial word (4) vs the rest (13)")
R = json.load(open("per_task_rates.json"))
hm = {"X-VLA": "X-VLA", "GR00T": "GR00T", "pi05": "pi05", "DP": "DP", "FastWAM": "FastWAM"}
for h in R:
    for a in ("expert_scratch", "expert_droid"):
        dd = {t: R[h][a][t] - R[h]["none"][t] for t in tasks}
        r = np.mean([dd[t] for t in REF]); o = np.mean([dd[t] for t in tasks if t not in REF])
        print(f"  {h:8s} {a[7:]:8s} referent {r:+5.1f}  others {o:+5.1f}")
print("\nlibero_spatial k5 (3 seeds): per-task delta vs none")
for arm in ("expert_scratch", "expert_droid"):
    ts = sorted(d["e13_libero_spatial_none_k5_s0"])
    for t in ts:
        a = np.mean([v for s in range(3) for _, v in d[f"e13_libero_spatial_{arm}_k5_s{s}"][t]]) * 100
        b = np.mean([v for s in range(3) for _, v in d[f"e13_libero_spatial_none_k5_s{s}"][t]]) * 100
        print(f"  {arm[7:]:8s} {t[:75]:75s} none {b:5.1f} delta {a-b:+5.1f}")
