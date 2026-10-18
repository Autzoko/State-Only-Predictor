# Per-task drop-pattern analysis (2026-10-18). Input: per_task_dump.json = {run: {task: [[init, success], ...]}} dumped from outputs/*/rollouts_n50.json; run from the directory holding it.
import json, numpy as np, math
d = json.load(open("per_task_dump.json"))
tasks = json.load(open("/Users/ll5582/Desktop/NYU AIR Lab/State-Only Pretrain/configs/libero90_17_tasks.json"))
H = {"DP": ("e21_dp_%s_k5_t17_s%d", 3), "FastWAM": ("e12_wam_%s_k5_t17_s%d", 3), "GR00T": ("e22_groot_%s_k5_t17_s%d", 3),
     "pi05": ("e14_pi05_%s_k5_lr1e-4_t17_s%d", 2), "X-VLA": ("e20_xvla_%s_k5_t17_s%d", 3)}
def rate(h, arm):
    pat, n = H[h]; out = {}
    for t in tasks:
        v = [s for k in range(n) for _, s in d[pat % (arm, k)][t]]
        out[t] = 100 * np.mean(v)
    return out
short = {t: t.replace("pick up the ", "").replace("put the ", "")[:38] for t in tasks}
R = {h: {a: rate(h, a) for a in ("none", "expert_scratch", "expert_droid")} for h in H}
print(f"{'task':40s}" + "".join(f"{h+' S':>9s}{h+' D':>9s}" for h in H))
for t in tasks:
    print(f"{short[t]:40s}" + "".join(f"{R[h]['expert_scratch'][t]-R[h]['none'][t]:+9.1f}{R[h]['expert_droid'][t]-R[h]['none'][t]:+9.1f}" for h in H))
# correlations of per-task deltas
cases = {f"{h} {a[7]}": np.array([R[h][a][t] - R[h]["none"][t] for t in tasks]) for h in H for a in ("expert_scratch", "expert_droid")}
names = list(cases)
print("\ncorrelation of per-task deltas (Spearman)")
from itertools import combinations
def rk(x): return np.argsort(np.argsort(x))
M = np.array([[np.corrcoef(rk(cases[a]), rk(cases[b]))[0, 1] for b in names] for a in names])
print(" " * 12 + "".join(f"{n:>11s}" for n in names))
for i, n in enumerate(names): print(f"{n:12s}" + "".join(f"{M[i,j]:11.2f}" for j in range(len(names))))
# baseline difficulty vs delta
print("\ncorr(none success, delta) per case")
for n in names:
    h = n.rsplit(" ", 1)[0]
    base = np.array([R[h]["none"][t] for t in tasks]); print(f"{n:12s} {np.corrcoef(base, cases[n])[0,1]:+.2f}")
json.dump({h: {a: R[h][a] for a in R[h]} for h in R}, open("per_task_rates.json", "w"), indent=1)
