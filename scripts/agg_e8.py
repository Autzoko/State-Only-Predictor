import json, sys
from math import comb
ph = sys.argv[1] if len(sys.argv) > 1 else "k5"
arms = {"bc": "e8_dino_bc", "scratch": "e8_dino_trunk_scratch", "droid": "e8_dino_trunk_droid"}
R = {}
for a, p in arms.items():
    R[a] = {}
    for s in range(3):
        for e in json.load(open(f"outputs/{p}_{ph}_s{s}/rollouts_n50.json"))["episodes"]:
            R[a][(s, e["task"], e["init"])] = e["success"]
def mc(a, b):
    ks = R[a].keys() & R[b].keys()
    x = sum(R[a][k] and not R[b][k] for k in ks); y = sum(R[b][k] and not R[a][k] for k in ks)
    n = x + y; t = min(x, y)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(t + 1)) / 2 ** n) if n else 1.0
    return x, y, p
for a in arms:
    per = [100 * sum(v for k, v in R[a].items() if k[0] == s) / sum(1 for k in R[a] if k[0] == s) for s in range(3)]
    m = sum(per) / 3; sd = (sum((q - m) ** 2 for q in per) / 2) ** .5
    print(f"{a:8s}", [round(q, 1) for q in per], f"{m:.1f} ± {sd:.1f}")
for a, b in [("droid", "scratch"), ("droid", "bc"), ("scratch", "bc")]:
    x, y, p = mc(a, b); print(f"{a} vs {b}: {x} vs {y} p={p:.3g}")
