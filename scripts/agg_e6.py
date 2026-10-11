import json, statistics as st, sys
from math import comb
tag = sys.argv[1]  # k5 | all
arms = {"bc": "e4b_bc_{t}{s}", "trunk_scratch": "e4b_trunkbc_scratch_{t}{s}", "trunk_droid": "e4b_trunkbc_droid_{t}{s}",
        "md_scratch": "e6_md_scratch_{t}_s{n}", "md_droid": "e6_md_droid_{t}_s{n}"}
if tag == "k5": arms["md_droid_noaux"] = "e6_md_droid_noaux_{t}_s{n}"
R = {}
for a, pat in arms.items():
    R[a] = {}
    for n in range(3):
        d = json.load(open(f"outputs/{pat.format(t=tag, s='' if n == 0 else f'_s{n}', n=n)}/rollouts.json"))
        R[a].update({(n, e["task"], e["init"]): e["success"] for e in d["episodes"]})
for a in arms:
    r = [100 * st.mean(v for k, v in R[a].items() if k[0] == n) for n in range(3)]
    print(f"{a:16s}", [round(x, 1) for x in r], f"{st.mean(r):.1f} ± {st.stdev(r):.1f}")
def mc(x, y):
    b = sum(R[x][k] and not R[y][k] for k in R[x]); c = sum(R[y][k] and not R[x][k] for k in R[x]); n = b + c
    p = min(1, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
    print(f"{x} vs {y}: {b} vs {c}  p={p:.3g}")
pairs = [("md_droid", "md_scratch"), ("md_droid", "trunk_droid"), ("md_droid", "bc"), ("md_scratch", "trunk_scratch"), ("md_scratch", "bc")]
if tag == "k5": pairs.append(("md_droid", "md_droid_noaux"))
for p in pairs: mc(*p)
tasks = sorted({k[1] for k in R["md_droid"]})
for t in tasks:
    print(t[:40].ljust(40), {a: round(100 * st.mean(v for k, v in R[a].items() if k[1] == t)) for a in arms})
