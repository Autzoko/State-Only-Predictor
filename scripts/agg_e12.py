#!/usr/bin/env python
# E12 aggregation: FastWAM none / +SOPT scratch / +SOPT DROID, pooled McNemar over seeds
import json, math
def comb(n,k): return math.factorial(n)//(math.factorial(k)*math.factorial(n-k))
import sys
SEEDS=[int(x) for x in sys.argv[1:]] or [0,1,2]  # E12 FastWAM; pass the finished seeds
arms={a:["e12_wam_%s_k5_s%d"%(a if a=="none" else "expert_"+a,s) for s in SEEDS] for a in ("none","scratch","droid")}
R={}
for a,runs in arms.items():
    R[a]={}; per=[]
    for s,r in enumerate(runs):
        d=json.load(open("outputs/%s/rollouts_n50.json"%r)); per.append(100*d["success_rate"])
        for e in d["episodes"]: R[a][(s,e["task"],e["init"])]=e["success"]
    m=sum(per)/len(per); sd=(sum((q-m)**2 for q in per)/max(1,len(per)-1))**.5
    print("%-8s %s %.1f +- %.1f"%(a,[round(q,1) for q in per],m,sd))
for a,b in [("droid","none"),("droid","scratch"),("scratch","none")]:
    ks=R[a].keys()&R[b].keys(); x=sum(R[a][k] and not R[b][k] for k in ks); y=sum(R[b][k] and not R[a][k] for k in ks)
    n=x+y; t=min(x,y); p=min(1.0,2*sum(comb(n,i) for i in range(t+1))/2.0**n) if n else 1
    print("%s vs %s: %d vs %d p=%.3g"%(a,b,x,y,p))
