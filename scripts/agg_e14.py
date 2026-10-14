#!/usr/bin/env python
# E14 aggregation: pi0.5-DROID none / +SOPT scratch / +SOPT DROID, pooled McNemar over seeds.
#   python scripts/agg_e14.py [--tag _lr1e-4] [--x 25] [seeds ...]
import argparse, json, math
def comb(n,k): return math.factorial(n)//(math.factorial(k)*math.factorial(n-k))
p=argparse.ArgumentParser(); p.add_argument("--tag",default=""); p.add_argument("--x",default="10")
p.add_argument("seeds",nargs="*",type=int); a_=p.parse_args()
SEEDS=a_.seeds or [0,1,2]; X="" if a_.x=="10" else "_x"+a_.x
arms={a:["e14_pi05_%s_k5%s_s%d"%(a if a=="none" else "expert_"+a,a_.tag,s) for s in SEEDS] for a in ("none","scratch","droid")}
print("tag=%r n_exec=%s seeds=%s"%(a_.tag,a_.x,SEEDS))
R={}
for a,runs in arms.items():
    R[a]={}; per=[]
    for s,r in enumerate(runs):
        d=json.load(open("outputs/%s/rollouts_n50%s.json"%(r,X))); per.append(100*d["success_rate"])
        for e in d["episodes"]: R[a][(s,e["task"],e["init"])]=e["success"]
    m=sum(per)/len(per); sd=(sum((q-m)**2 for q in per)/max(1,len(per)-1))**.5
    print("%-8s %s %.1f +- %.1f"%(a,[round(q,1) for q in per],m,sd))
for a,b in [("droid","none"),("droid","scratch"),("scratch","none")]:
    ks=R[a].keys()&R[b].keys(); x=sum(R[a][k] and not R[b][k] for k in ks); y=sum(R[b][k] and not R[a][k] for k in ks)
    n=x+y; t=min(x,y); p=min(1.0,2*sum(comb(n,i) for i in range(t+1))/2.0**n) if n else 1
    print("%s vs %s: %d vs %d p=%.3g"%(a,b,x,y,p))
