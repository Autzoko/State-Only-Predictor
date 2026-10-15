#!/usr/bin/env python
# E19 aggregation: SOPT dropout vs plain expert_droid vs none, pooled McNemar over seeds.
#   python scripts/agg_e19.py [libero90|libero_spatial|...] [cd05]
import json, math, sys
def comb(n,k): return math.factorial(n)//(math.factorial(k)*math.factorial(n-k))
suite=sys.argv[1] if len(sys.argv)>1 else "libero90"; cdt=sys.argv[2] if len(sys.argv)>2 else "cd05"
if suite=="libero90":
    arms={"none":["e5_vla_none_k5","e5_vla_none_k5_s1","e5_vla_none_k5_s2"],
          "droid":["e7_vla_expert_droid_k5_s%d"%s for s in range(3)],
          "droid_"+cdt:["e19_vla_expert_droid_%s_k5_s%d"%(cdt,s) for s in range(3)]}
else:
    arms={"none":["e13_%s_none_k5_s%d"%(suite,s) for s in range(3)],
          "droid":["e13_%s_expert_droid_k5_s%d"%(suite,s) for s in range(3)],
          "droid_"+cdt:["e19_%s_expert_droid_%s_k5_s%d"%(suite,cdt,s) for s in range(3)]}
R={}
for a,runs in arms.items():
    R[a]={}; per=[]
    for s,r in enumerate(runs):
        try: d=json.load(open("outputs/%s/rollouts_n50.json"%r))
        except FileNotFoundError: continue
        per.append(100*d["success_rate"])
        for e in d["episodes"]: R[a][(s,e["task"],e["init"])]=e["success"]
    m=sum(per)/max(1,len(per)); sd=(sum((q-m)**2 for q in per)/max(1,len(per)-1))**.5
    print("%-12s %s %.1f +- %.1f"%(a,[round(q,1) for q in per],m,sd))
c="droid_"+cdt
for a,b in [(c,"none"),(c,"droid"),("droid","none")]:
    ks=R[a].keys()&R[b].keys(); x=sum(R[a][k] and not R[b][k] for k in ks); y=sum(R[b][k] and not R[a][k] for k in ks)
    n=x+y; t=min(x,y); p=min(1.0,2*sum(comb(n,i) for i in range(t+1))/2.0**n) if n else 1
    print("%s vs %s: %d vs %d p=%.3g (n=%d)"%(a,b,x,y,p,len(ks)))
