import json, math
def comb(n,k): return math.factorial(n)//(math.factorial(k)*math.factorial(n-k))
arms={"none":["e5_vla_none_k5","e5_vla_none_k5_s1","e5_vla_none_k5_s2"],
 "scratch":["e7_vla_expert_scratch_k5_s%d"%s for s in range(3)],
 "droid":["e7_vla_expert_droid_k5_s%d"%s for s in range(3)]}
R={}
for a,runs in arms.items():
    R[a]={}; per=[]
    for s,r in enumerate(runs):
        d=json.load(open("outputs/%s/rollouts_n50.json"%r)); per.append(100*d["success_rate"])
        for e in d["episodes"]: R[a][(s,e["task"],e["init"])]=e["success"]
    m=sum(per)/3; sd=(sum((q-m)**2 for q in per)/2)**.5
    print("%-8s %s %.1f +- %.1f"%(a,[round(q,1) for q in per],m,sd))
for a,b in [("droid","none"),("droid","scratch"),("scratch","none")]:
    ks=R[a].keys()&R[b].keys(); x=sum(R[a][k] and not R[b][k] for k in ks); y=sum(R[b][k] and not R[a][k] for k in ks)
    n=x+y; t=min(x,y); p=min(1.0,2*sum(comb(n,i) for i in range(t+1))/2.0**n) if n else 1
    print("%s vs %s: %d vs %d p=%.3g"%(a,b,x,y,p))
