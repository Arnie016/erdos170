import json, math, time, sys
from pathlib import Path
N=69; K=6; H=34; MASK=(1<<N)-1
nodes=[0]*(K+1); rejects=[0]*(K+1); accepted=[]
started=time.monotonic()
def dfs(prefix, sums, lo):
    d=len(prefix); nodes[d]+=1
    if d==K:
        accepted.append(prefix); return
    end=H-(K-d)+1
    for a in range(lo,end+1):
        shifted=((sums<<a)|(sums>>(N-a)))&MASK
        if sums&shifted:
            rejects[d+1]+=1; continue
        dfs(prefix+[a],sums|shifted,a+1)
dfs([],1,1)
units=[u for u in range(1,N) if math.gcd(u,N)==1]
orbits={}
for u in units:
    raw=[(u*(1<<i))%N for i in range(K)]
    can=tuple(sorted(min(x,N-x) for x in raw))
    assert len(set(can))==K
    orbits.setdefault(can,u)
results=[]
for A in accepted:
    S=set(A)
    base,u=max(orbits.items(), key=lambda item:len(S&set(item[0])))
    intersection=sorted(S&set(base))
    results.append({"canonical_set":A,"best_overlap":len(intersection),
                    "unit_witness":u,"canonical_dyadic":list(base),
                    "intersection":intersection})
out={"schema":"modular69.complete-canonical-v1","N":N,"n":K,
     "domain":{"canonical_min":1,"canonical_max":H,
               "canonical_full_combinations":math.comb(H,K),
               "original_full_combinations":math.comb(N-1,K),
               "method":"complete increasing DFS, collision pruning"},
     "nodes_by_depth":nodes,"rejected_extensions_by_depth":rejects,
     "canonical_count":len(results),"full_count":(1<<K)*len(results),
     "canonical_dyadic_orbits":len(orbits),
     "overlap_histogram":{str(x):sum(r["best_overlap"]==x for r in results) for x in range(K+1)},
     "records":results,"elapsed_seconds":time.monotonic()-started}
path=Path(sys.argv[1]);path.write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!="records"},indent=2))
