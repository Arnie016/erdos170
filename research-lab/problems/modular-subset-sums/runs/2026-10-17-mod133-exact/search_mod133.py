#!/usr/bin/env python3
"""Exact canonical n=7, N=133 search. Self-contained; no external dependencies."""
import json, time, sys
from math import gcd
N=133; n=7; HALF=66; FULL=(1<<N)-1
reps=set()
for u in range(1,N):
    if gcd(u,N)!=1: continue
    b=frozenset(min((u*(1<<i))%N, N-(u*(1<<i))%N) for i in range(n))
    if len(b)!=n: raise RuntimeError('degenerate orbit')
    reps.add(b)
reps=[set(r) for r in reps]
start=time.monotonic(); counters=[0]*(n+1); samples=[]; first_bad=None
trace=None

def dfs(last, depth, sums, row):
    global first_bad
    counters[depth]+=1
    if depth==n:
        best=max(len(row_set & b) for b in reps for row_set in [set(row)])
        if best<n-1 and first_bad is None:
            first_bad={'canonical':list(row),'best_overlap':best, 'num_subset_sums':sums.bit_count()}
            print('FALSIFIER',json.dumps(first_bad), flush=True)
        if len(samples)<5:samples.append({'a':list(row),'best_overlap':best})
        return
    for a in range(last+1, HALF-(n-depth-1)+1):
        shift=((sums<<a)&FULL)|(sums>>(N-a))
        if (sums & shift)==0:
            dfs(a, depth+1, sums|shift, row+(a,))

dfs(0,0,1,())
result={'N':N,'n':n,'search_space':'all strictly increasing 7-subsets of 1..66','canonical_orbit_count':len(reps),'nodes_by_depth':counters,'first_bad':first_bad,'samples':samples,'wall_s':time.monotonic()-start,'all_completed':True}
with open(sys.argv[1],'w') as f: json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
