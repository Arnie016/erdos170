#!/usr/bin/env python3
"""Independent kernel enumeration via span-closure BFS; clique DP and isotropic subspace covering.
No imports from worker1. Domain all linear K<=F2^6 (2825).
"""
from collections import Counter,deque
from functools import lru_cache
from pathlib import Path
import json,time
T0=time.monotonic();OUT=Path(__file__).resolve().parents[1]/'results'; OUT.mkdir(exist_ok=True)

def bits(mask):
    a=[]
    while mask:
        p=mask&-mask;a.append(p.bit_length()-1);mask-=p
    return a

def enumerate_subspaces(n):
    start=1
    seen={start};queue=deque([start])
    while queue:
        sm=queue.popleft(); elems=bits(sm)
        for v in range(1,1<<n):
            if (sm>>v)&1:continue
            tm=sm
            for a in elems:tm |= 1<<(a^v)
            if tm not in seen:
                seen.add(tm);queue.append(tm)
    return seen

KERNELS=enumerate_subspaces(6)
assert len(KERNELS)==2825,('wrong kernel count',len(KERNELS))
SUBS=enumerate_subspaces(4)
assert len(SUBS)==67,('wrong quotient subspace count',len(SUBS))
PAIRS=[(i,j) for i in range(4) for j in range(i+1,4)]

def form_eval(c,v,w):
    res=0
    for t,(i,j) in enumerate(PAIRS):
        if (c>>t)&1:
            res ^= ((v>>i)&1)&((w>>j)&1)
            res ^= ((v>>j)&1)&((w>>i)&1)
    return res

def graph_and_independents(k):
    kb=bits(k)
    dual=[c for c in range(1,64) if all(((c&s).bit_count()%2)==0 for s in kb)]
    edge=[0]*15
    for v in range(1,16):
        for w in range(v+1,16):
            if any(form_eval(c,v,w) for c in dual):
                edge[v-1]|=1<<(w-1);edge[w-1]|=1<<(v-1)
    good=[]
    for sub in SUBS:
        m=sub>>1
        if not m:continue
        mem=bits(m)
        if all((edge[i]&m)==0 for i in mem):good.append(m)
    maximal=[u for u in good if not any(u!=v and (u&v)==u for v in good)]
    return edge,maximal

def max_clique_dp(adj):
    @lru_cache(None)
    def f(s):
        if not s:return 0
        vbit=s&-s;v=vbit.bit_length()-1
        rest=s^vbit
        return max(f(rest),1+f(rest&adj[v]))
    return f((1<<15)-1)

def min_isotropic_cover(subspaces):
    choices=[[] for _ in range(15)]
    for u in subspaces:
        for i in bits(u):choices[i].append(u)
    assert all(choices)
    @lru_cache(None)
    def solve(left):
        if left==0:return 0
        b=left&-left;v=b.bit_length()-1
        return 1+min(solve(left&~u) for u in choices[v])
    return solve((1<<15)-1)

hist=Counter(); results=[]; first_gap=None
for k in sorted(KERNELS):
    adj,iso=graph_and_independents(k)
    omega=max_clique_dp(adj)
    a=min_isotropic_cover(iso)
    hist[(omega,a)]+=1
    if a>omega and first_gap is None:first_gap={'kernel':str(k),'omega':omega,'a':a,'adj':adj,'isotropic_subspace_masks':iso}
    if a<omega:raise AssertionError('cover smaller than clique')
    results.append([str(k),omega,a])
original=OUT/'worker1_kernels.jsonl'
reference={v[0]:(v[1],v[2]) for v in (json.loads(line) for line in original.read_text().splitlines())}
disagree=[p for p in results if reference.get(p[0])!=(p[1],p[2])]
assert not disagree, ('independent method discrepancy',disagree[:3])
with (OUT/'worker2_kernels.jsonl').open('w') as out:
    for p in results:out.write(json.dumps(p)+'\n')
summary={'status':'PASS','method':'span-closure BFS kernel enumeration; dual scalar-form evaluation; clique subset DP; minimum isotropic-subspace cover','kernels_checked':len(KERNELS),'subspaces_of_V':len(SUBS),'histogram':{f'{w},{a}':count for (w,a),count in sorted(hist.items())},'first_gap':first_gap,'disagreements_with_worker1':len(disagree),'seconds':time.monotonic()-T0}
(OUT/'worker2_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary))
