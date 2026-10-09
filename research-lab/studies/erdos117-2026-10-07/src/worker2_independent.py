import json,math,time
from pathlib import Path
from collections import Counter
from functools import lru_cache
from itertools import combinations
OUT=Path(__file__).resolve().parents[1]/'results';OUT.mkdir(parents=True,exist_ok=True)
PAIRS=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)];NZ=list(range(1,16));FULL=(1<<15)-1

def bil(mask,u,v):
    s=0
    for k,(i,j) in enumerate(PAIRS):
        if (mask>>k)&1:s^=(((u>>i)&1)&((v>>j)&1))^(((u>>j)&1)&((v>>i)&1))
    return s

def span_mask(gens):
    xs={0}
    for g in gens:xs|={x^g for x in list(xs)}
    return sum(1<<(x-1) for x in xs if x)
subs={0}
for r in range(1,5):
 for gs in combinations(NZ,r):subs.add(span_mask(gs))
SUBS=sorted(subs);assert len(SUBS)==67

def graph_adj(m1,m2):
    A=[0]*15
    for i,u in enumerate(NZ):
     for j in range(i+1,15):
        v=NZ[j]
        if bil(m1,u,v) or bil(m2,u,v):A[i]|=1<<j;A[j]|=1<<i
    return A

def clique_dp(A):
    @lru_cache(None)
    def f(S):
        if not S:return 0
        lb=S&-S;v=lb.bit_length()-1
        return max(f(S^lb),1+f((S^lb)&A[v]))
    return max(1,f(FULL))

def maximal_isotropic_masks(m1,m2):
    iso=[]
    for S in SUBS:
        elems=[i+1 for i in range(15) if (S>>i)&1];ok=True
        for ii,u in enumerate(elems):
            for v in elems[ii+1:]:
                if bil(m1,u,v) or bil(m2,u,v):ok=False;break
            if not ok:break
        if ok:iso.append(S)
    return [S for S in iso if not any(S!=T and (S&T)==S for T in iso)]

def cover_number(m1,m2):
    M=maximal_isotropic_masks(m1,m2)
    if any(S==FULL for S in M):return 1
    bypoint=[[] for _ in range(15)]
    for S in M:
     for i in range(15):
        if (S>>i)&1:bypoint[i].append(S)
    maxsz=max(s.bit_count() for s in M);best=[16];seen={}
    def rec(covered,k):
        if k>=best[0]:return
        if covered==FULL:best[0]=k;return
        prev=seen.get(covered)
        if prev is not None and prev<=k:return
        seen[covered]=k;rem=FULL^covered
        if k+(rem.bit_count()+maxsz-1)//maxsz>=best[0]:return
        p=min((i for i in range(15) if (rem>>i)&1),key=lambda i:len(bypoint[i]))
        for S in sorted(bypoint[p],key=lambda S:(S&rem).bit_count(),reverse=True):rec(covered|S,k+1)
    rec(0,0);return best[0]

t0=time.time();C=Counter();maxratio=-1;maxpair=None;failures=[]
for m1 in range(64):
 for m2 in range(64):
    A=graph_adj(m1,m2);om=clique_dp(A);cov=cover_number(m1,m2);C[(om,cov)]+=1;r=cov/(2**(om/2))
    if r>maxratio:maxratio=r;maxpair=(om,cov,m1,m2)
    if cov<om:failures.append([m1,m2,om,cov])
out={'status':'PASS' if not failures else 'FAIL','method':'independent subset-DP clique + exact cover by maximal vector-valued isotropic subspaces','subspaces_enumerated':len(SUBS),'maps_checked':4096,'pair_counts':{f'{a},{b}':n for (a,b),n in sorted(C.items())},'max_ratio':maxratio,'max_pair_and_example':maxpair,'benchmark_C0':3/(2*math.sqrt(2)),'failures':failures[:10],'elapsed_seconds':time.time()-t0}
(OUT/'worker2_independent.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
