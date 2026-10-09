import json,math,time
from collections import Counter
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/'results';OUT.mkdir(parents=True,exist_ok=True)
PAIRS=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)];VERTS=list(range(1,16))
def pair(mask,u,v):
    s=0
    for k,(i,j) in enumerate(PAIRS):
        if (mask>>k)&1:s^=(((u>>i)&1)&((v>>j)&1))^(((u>>j)&1)&((v>>i)&1))
    return s

def graph(m1,m2):
    adj=[0]*15
    for i,u in enumerate(VERTS):
     for j in range(i+1,15):
        v=VERTS[j]
        if pair(m1,u,v) or pair(m2,u,v):adj[i]|=1<<j;adj[j]|=1<<i
    return adj

def max_clique(adj):
    best=0
    def bk(Rsz,P,X):
        nonlocal best
        if Rsz+P.bit_count()<=best:return
        if not P and not X:best=max(best,Rsz);return
        union=P|X
        if union:
            uu=max((u for u in range(15) if (union>>u)&1),key=lambda u:(P&adj[u]).bit_count());cand=P&~adj[uu]
        else:cand=P
        while cand:
            lb=cand&-cand;v=lb.bit_length()-1
            bk(Rsz+1,P&adj[v],X&adj[v]);P&=~lb;X|=lb;cand&=~lb
            if Rsz+P.bit_count()<=best:break
    bk(0,(1<<15)-1,0);return max(best,1)

def greedy_upper(adj):
    colors=[-1]*15;mx=-1
    for v in sorted(range(15),key=lambda v:adj[v].bit_count(),reverse=True):
        used={colors[u] for u in range(15) if ((adj[v]>>u)&1) and colors[u]>=0};c=0
        while c in used:c+=1
        colors[v]=c;mx=max(mx,c)
    return mx+1

def chromatic(adj,lower):
    n=15;upper=greedy_upper(adj)
    if lower>=upper:return lower
    def kcolor(k):
        col=[-1]*n;sat=[0]*n;deg=[adj[v].bit_count() for v in range(n)]
        def rec(done):
            if done==n:return True
            v=max((v for v in range(n) if col[v]<0),key=lambda x:(sat[x].bit_count(),deg[x]));forbidden=sat[v]
            for c in range(k):
                if (forbidden>>c)&1:continue
                col[v]=c;changed=[];nb=adj[v]
                for u in range(n):
                    if ((nb>>u)&1) and col[u]<0 and not ((sat[u]>>c)&1):sat[u]|=1<<c;changed.append(u)
                if rec(done+1):return True
                for u in changed:sat[u]&=~(1<<c)
                col[v]=-1
            return False
        return rec(0)
    for k in range(lower,upper+1):
        if kcolor(k):return k
    raise RuntimeError

def common_rad_dim(m1,m2):
    rows=[]
    for mask in (m1,m2):
     for j in range(4):
        r=0
        for i in range(4):
            if pair(mask,1<<i,1<<j):r|=1<<i
        rows.append(r)
    basis=[]
    for x in rows:
        y=x
        for b in basis:y=min(y,y^b)
        if y:basis.append(y);basis.sort(reverse=True)
    return 4-len(basis)

t0=time.time();counts=Counter();max_ratio=-1;max_maps=[];max_pair=None
for m1 in range(64):
 for m2 in range(64):
    adj=graph(m1,m2);om=max_clique(adj);a=max(1,chromatic(adj,om));counts[(om,a)]+=1;ratio=a/(2**(om/2))
    if ratio>max_ratio+1e-15:max_ratio=ratio;max_maps=[(m1,m2,common_rad_dim(m1,m2))];max_pair=(om,a)
    elif abs(ratio-max_ratio)<1e-15:max_maps.append((m1,m2,common_rad_dim(m1,m2)))
out={'status':'PASS','family':'ordered pairs of alternating forms on F2^4, 64^2=4096 maps','maps_checked':4096,'vertex_model':'15 nonzero vectors; common-radical twins retained','pair_counts':{f'{k[0]},{k[1]}':v for k,v in sorted(counts.items())},'max_pair':max_pair,'max_ratio':max_ratio,'benchmark_C0':3/(2*math.sqrt(2)),'max_maps_count':len(max_maps),'max_maps_first_50':max_maps[:50],'elapsed_seconds':time.time()-t0}
(OUT/'worker1_census.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
