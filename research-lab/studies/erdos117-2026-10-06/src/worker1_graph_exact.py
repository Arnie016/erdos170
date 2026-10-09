import json,itertools,time
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/'results';OUT.mkdir(parents=True,exist_ok=True)
def bits4(v):return tuple((v>>i)&1 for i in range(4))
def f(case,v,w):
    x,y,z,t=bits4(v);xp,yp,zp,tp=bits4(w)
    if case=='A':return x&yp,z&tp
    if case=='B':return x&yp,y&zp
    raise ValueError(case)
def mul(case,g,h):
    v,c=g;w,d=h;a,b=f(case,v,w)
    return v^w,c^d^a^(b<<1)
def inv(case,g):
    for v in range(16):
     for c in range(4):
        h=(v,c)
        if mul(case,g,h)==(0,0) and mul(case,h,g)==(0,0):return h
    raise AssertionError('no inverse')
def comm(case,g,h):return mul(case,mul(case,mul(case,inv(case,g),inv(case,h)),g),h)
def adjacent(case,g,h):return comm(case,g,h)!=(0,0)
def max_clique(adj):
    n=len(adj);best=[];neigh=[sum(1<<j for j in range(n) if adj[i][j]) for i in range(n)]
    def bk(R,P,X):
        nonlocal best
        if not P and not X:
            if R.bit_count()>len(best):best=[i for i in range(n) if R>>i&1]
            return
        if R.bit_count()+P.bit_count()<=len(best):return
        union=P|X
        u=max((i for i in range(n) if union>>i&1),key=lambda i:(P&neigh[i]).bit_count(),default=0)
        cand=P&~neigh[u]
        while cand:
            v=(cand&-cand).bit_length()-1
            bk(R|1<<v,P&neigh[v],X&neigh[v])
            P&=~(1<<v);X|=1<<v;cand&=~(1<<v)
    bk(0,(1<<n)-1,0)
    return best

def chromatic_number(adj,lower):
    n=len(adj);deg=[sum(row) for row in adj]
    def kcolor(k):
        color=[-1]*n;sat=[set() for _ in range(n)];un=set(range(n))
        def rec():
            if not un:return True
            v=max(un,key=lambda x:(len(sat[x]),deg[x]));forbidden=sat[v];un.remove(v)
            for c in range(k):
                if c in forbidden:continue
                color[v]=c;changed=[]
                for u in un:
                    if adj[v][u] and c not in sat[u]:sat[u].add(c);changed.append(u)
                if rec():return True
                for u in changed:sat[u].remove(c)
                color[v]=-1
            un.add(v);return False
        ok=rec();return ok,color[:] if ok else None
    for k in range(lower,n+1):
        ok,col=kcolor(k)
        if ok:return k,col
    raise AssertionError

def run(case):
    els=[(v,c) for v in range(16) for c in range(4)];e=(0,0)
    for a in els:
        assert mul(case,e,a)==a and mul(case,a,e)==a
        inv(case,a)
    assoc=0
    for a in els:
     for b in els:
      ab=mul(case,a,b)
      for c in els:
        assert mul(case,ab,c)==mul(case,a,mul(case,b,c));assoc+=1
    center=[g for g in els if all(not adjacent(case,g,h) for h in els)]
    reps=[(v,0) for v in range(16)]
    adj=[[i!=j and adjacent(case,g,h) for j,h in enumerate(reps)] for i,g in enumerate(reps)]
    cl=max_clique(adj);chi,col=chromatic_number(adj,len(cl))
    return {'case':case,'order':64,'associativity_triples_checked':assoc,'center_size':len(center),'center_elements':center,'quotient_graph_vertices':16,'omega':len(cl),'clique_v':[reps[i][0] for i in cl],'chromatic_number':chi,'coloring':col}
out={'status':'PASS','method':'full group multiplication + quotient graph exact Bron-Kerbosch and DSATUR','cases':[run('A'),run('B')]}
(OUT/'worker1_graph_exact.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
