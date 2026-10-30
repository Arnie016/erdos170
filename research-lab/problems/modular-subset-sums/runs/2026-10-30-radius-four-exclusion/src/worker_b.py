#!/usr/bin/env python3
"""Independent cyclic binary recurrence and exact four-cell graph; check against A."""
import json,sys,math,time
N=517
def compute(radius):
    H={0,1,N-1,radius,N-radius}
    S=set(range(N))-H
    records={}
    for a in range(1,N):
        if math.gcd(a,N)!=1: continue
        b=set();prev=0; valid=True
        for j in range(1,N):
            x=(j*a)%N
            v=(x in S)-prev
            if v not in (0,1):valid=False;break
            if v:b.add(x)
            prev=v
        if not valid or prev!=0: continue
        if len(b)!=256 or b.isdisjoint(b|{0}) is False and 0 in b: raise AssertionError('size')
        if not b.isdisjoint({(x+a)%N for x in b}):raise AssertionError('overlap')
        if b|{(x+a)%N for x in b}!=S:raise AssertionError('union')
        records[a]=b
    adj={a:set() for a in records}; edges=[]
    V=sorted(records)
    for i,a in enumerate(V):
        Ba=records[a]
        for b in V[i+1:]:
            C=Ba & records[b]
            if len(C)!=128:continue
            U=set()
            for t in (0,a,b,a+b):
                U.update({(x+t)%N for x in C})
            if len(U)!=512 or U!=S:continue
            # This verifies no overlaps, as each of the four blocks has 128 elements.
            adj[a].add(b);adj[b].add(a);edges.append([a,b])
    # Bron-Kerbosch with pivoting enumerates all maximal cliques, pruning only via a valid size bound.
    best=[]
    def bronk(R,P,X):
        nonlocal best
        if len(R)+len(P)<=len(best):return
        if not P and not X:
            if len(R)>len(best):best=R[:]
            return
        u=max(P|X,key=lambda u:len(P & adj[u])) if P or X else None
        for v in sorted(P-adj[u]) if u is not None else sorted(P):
            bronk(R+[v],P & adj[v],X & adj[v]);P.remove(v);X.add(v)
    bronk([],set(V),set())
    # Independent explicit certificate validation for the returned clique.
    assert all(b in adj[a] for i,a in enumerate(best) for b in best[i+1:])
    return {'vertices':V,'edges':edges,'clique_size':len(best),'clique':sorted(best)}
t0=time.monotonic()
ctrl=compute(3)
assert (len(ctrl['vertices']),len(ctrl['edges']),ctrl['clique_size'])==(156,308,9), (len(ctrl['vertices']),len(ctrl['edges']),ctrl['clique_size'])
tgt=compute(4)
a=json.load(open(sys.argv[1]))['target']
assert tgt['vertices']==a['vertices'] and tgt['edges']==a['edges'] and tgt['clique_size']==a['clique_size'], 'independent graph mismatch'
result={'control':{'eligible':156,'edges':308,'max_clique':9},'target':tgt,'full_graph_agreement':True,'elapsed_seconds':time.monotonic()-t0}
with open(sys.argv[2],'w') as f:json.dump(result,f,indent=2)
print('B eligible={} edges={} clique={}'.format(len(tgt['vertices']),len(tgt['edges']),tgt['clique_size']))
