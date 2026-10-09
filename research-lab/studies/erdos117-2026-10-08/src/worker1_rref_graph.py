#!/usr/bin/env python3
"""Independent exact GF(2) RREF subspace enumeration; graph clique + k-colouring.
Domain: all subspaces of F2^6. Exit nonzero if count != 2825.
"""
from itertools import combinations
from collections import Counter
from pathlib import Path
import json, time, sys

START=time.monotonic()
OUT=Path(__file__).resolve().parents[1]/'results'
OUT.mkdir(parents=True,exist_ok=True)
N=15; FULL=(1<<N)-1
pairs=list(combinations(range(4),2))

def wedge(v,w):
    z=0
    for k,(i,j) in enumerate(pairs):
        z |= ((((v>>i)&1)&((w>>j)&1)) ^ (((v>>j)&1)&((w>>i)&1)))<<k
    return z
wtab=[[wedge(i,j) for j in range(16)] for i in range(16)]

def all_kernels_rref():
    for d in range(7):
        for pivots in combinations(range(6),d):
            free=[(row,j) for row,p in enumerate(pivots) for j in range(p+1,6) if j not in pivots]
            for choice in range(1<<len(free)):
                basis=[1<<p for p in pivots]
                for i,(row,j) in enumerate(free):
                    if (choice>>i)&1: basis[row] |=1<<j
                space=[0]
                for b in basis:
                    space += [x^b for x in space]
                mask=sum(1<<x for x in space)
                yield mask,basis

def graph(kmask):
    adj=[0]*15
    for v in range(1,16):
        for w in range(v+1,16):
            if not (kmask>>wtab[v][w])&1:
                adj[v-1]|=1<<(w-1)
                adj[w-1]|=1<<(v-1)
    return adj

def clique(adj):
    best=[]
    def walk(candidates, chosen):
        nonlocal best
        if chosen.bit_count()+candidates.bit_count()<=len(best):return
        if candidates==0:
            if chosen.bit_count()>len(best):best=[i+1 for i in range(N) if (chosen>>i)&1]
            return
        while candidates:
            bit = candidates & -candidates
            idx=bit.bit_length()-1
            candidates^=bit
            walk(candidates & adj[idx], chosen|bit)
            if chosen.bit_count()+candidates.bit_count()<=len(best):return
    walk(FULL,0)
    return len(best),best

def colorable(adj,k):
    colors=[-1]*N
    neighbor_colors=[0]*N
    degrees=[a.bit_count() for a in adj]
    def dfs(uncolored, used):
        if not uncolored: return True
        choice=-1; score=(-1,-1)
        rem=uncolored
        while rem:
            b=rem & -rem; v=b.bit_length()-1; rem^=b
            s=(neighbor_colors[v].bit_count(), degrees[v])
            if s>score: score=s; choice=v
        v=choice; poss=~neighbor_colors[v] & ((1<<min(used+1,k))-1)
        while poss:
            bit=poss&-poss; c=bit.bit_length()-1; poss^=bit
            colors[v]=c
            affected=[]
            rem=adj[v]&uncolored
            while rem:
                q=rem&-rem; u=q.bit_length()-1; rem^=q
                old=neighbor_colors[u]
                if not (old>>c)&1:
                    neighbor_colors[u]=old | bit
                    affected.append(u)
            if dfs(uncolored^(1<<v), max(used,c+1)):return True
            for u in affected: neighbor_colors[u]^=bit
            colors[v]=-1
        return False
    return colors if dfs(FULL,0) else None

if __name__=='__main__':
    seen=set(); dist=Counter(); first_gap=None; data=[]; maxratio=(0,None)
    for kmask,basis in all_kernels_rref():
        if kmask in seen: raise RuntimeError(f'duplicate kernel {kmask}')
        seen.add(kmask)
        adj=graph(kmask)
        omega,wit=clique(adj)
        chi=omega
        col=colorable(adj,chi)
        while col is None:
            chi+=1
            col=colorable(adj,chi)
        dist[f'{omega},{chi}']+=1
        data.append([str(kmask),omega,chi])
        ratio=chi/(2**(omega/2))
        if ratio>maxratio[0]:maxratio=(ratio,(kmask,basis,omega,chi))
        if omega<chi and first_gap is None:
            first_gap={'kernel_mask_decimal':str(kmask),'rref_basis':basis,'omega':omega,'a':chi,'clique_witness':wit,'coloring':col,'adjacency_masks':adj}
    assert len(seen)==2825, ('missing RREF subspaces',len(seen))
    result={'status':'PASS','method':'RREF enumeration + clique backtracking + exact DSATUR k-colorability','finite_domain':'all 2825 kernels K <= Λ²(F2^4) ≅ F2^6','number_of_kernels':len(seen),'histogram':dict(sorted(dist.items())),'chromatic_gap_cases':sum(n for key,n in dist.items() if int(key.split(',')[0])<int(key.split(',')[1])),'first_chromatic_gap':first_gap,'max_normalized_ratio':maxratio,'seconds':time.monotonic()-START,'data_file':'worker1_kernels.jsonl'}
    with (OUT/'worker1_kernels.jsonl').open('w') as f:
        for row in data:f.write(json.dumps(row)+'\n')
    (OUT/'worker1_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'status':result['status'],'number_of_kernels':len(seen),'histogram':result['histogram'],'first_gap':first_gap,'max_ratio':maxratio,'seconds':result['seconds']}))
