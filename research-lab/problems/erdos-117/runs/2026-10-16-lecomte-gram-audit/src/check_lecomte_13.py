#!/usr/bin/env python3
"""Strict test of a printed finite table, designed to fail when table mismatches."""
import hashlib, json, platform, sys, time
from pathlib import Path
start=time.perf_counter()
root=Path(__file__).resolve().parents[1]
raw=(root/'data/lecomte_13_vectors.json').read_bytes()
d=json.loads(raw); V=d['vectors']; G=d['printed_gram_upper_rows']
def a(x,y):
    return sum(x[j]*y[j+1]-x[j+1]*y[j] for j in (0,2,4))%3
def b(x,y):
    J=[[0]*6 for _ in range(6)]
    for i in (0,2,4):
        J[i][i+1]=1;J[i+1][i]=2
    yy=[sum(J[i][j]*y[j] for j in range(6))%3 for i in range(6)]
    return sum(x[i]*yy[i] for i in range(6))%3
def rank(M):
    A=[list(v) for v in M];r=0
    for col in range(6):
        z=next((k for k in range(r,len(A)) if A[k][col]%3),None)
        if z is None:continue
        A[r],A[z]=A[z],A[r]
        iv=pow(A[r][col]%3,-1,3)
        A[r]=[(u*iv)%3 for u in A[r]]
        for k in range(len(A)):
            if k!=r:
                c=A[k][col];A[k]=[(x-c*y)%3 for x,y in zip(A[k],A[r])]
        r+=1
    return r
assert len(V)==13 and all(len(v)==6 for v in V)
assert [len(r) for r in G]==list(range(12,0,-1))
assert a(V[0],V[1])==b(V[0],V[1])==1,'smoke test failed'
entries=[]
for i in range(13):
    for j in range(i+1,13):
        entries.append((a(V[i],V[j]),b(V[i],V[j]),G[i][j-i-1]))
assert len(entries)==78
assert all(x==y==z and x in (1,2) for x,y,z in entries),'source verification FAIL'
# These downstream tests do not run on the observed failed source table.
assert rank(V)==6
negative=list(map(list,V));negative[1]=negative[0]
assert a(negative[0],negative[1])==b(negative[0],negative[1])==0
print('strict acceptance PASS; SHA256',hashlib.sha256(raw).hexdigest())
