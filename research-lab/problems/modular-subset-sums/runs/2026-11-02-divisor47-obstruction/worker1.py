#!/usr/bin/env python3
"""Exact companion-matrix method: Res(F_d,P) via det(C^d-I)/P(1)."""
import sys,json,hashlib,time
from pathlib import Path
TARGET=[1,0,0,1,1,1,0,0,1]
POS=[1,0,1]
NEG=[1]*11

def mul(a,b):
    n=len(a); z=[[0]*n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            ai=a[i][k]
            if ai:
                for j in range(n):z[i][j]+=ai*b[k][j]
    return z

def power(a,p):
    n=len(a); x=[[int(i==j) for j in range(n)] for i in range(n)]
    while p:
        if p&1:x=mul(x,a)
        p>>=1
        if p:a=mul(a,a)
    return x

def det_bareiss(M):
    a=[r[:] for r in M];n=len(a);sgn=1;old=1
    for k in range(n-1):
        t=next((i for i in range(k,n) if a[i][k]!=0),None)
        if t is None:return 0
        if t!=k:a[k],a[t]=a[t],a[k];sgn=-sgn
        pivot=a[k][k]
        for i in range(k+1,n):
            for j in range(k+1,n):
                numerator=a[i][j]*pivot-a[i][k]*a[k][j]
                assert numerator%old==0
                a[i][j]=numerator//old
            a[i][k]=0
        old=pivot
    return sgn*a[-1][-1]

def resultant(d,P):
    degree=len(P)-1
    assert degree%2==0 and P[-1]==1 and sum(P)!=0
    C=[[0]*degree for _ in range(degree)]
    for j in range(degree-1):C[j+1][j]=1
    for i in range(degree):C[i][-1]=-P[i]
    X=power(C,d)
    for i in range(degree):X[i][i]-=1
    z=det_bareiss(X);assert z%sum(P)==0
    return z//sum(P)

start=time.perf_counter()
out={str(d):str(resultant(d,TARGET)) for d in (11,47)}
checks={'positive_11':str(resultant(11,POS)),'positive_47':str(resultant(47,POS)), 'zero_11':str(resultant(11,NEG))}
assert checks=={'positive_11':'1','positive_47':'1','zero_11':'0'}, checks
receipt={'method':'companion power and Bareiss', 'resultants':out, 'controls':checks,'internal_wall_seconds':time.perf_counter()-start,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
Path(sys.argv[1]).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
