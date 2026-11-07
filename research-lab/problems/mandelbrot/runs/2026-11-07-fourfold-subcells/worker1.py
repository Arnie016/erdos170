#!/usr/bin/env python3
"""Predeclared domain: 9 previously unproved closed 1/1000 cells,
subdivide each into 4x4 closed 1/4000 cells. Certify every point when
|z_6(c)|^2 > 4 via exact rational interval Horner.
"""
import json
from fractions import Fraction as Q

HOLES=[(3,-9),(4,-8),(5,-8),(5,-7),(6,-7),(6,-6),(7,-6),(7,-5),(8,-4)]
C0=(Q(-3,4),Q(1,2)); ZERO=(Q(0),Q(0));ONE=(Q(1),Q(0))
def add(a,b): return (a[0]+b[0],a[1]+b[1])
def mul(a,b): return (a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0])
p=[ZERO]
for _ in range(6):
    n=[ZERO]*(2*len(p)-1)
    for i,a in enumerate(p):
        for j,b in enumerate(p): n[i+j]=add(n[i+j],mul(a,b))
    n[0]=add(n[0],C0)
    if len(n)==1:n.append(ONE)
    else:n[1]=add(n[1],ONE)
    p=n
assert len(p)==33

def interval_mul(a,b):
    q=(a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]);return (min(q),max(q))
def plus(a,b):return (a[0]+b[0],a[1]+b[1])
def minus(a,b):return (a[0]-b[1],a[1]-b[0])
def times(a,b):
    return (minus(interval_mul(a[0],b[0]),interval_mul(a[1],b[1])),
            plus(interval_mul(a[0],b[1]),interval_mul(a[1],b[0])))
def pt(a):return ((a[0],a[0]),(a[1],a[1]))
def eval_box(x,y):
    v=pt(p[-1]);z=(x,y)
    for a in reversed(p[:-1]):
        t=times(v,z);q=pt(a);v=(plus(t[0],q[0]),plus(t[1],q[1]))
    return v
def dmin(x):
    return Q(0) if x[0]<=0<=x[1] else min(abs(x[0]),abs(x[1]))
def accepted(x,y):
    v=eval_box(x,y);return dmin(v[0])**2+dmin(v[1])**2>Q(4)
def admissible(i,j):
    def near(k):return 0 if k<=0<=k+1 else min(abs(k),abs(k+1))
    return near(i)**2+near(j)**2>=16 and max(abs(i),abs(i+1))**2+max(abs(j),abs(j+1))**2<=100
assert len(HOLES)==len(set(HOLES))==9
assert all(admissible(i,j) for i,j in HOLES)
assert not accepted((Q(3,4),Q(3,4)),(Q(-1,2),Q(-1,2))) # c=0
assert accepted((Q(0),Q(0)),(Q(0),Q(0))) # c=c0
ok=[];allcells=[]
for i,j in HOLES:
  for a in range(4):
    for b in range(4):
      key=[i,j,a,b]
      # delta rectangle relative to c0, not absolute c
      x=(Q(4*i+a,4000),Q(4*i+a+1,4000))
      y=(Q(4*j+b,4000),Q(4*j+b+1,4000))
      allcells.append(key)
      if accepted(x,y):ok.append(key)
assert len(allcells)==144 and len(set(map(tuple,allcells)))==144
print(json.dumps({'id':'MAND-SUBGRID-2026-11-07-A','method':'Gaussian-rational interval Horner on z6(c0+delta)','domain':'9 frozen parents x 4x4 closed subdivision; denominator 4000','eligible':144,'accepted':ok,'accepted_count':len(ok),'unaccepted_count':144-len(ok),'controls':['c=0 negative','c=c0 positive']},sort_keys=True))
